# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import msgspec
import torch
import zmq
from zmq import asyncio as zmq_asyncio

from ..utils.logging import get_connector_logger
from ..utils.serialization import OmniSerializer
from .base import OmniConnectorBase

logger = get_connector_logger(__name__)

try:
    from mooncake.engine import TransferEngine
except ImportError:
    TransferEngine = None
    logger.warning("Mooncake not available, MooncakeRDMAConnector will not work")

# ZMQ Message constants
TRANS_DONE = b"trans_done"
TRANS_ERROR = b"trans_error"


@dataclass
class MooncakeAgentMetadata:
    """
    Metadata exchanged via ZMQ Handshake.
    """

    remote_hostname: str
    remote_port: int  # RDMA Port
    request_id: str
    dst_addrs: list[int]
    lengths: list[int]


class BufferAllocator:
    """
    Manages the allocation of memory segments within the registered pool.
    Thread-safe implementation using a simple free list.
    """

    def __init__(self, total_size: int, alignment: int = 4096):
        self.total_size = total_size
        self.alignment = alignment
        self.lock = threading.Lock()
        # Free list: [(start, size), ...] sorted by start
        self.free_blocks = [(0, total_size)]

    def alloc(self, size: int) -> int:
        """
        Allocates a block of 'size' bytes.
        Returns the starting offset.
        """
        # Align size upwards
        aligned_size = (size + self.alignment - 1) // self.alignment * self.alignment

        with self.lock:
            for i, (start, block_size) in enumerate(self.free_blocks):
                if block_size >= aligned_size:
                    # Found a block
                    new_start = start + aligned_size
                    new_size = block_size - aligned_size

                    if new_size > 0:
                        self.free_blocks[i] = (new_start, new_size)
                    else:
                        self.free_blocks.pop(i)
                    return start

        raise MemoryError(f"Out of memory in buffer pool. Requested {size} bytes (aligned {aligned_size}).")

    def free(self, offset: int, size: int):
        """
        Frees a previously allocated block.
        """
        aligned_size = (size + self.alignment - 1) // self.alignment * self.alignment

        with self.lock:
            # Check for double-free and corruption
            for start, length in self.free_blocks:
                # Case 1: Exact match = double free, safe to ignore
                if offset == start and aligned_size == length:
                    logger.warning(f"Double free detected at offset {offset}, size {aligned_size}. Ignoring.")
                    return
                # Case 2: Block is fully contained within an existing free block = also double free
                # This happens when the block was freed and then merged with adjacent blocks
                if offset >= start and offset + aligned_size <= start + length:
                    logger.warning(
                        f"Double free detected: block {offset}-{offset + aligned_size} "
                        f"is already within free block {start}-{start + length}. Ignoring."
                    )
                    return
                # Case 3: Partial overlap (but not fully contained) = memory corruption
                if not (offset + aligned_size <= start or start + length <= offset):
                    raise RuntimeError(
                        f"Memory corruption detected: freeing {offset}-{offset + aligned_size} "
                        f"partially overlaps with free block {start}-{start + length}"
                    )

            self.free_blocks.append((offset, aligned_size))
            self.free_blocks.sort()  # Sort by offset

            # Merge adjacent blocks
            i = 0
            while i < len(self.free_blocks) - 1:
                curr_start, curr_size = self.free_blocks[i]
                next_start, next_size = self.free_blocks[i + 1]

                if curr_start + curr_size == next_start:
                    self.free_blocks[i] = (curr_start, curr_size + next_size)
                    self.free_blocks.pop(i + 1)
                else:
                    i += 1


class ManagedBuffer:
    """
    A temporary view into the global memory pool.
    Must be kept alive while the data view is being used.
    """

    def __init__(self, allocator: BufferAllocator, offset: int, size: int, pool_tensor: torch.Tensor):
        self.allocator = allocator
        self.offset = offset
        self.size = size
        self.pool_tensor = pool_tensor
        self._released = False

    def release(self):
        """Explicitly release the buffer back to the pool."""
        if not self._released:
            self.allocator.free(self.offset, self.size)
            self._released = True

    def __del__(self):
        self.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    @property
    def tensor(self) -> torch.Tensor:
        """
        Returns a 1D uint8 zero-copy view of the buffer.
        """
        return self.pool_tensor[self.offset : self.offset + self.size]

    def as_tensor(self, dtype: torch.dtype, shape: tuple) -> torch.Tensor:
        """
        Returns a typed, shaped zero-copy view.
        Validates size, shape, and alignment.
        """
        itemsize = torch.tensor([], dtype=dtype).element_size()

        # Calculate expected size
        expected_bytes = itemsize
        for dim in shape:
            if dim < 0:
                raise ValueError("Dynamic dimension (-1) is not supported in as_tensor")
            expected_bytes *= dim

        if expected_bytes != self.size:
            raise ValueError(
                f"Shape {shape} with dtype {dtype} requires {expected_bytes} bytes, but buffer size is {self.size}"
            )

        # Check alignment (offset must be divisible by itemsize)
        if self.offset % itemsize != 0:
            raise RuntimeError(f"Buffer offset {self.offset} is not aligned for dtype {dtype} (itemsize {itemsize})")

        raw_view = self.tensor
        # view() requires contiguous memory, slice of contiguous tensor is contiguous
        typed_view = raw_view.view(dtype)
        return typed_view.reshape(shape)

    def to_bytes(self) -> bytes:
        """
        Returns a copy of the data as python bytes.
        Performs D2H copy if pool is on GPU.
        """
        t = self.tensor
        if t.is_cuda:
            t = t.cpu()
        return t.numpy().tobytes()


class MooncakeRDMAConnector(OmniConnectorBase):
    """
    OmniConnector implementation using Mooncake RDMA transfer engine with a managed memory pool.
    Supports both CPU (Pinned) and GPU memory pools.
    """

    def __init__(self, config: dict[str, Any]):
        if TransferEngine is None:
            raise ImportError("Mooncake not available")

        self.config = config
        self.host = config.get("host", "127.0.0.1")
        self.zmq_port = config.get("zmq_port", 50051)
        self.protocol = config.get("protocol", "rdma")

        # --- RDMA Device Configuration ---
        # Specify device names to filter (comma-separated), or empty for all devices.
        # Example: "mlx5_0,mlx5_1" to use only specific NICs.
        # This is important for environments with mixed InfiniBand/RoCE NICs.
        self.device_name = config.get("device_name", "")

        # --- Memory Pool Configuration ---
        self.pool_size = config.get("memory_pool_size", 1024**3)  # Default 1GB
        self.pool_device = config.get("memory_pool_device", "cpu")

        self.engine_id = str(uuid.uuid4())

        # --- Mooncake Engine Init ---
        self.engine = TransferEngine()
        # Note: For P2P handshake mode, local_hostname should be just the IP address.
        # Mooncake will auto-assign an RPC port, retrievable via get_rpc_port().
        ret = self.engine.initialize(self.host, "P2PHANDSHAKE", self.protocol, self.device_name)
        if ret != 0:
            raise RuntimeError(f"Mooncake Engine initialization failed with code {ret}")

        self.rpc_port = self.engine.get_rpc_port()
        logger.info(f"MooncakeRDMAConnector initialized at {self.host}:{self.rpc_port}")

        # --- Pool Allocation & Registration ---
        logger.info(f"Allocating RDMA Memory Pool: {self.pool_size / 1024**2:.2f} MB on {self.pool_device}")
        try:
            if self.pool_device == "cpu":
                self.pool = torch.empty(self.pool_size, dtype=torch.uint8).pin_memory()
                self.base_ptr = self.pool.data_ptr()
            else:
                self.pool = torch.empty(self.pool_size, dtype=torch.uint8, device=self.pool_device)
                self.base_ptr = self.pool.data_ptr()

            # Register the entire pool
            ret = self.engine.register_memory(self.base_ptr, self.pool_size)
            if ret != 0:
                raise RuntimeError("Failed to register memory pool with Mooncake Engine")

        except Exception as e:
            logger.error(f"Failed to allocate/register memory pool: {e}")
            raise

        self.allocator = BufferAllocator(self.pool_size, alignment=4096)  # 4KB alignment for safety

        # --- State Management ---
        self.zmq_ctx = zmq.Context()
        self.async_zmq_ctx = zmq_asyncio.Context()

        # Producer buffers: {req_id: (src_addrs, lengths, holder)}
        # 'holder' keeps the object alive (ManagedBuffer or original Tensor)
        self._local_buffers: dict[str, Any] = {}
        self._local_buffers_lock = threading.Lock()

        # --- Background Threads ---
        self._sender_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="mooncake-sender")
        self._stop_event = threading.Event()
        self._listener_thread = threading.Thread(target=self._zmq_listener_loop, daemon=True)
        self._listener_thread.start()

    def put(
        self, from_stage: str, to_stage: str, request_id: str, data: Any
    ) -> tuple[bool, int, dict[str, Any] | None]:
        """
        Producer Side.
        Exposes data for RDMA transfer.
        Tries to use Zero-Copy if data is ManagedBuffer, otherwise copies to pool.
        """
        if getattr(self, "_closed", False):
            raise RuntimeError("Cannot put data: MooncakeRDMAConnector is closed")

        try:
            src_addr = 0
            size = 0
            holder = None
            is_fast_path = False
            should_release_holder = False
            is_bytes_payload = False

            if isinstance(data, ManagedBuffer):
                # Zero-Copy Path
                # Validate that the buffer belongs to this connector's pool or is compatible
                # Since we register self.pool base_ptr, we can only safely use offsets relative to it
                # if the buffer is from the same pool.
                if data.pool_tensor.data_ptr() != self.pool.data_ptr():
                    # Fallback to copy path if buffer is from a different pool/device context
                    logger.warning("ManagedBuffer from different pool detected. Falling back to copy path.")
                    # Ensure contiguous before fallback put
                    return self.put(from_stage, to_stage, request_id, data.tensor.contiguous())

                src_addr = self.base_ptr + data.offset
                size = data.size
                holder = data  # Keep the buffer alive
                should_release_holder = False  # Caller owns it
                is_fast_path = True

            elif isinstance(data, (torch.Tensor, bytes)):
                # Copy Path
                # 1. Determine size
                if isinstance(data, torch.Tensor):
                    size = data.nbytes
                    tensor_data = data
                else:
                    size = len(data)
                    # Convert bytes to tensor for copy
                    tensor_data = torch.frombuffer(data, dtype=torch.uint8)

                # 2. Alloc from pool
                try:
                    offset = self.allocator.alloc(size)
                    holder = ManagedBuffer(self.allocator, offset, size, self.pool)
                    should_release_holder = True  # We created it, we release it
                except MemoryError:
                    logger.error(f"Pool exhausted, cannot put data size {size}")
                    return False, 0, None

                # 3. Copy data to pool
                # Handle device mismatch for copy
                try:
                    dst_tensor = holder.tensor
                    if isinstance(data, torch.Tensor):
                        if not data.is_contiguous():
                            data = data.contiguous()

                        # View as flat uint8
                        src_view = data.view(torch.uint8).flatten()
                        if src_view.device != dst_tensor.device:
                            dst_tensor.copy_(src_view, non_blocking=True)
                            # Ensure copy is complete before exposing buffer to RDMA
                            # Must sync on SOURCE device for D2H copies, or DST for H2D
                            if src_view.is_cuda:
                                with torch.cuda.device(src_view.device):
                                    torch.cuda.current_stream().synchronize()
                            elif dst_tensor.is_cuda:
                                with torch.cuda.device(dst_tensor.device):
                                    torch.cuda.current_stream().synchronize()
                        else:
                            dst_tensor.copy_(src_view)
                            if dst_tensor.is_cuda:
                                with torch.cuda.device(dst_tensor.device):
                                    torch.cuda.current_stream().synchronize()
                    else:
                        # bytes -> tensor copy
                        # torch.frombuffer creates CPU tensor. If pool is GPU, copy_ handles H2D.
                        dst_tensor.copy_(tensor_data)
                        if dst_tensor.is_cuda:
                            with torch.cuda.device(dst_tensor.device):
                                torch.cuda.current_stream().synchronize()
                except Exception as e:
                    # Copy failed, release the allocated buffer to prevent leak
                    holder.release()
                    logger.error(f"Failed to copy data to pool: {e}")
                    return False, 0, None

                src_addr = self.base_ptr + offset
                is_fast_path = True  # Raw transfer, no deserialization needed on receiver side

                # Special handling: if input was bytes, receiver might expect bytes back, not ManagedBuffer.
                is_bytes_payload = isinstance(data, bytes)
            else:
                # Fallback: Serialize object and put as bytes
                # This incurs double copy (serialize -> bytes -> pool)
                # Only serialized objects set is_fast_path=False
                serialized = OmniSerializer.serialize(data)
                # Recursively call put with bytes data
                # We need to manually handle the recursive return to set is_fast_path correctly in metadata
                success, size, meta = self.put(from_stage, to_stage, request_id, serialized)
                if success and meta:
                    meta["is_fast_path"] = False  # Override to indicate deserialization needed
                return success, size, meta

            with self._local_buffers_lock:
                self._local_buffers[request_id] = ([src_addr], [size], holder, should_release_holder)

            # Metadata for Consumer
            metadata = {
                "source_host": self.host,
                "source_port": self.zmq_port,
                "data_size": size,
                "is_fast_path": is_fast_path,  # Hint if receiver can skip deserialize
                "is_bytes": is_bytes_payload,  # Hint to return bytes
            }

            return True, size, metadata

        except Exception as e:
            logger.error(f"RDMA Put failed for {request_id}: {e}", exc_info=True)
            return False, 0, None

    def get(
        self,
        from_stage: str,
        to_stage: str,
        request_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Any, int] | None:
        """
        Consumer Side.
        Allocates from local pool and pulls data via RDMA.

        Returns:
            Tuple[Any, int] | None:
            - If metadata['is_fast_path'] is True and not bytes: Returns (ManagedBuffer, size).
              **CALLER MUST RELEASE ManagedBuffer**.
            - Otherwise: Returns (DeserializedObject|bytes, size). Buffer auto-released.
        """
        if getattr(self, "_closed", False):
            raise RuntimeError("Cannot get data: MooncakeRDMAConnector is closed")

        if not metadata:
            return None

        src_host = metadata.get("source_host")
        src_port = metadata.get("source_port")
        data_size = metadata.get("data_size", 0)
        is_fast_path = metadata.get("is_fast_path", False)
        is_bytes = metadata.get("is_bytes", False)

        if data_size == 0:
            return None, 0

        # 1. Allocate Destination Buffer from Pool
        try:
            offset = self.allocator.alloc(data_size)
            recv_buffer = ManagedBuffer(self.allocator, offset, data_size, self.pool)
            dst_ptr = self.base_ptr + offset
        except MemoryError:
            logger.error(f"Failed to allocate {data_size} bytes in receive pool")
            return None

        # 2. Prepare Handshake
        agent_meta = MooncakeAgentMetadata(
            remote_hostname=self.host,
            remote_port=self.rpc_port,
            request_id=request_id,
            dst_addrs=[dst_ptr],
            lengths=[data_size],
        )

        # 3. ZMQ Transaction
        zmq_addr = f"tcp://{src_host}:{src_port}"
        req_socket = self.zmq_ctx.socket(zmq.REQ)
        req_socket.setsockopt(zmq.RCVTIMEO, 10000)  # 10s timeout for RDMA transfers
        req_socket.connect(zmq_addr)

        try:
            req_socket.send(msgspec.msgpack.encode(agent_meta))
            resp = req_socket.recv()

            if resp == TRANS_DONE:
                # Success
                # Ensure data is visible on GPU
                # Note: RDMA write visibility on GPU usually requires some form of fence/sync.
                # torch.cuda.synchronize() is a heavy hammer but safe.
                # Ideally Mooncake engine provides a way to poll for completion that guarantees visibility.
                # TODO(wzliu): Replace synchronize with cuda event in the future for better performance.
                if self.pool.is_cuda:
                    with torch.cuda.device(self.pool.device):
                        torch.cuda.current_stream().synchronize()

                if is_fast_path:
                    if is_bytes:
                        # Return as bytes (copy) and release buffer
                        try:
                            return recv_buffer.to_bytes(), data_size
                        finally:
                            recv_buffer.release()
                    else:
                        # If sender said it was a raw transfer (ManagedBuffer or
                        # Tensor), return the ManagedBuffer directly.
                        return recv_buffer, data_size
                else:
                    # If it was a serialized object or generic bytes, we assume standard Omni behavior:
                    # Deserialize and return object. This requires a copy (to_bytes).
                    # We MUST release the buffer after deserialization.
                    try:
                        val = OmniSerializer.deserialize(recv_buffer.to_bytes())
                        return val, data_size
                    finally:
                        recv_buffer.release()
            else:
                logger.error(f"RDMA Get failed: received {resp} instead of TRANS_DONE")
                recv_buffer.release()
                return None
        except Exception as e:
            logger.error(f"RDMA Get error: {e}", exc_info=True)
            recv_buffer.release()
            return None
        finally:
            req_socket.close()

    def cleanup(self, request_id: str) -> None:
        """Release the producer-side buffer associated with the request."""
        with self._local_buffers_lock:
            item = self._local_buffers.pop(request_id, None)
            if item:
                # item is (src_addrs, lengths, holder, should_release)
                _, _, holder, should_release = item
                if should_release and isinstance(holder, ManagedBuffer):
                    # We own this buffer (created internally), so we must release it.
                    holder.release()
                # If holder was externally owned (should_release=False), we do nothing.
                # If holder was something else (e.g. Tensor), GC handles it.

    def health(self) -> dict[str, Any]:
        return {
            "status": "healthy",
            "protocol": self.protocol,
            "pool_device": self.pool_device,
            "pool_size": self.pool_size,
        }

    def close(self) -> None:
        """
        Gracefully shutdown the connector and release all resources.
        This method should be called when the connector is no longer needed.
        Idempotent: safe to call multiple times.
        """
        # Idempotent guard
        if getattr(self, "_closed", False):
            return
        self._closed = True

        logger.info("Closing MooncakeRDMAConnector...")

        # 1. Signal listener thread to stop
        self._stop_event.set()

        # 2. Wait for listener thread to finish
        if self._listener_thread.is_alive():
            self._listener_thread.join(timeout=2.0)
            if self._listener_thread.is_alive():
                logger.warning("Listener thread did not stop gracefully")

        # 3. Shutdown sender executor
        self._sender_executor.shutdown(wait=True, cancel_futures=False)

        # 4. Release all pending buffers
        with self._local_buffers_lock:
            for req_id, item in list(self._local_buffers.items()):
                _, _, holder, should_release = item
                if should_release and isinstance(holder, ManagedBuffer):
                    holder.release()
            self._local_buffers.clear()

        # 5. Unregister memory from engine (if supported)
        try:
            if hasattr(self, "engine") and hasattr(self.engine, "unregister_memory"):
                # Mooncake API only takes address, not size
                self.engine.unregister_memory(self.base_ptr)
        except Exception as e:
            logger.warning(f"Failed to unregister memory: {e}")

        # 6. Close ZMQ contexts
        try:
            if hasattr(self, "zmq_ctx"):
                self.zmq_ctx.term()
            if hasattr(self, "async_zmq_ctx"):
                self.async_zmq_ctx.term()
        except Exception as e:
            logger.warning(f"Failed to terminate ZMQ contexts: {e}")

        # 7. Release pool tensor reference (let GC handle actual deallocation)
        # Note: We set to None instead of del to avoid AttributeError on repeated access
        self.pool = None

        logger.info("MooncakeRDMAConnector closed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # -------------------------------------------------------
    # Listener Logic (Unchanged mostly)
    # -------------------------------------------------------

    def _zmq_listener_loop(self):
        socket = self.zmq_ctx.socket(zmq.ROUTER)
        try:
            socket.bind(f"tcp://{self.host}:{self.zmq_port}")
        except zmq.ZMQError:
            logger.error(f"Failed to bind ZMQ listener on {self.host}:{self.zmq_port}")
            return

        # Create inproc socket pair for worker thread notifications
        # This allows workers to wake up the listener immediately when done
        notify_addr = f"inproc://notify-{id(self)}"
        notify_recv = self.zmq_ctx.socket(zmq.PULL)
        notify_recv.bind(notify_addr)

        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        poller.register(notify_recv, zmq.POLLIN)

        # Response queue for thread-safe socket operations
        response_queue: queue.Queue = queue.Queue()

        try:
            while not self._stop_event.is_set():
                try:
                    # Poll for incoming requests or notifications (with timeout)
                    events = dict(poller.poll(1000))

                    # Process notifications (drain all)
                    if notify_recv in events:
                        while True:
                            try:
                                notify_recv.recv(zmq.NOBLOCK)
                            except zmq.Again:
                                break

                    # Process any pending responses (non-blocking)
                    while True:
                        try:
                            identity, response = response_queue.get_nowait()
                            socket.send_multipart([identity, b"", response])
                        except queue.Empty:
                            break

                    # Process incoming requests
                    if socket in events:
                        frames = socket.recv_multipart()
                        if len(frames) >= 2:
                            # Submit to thread pool
                            self._sender_executor.submit(
                                self._handle_pull_request,
                                response_queue,
                                notify_addr,
                                frames[0],
                                frames[-1],
                            )
                except zmq.ContextTerminated:
                    break
                except Exception:
                    pass
        finally:
            try:
                notify_recv.close(linger=0)
                socket.close(linger=0)
            except Exception:
                pass

    def _handle_pull_request(
        self, response_queue: queue.Queue, notify_addr: str, identity, payload
    ):
        """
        Handle pull request in worker thread.
        Results are put into response_queue and listener is notified via inproc.
        """
        try:
            meta = msgspec.msgpack.decode(payload, type=MooncakeAgentMetadata)

            with self._local_buffers_lock:
                item = self._local_buffers.get(meta.request_id)

            if not item:
                response_queue.put((identity, TRANS_ERROR))
                self._notify_listener(notify_addr)
                return

            src_addrs, src_lengths, _, _ = item
            remote_session = f"{meta.remote_hostname}:{meta.remote_port}"

            # RDMA Write
            ret = self.engine.batch_transfer_sync_write(
                remote_session, src_addrs, meta.dst_addrs, src_lengths
            )

            if ret == 0:
                self.cleanup(meta.request_id)
                response_queue.put((identity, TRANS_DONE))
            else:
                response_queue.put((identity, TRANS_ERROR))

        except Exception as e:
            logger.error(f"Push failed: {e}")
            response_queue.put((identity, TRANS_ERROR))

        # Notify listener thread that response is ready
        self._notify_listener(notify_addr)

    def _notify_listener(self, notify_addr: str):
        """Send notification to wake up listener thread."""
        try:
            notify_socket = self.zmq_ctx.socket(zmq.PUSH)
            notify_socket.connect(notify_addr)
            notify_socket.send(b"", zmq.NOBLOCK)
            notify_socket.close(linger=0)
        except Exception:
            pass
