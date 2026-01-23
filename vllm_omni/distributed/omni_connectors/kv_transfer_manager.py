# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified OmniConnector and KV cache transfer management."""

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import torch
from vllm.logger import init_logger

from .factory import OmniConnectorFactory
from .utils.config import ConnectorSpec

logger = init_logger(__name__)


@dataclass
class OmniKVCacheConfig:
    """Configuration for OmniKVTransferManager."""

    connector_config: dict[str, Any] | None = None
    from_stage: str | None = None
    to_stage: str | None = None
    stage_id: str | int | None = None
    engine_input_source: list[str | int] | None = None
    need_recv_cache: bool = False
    need_send_cache: bool = False
    recv_timeout: float = 30.0


@dataclass
class KVCacheTransferData:
    """Container for KV cache transfer data."""

    request_id: str
    layer_blocks: dict[str, Any]
    block_ids: list[int]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def to_bytes(self) -> bytes:
        """Convert to compact binary format for fast transfer.

        Format: [4-byte header_length][JSON header][tensor data]

        The header contains metadata and tensor layout descriptors.
        Tensor data is a flat concatenation of raw tensor bytes.
        On the receiver side, tensors can be reconstructed with
        torch.frombuffer() for near-zero-copy deserialization.
        """
        import json
        import struct

        tensors_desc: list[dict[str, Any]] = []
        tensor_bufs: list[bytes] = []
        data_offset = 0

        for cache_name in ("key_cache", "value_cache"):
            cache_list = self.layer_blocks.get(cache_name, [])
            for layer_idx, tensor in enumerate(cache_list):
                if tensor is None:
                    tensors_desc.append({"n": f"{cache_name}_{layer_idx}", "x": True})
                    continue

                t = tensor.detach().cpu().contiguous()
                dtype_str = str(t.dtype).removeprefix("torch.")
                shape = list(t.shape)

                # View as uint8 then use numpy tobytes for reliable byte extraction
                raw = t.view(torch.uint8).numpy().tobytes()

                tensors_desc.append(
                    {
                        "n": f"{cache_name}_{layer_idx}",
                        "d": dtype_str,
                        "s": shape,
                        "o": data_offset,
                        "b": len(raw),
                    }
                )
                tensor_bufs.append(raw)
                data_offset += len(raw)

        header = json.dumps(
            {
                "rid": self.request_id,
                "bids": self.block_ids,
                "meta": self.metadata,
                "td": tensors_desc,
                "nl": len(self.layer_blocks.get("key_cache", [])),
            },
            separators=(",", ":"),
        ).encode("utf-8")

        # Assemble: [4-byte header_length][JSON header][tensor data...]
        return b"".join([struct.pack(">I", len(header)), header] + tensor_bufs)

    def to_gpu_tensor(self) -> torch.Tensor:
        """Convert to a GPU tensor in the same binary format as to_bytes().

        Instead of GPU→CPU copy per tensor (~90ms for 115MB), this keeps
        all tensor data on GPU and assembles the buffer via GPU→GPU copies.
        Only the small JSON header (~few KB) is built on CPU.

        The returned tensor can be passed directly to connector.put(),
        which will copy it to the RDMA pool (GPU→GPU if pool is on GPU).
        """
        import json
        import struct

        tensors_desc: list[dict[str, Any]] = []
        gpu_tensors: list[torch.Tensor] = []
        data_offset = 0
        device = None

        for cache_name in ("key_cache", "value_cache"):
            cache_list = self.layer_blocks.get(cache_name, [])
            for layer_idx, tensor in enumerate(cache_list):
                if tensor is None:
                    tensors_desc.append({"n": f"{cache_name}_{layer_idx}", "x": True})
                    continue

                t = tensor.detach().contiguous()
                if device is None and t.is_cuda:
                    device = t.device
                dtype_str = str(t.dtype).removeprefix("torch.")
                shape = list(t.shape)
                nbytes = t.numel() * t.element_size()

                tensors_desc.append(
                    {
                        "n": f"{cache_name}_{layer_idx}",
                        "d": dtype_str,
                        "s": shape,
                        "o": data_offset,
                        "b": nbytes,
                    }
                )
                gpu_tensors.append(t.view(torch.uint8).flatten())
                data_offset += nbytes

        # If no GPU tensors found, fall back to CPU to_bytes
        if device is None:
            raise RuntimeError("No CUDA tensors found, use to_bytes() instead")

        header = json.dumps(
            {
                "rid": self.request_id,
                "bids": self.block_ids,
                "meta": self.metadata,
                "td": tensors_desc,
                "nl": len(self.layer_blocks.get("key_cache", [])),
            },
            separators=(",", ":"),
        ).encode("utf-8")

        header_prefix = struct.pack(">I", len(header)) + header
        total_size = len(header_prefix) + data_offset

        # Allocate output buffer on GPU
        output = torch.empty(total_size, dtype=torch.uint8, device=device)

        # Copy header to GPU (tiny, a few KB)
        header_tensor = torch.frombuffer(bytearray(header_prefix), dtype=torch.uint8)
        output[: len(header_prefix)].copy_(header_tensor)

        # Copy each tensor directly on GPU (GPU→GPU, ~0.1ms for 115MB)
        pos = len(header_prefix)
        for t_flat in gpu_tensors:
            n = t_flat.numel()
            output[pos : pos + n].copy_(t_flat)
            pos += n

        return output

    @staticmethod
    def from_bytes(raw: "bytes | bytearray | memoryview") -> dict[str, Any]:
        """Reconstruct KV cache data dict from compact binary format.

        Uses memoryview + torch.frombuffer for true zero-copy tensor
        reconstruction.  The returned tensors are read-only views into
        the original *raw* buffer – no large memory copies are made.
        Since the tensors are typically moved to GPU immediately after
        this call, the read-only restriction is harmless.
        """
        import json
        import struct

        # 1. Wrap in memoryview for zero-copy slicing
        raw_mv = memoryview(raw) if not isinstance(raw, memoryview) else raw

        # 2. Parse header (tiny copy for JSON string only)
        header_len = struct.unpack(">I", raw_mv[:4])[0]
        header = json.loads(bytes(raw_mv[4 : 4 + header_len]))
        data_start = 4 + header_len

        # 3. Zero-copy view into the tensor data region
        #    No bytearray() copy – tensors will be non-writable but that is
        #    fine because they are moved to GPU via .to(device) right after.
        tensor_data_mv = raw_mv[data_start:]

        # 4. Reconstruct tensors via zero-copy views into the shared buffer
        num_layers = header["nl"]
        key_cache: list[torch.Tensor | None] = [None] * num_layers
        value_cache: list[torch.Tensor | None] = [None] * num_layers

        for info in header["td"]:
            if info.get("x"):
                continue

            name: str = info["n"]
            dtype_str: str = info["d"]
            shape: list[int] = info["s"]
            offset: int = info["o"]
            nbytes: int = info["b"]

            torch_dtype = getattr(torch, dtype_str)

            # True zero-copy: frombuffer from memoryview (no data copied)
            t = torch.frombuffer(tensor_data_mv, dtype=torch.uint8, offset=offset, count=nbytes)
            t = t.view(torch_dtype).reshape(shape)

            # Parse layer index from name (e.g. "key_cache_3" -> 3)
            layer_idx = int(name.split("_")[-1])
            if name.startswith("key_cache_"):
                key_cache[layer_idx] = t
            elif name.startswith("value_cache_"):
                value_cache[layer_idx] = t

        return {
            "request_id": header["rid"],
            "layer_blocks": {"key_cache": key_cache, "value_cache": value_cache},
            "block_ids": header["bids"],
            "metadata": header["meta"],
        }

    @staticmethod
    def from_bytes_gpu(gpu_tensor: torch.Tensor) -> dict[str, Any]:
        """Reconstruct KV cache data dict from a GPU tensor in binary format.

        Optimized for GPUDirect RDMA: only the small header (a few KB) is
        copied to CPU for JSON parsing.  The ~115 MB tensor data stays on
        GPU via direct slicing + clone – **no GPU-to-CPU bulk copy**.

        Each tensor slice is `.clone()`-d so that:
        1. storage_offset is reset to 0 (required for `.view()` alignment),
        2. the returned tensors are independent of the RDMA buffer, so the
           caller can release the buffer immediately after this call.
        """
        import json
        import struct

        # 1. Copy only the 4-byte length prefix to CPU
        header_len = struct.unpack(">I", gpu_tensor[:4].cpu().numpy().tobytes())[0]

        # 2. Copy only the JSON header to CPU (typically a few KB)
        header_bytes = gpu_tensor[4 : 4 + header_len].cpu().numpy().tobytes()
        header = json.loads(header_bytes)
        data_start = 4 + header_len

        # 3. Reconstruct tensors via GPU slicing + clone
        #    clone() is needed because the slice's storage_offset may not
        #    be aligned to the target dtype's element size (e.g. bfloat16
        #    requires offset divisible by 2).  clone() resets offset to 0
        #    and also detaches the tensor from the RDMA pool buffer.
        num_layers = header["nl"]
        key_cache: list[torch.Tensor | None] = [None] * num_layers
        value_cache: list[torch.Tensor | None] = [None] * num_layers

        for info in header["td"]:
            if info.get("x"):
                continue

            name: str = info["n"]
            dtype_str: str = info["d"]
            shape: list[int] = info["s"]
            offset: int = info["o"]
            nbytes: int = info["b"]

            torch_dtype = getattr(torch, dtype_str)

            # Slice on GPU, clone to fix alignment & detach from pool
            t = gpu_tensor[data_start + offset : data_start + offset + nbytes].clone()
            t = t.view(torch_dtype).reshape(shape)

            layer_idx = int(name.split("_")[-1])
            if name.startswith("key_cache_"):
                key_cache[layer_idx] = t
            elif name.startswith("value_cache_"):
                value_cache[layer_idx] = t

        return {
            "request_id": header["rid"],
            "layer_blocks": {"key_cache": key_cache, "value_cache": value_cache},
            "block_ids": header["bids"],
            "metadata": header["meta"],
        }


class OmniKVTransferManager:
    """Unified management for OmniConnector and KV cache transfer.

    This class encapsulates all KV cache related operations:
    - Connector initialization and lazy creation
    - KV cache extraction from GPU blocks
    - KV cache transfer with retry logic
    - KV cache receiving with timeout
    """

    def __init__(self, config: OmniKVCacheConfig):
        self.config = config
        self._connector = None

        # Pre-calculate send stages (from_stage, to_stage)
        self.send_stages = (
            (str(config.from_stage), str(config.to_stage)) if config.from_stage and config.to_stage else (None, None)
        )

        # Pre-calculate receive stages (from_stage, to_stage)
        recv_from = config.from_stage
        if config.engine_input_source:
            recv_from = config.engine_input_source[0]
        elif isinstance(config.stage_id, int):
            recv_from = config.stage_id - 1

        self.recv_stages = (
            (str(recv_from), str(config.stage_id))
            if recv_from is not None and config.stage_id is not None
            else (None, None)
        )

        # For sender mode, eagerly initialize connector so sender info is available early
        # This allows the info to be passed to receiver before it needs to connect
        if config.need_send_cache and config.connector_config:
            try:
                _ = self.connector  # Trigger lazy init
                logger.info("Sender connector eagerly initialized")
            except Exception as e:
                logger.warning(f"Failed to eagerly initialize sender connector: {e}")

    @classmethod
    def _create(cls, cfg: dict | None) -> "OmniKVTransferManager":
        """Create manager from raw config dict."""
        if not cfg or not isinstance(cfg, dict):
            return cls(OmniKVCacheConfig())
        return cls(
            OmniKVCacheConfig(
                connector_config=cfg.get("connector_config"),
                from_stage=cfg.get("omni_from_stage"),
                to_stage=cfg.get("omni_to_stage"),
                stage_id=cfg.get("stage_id"),
                engine_input_source=cfg.get("engine_input_source", []),
                need_recv_cache=cfg.get("need_recv_cache", False),
                need_send_cache=cfg.get("need_send_cache", False),
                recv_timeout=cfg.get("recv_timeout", 30.0),
            )
        )

    @classmethod
    def from_model_config(cls, config: Any) -> "OmniKVTransferManager":
        """Create from model config (for AR model runner)."""
        return cls._create(getattr(config, "omni_kv_config", None))

    @classmethod
    def from_od_config(cls, config: Any) -> "OmniKVTransferManager":
        """Create from OmniDiffusion config (for diffusion runner)."""
        return cls._create(getattr(config, "omni_kv_config", None))

    @classmethod
    def from_vllm_config(cls, vllm_config: Any, model_config: Any) -> "OmniKVTransferManager":
        """Create from vllm config with fallback to kv_transfer_config."""
        # Primary: omni_kv_config from model_config
        omni_kv = getattr(model_config, "omni_kv_config", None)
        if isinstance(omni_kv, dict):
            return cls._create(omni_kv)

        # Fallback: check kv_transfer_config
        kv_cfg = getattr(vllm_config, "kv_transfer_config", None)
        if kv_cfg:
            direct = getattr(kv_cfg, "omni_connector_config", None)
            if isinstance(direct, dict) and direct:
                return cls._create({"connector_config": direct})
            extra = getattr(kv_cfg, "kv_connector_extra_config", None)
            if isinstance(extra, dict):
                omni = extra.get("omni_connector_config")
                if isinstance(omni, dict) and omni:
                    return cls._create({"connector_config": omni})

        return cls(OmniKVCacheConfig())

    @property
    def connector(self):
        """Lazy initialization of connector."""
        # If a previous initialization attempt failed, don't retry on every access.
        if self._connector is False:
            return None

        if self._connector is None:
            cfg = self.config.connector_config
            if cfg and (c_type := cfg.get("type")):
                try:
                    c_extra = {k: v for k, v in cfg.items() if k != "type"}

                    # Auto-set role and port based on need_send_cache/need_recv_cache for RDMA connector
                    # Port offset strategy:
                    #   - request_forwarding: base_port + 0 + from_stage_id
                    #   - kv_transfer: base_port + 100 + from_stage_id
                    # This allows orchestrator/stage main processes and workers to each have
                    # their own RDMA connections without port conflicts.
                    KV_TRANSFER_PORT_OFFSET = 100

                    if c_type == "MooncakeRDMAConnector" and c_extra.get("role") == "auto":
                        base_port = c_extra.get("zmq_port", 50051)

                        # Pass stage info for dynamic sender discovery
                        c_extra["from_stage"] = (
                            str(self.config.from_stage) if self.config.from_stage is not None else "0"
                        )
                        c_extra["to_stage"] = str(self.config.to_stage) if self.config.to_stage is not None else "1"

                        if self.config.need_send_cache:
                            c_extra["role"] = "sender"
                            # Sender port = base_port + KV_TRANSFER_PORT_OFFSET + from_stage_id
                            from_stage = self.config.from_stage
                            if from_stage is not None:
                                try:
                                    c_extra["zmq_port"] = base_port + KV_TRANSFER_PORT_OFFSET + int(from_stage)
                                except (ValueError, TypeError):
                                    c_extra["zmq_port"] = base_port + KV_TRANSFER_PORT_OFFSET

                        elif self.config.need_recv_cache:
                            c_extra["role"] = "receiver"
                            # Receiver connects to sender's port = base_port + KV_TRANSFER_PORT_OFFSET + from_stage_id
                            from_stage = self.config.from_stage
                            sender_port = base_port + KV_TRANSFER_PORT_OFFSET
                            if from_stage is not None:
                                try:
                                    sender_port = base_port + KV_TRANSFER_PORT_OFFSET + int(from_stage)
                                except (ValueError, TypeError):
                                    pass

                            if "sender_host" not in c_extra:
                                c_extra["sender_host"] = c_extra.get("host", "127.0.0.1")
                            if "sender_zmq_port" not in c_extra:
                                c_extra["sender_zmq_port"] = sender_port

                    logger.info(
                        f"Initializing OmniConnector (purpose=kv_transfer) "
                        f"with config: {cfg}, role: {c_extra.get('role', 'N/A')}"
                    )
                    self._connector = OmniConnectorFactory.create_connector(ConnectorSpec(name=c_type, extra=c_extra))
                except Exception as e:
                    logger.error(f"Failed to initialize OmniConnector: {e}")
                    import traceback

                    traceback.print_exc()
                    # Cache failure sentinel to avoid repeated initialization attempts in hot paths.
                    self._connector = False

        return self._connector if self._connector else None

    def get_connector(self):
        """Get connector (compatibility wrapper for existing code)."""
        return self.connector

    def get_sender_connection_info(self) -> dict[str, Any] | None:
        """Get sender connector's connection info for passing to receiver.

        Returns:
            Dict with 'host', 'zmq_port', 'rpc_port' if sender connector is initialized,
            None otherwise.
        """
        if not self.config.need_send_cache:
            return None
        conn = self.connector
        if conn and hasattr(conn, "get_connection_info"):
            return conn.get_connection_info()
        return None

    def update_sender_info(self, sender_info: dict[str, Any]) -> None:
        """Update sender connection info for receiver connector.

        This should be called before receive_kv_cache() when sender info
        is dynamically provided (e.g., via task from orchestrator).

        Args:
            sender_info: Dict with 'host' and 'zmq_port' keys, or
                         Dict mapping rank_id to {"host": ..., "zmq_port": ...}
        """
        if not self.config.need_recv_cache:
            return

        # Handle nested format: {rank_id: {"host": ..., "zmq_port": ...}}
        # For TP=1, rank_id=0; for TP>1, use matching rank or first available
        actual_info = sender_info
        if sender_info and "host" not in sender_info:
            # It's the nested format, extract the first rank's info
            # TODO: For TP>1, match receiver rank to sender rank
            for rank_id, info in sender_info.items():
                if isinstance(info, dict) and "host" in info:
                    actual_info = info
                    logger.debug(f"Extracted sender info for rank {rank_id}: {info}")
                    break

        if not actual_info or "host" not in actual_info:
            logger.warning(f"Invalid sender_info format: {sender_info}")
            return

        # Update connector config so new connector uses correct sender info
        if self.config.connector_config:
            self.config.connector_config["sender_host"] = actual_info.get("host")
            self.config.connector_config["sender_zmq_port"] = actual_info.get("zmq_port")
            logger.info(
                f"Updated sender info in config: host={actual_info.get('host')}, zmq_port={actual_info.get('zmq_port')}"
            )

        # If connector already exists, try to update its sender info
        if self._connector and hasattr(self._connector, "sender_host"):
            self._connector.sender_host = actual_info.get("host")
            self._connector.sender_zmq_port = actual_info.get("zmq_port")
            logger.info(
                f"Updated existing connector's sender info: host={actual_info.get('host')}, "
                f"zmq_port={actual_info.get('zmq_port')}"
            )

    def handle_finished_requests_kv_transfer(
        self,
        finished_reqs: dict[str, dict[str, Any]],
        kv_caches: list[torch.Tensor],
        block_size: int,
        cache_dtype: str,
        request_id_resolver: Callable[[str], str] | None = None,
    ) -> list[str]:
        """Handle KV cache transfer for finished requests.

        This method extracts KV cache from GPU blocks and transfers them
        to the downstream stage via the connector.

        Args:
            finished_reqs: Dict mapping request_id to {block_ids, seq_len}
            kv_caches: List of KV cache tensors per layer
            block_size: Size of each cache block
            cache_dtype: Data type of the cache
            request_id_resolver: Optional function to resolve global request ID

        Returns:
            List of request IDs that were processed
        """
        if not finished_reqs:
            return []

        if not self.config.need_send_cache:
            return list(finished_reqs.keys())

        if not self.connector:
            logger.warning("No connector available, skipping KV transfer but freeing resources")
            return list(finished_reqs.keys())

        logger.debug(f"Processing KV transfer for {len(finished_reqs)} requests")

        extracted_ids = []
        for req_id, data in finished_reqs.items():
            try:
                seq_len = data.get("seq_len", 0)
                block_ids = data.get("block_ids", [])
                if not block_ids:
                    logger.warning(f"Request {req_id} has no block IDs, skipping")
                    continue

                # Extract KV cache from GPU blocks -> CPU tensors
                kv_data = self._extract_kv_cache(req_id, block_ids, seq_len, kv_caches, block_size, cache_dtype)
                if kv_data:
                    # Resolve global request ID if available
                    transfer_req_id = request_id_resolver(req_id) if request_id_resolver else req_id

                    # Transfer to downstream stage via connector
                    self._transfer_kv_cache(kv_data, transfer_req_id)

            except Exception as e:
                logger.error(f"Failed KV transfer for {req_id}: {e}")
            finally:
                extracted_ids.append(req_id)

        return extracted_ids

    def _extract_kv_cache(
        self,
        req_id: str,
        block_ids: list[int],
        seq_len: int,
        kv_caches: list[torch.Tensor],
        block_size: int,
        cache_dtype: str,
    ) -> KVCacheTransferData | None:
        """Extract KV cache from GPU blocks for a single request.

        Args:
            req_id: Request identifier
            block_ids: List of block IDs to extract
            seq_len: Sequence length
            kv_caches: List of KV cache tensors per layer
            block_size: Size of each cache block
            cache_dtype: Data type of the cache

        Returns:
            KVCacheTransferData if extraction successful, None otherwise
        """
        num_layers = len(kv_caches)
        key_cache: list[torch.Tensor | None] = [None] * num_layers
        value_cache: list[torch.Tensor | None] = [None] * num_layers

        for layer_idx, kv_tensor in enumerate(kv_caches):
            # Validate block IDs - shape: [2, num_blocks, block_size, n_heads, head_dim]
            max_block = kv_tensor.shape[1] - 1
            valid_ids = [bid for bid in block_ids if 0 <= bid <= max_block]
            if not valid_ids:
                continue

            # Extract and reshape: [2, n_blocks, block_size, n_heads, head_dim]
            # -> [2, seq_len, n_heads, head_dim]
            selected = kv_tensor[:, valid_ids]  # [2, n_valid, block_size, n_heads, head_dim]
            n_kv, n_blks, blk_sz, n_heads, d_head = selected.shape
            flat = selected.reshape(n_kv, n_blks * blk_sz, n_heads, d_head)
            if seq_len < flat.shape[1]:
                flat = flat[:, :seq_len]

            # Keep on original device (GPU) – sender serialization path
            # will handle device placement (GPU direct or CPU fallback)
            flat_gpu = flat.detach().contiguous()
            key_cache[layer_idx] = flat_gpu[0]
            value_cache[layer_idx] = flat_gpu[1]

        if not any(k is not None for k in key_cache):
            return None

        return KVCacheTransferData(
            request_id=req_id,
            layer_blocks={"key_cache": key_cache, "value_cache": value_cache},
            block_ids=block_ids,
            metadata={
                "block_size": block_size,
                "num_layers": num_layers,
                "dtype": str(cache_dtype),
                "seq_len": seq_len,
            },
        )

    def _transfer_kv_cache(self, kv_data: KVCacheTransferData, transfer_req_id: str) -> None:
        """Transfer KV cache data to downstream stage via OmniConnector.

        Args:
            kv_data: The extracted KV cache data
            transfer_req_id: The request ID to use for transfer
        """
        from_stage, to_stage = self.send_stages
        if not from_stage or not to_stage:
            raise ValueError("Transfer stages (omni_from_stage, omni_to_stage) not configured")

        import time as _time

        # Serialization priority depends on connector capability:
        #
        # Connectors with supports_raw_data=True (e.g. MooncakeRDMAConnector):
        #   1. to_gpu_tensor() – keeps data on GPU, ~0.1ms (requires CUDA tensors)
        #   2. to_bytes()      – CPU path, ~90ms (GPU→CPU copy + tobytes)
        #   3. to_dict()       – legacy fallback via OmniSerializer
        #
        # Connectors with supports_raw_data=False (e.g. MooncakeConnector/TCP):
        #   1. to_bytes()      – compact binary, connector serializes via OmniSerializer
        #   2. to_dict()       – legacy fallback
        #
        # Why skip to_gpu_tensor() for non-raw connectors?
        # These connectors call OmniSerializer.serialize() on the data.
        # A torch.Tensor gets serialized as a single opaque tensor blob, and
        # the receiver deserializes it as a raw tensor instead of a KV cache dict.
        kv_data.request_id = transfer_req_id
        _set_start = _time.perf_counter()
        transfer_data: torch.Tensor | bytes | dict[str, Any]
        _supports_raw = getattr(self.connector, "supports_raw_data", False)

        try:
            if _supports_raw:
                transfer_data = kv_data.to_gpu_tensor()
                _set_ms = (_time.perf_counter() - _set_start) * 1000
                logger.info(f"KV cache serialized: {transfer_data.numel()} bytes, {_set_ms:.1f}ms (gpu_direct)")
            else:
                raise RuntimeError("Connector does not support raw tensor")
        except Exception:
            # Fallback: GPU tensor assembly failed or not supported, try CPU bytes path
            try:
                transfer_data = kv_data.to_bytes()
                _set_ms = (_time.perf_counter() - _set_start) * 1000
                logger.info(f"KV cache serialized: {len(transfer_data)} bytes, {_set_ms:.1f}ms (fast_path)")
            except Exception as e:
                logger.warning(f"Fast serialization failed, falling back to dict: {e}")
                import traceback

                traceback.print_exc()
                data_dict = kv_data.to_dict()
                data_dict["request_id"] = transfer_req_id
                transfer_data = data_dict
                _set_ms = (_time.perf_counter() - _set_start) * 1000
                logger.info(f"KV cache serialized: {_set_ms:.1f}ms (dict fallback)")

        _start = _time.perf_counter()
        success, size, _ = self._transfer_with_retry(from_stage, to_stage, f"kv_cache_{transfer_req_id}", transfer_data)
        _elapsed = _time.perf_counter() - _start

        if success:
            _mbps = (size / 1024 / 1024) / _elapsed if _elapsed > 0 else 0
            logger.info(f"KV transfer OK: {transfer_req_id}, {size} bytes, {_elapsed:.3f}s, {_mbps:.1f} MB/s")
        else:
            logger.error(f"KV transfer FAILED: {transfer_req_id}")

    def _transfer_with_retry(
        self,
        from_stage: str,
        to_stage: str,
        request_id: str,
        data: "dict[str, Any] | bytes | torch.Tensor",
        max_retries: int = 3,
    ) -> tuple[bool, int, dict[str, Any] | None]:
        """Transfer data with retry and exponential backoff.

        Args:
            from_stage: Source stage identifier
            to_stage: Target stage identifier
            request_id: Request identifier for the key
            data: Data to transfer
            max_retries: Maximum number of retry attempts

        Returns:
            Tuple of (success, size, metadata)
        """
        for attempt in range(max_retries):
            try:
                # Build the full key for connector
                full_request_id = f"omni_{from_stage}_to_{to_stage}_{request_id}"
                success, size, metadata = self.connector.put(
                    from_stage=from_stage, to_stage=to_stage, put_key=full_request_id, data=data
                )
                if success:
                    return success, size, metadata
                logger.warning(f"Transfer attempt {attempt + 1} failed for {request_id}")
            except Exception as e:
                logger.warning(f"Transfer attempt {attempt + 1} exception: {e}")

            if attempt < max_retries - 1:
                time.sleep(0.1 * (2**attempt))

        return False, 0, None

    @torch.inference_mode()
    def receive_kv_cache_for_request(
        self,
        request_id: str,
        target_device: torch.device | None = None,
    ) -> tuple[dict[str, Any] | None, int]:
        """Receive KV cache for a specific request.

        This implements the receiving logic from gpu_diffusion_model_runner.py.

        Args:
            request_id: The request ID to receive KV cache for
            target_device: Optional device to move tensors to

        Returns:
            Tuple of (data dict, size) if successful, (None, 0) otherwise
        """
        if not self.connector:
            logger.warning("No connector available for receiving KV cache")
            return None, 0

        from_stage, to_stage = self.recv_stages
        if not from_stage or not to_stage:
            logger.warning("Receive stages not configured")
            return None, 0

        # Check if we should receive KV cache based on config
        if not self.config.need_recv_cache:
            logger.info(f"Skip receiving KV cache for {request_id} (need_recv_cache=False)")
            return None, 0

        timeout = self.config.recv_timeout
        start_time = time.time()

        logger.info(f"Wait for KV cache for request {request_id} from stage {from_stage} to {to_stage}...")

        try:
            while True:
                # Build the full key for connector
                full_request_id = f"omni_{from_stage}_to_{to_stage}_kv_cache_{request_id}"
                _link_start = time.perf_counter()
                result = self.connector.get(
                    from_stage=from_stage,
                    to_stage=to_stage,
                    get_key=full_request_id,
                )
                if result:
                    raw_data, size = result
                    _link_ms = (time.perf_counter() - _link_start) * 1000
                    _elapsed = time.time() - start_time
                    managed_buffer = None  # Track RDMA buffer for deferred release

                    # Handle ManagedBuffer (RDMA zero-copy), bytes (fast_path), or dict (legacy)
                    if hasattr(raw_data, "tensor") and hasattr(raw_data, "release"):
                        # ManagedBuffer from RDMA connector – zero-copy path
                        # Parse directly from the RDMA pool memory, no copy needed
                        import time as _time

                        managed_buffer = raw_data
                        _deser_start = _time.perf_counter()
                        try:
                            buf_tensor = raw_data.tensor
                            if buf_tensor.is_cuda:
                                # GPU pool (GPUDirect RDMA): only copy header to CPU,
                                # tensor data stays on GPU – true GPUDirect path.
                                # from_bytes_gpu() clones each tensor, so they are
                                # independent of the buffer – release immediately.
                                data = KVCacheTransferData.from_bytes_gpu(buf_tensor)
                                raw_data.release()
                                managed_buffer = None  # already released
                            else:
                                # CPU pool: zero-copy via memoryview
                                np_view = buf_tensor.numpy()
                                data = KVCacheTransferData.from_bytes(memoryview(np_view))
                        except Exception as e:
                            logger.error(f"Failed to deserialize KV cache from ManagedBuffer: {e}")
                            import traceback

                            traceback.print_exc()
                            if managed_buffer is not None:
                                raw_data.release()
                            return None, 0
                        _deser_ms = (_time.perf_counter() - _deser_start) * 1000
                        logger.info(
                            f"Successfully received KV cache for {request_id}, "
                            f"{size} bytes, wait={_elapsed:.3f}s, link={_link_ms:.1f}ms, "
                            f"deser={_deser_ms:.1f}ms (rdma {'gpu-direct' if buf_tensor.is_cuda else 'zero-copy'})"
                        )
                    elif isinstance(raw_data, (bytes, bytearray)):
                        import time as _time

                        _deser_start = _time.perf_counter()
                        try:
                            data = KVCacheTransferData.from_bytes(raw_data)
                        except Exception as e:
                            logger.error(f"Failed to deserialize KV cache bytes: {e}")
                            import traceback

                            traceback.print_exc()
                            return None, 0
                        _deser_ms = (_time.perf_counter() - _deser_start) * 1000
                        logger.info(
                            f"Successfully received KV cache for {request_id}, "
                            f"{size} bytes, wait={_elapsed:.3f}s, link={_link_ms:.1f}ms, "
                            f"deser={_deser_ms:.1f}ms (fast_path)"
                        )
                    elif isinstance(raw_data, torch.Tensor) and raw_data.dtype == torch.uint8 and raw_data.dim() == 1:
                        # Safety net: raw tensor from OmniSerializer deserialization.
                        # This happens when to_gpu_tensor() output was sent through
                        # a non-raw connector (e.g. MooncakeConnector/TCP).
                        # The packed binary format is preserved inside the tensor.
                        import time as _time

                        _deser_start = _time.perf_counter()
                        try:
                            cpu_bytes = raw_data.numpy().tobytes()
                            data = KVCacheTransferData.from_bytes(cpu_bytes)
                        except Exception as e:
                            logger.error(f"Failed to deserialize KV cache from raw tensor: {e}")
                            import traceback

                            traceback.print_exc()
                            return None, 0
                        _deser_ms = (_time.perf_counter() - _deser_start) * 1000
                        logger.info(
                            f"Successfully received KV cache for {request_id}, "
                            f"{size} bytes, wait={_elapsed:.3f}s, link={_link_ms:.1f}ms, "
                            f"deser={_deser_ms:.1f}ms (tensor_fallback)"
                        )
                    else:
                        data = raw_data
                        logger.info(
                            f"Successfully received KV cache for {request_id}, "
                            f"{size} bytes, wait={_elapsed:.3f}s, link={_link_ms:.1f}ms (dict)"
                        )

                    # Move tensors to target device (or clone if no target)
                    # IMPORTANT: When using ManagedBuffer zero-copy path, tensors
                    # are views into the RDMA pool.  We MUST ensure all tensor data
                    # is copied out of the pool BEFORE releasing the buffer.
                    import time as _time2

                    _move_start = _time2.perf_counter()
                    _move_count = 0
                    try:
                        if isinstance(data, dict) and "layer_blocks" in data:
                            layer_blocks = data["layer_blocks"]
                            for cache_list in [
                                layer_blocks.get("key_cache", []),
                                layer_blocks.get("value_cache", []),
                            ]:
                                for i, tensor in enumerate(cache_list):
                                    if not isinstance(tensor, torch.Tensor):
                                        continue
                                    _move_count += 1
                                    if target_device is not None and tensor.device != target_device:
                                        cache_list[i] = tensor.to(target_device).contiguous()
                                    elif managed_buffer is not None:
                                        # No target device but tensor is a view into
                                        # the RDMA pool – must clone before release
                                        cache_list[i] = tensor.clone()
                    finally:
                        # Always release RDMA buffer, even if GPU transfer fails
                        if managed_buffer is not None:
                            managed_buffer.release()
                    _move_ms = (_time2.perf_counter() - _move_start) * 1000
                    if _move_ms > 0.1:
                        logger.info(
                            f"Tensor device placement for {request_id}: {_move_count} tensors, {_move_ms:.1f}ms"
                        )

                    return data, size

                if time.time() - start_time > timeout:
                    logger.error(f"Timeout waiting for KV cache for request {request_id} after {timeout}s")
                    return None, 0

                time.sleep(0.01)  # 10ms polling interval

        except Exception as e:
            logger.error(f"Error receiving KV cache for {request_id}: {e}")
            import traceback

            traceback.print_exc()
            return None, 0

    def apply_kv_cache_to_request(self, req: Any, data: dict[str, Any]) -> None:
        """Apply received KV cache data to a request object.

        Args:
            req: The request object to apply KV cache to
            data: The received KV cache data dictionary
        """
        if isinstance(data, dict) and "layer_blocks" in data:
            layer_blocks = data["layer_blocks"]
            from types import SimpleNamespace

            kv_obj = SimpleNamespace(**layer_blocks)
            req.past_key_values = kv_obj

            # [Omni] Also attach to sampling_params for BagelPipeline compatibility
            # BagelPipeline checks req.sampling_params.past_key_values
            if hasattr(req, "sampling_params") and req.sampling_params is not None:
                req.sampling_params.past_key_values = kv_obj

        if "metadata" in data:
            req.kv_metadata = data["metadata"]

    # Legacy compatibility method
    def receive_kv_cache(self, req: Any, target_device: torch.device | None = None) -> bool:
        """Receive KV cache and populate request object (legacy interface).

        Args:
            req: Request object with request_id attribute
            target_device: Optional device to move tensors to

        Returns:
            True if successful, False otherwise
        """
        # Check if request has sender info (for cross-node RDMA)
        kv_sender_info = getattr(req, "kv_sender_info", None)
        if kv_sender_info:
            self.update_sender_info(kv_sender_info)

        request_id = getattr(req, "request_id", None)
        if not request_id and hasattr(req, "request_ids") and req.request_ids:
            # Adaptation for new OmniDiffusionRequest which has list of prompts/ids
            request_id = req.request_ids[0]

        if not request_id:
            logger.warning("Request has no ID, cannot receive KV cache")
            return False

        data, size = self.receive_kv_cache_for_request(request_id, target_device)
        if data:
            self.apply_kv_cache_to_request(req, data)
            return True
        return False
