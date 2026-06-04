# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import fcntl
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, fields, is_dataclass
from multiprocessing import shared_memory as shm_pkg
from typing import Any

import torch

from vllm_omni.entrypoints.stage_utils import shm_read_bytes, shm_write_bytes

from ..utils.logging import get_connector_logger
from .base import OmniConnectorBase

logger = get_connector_logger(__name__)

_ASYNC_PAYLOAD_WRAPPER_MARKER = "__omni_async_put_payload__"
_ASYNC_PAYLOAD_DATA = "data"
_ASYNC_PAYLOAD_CUDA_EVENTS = "cuda_events"


def _shm_profile_enabled() -> bool:
    value = os.environ.get("OMNI_SHM_PROFILE") or os.environ.get("VLLM_OMNI_SHM_PROFILE", "")
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass
class _AsyncShmEntry:
    key: str
    data: Any
    cuda_events: list[torch.cuda.Event] = field(default_factory=list)
    cancelled: bool = False
    shm: dict[str, Any] | None = None
    size: int = 0
    error: str | None = None
    created_at: float = field(default_factory=time.monotonic)


class SharedMemoryConnector(OmniConnectorBase):
    """Key-addressed local shared-memory connector.

    SHM is a local-only transport: it reads/writes POSIX shared memory
    segments identified purely by *key*.  It does **not** understand
    remote-transport metadata such as ``source_host`` / ``source_port``
    (that is the RDMA connector's job).  When such metadata is passed in,
    the connector silently falls back to key-based lookup.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.stage_id = config.get("stage_id", -1)
        self.device = config.get("device", "cuda:0")
        self.threshold = int(config.get("shm_threshold_bytes", 65536))
        self.async_put = bool(config.get("async_put", False))
        self.role = str(config.get("role", "sender")).lower()
        self.host = str(config.get("host", "127.0.0.1"))
        self._pending_keys: set[str] = set()
        self._entries: dict[str, _AsyncShmEntry] = {}
        self._entries_lock = threading.Lock()
        self._writer_pool: ThreadPoolExecutor | None = None
        self._writer_init_lock = threading.Lock()
        self._closed = False
        self._get_first_attempt_times: dict[str, float] = {}
        self._metrics = {
            "puts": 0,
            "gets": 0,
            "bytes_transferred": 0,
            "shm_writes": 0,
            "inline_writes": 0,
            "event_waits": 0,
            "async_pending": 0,
            "async_ready": 0,
            "async_errors": 0,
        }

    def _ensure_async_writer(self) -> bool:
        if self._writer_pool is not None:
            return True
        with self._writer_init_lock:
            if self._writer_pool is not None:
                return True
            if self._closed:
                logger.warning("Cannot start async SHM writer after connector is closed")
                return False
            max_workers = int(self.config.get("async_writer_workers", 4))
            self._writer_pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="shm-writer")
            return True

    def _write_payload_to_shm(self, key: str, payload: bytes) -> dict[str, Any]:
        lock_file = f"/dev/shm/shm_{key}_lockfile.lock"
        with open(lock_file, "wb+") as lockf:
            fcntl.flock(lockf, fcntl.LOCK_EX)
            meta = shm_write_bytes(payload, name=key)
            fcntl.flock(lockf, fcntl.LOCK_UN)
        return meta

    def _put_sync_payload(self, put_key: str, payload: bytes, size: int) -> dict[str, Any]:
        meta = self._write_payload_to_shm(put_key, payload)
        metadata = {"shm": meta, "size": size}
        self._pending_keys.add(put_key)
        self._metrics["shm_writes"] += 1
        return metadata

    def _put_event_async(self, put_key: str, data: Any) -> tuple[bool, int, dict[str, Any] | None]:
        put_start = time.perf_counter()
        if not self._ensure_async_writer() or self._writer_pool is None:
            logger.error("async_put=True but writer pool could not be initialized")
            return False, 0, None

        data, cuda_events = self._unwrap_async_payload(data)
        entry = _AsyncShmEntry(key=put_key, data=data, cuda_events=cuda_events)
        with self._entries_lock:
            old = self._entries.get(put_key)
            if old is not None:
                old.cancelled = True
                if old.shm:
                    self._unlink_shm(old.shm.get("name"))
            self._entries[put_key] = entry
            self._pending_keys.add(put_key)

        self._writer_pool.submit(self._async_write_entry, entry)
        self._metrics["puts"] += 1
        self._metrics["async_pending"] += 1
        if _shm_profile_enabled():
            logger.info(
                "OMNI_SHM_PROFILE connector=SharedMemoryConnector stage=%s role=%s event=event_async_put_enqueue "
                "key=%s events=%d enqueue_ms=%.3f",
                self.stage_id,
                self.role,
                put_key,
                len(cuda_events),
                (time.perf_counter() - put_start) * 1000.0,
            )
        return True, 0, {"async_put": True, "shm_key": put_key}

    def _async_write_entry(self, entry: _AsyncShmEntry) -> None:
        write_start = time.perf_counter()
        wait_ms = 0.0
        serialize_ms = 0.0
        write_ms = 0.0
        try:
            wait_start = time.perf_counter()
            self._wait_cuda_events(entry.cuda_events)
            wait_ms = (time.perf_counter() - wait_start) * 1000.0
            serialize_start = time.perf_counter()
            payload = self.serialize_obj(entry.data)
            serialize_ms = (time.perf_counter() - serialize_start) * 1000.0
            size = len(payload)
            shm_write_start = time.perf_counter()
            meta = self._write_payload_to_shm(entry.key, payload)
            write_ms = (time.perf_counter() - shm_write_start) * 1000.0
            with self._entries_lock:
                current = self._entries.get(entry.key)
                if current is not entry or entry.cancelled:
                    self._unlink_shm(meta.get("name"))
                    return
                entry.shm = meta
                entry.size = size
                entry.data = None
                self._metrics["bytes_transferred"] += size
                self._metrics["shm_writes"] += 1
                self._metrics["async_ready"] += 1
                if entry.cuda_events:
                    self._metrics["event_waits"] += 1
            if _shm_profile_enabled():
                logger.info(
                    "OMNI_SHM_PROFILE connector=SharedMemoryConnector stage=%s role=%s event=event_async_write_done "
                    "key=%s events=%d wait_ms=%.3f serialize_ms=%.3f shm_write_ms=%.3f total_ms=%.3f size=%d",
                    self.stage_id,
                    self.role,
                    entry.key,
                    len(entry.cuda_events),
                    wait_ms,
                    serialize_ms,
                    write_ms,
                    (time.perf_counter() - write_start) * 1000.0,
                    size,
                )
        except Exception as e:
            with self._entries_lock:
                current = self._entries.get(entry.key)
                if current is entry:
                    entry.error = repr(e)
                    entry.data = None
                    self._metrics["async_errors"] += 1
            logger.error("SharedMemoryConnector async write failed for req %s: %s", entry.key, e, exc_info=True)

    def prepare_async_payload(self, data: Any) -> Any:
        """Record producer-stream CUDA events before payload leaves the model thread.

        ``connector.put()`` is normally called by a save thread, whose current
        CUDA stream is not necessarily the stream that produced the tensors.
        This sidecar keeps the payload unchanged for serialization while
        carrying the events the writer must wait on before CPU staging.
        """
        if not self.async_put:
            return data
        start = time.perf_counter()
        cuda_events = self._record_cuda_events(data)
        record_ms = (time.perf_counter() - start) * 1000.0
        if _shm_profile_enabled():
            logger.info(
                "OMNI_SHM_PROFILE connector=SharedMemoryConnector stage=%s role=%s event=prepare_async_payload "
                "events=%d record_ms=%.3f",
                self.stage_id,
                self.role,
                len(cuda_events),
                record_ms,
            )
        if not cuda_events:
            return data
        return {
            _ASYNC_PAYLOAD_WRAPPER_MARKER: True,
            _ASYNC_PAYLOAD_DATA: data,
            _ASYNC_PAYLOAD_CUDA_EVENTS: cuda_events,
        }

    def _unwrap_async_payload(self, data: Any) -> tuple[Any, list[torch.cuda.Event]]:
        if isinstance(data, dict) and data.get(_ASYNC_PAYLOAD_WRAPPER_MARKER) is True:
            events = data.get(_ASYNC_PAYLOAD_CUDA_EVENTS, [])
            if not isinstance(events, list):
                events = []
            return data.get(_ASYNC_PAYLOAD_DATA), events
        # Compatibility path for direct connector.put() callers. In the model
        # runner path, events should already be captured by prepare_async_payload().
        return data, self._record_cuda_events(data)

    def _record_cuda_events(self, data: Any) -> list[torch.cuda.Event]:
        devices: set[torch.device] = set()
        self._collect_cuda_devices(data, devices, set())
        events: list[torch.cuda.Event] = []
        for device in devices:
            with torch.cuda.device(device):
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(device))
                events.append(event)
        return events

    def _collect_cuda_devices(self, obj: Any, devices: set[torch.device], seen: set[int]) -> None:
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)
        if isinstance(obj, torch.Tensor):
            if obj.is_cuda:
                devices.add(obj.device)
            return
        if isinstance(obj, dict):
            for key, value in obj.items():
                self._collect_cuda_devices(key, devices, seen)
                self._collect_cuda_devices(value, devices, seen)
            return
        if isinstance(obj, (list, tuple)):
            for value in obj:
                self._collect_cuda_devices(value, devices, seen)
            return
        struct_fields = getattr(obj, "__struct_fields__", None)
        if struct_fields is not None:
            for field_name in struct_fields:
                self._collect_cuda_devices(getattr(obj, field_name), devices, seen)
            return
        if is_dataclass(obj) and not isinstance(obj, type):
            for item in fields(obj):
                self._collect_cuda_devices(getattr(obj, item.name), devices, seen)

    @staticmethod
    def _wait_cuda_events(events: list[torch.cuda.Event]) -> None:
        for event in events:
            event.synchronize()

    def put(
        self,
        from_stage: str,
        to_stage: str,
        put_key: str,
        data: Any,
    ) -> tuple[bool, int, dict[str, Any] | None]:
        try:
            if self.async_put:
                return self._put_event_async(put_key, data)

            # Always serialize first to check size (and for SHM writing)
            # Note: For extremely large objects in "inline" mode (e.g. Ray),
            # we might double-serialize if we're not careful, but here we assume
            # if it's huge we use SHM, or if Ray, threshold is maxsize.
            put_start = time.perf_counter()
            serialize_start = time.perf_counter()
            payload = self.serialize_obj(data)
            serialize_ms = (time.perf_counter() - serialize_start) * 1000.0
            size = len(payload)

            # Currently, we always use SHM.
            if True:
                write_start = time.perf_counter()
                metadata = self._put_sync_payload(put_key, payload, size)
                write_ms = (time.perf_counter() - write_start) * 1000.0
            else:
                # Inline - pass bytes directly to avoid double serialization of the object
                # We already serialized it to check size, so we pass the bytes.
                # The Queue will pickle these bytes (fast), avoiding re-serializing the complex object.
                metadata = {"inline_bytes": payload, "size": size}
                self._metrics["inline_writes"] += 1
                write_ms = 0.0

            self._metrics["puts"] += 1
            self._metrics["bytes_transferred"] += size
            if _shm_profile_enabled():
                logger.info(
                    "OMNI_SHM_PROFILE connector=SharedMemoryConnector stage=%s role=%s event=sync_put_done "
                    "key=%s serialize_ms=%.3f shm_write_ms=%.3f total_ms=%.3f size=%d",
                    self.stage_id,
                    self.role,
                    put_key,
                    serialize_ms,
                    write_ms,
                    (time.perf_counter() - put_start) * 1000.0,
                    size,
                )

            return True, size, metadata

        except Exception as e:
            logger.error(f"SharedMemoryConnector put failed for req {put_key}: {e}")
            return False, 0, None

    def _get_data_with_lock(self, lock_file: str, shm_handle: dict):
        obj = None
        try:
            with open(lock_file, "rb+") as lockf:
                fcntl.flock(lockf, fcntl.LOCK_EX)
                data_bytes = shm_read_bytes(shm_handle)
                fcntl.flock(lockf, fcntl.LOCK_UN)
            obj = self.deserialize_obj(data_bytes)
            return obj, int(shm_handle.get("size", 0))
        except Exception as e:
            logger.error(f"SharedMemoryConnector shm get failed for req : {e}")
            return None
        finally:
            # If data has been received, delete lock_file.
            if obj and os.path.exists(lock_file):
                os.remove(lock_file)

    def _get_by_key(self, get_key: str) -> tuple[Any, int] | None:
        """Read a SHM segment addressed purely by *get_key*."""
        shm = None
        try:
            shm = shm_pkg.SharedMemory(name=get_key)
            if shm is None or shm.size == 0:
                return None
            lock_file = f"/dev/shm/shm_{get_key}_lockfile.lock"
            shm_handle = {"name": get_key, "size": shm.size}
            result = self._get_data_with_lock(lock_file, shm_handle)
            if result is not None:
                self._pending_keys.discard(get_key)
            return result
        except FileNotFoundError:
            return None
        except Exception:
            logger.debug("_get_by_key: unexpected error reading SHM segment %s", get_key, exc_info=True)
            return None
        finally:
            if shm:
                shm.close()

    def _log_get_profile(
        self,
        get_key: str,
        result: tuple[Any, int] | None,
        call_start: float,
        first_attempt: float,
        status: str,
    ) -> None:
        if not _shm_profile_enabled():
            return
        ready = result is not None
        if ready:
            self._get_first_attempt_times.pop(get_key, None)
        logger.warning(
            "OMNI_SHM_PROFILE connector=SharedMemoryConnector stage=%s role=%s event=get_result "
            "key=%s status=%s ready=%s async_put=%s get_call_ms=%.3f wait_since_first_get_ms=%.3f size=%s",
            self.stage_id,
            self.role,
            get_key,
            status,
            ready,
            self.async_put,
            (time.perf_counter() - call_start) * 1000.0,
            (time.perf_counter() - first_attempt) * 1000.0,
            result[1] if result is not None else None,
        )

    def get(
        self,
        from_stage: str,
        to_stage: str,
        get_key: str,
        metadata=None,
    ) -> tuple[Any, int] | None:
        call_start = time.perf_counter()
        profile_enabled = _shm_profile_enabled()
        first_attempt = self._get_first_attempt_times.setdefault(get_key, call_start) if profile_enabled else call_start
        if metadata is not None:
            if isinstance(metadata, dict) and get_key in metadata:
                metadata = metadata.get(get_key)

            if not isinstance(metadata, dict):
                result = self._get_by_key(get_key)
                self._log_get_profile(get_key, result, call_start, first_attempt, "READY" if result else "MISS")
                return result

            if metadata.get("async_put"):
                key = str(metadata.get("shm_key", get_key))
                if profile_enabled and key != get_key:
                    first_attempt = self._get_first_attempt_times.setdefault(key, first_attempt)
                result = self._get_by_key(key)
                self._log_get_profile(key, result, call_start, first_attempt, "READY" if result else "MISS")
                return result

            if "inline_bytes" in metadata:
                try:
                    obj = self.deserialize_obj(metadata["inline_bytes"])
                    self._pending_keys.discard(get_key)
                    result = (obj, int(metadata.get("size", 0)))
                    self._log_get_profile(get_key, result, call_start, first_attempt, "READY")
                    return result
                except Exception as e:
                    logger.error(f"SharedMemoryConnector inline get failed for req {get_key}: {e}")
                    self._log_get_profile(get_key, None, call_start, first_attempt, "ERROR")
                    return None

            if "shm" in metadata:
                shm_handle = metadata["shm"]
                lock_file = f"/dev/shm/shm_{shm_handle['name']}_lockfile.lock"
                result = self._get_data_with_lock(lock_file, shm_handle)
                if result is not None:
                    self._pending_keys.discard(get_key)
                self._log_get_profile(get_key, result, call_start, first_attempt, "READY" if result else "MISS")
                return result

            # Metadata is a dict but has no SHM-specific handle (e.g. RDMA-
            # style source_host/source_port).  Fall back to key-based read.
            result = self._get_by_key(get_key)
            self._log_get_profile(get_key, result, call_start, first_attempt, "READY" if result else "MISS")
            return result

        result = self._get_by_key(get_key)
        self._log_get_profile(get_key, result, call_start, first_attempt, "READY" if result else "MISS")
        return result

    def cleanup(self, request_id: str) -> None:
        """Best-effort cleanup of unconsumed SHM segments for *request_id*.

        Matches pending keys where *request_id* appears as the full key,
        as a ``_``-delimited prefix, or as a ``_``-delimited suffix.
        If ``get()`` was never called, we unlink it here so /dev/shm
        doesn't leak.
        """
        stale = [
            k
            for k in self._pending_keys
            if k == request_id or k.startswith(request_id + "_") or k.endswith("_" + request_id)
        ]
        for key in tuple(self._get_first_attempt_times):
            if key == request_id or key.startswith(request_id + "_") or key.endswith("_" + request_id):
                self._get_first_attempt_times.pop(key, None)
        if self.async_put:
            with self._entries_lock:
                for key, entry in list(self._entries.items()):
                    if key == request_id or key.startswith(request_id + "_") or key.endswith("_" + request_id):
                        entry.cancelled = True
                        self._entries.pop(key, None)
                        if entry.shm:
                            self._unlink_shm(entry.shm.get("name"))
                        if key not in stale:
                            stale.append(key)

        for key in stale:
            self._pending_keys.discard(key)
            try:
                seg = shm_pkg.SharedMemory(name=key)
                seg.close()
                seg.unlink()
                logger.debug("cleanup: unlinked unconsumed SHM segment %s", key)
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.debug("cleanup: failed to unlink SHM segment %s: %s", key, e)
            lock_file = f"/dev/shm/shm_{key}_lockfile.lock"
            if os.path.exists(lock_file):
                try:
                    os.remove(lock_file)
                except OSError:
                    pass

    def close(self) -> None:
        """Unlink all remaining tracked SHM segments."""
        if getattr(self, "_closed", True):
            return
        self._closed = True

        if self._writer_pool is not None:
            self._writer_pool.shutdown(wait=True, cancel_futures=False)
            self._writer_pool = None

        with self._entries_lock:
            for entry in list(self._entries.values()):
                entry.cancelled = True
                if entry.shm:
                    self._unlink_shm(entry.shm.get("name"))
            self._entries.clear()

        for key in list(self._pending_keys):
            try:
                seg = shm_pkg.SharedMemory(name=key)
                seg.close()
                seg.unlink()
            except Exception:
                pass
            lock_file = f"/dev/shm/shm_{key}_lockfile.lock"
            if os.path.exists(lock_file):
                try:
                    os.remove(lock_file)
                except OSError:
                    pass
        self._pending_keys.clear()

    @staticmethod
    def _unlink_shm(name: Any) -> None:
        if not name:
            return
        try:
            seg = shm_pkg.SharedMemory(name=str(name))
            seg.close()
            seg.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            logger.debug("Failed to unlink SHM segment %s", name, exc_info=True)

    def health(self) -> dict[str, Any]:
        return {
            "status": "healthy",
            "threshold": self.threshold,
            "async_put": self.async_put,
            "host": self.host,
            **self._metrics,
        }
