# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified data-plane communication mixin for Model Runners.

All connector.put()/get() calls are consolidated here. Background I/O
threads handle async chunk and batch transfers; KV cache is delegated to
the existing OmniKVTransferManager (to be absorbed later).

The mixin reports transfer results via OmniConnectorOutput so that the
Scheduler can make scheduling decisions without ever touching a connector.
"""

from __future__ import annotations

import importlib
import os
import threading
import time
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any

import torch
from vllm.logger import init_logger

from vllm_omni.distributed.omni_connectors.factory import OmniConnectorFactory
from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec
from vllm_omni.outputs import OmniConnectorOutput

if TYPE_CHECKING:
    from vllm_omni.distributed.omni_connectors.connectors.base import (
        OmniConnectorBase,
    )
    from vllm_omni.distributed.omni_connectors.kv_transfer_manager import (
        OmniKVTransferManager,
    )

logger = init_logger(__name__)


class OmniConnectorModelRunnerMixin:
    """Unified data-plane communication mixin for Model Runners.

    Provides three transfer modes through a single pair of bg I/O threads:
      - **Batch**: ``recv_stage_inputs`` / ``send_stage_outputs``
      - **Streaming (async_chunk)**: ``recv_chunk`` / ``send_chunk``
      - **KV cache**: ``send_kv_cache`` / ``recv_kv_cache`` (delegates to
        the existing ``OmniKVTransferManager``)

    The mixin owns connector instances and background threads.  It never
    touches scheduling queues -- readiness is communicated to the Scheduler
    via ``OmniConnectorOutput``.
    """

    # ------------------------------------------------------------------ #
    #  Init / Shutdown
    # ------------------------------------------------------------------ #

    def init_omni_connectors(
        self,
        vllm_config: Any,
        model_config: Any,
        kv_transfer_manager: OmniKVTransferManager | None = None,
    ) -> None:
        """Initialize connectors and background threads.

        Args:
            vllm_config: Full vLLM config object.
            model_config: Stage-level model config with connector settings.
            kv_transfer_manager: Existing KV transfer manager to delegate to.
        """
        self._omni_connector: OmniConnectorBase | None = self._create_connector(model_config)
        self._kv_transfer_manager = kv_transfer_manager

        self._async_chunk: bool = getattr(model_config, "async_chunk", False)
        self._model_mode: str = getattr(model_config, "worker_type", "ar")
        self._stage_id: int = getattr(self._omni_connector, "stage_id", 0) if self._omni_connector else 0

        self._custom_process_func = self._load_custom_func(model_config)

        # -- next stage ID (from connector config or default stage_id + 1) --
        self._next_stage_id: int = self._resolve_next_stage_id(model_config)

        # -- heterogeneous TP rank support --
        rank_cfg = self._parse_rank_mapping(model_config)
        self._from_tp: int = rank_cfg["from_tp"]
        self._to_tp: int = rank_cfg["to_tp"]
        self._local_rank: int = rank_cfg["local_rank"]

        # -- chunk index tracking (ported from OmniChunkTransferAdapter) --
        self._put_req_chunk: dict[str, int] = defaultdict(int)
        self._get_req_chunk: dict[str, int] = defaultdict(int)
        self._request_payload: dict[str, dict[str, Any]] = {}
        self._code_prompt_token_ids: dict[str, list[list[int]]] = defaultdict(list)
        self._request_ids_mapping: dict[str, str] = {}

        # -- async I/O state (shared by chunk + batch) --
        self._pending_load_reqs: dict[str, Any] = {}
        self._finished_load_reqs: set[str] = set()
        self._pending_save_reqs: dict[str, deque] = {}
        self._finished_save_reqs: set[str] = set()

        # -- per-cycle output accumulator --
        self._chunk_ready_req_ids: set[str] = set()
        self._chunk_finished_req_ids: set[str] = set()
        self._chunk_data: dict[str, Any] = {}
        self._stage_recv_req_ids: set[str] = set()

        # -- batch mode: accumulate latest pooler_output per request,
        #    send only when the request finishes (next-cycle flush) --
        self._pending_batch_send: dict[str, tuple[Any, Any]] = {}

        # -- batch recv results (non-blocking) --
        self._batch_recv_results: dict[str, Any] = {}

        # -- KV sent accumulator --
        self._kv_sent_req_ids: list[str] = []

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._work_available = threading.Event()

        # Start background threads only when there's a connector
        self._recv_thread: threading.Thread | None = None
        self._save_thread: threading.Thread | None = None
        if self._omni_connector is not None:
            self._recv_thread = threading.Thread(
                target=self._recv_loop,
                daemon=True,
                name="omni-mixin-recv",
            )
            self._recv_thread.start()
            self._save_thread = threading.Thread(
                target=self._save_loop,
                daemon=True,
                name="omni-mixin-save",
            )
            self._save_thread.start()

    def shutdown_omni_connectors(self) -> None:
        """Stop background threads and release connector resources."""
        self._stop_event.set()
        if self._recv_thread is not None:
            self._recv_thread.join(timeout=5)
        if self._save_thread is not None:
            self._save_thread.join(timeout=5)
        if self._omni_connector is not None:
            try:
                self._omni_connector.close()
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    #  Batch mode  (recv_stage_inputs / send_stage_outputs)
    # ------------------------------------------------------------------ #

    def recv_stage_inputs(self, scheduler_output: Any) -> dict[str, Any] | None:
        """Check for incoming batch stage inputs (non-blocking).

        Returns a dict mapping ``request_id -> engine_inputs`` for data
        that has arrived, or ``None`` if nothing is ready.  Also stores
        the data in ``_chunk_data`` so the Scheduler's
        ``update_request_data`` writes it into request objects.
        """
        with self._lock:
            if not self._batch_recv_results:
                return None
            results = dict(self._batch_recv_results)
            self._batch_recv_results.clear()
        self._stage_recv_req_ids.update(results.keys())
        self._chunk_data.update(results)
        return results

    def accumulate_batch_output(
        self,
        req_id: str,
        pooler_output: Any,
        request: Any,
    ) -> None:
        """Store the latest pooler_output for a request (batch mode).

        The data is actually sent when ``flush_batch_outputs`` is called
        with the finished request IDs from the next scheduler cycle.
        """
        self._pending_batch_send[req_id] = (pooler_output, request)

    def flush_batch_outputs(self, finished_req_ids: set[str]) -> None:
        """Send accumulated batch outputs for requests that just finished."""
        to_send: dict[str, tuple[Any, Any]] = {}
        for req_id in finished_req_ids:
            entry = self._pending_batch_send.pop(req_id, None)
            if entry is not None:
                to_send[req_id] = entry
        if to_send:
            self.send_stage_outputs(scheduler_output=None, outputs=to_send)

    def send_stage_outputs(
        self,
        scheduler_output: Any,
        outputs: dict[str, tuple[Any, Any] | Any],
    ) -> list[str]:
        """Send batch stage outputs to the next stage via connector.

        Args:
            outputs: Mapping of ``req_id`` to either a
                ``(pooling_output, request)`` tuple (preferred) or a raw
                payload dict.  When a tuple is supplied the request object
                is forwarded to ``custom_process_stage_input_func``.

        Returns list of request IDs successfully enqueued.
        """
        if self._omni_connector is None:
            return []
        if not self.is_data_transfer_rank():
            return list(outputs.keys())
        sent_ids: list[str] = []
        next_stage_id = self._next_stage_id
        for req_id, value in outputs.items():
            if isinstance(value, tuple) and len(value) == 2:
                raw_output, request = value
            else:
                raw_output, request = value, None

            payload = raw_output
            if self._custom_process_func is not None:
                try:
                    payload = self._custom_process_func(
                        transfer_manager=self,
                        pooling_output=raw_output,
                        request=request,
                    )
                except Exception:
                    logger.exception("custom_process_stage_input_func failed for %s", req_id)
                    continue
            if payload is None:
                continue

            external_req_id = self._resolve_external_req_id(request, req_id)
            chunk_id = self._put_req_chunk[req_id]
            self._put_req_chunk[req_id] += 1
            connector_put_key = f"{external_req_id}_{self._stage_id}_{chunk_id}"

            task = {
                "stage_id": self._stage_id,
                "next_stage_id": next_stage_id,
                "put_key": connector_put_key,
                "data": payload,
                "request_id": req_id,
            }
            with self._lock:
                self._pending_save_reqs.setdefault(req_id, deque()).append(task)
            sent_ids.append(req_id)
        if sent_ids:
            self._work_available.set()
        return sent_ids

    # ------------------------------------------------------------------ #
    #  Streaming chunk mode  (recv_chunk / send_chunk)
    # ------------------------------------------------------------------ #

    def register_chunk_recv(self, request: Any) -> None:
        """Register a request for async chunk retrieval by the bg thread.

        Stage-0 has no upstream producer so this is a no-op there.
        """
        if self._stage_id == 0:
            return
        request_id = request.request_id
        self._request_ids_mapping[request_id] = getattr(
            request,
            "external_req_id",
            request_id,
        )
        if not hasattr(request, "additional_information"):
            request.additional_information = None
        with self._lock:
            self._pending_load_reqs[request_id] = request
        self._work_available.set()

    def recv_chunk(self) -> dict[str, Any]:
        """Collect chunks received by the bg thread since last call.

        Returns a dict ``{request_id: chunk_payload}`` for newly arrived
        chunks.  Empty dict when nothing is ready.

        This method reads from ``_finished_load_reqs`` without clearing
        it -- ``get_omni_connector_output()`` is the sole consumer that
        drains and resets ``_finished_load_reqs`` at the end of each
        ``execute_model`` cycle.
        """
        with self._lock:
            finished = set(self._finished_load_reqs)
        if not finished:
            return {}

        self._chunk_ready_req_ids.update(finished)
        return {rid: self._chunk_data.get(rid) for rid in finished}

    def send_chunk(
        self,
        request: Any,
        pooling_output: Any | None = None,
    ) -> bool:
        """Derive and enqueue one chunk for async sending.

        Payload extraction runs in the caller thread (via
        ``custom_process_stage_input_func``); the actual
        ``connector.put()`` is done by the background save thread.
        Non-KV data is identical across TP ranks; only rank 0 sends.
        """
        if self._omni_connector is None:
            return False
        if not self.is_data_transfer_rank():
            return True
        request_id = self._resolve_external_req_id(request, request.request_id)
        chunk_id = self._put_req_chunk[request_id]

        payload_data = None
        if self._custom_process_func is not None:
            try:
                payload_data = self._custom_process_func(
                    transfer_manager=self,
                    pooling_output=pooling_output,
                    request=request,
                )
            except Exception:
                logger.exception("custom_process_stage_input_func failed for chunk %s", request_id)
        if payload_data is None:
            return False

        self._put_req_chunk[request_id] += 1
        next_stage_id = self._next_stage_id
        connector_put_key = f"{request_id}_{self._stage_id}_{chunk_id}"

        task = {
            "stage_id": self._stage_id,
            "next_stage_id": next_stage_id,
            "put_key": connector_put_key,
            "data": payload_data,
            "request_id": request_id,
        }
        with self._lock:
            self._pending_save_reqs.setdefault(request_id, deque()).append(task)
        self._work_available.set()
        return True

    # ------------------------------------------------------------------ #
    #  KV cache  (delegates to OmniKVTransferManager)
    # ------------------------------------------------------------------ #

    def send_kv_cache(
        self,
        finished_reqs: dict[str, dict[str, Any]],
        kv_caches: list[torch.Tensor],
        block_size: int,
        cache_dtype: str,
        request_id_resolver: Any | None = None,
    ) -> list[str]:
        """Send KV cache for finished requests.

        Delegates to the existing ``OmniKVTransferManager``.
        """
        if self._kv_transfer_manager is None:
            return list(finished_reqs.keys()) if finished_reqs else []
        result = self._kv_transfer_manager.handle_finished_requests_kv_transfer(
            finished_reqs=finished_reqs,
            kv_caches=kv_caches,
            block_size=block_size,
            cache_dtype=cache_dtype,
            request_id_resolver=request_id_resolver,
        )
        if result:
            self._kv_sent_req_ids.extend(result)
        return result

    def recv_kv_cache(
        self,
        request_id: str,
        target_device: torch.device | None = None,
    ) -> tuple[dict[str, Any] | None, int]:
        """Receive KV cache for a request.

        Delegates to the existing ``OmniKVTransferManager``.
        """
        if self._kv_transfer_manager is None:
            return None, 0
        return self._kv_transfer_manager.receive_kv_cache_for_request(
            request_id=request_id,
            target_device=target_device,
        )

    # ------------------------------------------------------------------ #
    #  Output aggregation
    # ------------------------------------------------------------------ #

    def _empty_output_with_connector_signals(self) -> Any:
        """Return a minimal ModelRunnerOutput carrying pending connector signals.

        Used by early-return paths (e.g. ``num_scheduled_tokens == 0``)
        that still need to deliver ``omni_connector_output`` to the
        Scheduler so that WAITING_FOR_INPUT / WAITING_FOR_CHUNK
        transitions are not lost.
        """
        from copy import copy

        from vllm.v1.worker.gpu_model_runner import EMPTY_MODEL_RUNNER_OUTPUT

        output = copy(EMPTY_MODEL_RUNNER_OUTPUT)
        output.omni_connector_output = self.get_omni_connector_output()
        return output

    def get_omni_connector_output(self) -> OmniConnectorOutput:
        """Collect and reset transfer results for this execute_model cycle."""
        with self._lock:
            newly_finished = set(self._finished_load_reqs)
            self._finished_load_reqs.clear()
        self._chunk_ready_req_ids.update(newly_finished)

        output = OmniConnectorOutput(
            chunk_ready_req_ids=set(self._chunk_ready_req_ids),
            chunk_finished_req_ids=set(self._chunk_finished_req_ids),
            chunk_data=dict(self._chunk_data),
            kv_sent_req_ids=list(self._kv_sent_req_ids),
            stage_recv_req_ids=set(self._stage_recv_req_ids),
        )
        self._chunk_ready_req_ids.clear()
        self._chunk_finished_req_ids.clear()
        self._chunk_data.clear()
        self._kv_sent_req_ids.clear()
        self._stage_recv_req_ids.clear()
        return output

    # ------------------------------------------------------------------ #
    #  Properties for compatibility with custom_process funcs that access
    #  transfer_manager.put_req_chunk / request_payload / code_prompt_token_ids
    # ------------------------------------------------------------------ #

    @property
    def put_req_chunk(self) -> dict[str, int]:
        return self._put_req_chunk

    @property
    def request_payload(self) -> dict[str, dict[str, Any]]:
        return self._request_payload

    @request_payload.setter
    def request_payload(self, value: dict[str, dict[str, Any]]) -> None:
        self._request_payload = value

    @property
    def code_prompt_token_ids(self) -> dict[str, list[list[int]]]:
        return self._code_prompt_token_ids

    # ------------------------------------------------------------------ #
    #  Background I/O threads
    # ------------------------------------------------------------------ #

    def _recv_loop(self) -> None:
        """Background thread: poll connector for incoming data."""
        while not self._stop_event.is_set():
            with self._lock:
                pending_ids = list(self._pending_load_reqs.keys())

            if not pending_ids:
                self._work_available.wait(timeout=0.01)
                self._work_available.clear()
                continue

            for req_id in pending_ids:
                if self._stop_event.is_set():
                    break
                try:
                    self._poll_single_request(req_id)
                except Exception:
                    logger.warning("Error receiving data for %s", req_id, exc_info=True)

            time.sleep(0.001)

    def _save_loop(self) -> None:
        """Background thread: send outgoing data via connector."""
        while not self._stop_event.is_set():
            tasks: list[dict] = []
            with self._lock:
                for req_id in list(self._pending_save_reqs.keys()):
                    dq = self._pending_save_reqs[req_id]
                    while dq:
                        tasks.append(dq.popleft())
                    if not dq:
                        del self._pending_save_reqs[req_id]

            if tasks:
                for task in tasks:
                    if self._stop_event.is_set():
                        break
                    try:
                        self._send_single_request(task)
                    except Exception:
                        logger.error(
                            "Error saving data for %s",
                            task.get("request_id"),
                            exc_info=True,
                        )
            else:
                self._work_available.wait(timeout=0.01)
                self._work_available.clear()

    # ------------------------------------------------------------------ #
    #  Chunk-level poll / send  (ported from OmniChunkTransferAdapter)
    # ------------------------------------------------------------------ #

    def _poll_single_request(self, req_id: str) -> None:
        """Poll connector for one chunk of a request (non-blocking)."""
        connector = self._omni_connector
        if connector is None:
            return

        target_stage_id = self._stage_id - 1
        chunk_id = self._get_req_chunk[req_id]
        external_req_id = self._request_ids_mapping.get(req_id, req_id)
        connector_get_key = f"{external_req_id}_{target_stage_id}_{chunk_id}"

        result = connector.get(
            str(target_stage_id),
            str(self._stage_id),
            connector_get_key,
        )

        if result is None:
            return

        payload_data, _size = result
        if not payload_data:
            return

        self._get_req_chunk[req_id] += 1

        if self._async_chunk:
            is_finished = bool(payload_data.get("finished"))

            if self._model_mode == "ar":
                self._accumulate_payload(external_req_id, payload_data)
            else:
                new_ids = payload_data.get("code_predictor_codes", [])
                if not new_ids and not is_finished:
                    return

            if is_finished:
                self._chunk_finished_req_ids.add(req_id)

            self._chunk_data[req_id] = payload_data

            with self._lock:
                self._finished_load_reqs.add(req_id)
                # Streaming: keep polling for subsequent chunks; only
                # unregister when the final chunk arrives.
                if is_finished:
                    self._pending_load_reqs.pop(req_id, None)
        else:
            # Batch: the complete payload arrives in a single get(),
            # so always unregister immediately.
            if isinstance(payload_data, dict):
                engine_inputs = payload_data.get("engine_inputs", payload_data)
            else:
                engine_inputs = payload_data
            with self._lock:
                self._batch_recv_results[req_id] = engine_inputs
                self._pending_load_reqs.pop(req_id, None)

        logger.debug("[Stage-%s] Received data for key %s", self._stage_id, connector_get_key)

    def _send_single_request(self, task: dict) -> None:
        """Send one queued task via connector.put()."""
        connector = self._omni_connector
        if connector is None:
            return

        success, _size, _metadata = connector.put(
            from_stage=str(task["stage_id"]),
            to_stage=str(task["next_stage_id"]),
            put_key=task["put_key"],
            data=task["data"],
        )
        if success:
            logger.debug("[Stage-%s] Sent %s", task["stage_id"], task["put_key"])

    # ------------------------------------------------------------------ #
    #  Payload accumulation  (ported from OmniChunkTransferAdapter)
    # ------------------------------------------------------------------ #

    def _accumulate_payload(self, req_id: str, payload_data: dict[str, Any]) -> dict[str, Any]:
        """Accumulate chunk payloads (concat tensors, extend lists)."""
        if req_id not in self._request_payload:
            self._request_payload[req_id] = payload_data
            return payload_data

        origin = self._request_payload[req_id]
        for key, value in payload_data.items():
            if key == "finished":
                continue
            if isinstance(value, torch.Tensor) and key in origin:
                payload_data[key] = torch.cat([origin[key], value], dim=0)
            elif isinstance(value, list) and key in origin:
                payload_data[key] = origin[key] + value

        self._request_payload[req_id] = payload_data
        return payload_data

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _create_connector(model_config: Any) -> OmniConnectorBase | None:
        """Create a connector from model_config, or None if unconfigured."""
        connector_config = getattr(model_config, "stage_connector_config", None)
        if connector_config is None:
            return None

        if not isinstance(connector_config, dict):
            connector_config = {
                "name": getattr(connector_config, "name", None),
                "extra": getattr(connector_config, "extra", {}),
            }

        name = connector_config.get("name")
        if not name:
            return None

        spec = ConnectorSpec(name=name, extra=connector_config.get("extra", {}))
        try:
            return OmniConnectorFactory.create_connector(spec)
        except Exception:
            logger.warning("Failed to create connector %s", name, exc_info=True)
            return None

    @staticmethod
    def _load_custom_func(model_config: Any) -> Any | None:
        """Load custom_process_next_stage_input_func from config."""
        func_path = getattr(model_config, "custom_process_next_stage_input_func", None)
        if not func_path:
            return None
        try:
            module_path, func_name = func_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            return getattr(module, func_name)
        except Exception:
            logger.warning("Failed to load custom func: %s", func_path, exc_info=True)
            return None

    def _resolve_external_req_id(self, request: Any, fallback_req_id: str) -> str:
        """Resolve the external request ID consistently.

        Checks ``_request_ids_mapping`` first (populated by
        ``register_chunk_recv``), then falls back to the request's
        ``external_req_id`` attribute, and finally to the given
        ``fallback_req_id``.
        """
        mapped = self._request_ids_mapping.get(fallback_req_id)
        if mapped is not None:
            return mapped
        if request is not None:
            return getattr(request, "external_req_id", fallback_req_id)
        return fallback_req_id

    def _resolve_next_stage_id(self, model_config: Any) -> int:
        """Determine the downstream stage ID from connector config.

        Falls back to ``stage_id + 1`` when the config does not specify
        a ``to_stage`` explicitly.
        """
        connector_config = getattr(model_config, "stage_connector_config", None)
        if connector_config is not None:
            if isinstance(connector_config, dict):
                to_stage = connector_config.get("to_stage")
            else:
                to_stage = getattr(connector_config, "to_stage", None)
            if to_stage is not None:
                return int(to_stage)
        return self._stage_id + 1

    @staticmethod
    def _parse_rank_mapping(model_config: Any) -> dict[str, int]:
        """Parse rank_mapping from connector config (optional).

        Returns ``{"from_tp": int, "to_tp": int, "local_rank": int}``.
        When ``rank_mapping`` is absent, assumes 1:1 homogeneous mapping.
        """
        connector_config = getattr(model_config, "stage_connector_config", None)
        if connector_config is not None and not isinstance(connector_config, dict):
            connector_config = getattr(connector_config, "__dict__", {})

        rank_mapping: dict = {}
        if isinstance(connector_config, dict):
            rank_mapping = connector_config.get("rank_mapping", {})

        from_tp = int(rank_mapping.get("from_tp", 1))
        to_tp = int(rank_mapping.get("to_tp", 1))

        local_rank = 0
        try:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        except (ValueError, TypeError):
            pass

        return {"from_tp": from_tp, "to_tp": to_tp, "local_rank": local_rank}

    # ------------------------------------------------------------------ #
    #  Heterogeneous TP rank support
    # ------------------------------------------------------------------ #

    def get_kv_remote_ranks(self) -> list[int]:
        """Determine which remote ranks this local rank exchanges KV with.

        Follows vLLM's ``TpKVTopology.get_target_remote_ranks()`` pattern:
        - ``from_tp > to_tp``: each to-rank reads from multiple from-ranks
        - ``from_tp < to_tp``: multiple to-ranks read from the same from-rank
        - ``from_tp == to_tp``: 1:1 mapping
        """
        if self._from_tp == self._to_tp:
            return [self._local_rank]

        if self._from_tp > self._to_tp:
            tp_ratio = self._from_tp // self._to_tp
            return [self._local_rank * tp_ratio + i for i in range(tp_ratio)]
        else:
            tp_ratio = self._to_tp // self._from_tp
            return [self._local_rank // tp_ratio]

    def is_data_transfer_rank(self) -> bool:
        """Whether this rank should participate in data (non-KV) transfer.

        Data (stage inputs/outputs, chunks) is identical across all TP
        ranks after all-gather, so only rank 0 needs to transfer.
        """
        return self._local_rank == 0

    def get_kv_connector_key(
        self,
        req_id: str,
        from_stage: int,
        chunk_id: int,
        from_rank: int,
        to_rank: int,
    ) -> str:
        """Build connector key that includes rank info for KV transfers."""
        return f"{req_id}_{from_stage}_{chunk_id}_{from_rank}_{to_rank}"
