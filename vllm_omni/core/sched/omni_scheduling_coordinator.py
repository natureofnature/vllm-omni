# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduling-side coordination for chunk and batch input waiting.

Manages WAITING_FOR_CHUNK and WAITING_FOR_INPUT state transitions
based on readiness signals from OmniConnectorOutput, without ever
calling connector.put()/get().

This replaces the scheduling half of OmniChunkTransferAdapter; the
transport half lives in OmniConnectorModelRunnerMixin.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from vllm.logger import init_logger
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


class OmniSchedulingCoordinator:
    """Pure-scheduling coordinator for chunk and batch input waiting.

    The Scheduler owns an instance of this class.  It consumes readiness
    signals produced by the Model Runner's ``OmniConnectorModelRunnerMixin``
    (via ``OmniConnectorOutput``) and manages ``WAITING_FOR_CHUNK`` and
    ``WAITING_FOR_INPUT`` state transitions accordingly.
    """

    def __init__(self, scheduler_max_num_seqs: int, stage_id: int = 0, async_chunk: bool = False):
        self._stage_id = stage_id
        self._scheduler_max_num_seqs = scheduler_max_num_seqs
        self._async_chunk = async_chunk

        self.finished_requests: set[str] = set()
        self.requests_with_ready_chunks: set[str] = set()
        self._batch_input_received: set[str] = set()

        self._waiting_for_chunk_waiting: deque[Any] = deque()
        self._waiting_for_chunk_running: deque[Any] = deque()

        # Request IDs that were newly registered for chunk recv this cycle.
        # The engine/Model Runner should call register_chunk_recv() for these
        # so the bg thread starts polling.
        self.pending_chunk_registrations: list[Any] = []

        # Requests waiting for batch stage input (WAITING_FOR_INPUT).
        self._waiting_for_input: deque[Any] = deque()
        self.pending_input_registrations: list[Any] = []

    # ------------------------------------------------------------------ #
    #  Core scheduling methods
    # ------------------------------------------------------------------ #

    def process_pending_chunks(
        self,
        waiting_queue: Any,
        running_queue: list[Request],
        chunk_ready_req_ids: set[str],
        chunk_finished_req_ids: set[str],
    ) -> None:
        """Transition requests whose chunks have arrived.

        Args:
            waiting_queue: Scheduler's waiting request queue.
            running_queue: Scheduler's running request list.
            chunk_ready_req_ids: IDs with a newly arrived chunk this cycle.
            chunk_finished_req_ids: IDs whose final chunk has arrived.
        """
        if self._stage_id == 0 or not self._async_chunk:
            return

        self.finished_requests.update(chunk_finished_req_ids)
        self.pending_chunk_registrations = []

        self._process_chunk_queue(
            waiting_queue,
            self._waiting_for_chunk_waiting,
            RequestStatus.WAITING,
            chunk_ready_req_ids,
        )
        self._process_chunk_queue(
            running_queue,
            self._waiting_for_chunk_running,
            RequestStatus.RUNNING,
            chunk_ready_req_ids,
        )

        while len(running_queue) > self._scheduler_max_num_seqs:
            request = running_queue.pop()
            waiting_queue.prepend_requests([request])

    def process_pending_batch_inputs(
        self,
        waiting_queue: Any,
        running_queue: list[Request],
        stage_recv_req_ids: set[str],
    ) -> None:
        """Manage WAITING_FOR_INPUT lifecycle for batch mode.

        For non-Stage-0 stages in batch mode (``not async_chunk``):
        1. Fresh WAITING requests are transitioned to WAITING_FOR_INPUT
           and registered for bg-thread polling.
        2. WAITING_FOR_INPUT requests whose data has arrived (in
           ``stage_recv_req_ids``) are transitioned back to WAITING.
        """
        if self._stage_id == 0:
            return

        self._batch_input_received.update(stage_recv_req_ids)
        self.pending_input_registrations = []

        remaining: deque[Any] = deque()
        for request in self._waiting_for_input:
            if request.request_id in stage_recv_req_ids:
                request.status = RequestStatus.WAITING
                waiting_queue.add_request(request)
            else:
                remaining.append(request)
        self._waiting_for_input = remaining

        if not self._async_chunk:
            to_remove: list[Any] = []
            queue_snapshot = list(waiting_queue)
            for request in queue_snapshot:
                if request.status == RequestStatus.WAITING:
                    if request.request_id in self._batch_input_received:
                        continue
                    if request.request_id in self.requests_with_ready_chunks:
                        continue
                    if request.request_id in self.finished_requests:
                        continue
                    request.status = RequestStatus.WAITING_FOR_INPUT
                    to_remove.append(request)
                    self._waiting_for_input.append(request)
                    self.pending_input_registrations.append(request)
                elif request.status == RequestStatus.WAITING_FOR_INPUT:
                    if request.request_id in stage_recv_req_ids:
                        request.status = RequestStatus.WAITING
                    else:
                        to_remove.append(request)
                        self._waiting_for_input.append(request)
                        self.pending_input_registrations.append(request)
            for request in to_remove:
                waiting_queue.remove(request)

    def free_finished_request(self, request_id: str) -> None:
        """Prune internal tracking sets for a freed request to prevent unbounded growth."""
        self._batch_input_received.discard(request_id)
        self.finished_requests.discard(request_id)
        self.requests_with_ready_chunks.discard(request_id)

    def restore_queues(
        self,
        waiting_queue: Any,
        running_queue: list[Request],
    ) -> None:
        """Return waiting-for-chunk/input requests to scheduling queues."""
        for request in self._waiting_for_chunk_waiting:
            waiting_queue.add_request(request)
        self._waiting_for_chunk_waiting = deque()

        if self._waiting_for_chunk_running:
            running_queue.extend(self._waiting_for_chunk_running)
        self._waiting_for_chunk_running = deque()

        for request in self._waiting_for_input:
            waiting_queue.add_request(request)
        self._waiting_for_input = deque()

    def update_request_data(
        self,
        requests: dict[str, Request],
        chunk_data: dict[str, Any],
        model_mode: str = "ar",
    ) -> None:
        """Apply received chunk data to request objects.

        For AR mode: writes to ``request.additional_information``.
        For Generation mode: updates ``request.prompt_token_ids``.
        """
        for req_id, payload in chunk_data.items():
            request = requests.get(req_id)
            if request is None:
                continue

            if model_mode == "ar":
                request.additional_information = payload
            else:
                new_ids = payload.get("code_predictor_codes", [])
                if new_ids:
                    request.prompt_token_ids = new_ids
                    request.num_computed_tokens = 0

    def postprocess_scheduler_output(
        self,
        scheduler_output: Any,
        requests: dict[str, Request] | None = None,
    ) -> None:
        """Attach additional info for cached requests and clear ready state."""
        if requests is not None:
            self._attach_cached_additional_information(scheduler_output, requests)
        self._clear_chunk_ready(scheduler_output)

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _process_chunk_queue(
        self,
        queue: Any,
        waiting_for_chunk_list: deque[Any],
        target_status: RequestStatus,
        chunk_ready_req_ids: set[str],
    ) -> None:
        queue_snapshot = list(queue)
        for request in queue_snapshot:
            if request.status != RequestStatus.WAITING_FOR_CHUNK:
                if request.request_id in self.requests_with_ready_chunks:
                    continue
                if request.request_id in self.finished_requests:
                    continue
                if request.status == RequestStatus.WAITING_FOR_INPUT:
                    continue
                self.pending_chunk_registrations.append(request)
                request.status = RequestStatus.WAITING_FOR_CHUNK
            else:
                if request.request_id in chunk_ready_req_ids:
                    request.status = target_status
                    self.requests_with_ready_chunks.add(request.request_id)
                    continue
            queue.remove(request)
            waiting_for_chunk_list.append(request)

    def _clear_chunk_ready(self, scheduler_output: Any) -> None:
        if scheduler_output.scheduled_new_reqs:
            for req_data in scheduler_output.scheduled_new_reqs:
                self.requests_with_ready_chunks.discard(
                    getattr(req_data, "req_id", None),
                )

        if scheduler_output.scheduled_cached_reqs:
            for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
                self.requests_with_ready_chunks.discard(req_id)

    @staticmethod
    def _attach_cached_additional_information(
        scheduler_output: Any,
        requests: dict[str, Request],
    ) -> None:
        cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if not cached_reqs:
            return
        if not hasattr(cached_reqs, "additional_information"):
            cached_reqs.additional_information = {}
        for req_id in cached_reqs.req_ids:
            request = requests.get(req_id) if req_id else None
            info = getattr(request, "additional_information", None) if request else None
            cached_reqs.additional_information[req_id] = info


# Backward-compatible alias
ChunkSchedulingCoordinator = OmniSchedulingCoordinator
