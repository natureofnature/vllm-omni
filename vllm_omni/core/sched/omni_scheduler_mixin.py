from __future__ import annotations

from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput
from vllm.v1.request import Request, RequestStatus, StreamingUpdate


class OmniSchedulerMixin:
    """Shared scheduler helpers for omni-specific request handling."""

    def _replace_session_with_streaming_update(
        self,
        session: Request,
        update: StreamingUpdate,
    ) -> None:
        """For streaming input: Replace an existing streaming session payload with the latest update."""
        session._output_token_ids.clear()
        session._all_token_ids.clear()
        new_prompt = update.prompt_token_ids or ()
        session._all_token_ids.extend(new_prompt)
        session.num_computed_tokens = 0
        session.prompt_token_ids = update.prompt_token_ids or ()
        session.additional_information = update.additional_information or None
        # Update block hashes for the new tokens.
        session.update_block_hashes()
        session.num_prompt_tokens = len(session.prompt_token_ids)
        session.arrival_time = update.arrival_time
        session.sampling_params = update.sampling_params
        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            self.num_waiting_for_streaming_input -= 1
        session.status = RequestStatus.WAITING

        if self.log_stats:
            session.record_event(EngineCoreEventType.QUEUED)

    def _abort_timed_out_omni_requests(self, outputs: dict[int, list[EngineCoreOutput]]) -> None:
        """Finish requests orphaned in WAITING_FOR_CHUNK / WAITING_FOR_INPUT.

        AR and generation schedulers share the same timeout contract and queue
        removal sequence; keep it centralized so the behavior cannot drift.
        """
        if self.chunk_coordinator is None:
            return

        timeout_s = getattr(self, "_omni_chunk_timeout_s", 300.0)
        timed_out_reqs = self.chunk_coordinator.abort_timed_out_requests(
            timeout_s,
            self.requests,
            self.waiting,
            self.skipped_waiting,
        )
        if not timed_out_reqs:
            return

        timed_out_req_ids = {request.request_id for request in timed_out_reqs}
        self.running = [req for req in self.running if getattr(req, "request_id", None) not in timed_out_req_ids]
        for request in timed_out_reqs:
            self._free_request(request)
            outputs[request.client_index].append(
                EngineCoreOutput(
                    request_id=request.request_id,
                    new_token_ids=[],
                    finish_reason=request.get_finished_reason(),
                    events=request.take_events(),
                    trace_headers=request.trace_headers,
                    num_cached_tokens=max(getattr(request, "num_cached_tokens", 0), 0),
                )
            )
