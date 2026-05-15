from __future__ import annotations

from typing import Any

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.request import Request, RequestStatus, StreamingUpdate

from vllm_omni.core.sched.output import OmniInputRegistration, OmniSchedulerOutput


class OmniSchedulerMixin:
    """Shared scheduler helpers for omni-specific request handling."""

    def _free_input_coordinator_request(self, request_id: str) -> None:
        """Prune full-payload coordinator state for a completed request."""
        input_coordinator = getattr(self, "input_coordinator", None)
        if input_coordinator is not None:
            input_coordinator.free_finished_request(request_id)

    # ------------------------------------------------------------------ #
    #  Shared scheduler/output helpers (lift the AR / generation duplicates)
    # ------------------------------------------------------------------ #

    def _consume_pending_connector_output(self, model_mode: str) -> None:
        """Drain ``self._latest_omni_connector_output`` into the coordinator.

        Called at the top of every ``schedule()`` cycle.  Identical between
        AR and generation schedulers except for the ``model_mode`` argument
        forwarded to ``update_request_metadata``.
        """
        connector_output = getattr(self, "_latest_omni_connector_output", None)
        self._latest_omni_connector_output = None
        input_coordinator = getattr(self, "input_coordinator", None)
        if input_coordinator is None:
            return
        if connector_output and connector_output.request_metadata:
            input_coordinator.update_request_metadata(
                self.requests, connector_output.request_metadata, model_mode=model_mode
            )
        input_coordinator.process_pending_full_payload_inputs(
            self.waiting,
            self.running,
            connector_output.stage_recv_req_ids if connector_output else set(),
        )

    def _capture_omni_connector_output(self, model_runner_output: Any, model_mode: str) -> None:
        """Stash the model runner's omni_connector_output for next schedule().

        Called at the tail of every ``update_from_output()``.  Identical
        between AR and generation schedulers except for ``model_mode``.
        """
        omni_output = getattr(model_runner_output, "omni_connector_output", None)
        if omni_output is None:
            return
        self._latest_omni_connector_output = omni_output
        input_coordinator = getattr(self, "input_coordinator", None)
        if input_coordinator and omni_output.request_metadata:
            input_coordinator.update_request_metadata(
                self.requests,
                omni_output.request_metadata,
                model_mode=model_mode,
            )

    def _wrap_omni_scheduler_output(
        self,
        base: SchedulerOutput,
        *,
        finished_requests_needing_kv_transfer: dict | None = None,
        pending_input_registrations: list[OmniInputRegistration] | None = None,
    ) -> OmniSchedulerOutput:
        """Wrap a base ``SchedulerOutput`` in ``OmniSchedulerOutput``.

        Pulls each base ``SchedulerOutput`` dataclass field via ``getattr``
        and forwards optional omni-specific fields.  Lifted from 4 separate
        copy-pastes between AR (1) and generation (3) schedulers.
        """
        base_data = {name: getattr(base, name) for name in SchedulerOutput.__dataclass_fields__}
        input_coordinator = getattr(self, "input_coordinator", None)
        if pending_input_registrations is None:
            pending_input_registrations = input_coordinator.pending_input_registrations if input_coordinator else []
        return OmniSchedulerOutput(
            **base_data,
            finished_requests_needing_kv_transfer=finished_requests_needing_kv_transfer or {},
            pending_input_registrations=pending_input_registrations,
        )

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
