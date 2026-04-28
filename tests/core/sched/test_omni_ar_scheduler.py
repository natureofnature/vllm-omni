from collections import defaultdict
from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.request import RequestStatus

from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.core.sched.omni_scheduling_coordinator import OmniSchedulingCoordinator
from vllm_omni.outputs import OmniConnectorOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_scheduler_output():
    return SimpleNamespace(
        num_scheduled_tokens={},
        scheduled_spec_decode_tokens={},
        num_invalid_spec_tokens=0,
    )


def _make_runner_output(omni_output: OmniConnectorOutput):
    return SimpleNamespace(
        sampled_token_ids=[],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=None,
        num_nans_in_logits=None,
        kv_connector_output=None,
        cudagraph_stats=None,
        req_ids=[],
        req_id_to_index={},
        omni_connector_output=omni_output,
        kv_extracted_req_ids=None,
    )


def test_process_kv_transfer_trigger_stops_after_completed_extraction():
    scheduler = object.__new__(OmniARScheduler)
    scheduler.kv_transfer_criteria = {"type": "prefill_finished", "stop_after_transfer": True}
    scheduler.waiting_for_transfer_free = set()
    scheduler.transfer_triggered_requests = {"req-kv"}
    scheduler.pending_stop_after_extraction = {"req-kv"}
    scheduler.completed_kv_transfers = {"req-kv"}
    scheduler._request_omits_kv_transfer_to_next_stage = lambda request: False

    request = SimpleNamespace(request_id="req-kv", status=None)
    stopped = OmniARScheduler._process_kv_transfer_trigger(scheduler, request, [])

    assert stopped is True
    assert request.status is RequestStatus.FINISHED_STOPPED
    assert "req-kv" not in scheduler.pending_stop_after_extraction


def test_update_from_output_applies_prompt_resize_metadata():
    coordinator = OmniSchedulingCoordinator(
        scheduler_max_num_seqs=8,
        stage_id=1,
        async_chunk=False,
    )
    request = SimpleNamespace(
        request_id="req-ar",
        external_req_id="req-ar",
        status=RequestStatus.WAITING,
        prompt_token_ids=[7, 8],
        num_prompt_tokens=2,
        num_computed_tokens=3,
        _all_token_ids=[7, 8],
        _output_token_ids=[],
        sampling_params=SimpleNamespace(logprobs=None),
        pooling_params=None,
        client_index=0,
        stop_reason=None,
        trace_headers=None,
        num_cached_tokens=0,
        num_external_computed_tokens=0,
        num_nans_in_logits=0,
        structured_output_request=None,
        is_finished=lambda: False,
        take_events=lambda: [],
        get_finished_reason=lambda: "finished",
        _omni_initial_model_buffer=None,
    )
    scheduler = object.__new__(OmniARScheduler)
    scheduler.perf_metrics = None
    scheduler.connector = None
    scheduler.requests = {request.request_id: request}
    scheduler.chunk_coordinator = coordinator
    scheduler._latest_omni_connector_output = None
    scheduler._mixin_has_pending_kv_work = False
    scheduler.kv_cache_manager = SimpleNamespace(take_events=lambda: None)
    scheduler.kv_event_publisher = SimpleNamespace(publish=lambda batch: None)
    scheduler.make_stats = lambda *args, **kwargs: None
    scheduler.running = []
    scheduler.waiting = []
    scheduler.skipped_waiting = []
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = {}
    scheduler.completed_kv_transfers = set()
    scheduler.transfer_triggered_requests = set()
    scheduler.waiting_for_transfer_free = set()
    scheduler.requests_needing_kv_transfer = {}
    scheduler._omits_kv_transfer_cache = {}
    scheduler._update_from_kv_xfer_finished = lambda *args, **kwargs: None

    omni_output = OmniConnectorOutput(
        stage_recv_req_ids={"req-ar"},
        request_metadata={"req-ar": {"next_stage_prompt_len": 5, "left_context_size": 2}},
    )

    result = OmniARScheduler.update_from_output(scheduler, _make_scheduler_output(), _make_runner_output(omni_output))

    assert result == {}
    assert scheduler._latest_omni_connector_output is omni_output
    assert request.prompt_token_ids == [0, 0, 0, 0, 0]
    assert request.num_prompt_tokens == 5
    assert request.num_computed_tokens == 0
    assert request._all_token_ids == [0, 0, 0, 0, 0]
    assert request._output_token_ids == []


def test_has_pending_kv_work_survives_schedule_clear(monkeypatch):
    """Regression: has_unfinished_requests() must stay True when the mixin
    reports has_pending_kv_work=True, even after schedule() clears the
    connector output.  Without the latch, the engine could stop polling
    prematurely."""

    class _Queue(list):
        def remove_requests(self, reqs):
            req_ids = {getattr(req, "request_id", req) for req in reqs}
            self[:] = [req for req in self if getattr(req, "request_id", req) not in req_ids]

        def prepend_requests(self, reqs):
            self[:0] = list(reqs)

    scheduler = object.__new__(OmniARScheduler)
    scheduler.requests_needing_kv_transfer = {}
    scheduler.waiting_for_transfer_free = set()
    scheduler._mixin_has_pending_kv_work = False
    scheduler._latest_omni_connector_output = None
    scheduler.requests = {}
    scheduler.running = []
    scheduler.waiting = _Queue()
    scheduler.skipped_waiting = _Queue()
    scheduler.chunk_coordinator = OmniSchedulingCoordinator(
        scheduler_max_num_seqs=8,
        stage_id=1,
        async_chunk=False,
    )

    scheduler.perf_metrics = None
    scheduler.connector = None
    scheduler.kv_cache_manager = SimpleNamespace(take_events=lambda: None)
    scheduler.kv_event_publisher = SimpleNamespace(publish=lambda batch: None)
    scheduler.make_stats = lambda *args, **kwargs: None
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = {}
    scheduler.completed_kv_transfers = set()
    scheduler.transfer_triggered_requests = set()
    scheduler.pending_stop_after_extraction = set()
    scheduler._omits_kv_transfer_cache = {}
    scheduler._update_from_kv_xfer_finished = lambda *args, **kwargs: None
    scheduler.get_finished_requests_needing_kv_transfer = lambda: {}

    def _super_schedule(self):
        return SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData(
                req_ids=[],
                resumed_req_ids=set(),
                new_token_ids=[],
                all_token_ids={},
                new_block_ids=[],
                num_computed_tokens=[],
                num_output_tokens=[],
            ),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
            num_invalid_spec_tokens={},
        )

    monkeypatch.setattr(OmniARScheduler.__mro__[2], "schedule", _super_schedule)

    omni_output = OmniConnectorOutput(has_pending_kv_work=True)
    OmniARScheduler.update_from_output(scheduler, _make_scheduler_output(), _make_runner_output(omni_output))

    assert scheduler._latest_omni_connector_output is omni_output
    assert scheduler._mixin_has_pending_kv_work is True

    scheduled = OmniARScheduler.schedule(scheduler)

    assert scheduled.finished_requests_needing_kv_transfer == {}
    assert scheduler._latest_omni_connector_output is None
    assert scheduler._mixin_has_pending_kv_work is True
    assert scheduler._has_pending_kv_work() is True
    assert scheduler.has_unfinished_requests() is True


def test_abort_timed_out_omni_requests_uses_shared_scheduler_helper():
    scheduler = object.__new__(OmniARScheduler)
    freed: list[str] = []
    taken_events: list[str] = []

    request = SimpleNamespace(
        request_id="r1",
        client_index=3,
        trace_headers={"trace": "x"},
        num_cached_tokens=7,
        get_finished_reason=lambda: "error",
        take_events=lambda: taken_events.append("taken") or ["event"],
    )

    scheduler.chunk_coordinator = SimpleNamespace(
        abort_timed_out_requests=lambda timeout_s, requests, waiting, skipped_waiting: [request]
    )
    scheduler.requests = {"r1": request}
    scheduler.waiting = SimpleNamespace()
    scheduler.skipped_waiting = SimpleNamespace()
    scheduler.running = [request, "other"]
    scheduler._omni_chunk_timeout_s = 12.5
    scheduler._free_request = lambda req: freed.append(req.request_id)

    outputs = defaultdict(list)
    OmniARScheduler._abort_timed_out_omni_requests(scheduler, outputs)

    assert scheduler.running == ["other"]
    assert freed == ["r1"]
    assert taken_events == ["taken"]
    assert len(outputs[3]) == 1
    output = outputs[3][0]
    assert output.request_id == "r1"
    assert output.finish_reason == "error"
    assert output.num_cached_tokens == 7
