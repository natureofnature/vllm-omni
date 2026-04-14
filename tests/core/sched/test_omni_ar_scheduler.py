from types import SimpleNamespace

import pytest
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


def test_finish_requests_tolerates_missing_chunk_transfer_adapter(monkeypatch):
    scheduler = object.__new__(OmniARScheduler)
    scheduler.chunk_transfer_adapter = None
    scheduler.requests = {}

    calls = []

    def _super_finish(self, request_ids, finished_status):
        calls.append((tuple(request_ids), finished_status))
        return []

    monkeypatch.setattr(OmniARScheduler.__mro__[2], "finish_requests", _super_finish)

    result = OmniARScheduler.finish_requests(
        scheduler,
        ["req-abort"],
        RequestStatus.FINISHED_ABORTED,
    )

    assert result == []
    assert calls == [(("req-abort",), RequestStatus.FINISHED_ABORTED)]


def test_has_pending_kv_work_survives_schedule_clear():
    """Regression: has_unfinished_requests() must stay True when the mixin
    reports has_pending_kv_work=True, even after schedule() clears the
    connector output.  Without the latch, the engine could stop polling
    prematurely."""
    scheduler = object.__new__(OmniARScheduler)
    scheduler.requests_needing_kv_transfer = {}
    scheduler.waiting_for_transfer_free = set()
    scheduler._mixin_has_pending_kv_work = False
    scheduler._latest_omni_connector_output = None
    scheduler.requests = {}
    scheduler.running = []
    scheduler.waiting = SimpleNamespace(
        remove_requests=lambda reqs: None,
        __len__=lambda self: 0,
    )
    scheduler.skipped_waiting = SimpleNamespace(
        remove_requests=lambda reqs: None,
        __len__=lambda self: 0,
    )
    scheduler.chunk_coordinator = OmniSchedulingCoordinator(
        scheduler_max_num_seqs=8,
        stage_id=1,
        async_chunk=False,
    )

    # Mixin delivers connector output with has_pending_kv_work=True
    connector_output = OmniConnectorOutput(has_pending_kv_work=True)
    scheduler._latest_omni_connector_output = connector_output

    assert scheduler._has_pending_kv_work() is False, "latch not yet updated, should still be False"

    # Simulate update_from_output refreshing the latch
    scheduler._mixin_has_pending_kv_work = connector_output.has_pending_kv_work

    assert scheduler._has_pending_kv_work() is True

    # Now simulate schedule() consuming and clearing the connector output
    scheduler._mixin_has_pending_kv_work = (
        scheduler._latest_omni_connector_output is not None
        and scheduler._latest_omni_connector_output.has_pending_kv_work
    )
    scheduler._latest_omni_connector_output = None

    # The latch must survive the clear
    assert scheduler._has_pending_kv_work() is True, "keepalive signal lost after schedule() cleared connector output"
    assert scheduler.has_unfinished_requests() is True
