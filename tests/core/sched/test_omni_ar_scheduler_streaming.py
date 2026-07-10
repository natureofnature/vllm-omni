"""Unit tests for Omni AR streaming-session async placeholder handling."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest

# Imports must run in this order: vllm_omni applies patches to vllm.v1.request before
# Request / StreamingUpdate are bound in this module. Ruff isort would reorder them.
# isort: off
import vllm_omni  # noqa: F401 - import for side effects (patch vLLM)
from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.outputs import OmniConnectorOutput

# isort: on

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_scheduler(
    *,
    stage_id: int = 0,
    async_chunk: bool = False,
    model_stage: str = "thinker",
) -> OmniARScheduler:
    sched = OmniARScheduler.__new__(OmniARScheduler)
    sched._new_prompt_len_snapshot = {}
    sched._omni_pending_segment_finished = set()
    sched._omni_pending_upstream_finished = set()
    sched._omni_pending_upstream_segment_finished = set()
    sched.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            stage_id=stage_id,
            async_chunk=async_chunk,
            model_arch="Qwen3OmniMoeForConditionalGeneration",
            model_stage=model_stage,
        )
    )
    sched.num_waiting_for_streaming_input = 0
    sched.log_stats = False
    sched.chunk_transfer_adapter = None
    sched.input_coordinator = None
    return sched


def _empty_scheduler_output() -> SimpleNamespace:
    return SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=[],
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=0,
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        preempted_req_ids=None,
        has_structured_output_requests=False,
        pending_structured_output_tokens=False,
        num_invalid_spec_tokens=None,
        kv_connector_metadata=None,
        ec_connector_metadata=None,
        new_block_ids_to_zero=None,
        num_spec_tokens_to_schedule=0,
    )


def test_terminal_only_input_coordinator_request_finishes_without_model_run() -> None:
    sched = _make_scheduler(stage_id=1)
    sched.requests = {"req-live": object()}
    sched.input_coordinator = SimpleNamespace(
        _async_chunk=True,
        finished_requests={"req-live", "req-stale", "req-ready"},
        requests_with_ready_chunks={"req-ready"},
    )
    finished = []

    def fake_finish_requests(request_ids, status):
        finished.append((request_ids, status))
        for request_id in request_ids:
            sched.requests.pop(request_id, None)

    sched.finish_requests = fake_finish_requests

    sched._finish_input_coordinator_terminal_only_requests()

    assert finished == [(["req-live"], RequestStatus.FINISHED_STOPPED)]
    assert sched.input_coordinator.finished_requests == {"req-ready"}


def test_final_output_terminal_only_request_finishes_without_model_run() -> None:
    sched = _make_scheduler(stage_id=2, async_chunk=True, model_stage="code2wav")
    sched.requests = {"req-live": object()}
    sched.input_coordinator = SimpleNamespace(
        _async_chunk=True,
        finished_requests={"req-live", "req-stale", "req-ready"},
        requests_with_ready_chunks={"req-ready"},
    )
    finished = []

    def fake_finish_requests(request_ids, status):
        finished.append((request_ids, status))

    sched.finish_requests = fake_finish_requests

    sched._finish_input_coordinator_terminal_only_requests()

    assert finished == [(["req-live"], RequestStatus.FINISHED_STOPPED)]
    assert sched.input_coordinator.requests_with_ready_chunks == {"req-ready"}
    assert sched.input_coordinator.finished_requests == {"req-ready"}


def test_full_payload_input_coordinator_request_is_not_terminal_only() -> None:
    sched = _make_scheduler(stage_id=1)
    sched.requests = {"req-live": object()}
    sched.input_coordinator = SimpleNamespace(
        _async_chunk=False,
        finished_requests={"req-live"},
        requests_with_ready_chunks=set(),
    )
    finished = []

    def fake_finish_requests(request_ids, status):
        finished.append((request_ids, status))

    sched.finish_requests = fake_finish_requests

    sched._finish_input_coordinator_terminal_only_requests()

    assert finished == []
    assert sched.input_coordinator.finished_requests == {"req-live"}


def test_pending_segment_terminal_drops_stale_upstream_ready_state() -> None:
    sched = _make_scheduler(stage_id=1, async_chunk=True, model_stage="talker")
    sched._omni_pending_segment_finished = {"req-live"}
    sched.input_coordinator = SimpleNamespace(
        requests_with_ready_chunks={"req-live", "req-other"},
    )

    sched._clear_pending_segment_ready_state()

    assert sched.input_coordinator.requests_with_ready_chunks == {"req-other"}


def test_talker_segment_ready_continues_local_decode_before_forwarding() -> None:
    sched = _make_scheduler(stage_id=1, async_chunk=True, model_stage="talker")
    sched.requests = {"req-live": object()}
    sched.waiting = []
    sched.running = []
    sched._omni_pending_upstream_segment_finished = set()
    sched._latest_omni_connector_output = OmniConnectorOutput(
        chunk_ready_req_ids={"req-live"},
        chunk_segment_finished_req_ids={"req-live"},
    )
    chunk_calls = []
    mark_calls = []

    def process_pending_chunks(waiting, running, chunk_ready_req_ids, chunk_finished_req_ids, **_):
        chunk_calls.append((set(chunk_ready_req_ids), set(chunk_finished_req_ids)))

    def mark_chunk_segments_completed(req_ids, **kwargs):
        mark_calls.append((set(req_ids), kwargs))

    sched.input_coordinator = SimpleNamespace(
        process_pending_chunks=process_pending_chunks,
        process_pending_full_payload_inputs=lambda waiting, running, stage_recv_req_ids: None,
        mark_chunk_segments_completed=mark_chunk_segments_completed,
    )

    sched._consume_pending_connector_output(model_mode="ar")

    assert sched._omni_pending_upstream_segment_finished == set()
    assert sched._omni_pending_segment_finished == set()
    assert chunk_calls == [({"req-live"}, set())]
    assert mark_calls == [
        ({"req-live"}, {"continue_local_decode": True}),
    ]


def test_upstream_finished_terminal_only_request_drains_local_model() -> None:
    sched = _make_scheduler(stage_id=1, async_chunk=True, model_stage="talker")
    request = SimpleNamespace(resumable=True)
    sched.requests = {"req-live": request}
    sched.waiting = []
    sched.running = []
    sched._latest_omni_connector_output = OmniConnectorOutput(
        chunk_finished_req_ids={"req-live"},
    )
    chunk_calls = []
    finished = []

    def process_pending_chunks(waiting, running, chunk_ready_req_ids, chunk_finished_req_ids, **_):
        ready_ids = set(chunk_ready_req_ids)
        finished_ids = set(chunk_finished_req_ids)
        chunk_calls.append((ready_ids, finished_ids))
        sched.input_coordinator.requests_with_ready_chunks.update(ready_ids)
        sched.input_coordinator.finished_requests.update(finished_ids - ready_ids)

    def finish_requests(request_ids, status):
        finished.append((list(request_ids), status))
        for request_id in request_ids:
            sched.requests.pop(request_id, None)

    sched.finish_requests = finish_requests
    sched.input_coordinator = SimpleNamespace(
        _async_chunk=True,
        pending_connector_registrations=[],
        finished_requests=set(),
        requests_with_ready_chunks=set(),
        process_pending_chunks=process_pending_chunks,
        process_pending_full_payload_inputs=lambda waiting, running, stage_recv_req_ids: None,
    )

    sched._consume_pending_connector_output(model_mode="ar")
    sched._finish_input_coordinator_terminal_only_requests()
    output = sched._wrap_omni_scheduler_output(_empty_scheduler_output())
    output_after_drain = sched._wrap_omni_scheduler_output(_empty_scheduler_output())

    assert finished == []
    assert sched.requests["req-live"] is request
    assert request.resumable is False
    assert sched._omni_pending_upstream_finished == set()
    assert output.upstream_finished_req_ids == set()
    assert output_after_drain.upstream_finished_req_ids == set()
    assert sched.input_coordinator.finished_requests == {"req-live"}
    assert chunk_calls == [(set(), {"req-live"})]


def test_final_output_segment_only_records_control_signal() -> None:
    sched = _make_scheduler(stage_id=2, async_chunk=True, model_stage="code2wav")
    sched.requests = {"req-live": object()}
    sched.waiting = []
    sched.running = []
    sched._omni_pending_upstream_segment_finished = set()
    sched._latest_omni_connector_output = OmniConnectorOutput(
        chunk_segment_finished_req_ids={"req-live"},
    )
    chunk_calls = []
    mark_calls = []

    def process_pending_chunks(waiting, running, chunk_ready_req_ids, chunk_finished_req_ids, **_):
        chunk_calls.append((set(chunk_ready_req_ids), set(chunk_finished_req_ids)))

    sched.input_coordinator = SimpleNamespace(
        process_pending_chunks=process_pending_chunks,
        process_pending_full_payload_inputs=lambda waiting, running, stage_recv_req_ids: None,
        mark_chunk_segments_completed=lambda req_ids: mark_calls.append(set(req_ids)),
    )

    sched._consume_pending_connector_output(model_mode="generation")

    assert sched._omni_pending_upstream_segment_finished == {"req-live"}
    assert chunk_calls == [(set(), set())]
    assert mark_calls == [{"req-live"}]


def test_generation_segment_ready_records_flush_step() -> None:
    sched = _make_scheduler(stage_id=2, async_chunk=True, model_stage="code2wav")
    sched.requests = {"req-live": object()}
    sched.waiting = []
    sched.running = []
    sched._omni_pending_upstream_segment_finished = set()
    sched._latest_omni_connector_output = OmniConnectorOutput(
        chunk_ready_req_ids={"req-live"},
        chunk_segment_finished_req_ids={"req-live"},
    )
    chunk_calls = []
    mark_calls = []

    def process_pending_chunks(waiting, running, chunk_ready_req_ids, chunk_finished_req_ids, **_):
        chunk_calls.append((set(chunk_ready_req_ids), set(chunk_finished_req_ids)))

    sched.input_coordinator = SimpleNamespace(
        process_pending_chunks=process_pending_chunks,
        process_pending_full_payload_inputs=lambda waiting, running, stage_recv_req_ids: None,
        mark_chunk_segments_completed=lambda req_ids: mark_calls.append(set(req_ids)),
    )

    sched._consume_pending_connector_output(model_mode="generation")

    assert sched._omni_pending_upstream_segment_finished == {"req-live"}
    assert chunk_calls == [({"req-live"}, set())]
    assert mark_calls == [{"req-live"}]


def _make_request() -> Request:
    return Request(
        request_id="req-ar-streaming-test",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
        arrival_time=100.0,
        block_hasher=None,
    )


def _make_update(prompt_token_ids: list[int] | None = None) -> StreamingUpdate:
    return StreamingUpdate(
        mm_features=None,
        prompt_token_ids=[10, 20] if prompt_token_ids is None else prompt_token_ids,
        max_tokens=32,
        arrival_time=200.0,
        sampling_params=SamplingParams(max_tokens=16),
    )


def test_stage0_model_runner_final_commit_emits_segment_terminal() -> None:
    sched = _make_scheduler(stage_id=0, async_chunk=True, model_stage="thinker")
    session = _make_request()
    session.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    session.streaming_queue = deque()
    sched.requests = {session.request_id: session}
    finished = []

    def fake_finish_requests(request_ids, status):
        finished.append((request_ids, status))

    sched.finish_requests = fake_finish_requests
    final_commit = _make_request()

    sched.add_request(final_commit)

    assert sched._omni_pending_segment_finished == {session.request_id}
    assert finished == [(session.request_id, RequestStatus.FINISHED_STOPPED)]
    assert sched.has_requests()


def test_stage0_model_runner_local_segment_stop_emits_segment_terminal() -> None:
    sched = _make_scheduler(stage_id=0, async_chunk=True, model_stage="thinker")

    assert sched._should_emit_segment_terminal_after_local_stop(upstream_segment_finished=False)
    assert not sched._should_emit_segment_terminal_after_local_stop(upstream_segment_finished=True)


def test_stage0_non_runner_local_segment_stop_uses_legacy_adapter() -> None:
    sched = _make_scheduler(stage_id=0, async_chunk=False, model_stage="thinker")

    assert not sched._should_emit_segment_terminal_after_local_stop(upstream_segment_finished=False)


def test_stage0_running_final_commit_uses_base_stream_queue() -> None:
    sched = _make_scheduler(stage_id=0, async_chunk=True, model_stage="thinker")
    session = _make_request()
    session.status = RequestStatus.RUNNING
    session.streaming_queue = deque()
    sched.requests = {session.request_id: session}
    finish_called = False

    def fake_finish_requests(request_ids, status):
        nonlocal finish_called
        finish_called = True

    sched.finish_requests = fake_finish_requests
    final_commit = _make_request()

    assert not sched._maybe_finish_waiting_streaming_segment(final_commit)
    assert not finish_called
    assert sched._omni_pending_segment_finished == set()

    sched.add_request(final_commit)

    assert not finish_called
    assert list(session.streaming_queue) == [None]
    assert sched._omni_pending_segment_finished == set()


def test_stage0_queued_final_commit_emits_runner_segment_terminal() -> None:
    sched = _make_scheduler(stage_id=0, async_chunk=True, model_stage="thinker")
    session = _make_request()
    session.status = RequestStatus.FINISHED_STOPPED
    session.resumable = True
    session.streaming_queue = deque([None])

    assert sched._handle_stopped_request(session)
    assert sched._omni_pending_segment_finished == {session.request_id}
    assert list(session.streaming_queue) == []
    assert sched.has_requests()


def test_non_runner_queued_final_commit_uses_base_stream_finish() -> None:
    sched = _make_scheduler(stage_id=0, async_chunk=False, model_stage="thinker")
    session = _make_request()
    session.status = RequestStatus.FINISHED_STOPPED
    session.resumable = True
    session.streaming_queue = deque([None])

    assert sched._handle_stopped_request(session)
    assert sched._omni_pending_segment_finished == set()


def test_final_output_stage_segment_stop_waits_for_upstream_chunk() -> None:
    sched = _make_scheduler(stage_id=2, async_chunk=True, model_stage="code2wav")
    sched.input_coordinator = object()

    assert sched._wait_for_upstream_chunk_after_segment_stop(upstream_segment_finished=True)
    assert not sched._wait_for_upstream_chunk_after_segment_stop(upstream_segment_finished=False)

    stage0 = _make_scheduler(stage_id=0, async_chunk=True, model_stage="thinker")
    stage0.input_coordinator = object()
    assert not stage0._wait_for_upstream_chunk_after_segment_stop(True)

    talker = _make_scheduler(stage_id=1, async_chunk=True, model_stage="talker")
    talker.input_coordinator = object()
    assert not talker._wait_for_upstream_chunk_after_segment_stop(True)


def test_downstream_model_runner_streaming_update_waits_for_connector_payload() -> None:
    sched = _make_scheduler(stage_id=1, async_chunk=True, model_stage="talker")
    reset_req_ids = []
    sched.input_coordinator = SimpleNamespace(
        reset_request_segment_state=reset_req_ids.append,
    )
    sched.skipped_waiting = []
    sched.num_waiting_for_streaming_input = 1
    session = _make_request()
    session.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    session.append_output_token_ids([7, 8, 9])
    session.num_computed_tokens = 12
    session.num_output_placeholders = 1
    session.spec_token_ids = [-1]

    sched._update_request_as_session(session, _make_update([10, 20]))

    assert reset_req_ids == [session.request_id]
    assert session.status == RequestStatus.WAITING
    assert session.prompt_token_ids == [1, 2, 3]
    assert list(session._all_token_ids) == [1, 2, 3]
    assert session.num_prompt_tokens == 3
    assert session._output_token_ids == []
    assert session.num_computed_tokens == 0
    assert session.num_output_placeholders == 0
    assert session.spec_token_ids == []
    assert sched.num_waiting_for_streaming_input == 0
    assert session.arrival_time == 200.0
    assert session.sampling_params.max_tokens == 16


def test_stage0_streaming_update_discards_outstanding_async_placeholder_token() -> None:
    sched = _make_scheduler(stage_id=0)
    session = _make_request()
    session.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    session.append_output_token_ids([7, 8, 9])
    session.num_computed_tokens = 6
    session.num_output_placeholders = 1
    session.spec_token_ids = [-1]

    sched._update_request_as_session(session, _make_update([10, 20]))

    assert session.async_tokens_to_discard == 1
    assert session.num_output_placeholders == 0
    assert session.spec_token_ids == []
    # The async placeholder makes token 9 unconfirmed, so only 7 and 8 are
    # carried into the next streaming prompt before the new chunk tokens.
    assert session.prompt_token_ids == [1, 2, 3, 7, 8, 10, 20]
    assert list(session._all_token_ids) == [1, 2, 3, 7, 8, 10, 20]
    assert session._output_token_ids == []
    assert session.num_prompt_tokens == 7
    assert sched._new_prompt_len_snapshot[session.request_id] == 2


def test_stage0_streaming_update_keeps_all_computed_tokens_without_placeholder() -> None:
    sched = _make_scheduler(stage_id=0)
    session = _make_request()
    session.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    session.append_output_token_ids([7, 8, 9])
    session.num_computed_tokens = 6
    session.num_output_placeholders = 0

    sched._update_request_as_session(session, _make_update([10, 20]))

    assert getattr(session, "async_tokens_to_discard", 0) == 0
    assert session.num_output_placeholders == 0
    assert session.prompt_token_ids == [1, 2, 3, 7, 8, 9, 10, 20]
    assert list(session._all_token_ids) == [1, 2, 3, 7, 8, 9, 10, 20]
    assert session._output_token_ids == []
    assert session.num_prompt_tokens == 8
    assert sched._new_prompt_len_snapshot[session.request_id] == 2
