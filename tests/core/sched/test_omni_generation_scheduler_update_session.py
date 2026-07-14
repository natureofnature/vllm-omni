"""Unit tests for generation streaming session replacement.

These tests pin the behavior of `_update_request_as_session` against
current vLLM `Request` / `StreamingUpdate` (and Omni patches). When upgrading
vLLM, failures here should highlight incompatible changes to request state or
update payloads early.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

# Imports must run in this order: vllm_omni applies patches to vllm.v1.request before
# Request / StreamingUpdate are bound in this module. Ruff isort would reorder them.
# isort: off
import vllm_omni  # noqa: F401 - import for side effects (patch vLLM)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler
from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin
from vllm_omni.outputs import OmniConnectorOutput

# isort: on

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _SkippedWaitingStub:
    def __contains__(self, request: Request) -> bool:
        return False


class _ChunkTransferAdapterStub:
    def __init__(self) -> None:
        self.segment_finished_requests: set[str] = set()


class _SchedulerStub(OmniGenerationScheduler):
    """Minimal scheduler surface required by OmniGenerationScheduler."""

    def __init__(self, *, log_stats: bool = False) -> None:
        self.num_waiting_for_streaming_input = 0
        self.log_stats = log_stats
        self.chunk_transfer_adapter = _ChunkTransferAdapterStub()
        self.skipped_waiting = _SkippedWaitingStub()

    def _enqueue_waiting_request(self, session: Request) -> None:
        raise AssertionError("unexpected enqueue for skipped_waiting miss")


def _make_request(**kwargs) -> Request:
    sp = SamplingParams(max_tokens=8)
    defaults = dict(
        request_id="req-mixin-test",
        prompt_token_ids=[1, 2, 3],
        sampling_params=sp,
        pooling_params=None,
        arrival_time=100.0,
        block_hasher=None,
    )
    defaults.update(kwargs)
    return Request(**defaults)


def _make_update(**kwargs) -> StreamingUpdate:
    sp_new = SamplingParams(max_tokens=16)
    defaults = dict(
        mm_features=None,
        prompt_token_ids=[10, 20],
        max_tokens=32,
        arrival_time=200.0,
        sampling_params=sp_new,
    )
    defaults.update(kwargs)
    return StreamingUpdate(**defaults)


class TestReplaceSessionWithStreamingUpdate:
    def test_resets_tokens_and_prompt_from_update(self) -> None:
        sched = _SchedulerStub()
        session = _make_request()
        session.append_output_token_ids([7, 8])
        session.num_computed_tokens = 99
        session.status = RequestStatus.WAITING_FOR_STREAMING_REQ

        update = _make_update(prompt_token_ids=[40, 41, 42])
        sched.num_waiting_for_streaming_input = 3
        sched._update_request_as_session(session, update)

        assert session._output_token_ids == []
        assert list(session._all_token_ids) == [40, 41, 42]
        assert session.prompt_token_ids == [40, 41, 42]
        assert session.num_computed_tokens == 0
        assert session.num_prompt_tokens == 3
        assert session.arrival_time == 200.0
        assert session.sampling_params is update.sampling_params
        assert session.status == RequestStatus.WAITING
        assert sched.num_waiting_for_streaming_input == 2

    def test_none_prompt_token_ids_becomes_empty(self) -> None:
        sched = _SchedulerStub()
        session = _make_request()
        session.status = RequestStatus.RUNNING
        update = _make_update(prompt_token_ids=None)
        sched._update_request_as_session(session, update)

        assert session.prompt_token_ids == ()
        assert list(session._all_token_ids) == []
        assert session.num_prompt_tokens == 0
        assert sched.num_waiting_for_streaming_input == 0

    def test_additional_information_cleared_when_update_omits_it(self) -> None:
        sched = _SchedulerStub()
        session = _make_request()
        if not hasattr(session, "additional_information"):
            pytest.skip("Request has no additional_information (Omni patch inactive?)")
        session.additional_information = {"keep": True}
        session.status = RequestStatus.RUNNING

        base = _make_update()
        if not hasattr(base, "additional_information"):
            pytest.skip("StreamingUpdate has no additional_information (Omni patch inactive?)")
        update = replace(base, additional_information=None)

        sched._update_request_as_session(session, update)
        assert session.additional_information is None

    def test_does_not_decrement_waiting_when_not_streaming_status(self) -> None:
        sched = _SchedulerStub()
        session = _make_request()
        session.status = RequestStatus.RUNNING
        sched.num_waiting_for_streaming_input = 5
        sched._update_request_as_session(session, _make_update())
        assert sched.num_waiting_for_streaming_input == 5

    def test_records_queued_event_when_log_stats_enabled(self) -> None:
        sched = _SchedulerStub(log_stats=True)
        session = _make_request()
        session.status = RequestStatus.WAITING_FOR_STREAMING_REQ
        sched._update_request_as_session(session, _make_update())

        assert session.events
        assert session.events[-1].type == EngineCoreEventType.QUEUED


class _RealignSchedulerStub(OmniSchedulerMixin):
    """Minimal scheduler surface for ``_realign_request_status_to_queues``.

    Pins ``self.requests`` / ``self.running`` / ``self.waiting`` to the
    same shapes the upstream ``Scheduler.finish_requests`` reads, so the
    helper sees realistic state without spinning a real scheduler.
    """

    def __init__(
        self,
        *,
        requests: dict,
        running: list,
        waiting: list,
    ) -> None:
        self.requests = requests
        self.running = running
        self.waiting = waiting


class TestRealignRequestStatusToQueues:
    """Regression for the residual hang described in
    https://github.com/vllm-project/vllm-omni/pull/3774 -- chunk-transfer
    adapter's ``requests_origin_status`` table goes stale on the
    ``waiting → running`` admit transition, and an abort that lands
    before the next deque round-trip stomps stale ``WAITING`` onto a
    request that actually lives in ``self.running``. After
    ``max_num_seqs`` such aborts every ``input_batch`` slot is leaked and
    new requests hang at ``chunks=0``.
    """

    def test_running_with_stale_waiting_status_is_realigned_to_running(
        self,
    ) -> None:
        """The exact race the hang reproduces: a request lives in
        ``self.running`` but its ``status`` is ``WAITING``. After
        realign, status must be ``RUNNING`` so upstream
        ``Scheduler.finish_requests`` removes it from ``self.running``.
        """
        req = _make_request(request_id="req-stale")
        req.status = RequestStatus.WAITING  # stale -- actually in running

        sched = _RealignSchedulerStub(
            requests={req.request_id: req},
            running=[req],
            waiting=[],
        )
        sched._realign_request_status_to_queues([req.request_id])

        assert req.status == RequestStatus.RUNNING

    def test_waiting_with_stale_running_status_is_realigned_to_waiting(
        self,
    ) -> None:
        """Symmetric case: request lives in ``self.waiting`` but status
        is ``RUNNING``. Upstream's else branch should run, so realign to
        ``WAITING``.
        """
        req = _make_request(request_id="req-stale-2")
        req.status = RequestStatus.RUNNING  # stale -- actually in waiting

        sched = _RealignSchedulerStub(
            requests={req.request_id: req},
            running=[],
            waiting=[req],
        )
        sched._realign_request_status_to_queues([req.request_id])

        assert req.status == RequestStatus.WAITING

    def test_already_aligned_status_is_left_unchanged(self) -> None:
        """Healthy case: status matches actual queue. No status mutation
        means no spurious side effects.
        """
        req_running = _make_request(request_id="req-r")
        req_running.status = RequestStatus.RUNNING
        req_waiting = _make_request(request_id="req-w")
        req_waiting.status = RequestStatus.WAITING

        sched = _RealignSchedulerStub(
            requests={
                req_running.request_id: req_running,
                req_waiting.request_id: req_waiting,
            },
            running=[req_running],
            waiting=[req_waiting],
        )
        sched._realign_request_status_to_queues([req_running.request_id, req_waiting.request_id])

        assert req_running.status == RequestStatus.RUNNING
        assert req_waiting.status == RequestStatus.WAITING

    def test_unknown_request_id_is_skipped_silently(self) -> None:
        """Aligning ids that aren't in ``self.requests`` (already freed
        upstream, or never existed) must be a no-op.
        """
        sched = _RealignSchedulerStub(requests={}, running=[], waiting=[])
        sched._realign_request_status_to_queues(["never-existed"])  # no raise

    def test_request_in_neither_queue_is_left_unchanged(self) -> None:
        """If a tracked request is in neither queue (e.g. parked in a
        chunk-transfer deque), realign must not invent a status. The
        adapter / deque purge owns that surface; the realign helper is
        only for the admit-transition staleness.
        """
        req = _make_request(request_id="req-parked")
        req.status = RequestStatus.WAITING_FOR_CHUNK

        sched = _RealignSchedulerStub(
            requests={req.request_id: req},
            running=[],
            waiting=[],
        )
        sched._realign_request_status_to_queues([req.request_id])

        assert req.status == RequestStatus.WAITING_FOR_CHUNK

    def test_request_ids_str_is_treated_as_single_id(self) -> None:
        """Match upstream ``Scheduler.finish_requests`` resolution: a
        bare string is treated as one id, not iterated as characters.
        """
        req = _make_request(request_id="req-s")
        req.status = RequestStatus.WAITING

        sched = _RealignSchedulerStub(
            requests={req.request_id: req},
            running=[req],
            waiting=[],
        )
        sched._realign_request_status_to_queues(req.request_id)

        assert req.status == RequestStatus.RUNNING

    def test_request_ids_none_aligns_every_known_request(self) -> None:
        """``request_ids=None`` matches upstream's "all requests" path.
        Realign must walk every entry in ``self.requests`` and fix any
        stale status it finds.
        """
        stale = _make_request(request_id="req-stale-none")
        stale.status = RequestStatus.WAITING  # but actually in running

        clean = _make_request(request_id="req-clean-none")
        clean.status = RequestStatus.RUNNING

        sched = _RealignSchedulerStub(
            requests={
                stale.request_id: stale,
                clean.request_id: clean,
            },
            running=[stale, clean],
            waiting=[],
        )
        sched._realign_request_status_to_queues(None)

        assert stale.status == RequestStatus.RUNNING
        assert clean.status == RequestStatus.RUNNING

    def test_finished_request_is_skipped(self) -> None:
        """Already-finished requests must not be touched -- they may
        have legitimate finished statuses (FINISHED_STOPPED etc.) that
        a status flip would corrupt.
        """
        req = _make_request(request_id="req-finished")
        req.status = RequestStatus.FINISHED_STOPPED

        sched = _RealignSchedulerStub(
            requests={req.request_id: req},
            running=[req],  # not realistic, but we want to prove the guard fires
            waiting=[],
        )
        sched._realign_request_status_to_queues([req.request_id])

        assert req.status == RequestStatus.FINISHED_STOPPED


class TestPurgeFinishedFromRunning:
    """Regression for the residual ``self.running`` slot leak surface
    paired with ``_realign_request_status_to_queues``: even after
    realign + ``super().finish_requests`` runs, corner cases can leave
    already-finished or untracked entries in ``self.running`` -- e.g.
    a connector cleanup that pops from ``self.requests`` without
    unwinding ``self.running``, or a ``status`` mid-transition when
    finish ran. The defensive post-finish purge sweeps those residues
    so the worker's ``input_batch`` slot never pins a freed request.

    See https://github.com/vllm-project/vllm-omni/pull/3774 discussion
    and the residual-hang reproduction in that PR.
    """

    def test_finished_request_is_purged_from_running(self) -> None:
        """``is_finished()`` request lingering in ``self.running`` must
        be swept so its ``input_batch`` slot is freed."""
        finished = _make_request(request_id="req-finished")
        finished.status = RequestStatus.FINISHED_STOPPED  # is_finished() True

        sched = _RealignSchedulerStub(
            requests={finished.request_id: finished},
            running=[finished],
            waiting=[],
        )
        sched._purge_finished_from_running()

        assert sched.running == []

    def test_untracked_request_is_purged_from_running(self) -> None:
        """Request lingering in ``self.running`` but no longer present
        in ``self.requests`` (already freed by upstream / connector)
        must be swept."""
        untracked = _make_request(request_id="req-untracked")
        untracked.status = RequestStatus.RUNNING

        sched = _RealignSchedulerStub(
            requests={},  # already deleted from self.requests
            running=[untracked],
            waiting=[],
        )
        sched._purge_finished_from_running()

        assert sched.running == []

    def test_healthy_running_is_left_unchanged(self) -> None:
        """Live tracked running requests must not be touched -- the
        purge is defensive, not aggressive."""
        alive = _make_request(request_id="req-alive")
        alive.status = RequestStatus.RUNNING

        sched = _RealignSchedulerStub(
            requests={alive.request_id: alive},
            running=[alive],
            waiting=[],
        )
        sched._purge_finished_from_running()

        assert sched.running == [alive]

    def test_empty_running_is_noop(self) -> None:
        """Empty ``self.running`` must short-circuit cleanly."""
        sched = _RealignSchedulerStub(requests={}, running=[], waiting=[])
        sched._purge_finished_from_running()

        assert sched.running == []

    def test_mixed_alive_and_dead_keeps_only_alive_in_order(self) -> None:
        """Mixed ``self.running`` -- some alive, some finished, some
        untracked. Sweep keeps only the alive ones and preserves their
        relative order."""
        alive_a = _make_request(request_id="req-alive-a")
        alive_a.status = RequestStatus.RUNNING
        finished_b = _make_request(request_id="req-finished-b")
        finished_b.status = RequestStatus.FINISHED_STOPPED
        untracked_c = _make_request(request_id="req-untracked-c")
        untracked_c.status = RequestStatus.RUNNING
        alive_d = _make_request(request_id="req-alive-d")
        alive_d.status = RequestStatus.RUNNING

        sched = _RealignSchedulerStub(
            requests={
                alive_a.request_id: alive_a,
                finished_b.request_id: finished_b,  # tracked but is_finished
                # untracked_c is NOT in self.requests
                alive_d.request_id: alive_d,
            },
            running=[alive_a, finished_b, untracked_c, alive_d],
            waiting=[],
        )
        sched._purge_finished_from_running()

        assert sched.running == [alive_a, alive_d]


class _FinishOnlyInputCoordinatorStub:
    def __init__(self) -> None:
        self.freed: list[str] = []
        self.finished_requests: set[str] = set()
        self.requests_with_ready_chunks: set[str] = set()
        self._completed_chunk_streams: set[str] = set()
        self.restored = False
        self.postprocessed = False

    def free_finished_request(self, request_id: str) -> None:
        self.freed.append(request_id)

    def chunk_stream_completed(self, request_id: str) -> bool:
        return request_id in self.finished_requests or request_id in self._completed_chunk_streams

    def postprocess_scheduler_output(self, _scheduler_output: SchedulerOutput) -> None:
        self.postprocessed = True

    def restore_queues(self, _waiting, _running) -> None:
        self.restored = True


class _FinishOnlySchedulerStub(OmniGenerationScheduler):
    def __init__(
        self,
        request: Request,
        *,
        has_generation_output: bool = True,
        stop_finished: bool = True,
        model_stage: str = "code2wav",
        stage_id: int = 2,
    ) -> None:
        self._async_chunk_coordinator_active = True
        self._pending_finish_reqs: list[Request] = []
        self._deferred_terminal_chunk_req_ids: set[str] = set()
        self._deferred_terminal_request_metadata: dict[str, dict] = {}
        self._reqs_with_generation_output = {request.request_id} if has_generation_output else set()
        self._omni_pending_upstream_segment_finished: set[str] = set()
        self._stop_finished = stop_finished
        self.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(
                model_arch="Qwen3OmniMoeForConditionalGeneration",
                model_stage=model_stage,
                stage_id=stage_id,
                async_chunk=True,
            )
        )
        self.requests = {request.request_id: request}
        self.running = [request]
        self.waiting = SimpleNamespace(remove_requests=lambda _reqs: None)
        self.skipped_waiting = SimpleNamespace(remove_requests=lambda _reqs: None)
        self.input_coordinator = _FinishOnlyInputCoordinatorStub()
        self.chunk_transfer_adapter = None
        self.connector = None
        self.perf_metrics = None
        self.kv_cache_manager = SimpleNamespace(take_events=lambda: None)
        self.kv_event_publisher = SimpleNamespace(publish=lambda _batch: None)
        self.structured_output_manager = SimpleNamespace(should_advance=lambda _req: False)
        self.recompute_kv_load_failures = False
        self.finished_req_ids_dict: dict[int, set[str]] = {}
        self.log_stats = False
        self.freed: list[str] = []
        self.handled_stops = 0
        self.reenqueued: list[str] = []

    def _handle_stopped_request(self, request: Request) -> bool:
        self.handled_stops += 1
        return self._stop_finished

    def _free_request(self, request: Request, delay_free_blocks: bool = False):
        self.freed.append(request.request_id)
        self.requests.pop(request.request_id, None)
        return None

    def _enqueue_waiting_request(self, request: Request) -> None:
        self.reenqueued.append(request.request_id)

    def make_stats(self, *args, **kwargs):
        return None


class _ScheduleQueueStub:
    def __init__(self, requests: list[Request] | None = None) -> None:
        self._requests = list(requests or [])

    def __bool__(self) -> bool:
        return bool(self._requests)

    def __len__(self) -> int:
        return len(self._requests)

    def peek_request(self) -> Request:
        return self._requests[0]

    def pop_request(self) -> Request:
        return self._requests.pop(0)

    def prepend_request(self, request: Request) -> None:
        self._requests.insert(0, request)

    def prepend_requests(self, requests) -> None:
        while requests:
            self.prepend_request(requests.pop_request())

    def remove_requests(self, requests: set[Request]) -> None:
        self._requests = [request for request in self._requests if request not in requests]


class _KVBlockStub:
    def get_block_ids(self):
        return ([0],)


class _ScheduleKVCacheManagerStub:
    def __init__(self) -> None:
        self.allocated: list[tuple[str, int]] = []

    def new_step_starts(self) -> None:
        return None

    def allocate_slots(self, request: Request, num_new_tokens: int, **_kwargs):
        self.allocated.append((request.request_id, num_new_tokens))
        return _KVBlockStub()

    def get_num_common_prefix_blocks(self, _request_id: str) -> list[int]:
        return [0]

    def take_new_block_ids(self):
        return []


class _FinalOnlyScheduleStub(OmniGenerationScheduler):
    def __init__(self, request: Request, *, in_waiting: bool) -> None:
        self._async_chunk_coordinator_active = True
        self._pending_finish_reqs: list[Request] = []
        self._deferred_terminal_chunk_req_ids: set[str] = set()
        self._deferred_terminal_request_metadata: dict[str, dict] = {}
        self._reqs_with_generation_output: set[str] = set()
        self._omni_pending_upstream_segment_finished: set[str] = set()
        self.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(
                model_arch="Qwen3OmniMoeForConditionalGeneration",
                model_stage="code2wav",
                stage_id=2,
                async_chunk=True,
            )
        )
        self.requests = {request.request_id: request}
        self.running = [] if in_waiting else [request]
        self.waiting = _ScheduleQueueStub([request] if in_waiting else [])
        self.skipped_waiting = _ScheduleQueueStub()
        self.input_coordinator = _FinishOnlyInputCoordinatorStub()
        self.chunk_transfer_adapter = None
        self.connector = None
        self.ec_connector = None
        self.scheduler_config = SimpleNamespace(enable_chunked_prefill=True)
        self.max_num_scheduled_tokens = 8
        self.max_num_running_reqs = 8
        self._pause_state = PauseState.UNPAUSED
        self.policy = SchedulingPolicy.FCFS
        self.num_lookahead_tokens = 0
        self.kv_cache_manager = _ScheduleKVCacheManagerStub()
        self.kv_cache_config = SimpleNamespace(kv_cache_groups=[object()])
        self.use_v2_model_runner = False
        self.prev_step_scheduled_req_ids: set[str] = set()
        self.needs_kv_cache_zeroing = False
        self.finished_req_ids: set[str] = set()
        self.encoder_cache_manager = SimpleNamespace(get_freed_mm_hashes=lambda: [])
        self.log_stats = False

    def _consume_pending_connector_output(self, model_mode: str) -> None:
        return None

    def _process_pending_input_timeouts(self) -> None:
        return None

    def _make_cached_request_data(self, **_kwargs) -> CachedRequestData:
        return CachedRequestData.make_empty()

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        for req_id, num_tokens in scheduler_output.num_scheduled_tokens.items():
            self.requests[req_id].num_computed_tokens += num_tokens

    def _wrap_omni_scheduler_output(self, scheduler_output: SchedulerOutput) -> SchedulerOutput:
        return scheduler_output


def _empty_scheduler_output() -> SchedulerOutput:
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _model_output(
    *,
    payload_request_id: str | None = None,
    audio: bytes | None = None,
    connector_output: OmniConnectorOutput | None = None,
):
    has_payload = payload_request_id is not None
    return SimpleNamespace(
        sampled_token_ids=[[]] if has_payload else [],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=None,
        multimodal_outputs=[{"audio": audio}] if has_payload else None,
        num_nans_in_logits=None,
        kv_connector_output=None,
        cudagraph_stats=None,
        req_id_to_index={payload_request_id: 0} if has_payload else {},
        routed_experts=None,
        req_ids=[payload_request_id] if has_payload else [],
        omni_connector_output=connector_output or OmniConnectorOutput(),
    )


def _finish_only_model_output(request_id: str):
    return _model_output(
        connector_output=OmniConnectorOutput(chunk_finished_req_ids={request_id}),
    )


def _segment_multimodal_model_output(request_id: str):
    return _model_output(payload_request_id=request_id, audio=b"segment")


def _terminal_multimodal_model_output(request_id: str):
    return _model_output(
        payload_request_id=request_id,
        audio=b"done",
        connector_output=OmniConnectorOutput(
            chunk_finished_req_ids={request_id},
            request_metadata={request_id: {"next_stage_prompt_len": 3}},
        ),
    )


def _other_request_output_with_terminal_signal(request_id: str):
    return _model_output(
        payload_request_id="other",
        audio=b"other",
        connector_output=OmniConnectorOutput(
            chunk_ready_req_ids={request_id},
            chunk_finished_req_ids={request_id},
            request_metadata={request_id: {"next_stage_prompt_len": 3}},
        ),
    )


def _scheduled_output(request_id: str):
    return SimpleNamespace(
        num_scheduled_tokens={request_id: 1},
        scheduled_spec_decode_tokens={},
        num_invalid_spec_tokens=0,
    )


def _assert_stopped(outputs, request: Request):
    client_outputs = outputs[request.client_index].outputs
    assert len(client_outputs) == 1
    output = client_outputs[0]
    assert output.request_id == request.request_id
    assert output.finish_reason.name == "STOP"
    return output


class TestAsyncChunkFinalOnlyScheduling:
    @pytest.mark.parametrize(
        ("in_waiting", "has_output", "prompt_token_ids", "num_computed_tokens"),
        [
            (True, False, [], 0),
            (True, True, [0] * 81, 80),
            (False, False, [1, 2, 3], 3),
            (False, True, [0] * 81, 80),
        ],
        ids=["waiting-placeholder", "waiting-finish", "running-placeholder", "running-finish"],
    )
    def test_completed_final_stream(
        self,
        in_waiting: bool,
        has_output: bool,
        prompt_token_ids: list[int],
        num_computed_tokens: int,
    ) -> None:
        request = _make_request(
            request_id=f"req-final-{in_waiting}-{has_output}",
            prompt_token_ids=prompt_token_ids,
        )
        request.status = RequestStatus.WAITING if in_waiting else RequestStatus.RUNNING
        request.num_computed_tokens = num_computed_tokens
        sched = _FinalOnlyScheduleStub(request, in_waiting=in_waiting)
        sched.input_coordinator.finished_requests.add(request.request_id)
        if has_output:
            sched._reqs_with_generation_output.add(request.request_id)

        output = sched.schedule()

        if has_output:
            assert output.num_scheduled_tokens == {}
            assert sched._pending_finish_reqs == [request]
            assert sched.kv_cache_manager.allocated == []
        else:
            assert output.num_scheduled_tokens == {request.request_id: 1}
            assert sched._pending_finish_reqs == []
            assert sched.kv_cache_manager.allocated == [(request.request_id, 1)]


class TestAsyncChunkFinishOnlyOutput:
    def test_multimodal_request_finishes_on_connector_only_terminal(self) -> None:
        request = _make_request(request_id="req-mm-terminal")
        request.status = RequestStatus.RUNNING
        request.num_computed_tokens = len(request.prompt_token_ids)
        sched = _FinishOnlySchedulerStub(request, has_generation_output=False)

        outputs = sched.update_from_output(
            _empty_scheduler_output(),
            _finish_only_model_output(request.request_id),
        )

        _assert_stopped(outputs, request)
        assert sched.freed == [request.request_id]
        assert sched.input_coordinator.freed == [request.request_id]

    def test_connector_only_segment_terminal_marks_pending_finish_output(self) -> None:
        request = _make_request(request_id="req-mm-segment-terminal")
        request.status = RequestStatus.RUNNING
        request.num_computed_tokens = len(request.prompt_token_ids)
        sched = _FinishOnlySchedulerStub(request, has_generation_output=False)
        model_output = _finish_only_model_output(request.request_id)
        model_output.omni_connector_output.chunk_segment_finished_req_ids = {request.request_id}

        outputs = sched.update_from_output(
            _empty_scheduler_output(),
            model_output,
        )

        output = _assert_stopped(outputs, request)
        assert output.is_segment_finished is True
        assert request.request_id not in sched._omni_pending_upstream_segment_finished

    def test_intermediate_stage_preserves_upstream_terminal(self) -> None:
        request = _make_request(request_id="req-talk-terminal")
        request.status = RequestStatus.RUNNING
        request.num_computed_tokens = len(request.prompt_token_ids)
        sched = _FinishOnlySchedulerStub(
            request,
            has_generation_output=False,
            model_stage="talker",
            stage_id=1,
        )

        outputs = sched.update_from_output(
            _empty_scheduler_output(),
            _finish_only_model_output(request.request_id),
        )

        assert outputs.get(request.client_index) is None
        assert sched._pending_finish_reqs == []
        assert request.request_id in sched.requests

    def test_connector_terminal_with_ready_chunk_waits_for_payload(self) -> None:
        request = _make_request(request_id="req-mm-terminal-codes")
        request.status = RequestStatus.RUNNING
        request.num_computed_tokens = len(request.prompt_token_ids)
        sched = _FinishOnlySchedulerStub(request, has_generation_output=False)

        model_output = _finish_only_model_output(request.request_id)
        model_output.omni_connector_output.chunk_ready_req_ids = {request.request_id}
        model_output.omni_connector_output.request_metadata = {
            request.request_id: {
                "code_predictor_codes": list(request.prompt_token_ids),
            }
        }

        first_outputs = sched.update_from_output(
            _empty_scheduler_output(),
            model_output,
        )

        target_outputs = first_outputs.get(request.client_index)
        assert target_outputs is None or all(
            output.request_id != request.request_id for output in target_outputs.outputs
        )
        assert sched._deferred_terminal_chunk_req_ids == {request.request_id}
        assert sched._deferred_terminal_request_metadata[request.request_id]["code_predictor_codes"] == list(
            request.prompt_token_ids
        )

        scheduler_output = _scheduled_output(request.request_id)
        outputs = sched.update_from_output(
            scheduler_output,
            _segment_multimodal_model_output(request.request_id),
        )

        _assert_stopped(outputs, request)
        assert sched._deferred_terminal_chunk_req_ids == set()

    def test_completed_stream_uses_terminal_metadata_with_padded_prompt(self) -> None:
        request = _make_request(
            request_id="req-mm-terminal-padded-codes",
            prompt_token_ids=[1, 2, 3, 0, 0],
        )
        request.status = RequestStatus.RUNNING
        request.num_computed_tokens = 3
        sched = _FinishOnlySchedulerStub(request, has_generation_output=False)
        sched.input_coordinator._completed_chunk_streams.add(request.request_id)
        sched._deferred_terminal_request_metadata[request.request_id] = {
            "code_predictor_codes": [1, 2, 3],
        }

        outputs = sched.update_from_output(
            _empty_scheduler_output(),
            _model_output(),
        )

        _assert_stopped(outputs, request)

    def test_completed_stream_with_output_finishes_despite_stale_prompt_len(self) -> None:
        request = _make_request(
            request_id="req-mm-terminal-stale-prompt",
            prompt_token_ids=[0] * 81,
        )
        request.status = RequestStatus.RUNNING
        request.num_computed_tokens = 32
        sched = _FinishOnlySchedulerStub(request, has_generation_output=False)
        sched.input_coordinator._completed_chunk_streams.add(request.request_id)
        scheduler_output = _scheduled_output(request.request_id)

        outputs = sched.update_from_output(
            scheduler_output,
            _segment_multimodal_model_output(request.request_id),
        )

        _assert_stopped(outputs, request)
        assert sched.freed == [request.request_id]

    def test_multimodal_request_marks_pending_segment_flush(self) -> None:
        request = _make_request(request_id="req-mm-segment-flush")
        request.status = RequestStatus.RUNNING
        request.num_computed_tokens = len(request.prompt_token_ids)
        sched = _FinishOnlySchedulerStub(request, has_generation_output=False)
        sched._omni_pending_upstream_segment_finished.add(request.request_id)
        scheduler_output = _scheduled_output(request.request_id)

        outputs = sched.update_from_output(
            scheduler_output,
            _segment_multimodal_model_output(request.request_id),
        )

        output = _assert_stopped(outputs, request)
        assert output.is_segment_finished is True
        assert sched.handled_stops == 0
        assert sched.freed == []
        assert sched.reenqueued == [request.request_id]
        assert request.request_id in sched.requests
        assert request.status == RequestStatus.WAITING
        assert sched._omni_pending_upstream_segment_finished == set()

    def test_multimodal_request_finishes_from_stored_coordinator_terminal(self) -> None:
        request = _make_request(request_id="req-mm-stored-terminal")
        request.status = RequestStatus.RUNNING
        request.num_computed_tokens = len(request.prompt_token_ids)
        sched = _FinishOnlySchedulerStub(request, has_generation_output=False)
        sched.input_coordinator.finished_requests.add(request.request_id)

        outputs = sched.update_from_output(
            _empty_scheduler_output(),
            _model_output(),
        )

        _assert_stopped(outputs, request)
        assert sched.freed == [request.request_id]
        assert sched.input_coordinator.freed == [request.request_id]

    def test_multimodal_request_finishes_when_terminal_arrives_with_output(self) -> None:
        request = _make_request(request_id="req-mm-terminal-output")
        request.status = RequestStatus.RUNNING
        request.num_computed_tokens = len(request.prompt_token_ids)
        sched = _FinishOnlySchedulerStub(request, has_generation_output=False)
        scheduler_output = _scheduled_output(request.request_id)

        outputs = sched.update_from_output(
            scheduler_output,
            _terminal_multimodal_model_output(request.request_id),
        )

        _assert_stopped(outputs, request)

    def test_terminal_output_with_segment_finish_does_not_wait_for_next_chunk(self) -> None:
        request = _make_request(request_id="req-mm-terminal-segment-output")
        request.status = RequestStatus.RUNNING
        request.num_computed_tokens = len(request.prompt_token_ids)
        sched = _FinishOnlySchedulerStub(request, has_generation_output=False)
        sched._omni_pending_upstream_segment_finished.add(request.request_id)
        scheduler_output = _scheduled_output(request.request_id)

        outputs = sched.update_from_output(
            scheduler_output,
            _terminal_multimodal_model_output(request.request_id),
        )

        output = _assert_stopped(outputs, request)
        assert output.is_segment_finished is True
        assert sched.handled_stops == 1
        assert sched.freed == [request.request_id]
        assert sched.reenqueued == []
        assert request.request_id not in sched.requests

    def test_terminal_output_uses_current_cycle_segment_finish_signal(self) -> None:
        request = _make_request(request_id="req-mm-terminal-current-segment")
        request.status = RequestStatus.RUNNING
        request.num_computed_tokens = len(request.prompt_token_ids)
        sched = _FinishOnlySchedulerStub(request, has_generation_output=False)
        scheduler_output = _scheduled_output(request.request_id)
        model_output = _terminal_multimodal_model_output(request.request_id)
        model_output.omni_connector_output.chunk_segment_finished_req_ids = {request.request_id}

        outputs = sched.update_from_output(scheduler_output, model_output)

        output = _assert_stopped(outputs, request)
        assert output.is_segment_finished is True
        assert sched.freed == [request.request_id]

    def test_terminal_ready_signal_survives_mixed_batch_output(self) -> None:
        request = _make_request(request_id="req-mm-mixed-terminal")
        request.status = RequestStatus.RUNNING
        request.num_computed_tokens = len(request.prompt_token_ids)
        sched = _FinishOnlySchedulerStub(request, has_generation_output=False)

        first_outputs = sched.update_from_output(
            _empty_scheduler_output(),
            _other_request_output_with_terminal_signal(request.request_id),
        )

        target_outputs = first_outputs.get(request.client_index)
        assert target_outputs is None or all(
            output.request_id != request.request_id for output in target_outputs.outputs
        )
        assert sched._deferred_terminal_chunk_req_ids == {request.request_id}

        scheduler_output = _scheduled_output(request.request_id)
        outputs = sched.update_from_output(
            scheduler_output,
            _segment_multimodal_model_output(request.request_id),
        )

        _assert_stopped(outputs, request)
