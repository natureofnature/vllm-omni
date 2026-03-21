from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_omni.payload_span import (
    CACHED_THINKER_DECODE_EMBEDDINGS_KEY,
    CACHED_THINKER_DECODE_TOKEN_END_KEY,
    CACHED_THINKER_DECODE_TOKEN_START_KEY,
)
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner
from vllm_omni.worker.gpu_generation_model_runner import GPUGenerationModelRunner
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class DummyBuffer:
    """A minimal buffer wrapper that exposes the `.gpu` attribute."""

    def __init__(self, t: torch.Tensor):
        self.gpu = t


class DummyInputBatch:
    """A minimal input batch that only provides `req_ids`."""

    def __init__(self, req_ids):
        self.req_ids = req_ids


class DummyPersistentInputBatch:
    """A minimal persistent batch for exercising _update_states."""

    def __init__(self, req_ids):
        self.req_ids = list(req_ids)
        self.req_id_to_index = {req_id: idx for idx, req_id in enumerate(self.req_ids)}
        self.prev_req_id_to_index = None
        self.removed_req_ids = []

    def remove_request(self, req_id):
        self.removed_req_ids.append(req_id)
        if req_id in self.req_id_to_index:
            del self.req_id_to_index[req_id]
        self.req_ids = [rid for rid in self.req_ids if rid != req_id]

    def add_request(self, request):
        self.req_ids.append(request.req_id)
        self.req_id_to_index[request.req_id] = len(self.req_ids) - 1

    def update_req_spec_token_ids(self, request, scheduled_spec_tokens):
        return None

    def condense(self):
        return None

    def refresh_metadata(self):
        return None


class DummyReqState:
    """A minimal request state container."""

    pass


class MiMoAudioForConditionalGeneration(torch.nn.Module):
    """Dummy model whose class name must exactly match the production check."""

    def __init__(self):
        super().__init__()

    # No real forward needed for these tests.


class DummyTalkerMTP(torch.nn.Module):
    """A fake talker_mtp module for deterministic CPU testing."""

    def forward(self, req_input_ids, req_embeds, last_talker_hidden, text_step):
        # Deterministic behavior:
        # - output embeds = input embeds + 1
        # - output codes = [[0], [1], ...]
        bsz = req_embeds.shape[0]
        new_embeds = req_embeds + 1.0
        codes = torch.arange(bsz, dtype=torch.int64).view(bsz, 1)
        return new_embeds, codes


@contextmanager
def _noop_forward_context(*args, **kwargs):
    """A no-op context manager to replace vLLM forward context in CPU tests."""
    yield


def _make_runner(req_ids=("r1", "r2"), hidden_size=4):
    # Create an instance without calling OmniGPUModelRunner.__init__
    runner = object.__new__(OmniGPUModelRunner)

    # Minimal attributes used by OmniGPUModelRunner._talker_mtp_forward
    runner.input_batch = DummyInputBatch(list(req_ids))
    runner.requests = {rid: DummyReqState() for rid in req_ids}
    runner.model_intermediate_buffer = {}

    # query_start_loc.cpu[req_index] is used to locate the token position
    # in the flattened `inputs_embeds`.
    runner.query_start_loc = type("QSL", (), {})()
    # Map: r1 -> offset 0, r2 -> offset 3
    runner.query_start_loc.cpu = torch.tensor([0, 3], dtype=torch.int32)

    bsz = len(req_ids)
    runner.talker_mtp_input_ids = DummyBuffer(torch.zeros((bsz,), dtype=torch.int64))
    runner.talker_mtp_inputs_embeds = DummyBuffer(torch.zeros((bsz, hidden_size), dtype=torch.float32))
    runner.last_talker_hidden = DummyBuffer(torch.zeros((bsz, hidden_size), dtype=torch.float32))
    runner.text_step = DummyBuffer(torch.zeros((bsz, hidden_size), dtype=torch.float32))

    runner.talker_mtp = DummyTalkerMTP()
    runner.model = SimpleNamespace(talker_mtp_output_key="code_predictor_codes")
    runner.vllm_config = object()

    # Provide a minimal implementation that returns the expected 4-tuple.
    def _determine_batch_execution_and_padding(**kwargs):
        return None, object(), None, None, None

    runner._determine_batch_execution_and_padding = _determine_batch_execution_and_padding

    # Use the real merge method from OmniGPUModelRunner.
    return runner


def _make_runner_for_update_states(req_ids=("r1", "r2")):
    runner = object.__new__(OmniGPUModelRunner)
    runner.requests = {rid: DummyReqState() for rid in req_ids}
    runner.model_intermediate_buffer = {rid: {"persisted": rid} for rid in req_ids}
    runner.num_prompt_logprobs = {rid: 1 for rid in req_ids}
    runner.input_batch = DummyPersistentInputBatch(req_ids)
    runner.encoder_cache = {}
    runner.use_async_scheduling = False
    runner.is_pooling_model = False
    runner.uses_mrope = False
    runner.uses_xdrope_dim = 0
    runner._may_reorder_batch = lambda scheduler_output: None
    return runner


def _make_scheduler_output(*, scheduled_req_ids=(), finished_req_ids=()):
    return SimpleNamespace(
        finished_req_ids=list(finished_req_ids),
        free_encoder_mm_hashes=[],
        num_scheduled_tokens={req_id: 1 for req_id in scheduled_req_ids},
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[],
            resumed_req_ids=[],
            num_computed_tokens=[],
            new_block_ids=[],
            num_output_tokens=[],
            new_token_ids=[],
            all_token_ids={},
        ),
        scheduled_new_reqs=[],
        scheduled_spec_decode_tokens={},
    )


def _make_runner_for_mimo(req_id="r_mimo"):
    """Create a minimal runner with MiMoAudio-like model and request state."""
    runner = object.__new__(OmniGPUModelRunner)
    runner.model = MiMoAudioForConditionalGeneration()

    # Minimal vllm_config / model_config used by helper.
    class _DummyModelConfig:
        async_chunk = False

    class _DummyVllmConfig:
        model_config = _DummyModelConfig()

    runner.vllm_config = _DummyVllmConfig()

    # Attach a single request state with mm_features.
    req_state = DummyReqState()
    req_state.mm_features = ["mm_feature_obj"]

    runner.requests = {req_id: req_state}

    return runner


def test_talker_mtp_forward_cpu_updates_inputs_and_info(monkeypatch):
    # Patch the module-level `set_forward_context` symbol used inside
    # OmniGPUModelRunner._talker_mtp_forward.
    import vllm_omni.worker.gpu_model_runner as mod  # Must be the same module that defines OmniGPUModelRunner

    monkeypatch.setattr(mod, "set_forward_context", _noop_forward_context)

    runner = _make_runner(req_ids=("r1", "r2"), hidden_size=4)

    def fake_determine(self, num_tokens, num_reqs, num_scheduled_tokens_np, max_num_scheduled_tokens, use_cascade_attn):
        batch_desc = SimpleNamespace(num_tokens=int(num_tokens))
        return (False, batch_desc, None, None, None)

    monkeypatch.setattr(runner, "_determine_batch_execution_and_padding", fake_determine.__get__(runner, type(runner)))

    # Initialize per-request embeds (batch-major inside talker_mtp_inputs_embeds)
    runner.talker_mtp_inputs_embeds.gpu[0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    runner.talker_mtp_inputs_embeds.gpu[1] = torch.tensor([10.0, 20.0, 30.0, 40.0])

    # Flattened `inputs_embeds`: offsets 0 and 3 will be overwritten
    inputs_embeds = torch.zeros((6, 4), dtype=torch.float32)

    # Call the original implementation from OmniGPUModelRunner (no re-implementation)
    OmniGPUModelRunner._talker_mtp_forward(runner, ["r1", "r2"], inputs_embeds)

    # Validate embeds were written back (+1)
    assert torch.allclose(inputs_embeds[0], torch.tensor([2.0, 3.0, 4.0, 5.0]))
    assert torch.allclose(inputs_embeds[3], torch.tensor([11.0, 21.0, 31.0, 41.0]))

    # Validate per-request runtime buffer was updated
    info_r1 = runner.model_intermediate_buffer["r1"]
    info_r2 = runner.model_intermediate_buffer["r2"]
    assert int(info_r1["code_predictor_codes"][0, 0]) == 0
    assert int(info_r2["code_predictor_codes"][0, 0]) == 1


def test_talker_mtp_forward_cpu_empty_batch_noop(monkeypatch):
    import vllm_omni.worker.gpu_model_runner as mod

    monkeypatch.setattr(mod, "set_forward_context", _noop_forward_context)

    runner = _make_runner(req_ids=("r1",), hidden_size=4)

    inputs_embeds = torch.randn((2, 4))
    before = inputs_embeds.clone()

    OmniGPUModelRunner._talker_mtp_forward(runner, [], inputs_embeds)

    # Ensure no changes were made
    assert torch.allclose(inputs_embeds, before)


def test_update_intermediate_buffer_writes_to_buffer(monkeypatch):
    """Validate that _update_intermediate_buffer writes request runtime state."""
    import vllm_omni.worker.gpu_model_runner as mod

    monkeypatch.setattr(mod, "set_forward_context", _noop_forward_context)

    runner = _make_runner(req_ids=("r1",), hidden_size=4)

    update = {"my_tensor": torch.tensor([1.0, 2.0]), "my_list": [3, 4]}
    OmniGPUModelRunner._update_intermediate_buffer(runner, "r1", update)

    # Forward: buffer is populated
    assert "r1" in runner.model_intermediate_buffer
    buf = runner.model_intermediate_buffer["r1"]
    assert torch.allclose(buf["my_tensor"], torch.tensor([1.0, 2.0]))
    assert buf["my_list"] == [3, 4]


def test_update_intermediate_buffer_accumulates():
    """Validate that successive merges accumulate keys in the buffer."""
    runner = _make_runner(req_ids=("r1",), hidden_size=4)

    OmniGPUModelRunner._update_intermediate_buffer(runner, "r1", {"a": torch.tensor([1.0])})
    OmniGPUModelRunner._update_intermediate_buffer(runner, "r1", {"b": torch.tensor([2.0])})

    buf = runner.model_intermediate_buffer["r1"]
    assert "a" in buf and "b" in buf
    assert torch.allclose(buf["a"], torch.tensor([1.0]))
    assert torch.allclose(buf["b"], torch.tensor([2.0]))


def test_update_intermediate_buffer_skips_empty_update():
    """Validate that an empty update dict is a no-op."""
    runner = _make_runner(req_ids=("r1",), hidden_size=4)

    OmniGPUModelRunner._update_intermediate_buffer(runner, "r1", {})

    assert "r1" not in runner.model_intermediate_buffer


def test_update_intermediate_buffer_skips_unknown_req_id():
    """Validate that merge is a no-op when req_id is not in self.requests."""
    runner = _make_runner(req_ids=("r1",), hidden_size=4)

    OmniGPUModelRunner._update_intermediate_buffer(runner, "unknown_req", {"key": torch.tensor([1.0])})

    assert "unknown_req" not in runner.model_intermediate_buffer


def test_update_intermediate_buffer_replaces_stale_cached_decode_span():
    runner = _make_runner(req_ids=("r1",), hidden_size=4)
    runner.model_intermediate_buffer["r1"] = {
        CACHED_THINKER_DECODE_EMBEDDINGS_KEY: torch.tensor([[1.0], [2.0]], dtype=torch.float32),
        CACHED_THINKER_DECODE_TOKEN_START_KEY: 0,
        CACHED_THINKER_DECODE_TOKEN_END_KEY: 2,
    }

    OmniGPUModelRunner._update_intermediate_buffer(
        runner,
        "r1",
        {
            CACHED_THINKER_DECODE_EMBEDDINGS_KEY: torch.tensor([[73.0], [74.0]], dtype=torch.float32),
            CACHED_THINKER_DECODE_TOKEN_START_KEY: 73,
            CACHED_THINKER_DECODE_TOKEN_END_KEY: 75,
        },
    )

    buf = runner.model_intermediate_buffer["r1"]
    assert buf[CACHED_THINKER_DECODE_TOKEN_START_KEY] == 73
    assert buf[CACHED_THINKER_DECODE_TOKEN_END_KEY] == 75
    assert torch.allclose(
        buf[CACHED_THINKER_DECODE_EMBEDDINGS_KEY].to(torch.float32),
        torch.tensor([[73.0], [74.0]], dtype=torch.float32),
    )


def test_maybe_attach_mimo_audio_req_infos_enriches_dict():
    runner = _make_runner_for_mimo()
    req_id = "r_mimo"
    req_state = runner.requests[req_id]

    # Existing req_infos should be copied and enriched, not mutated in place.
    original_req_infos = {"existing": 1}
    enriched = OmniGPUModelRunner._maybe_attach_mimo_audio_req_infos(runner, req_state, original_req_infos, req_id)

    assert enriched is not original_req_infos
    assert enriched["existing"] == 1
    # mm_features should be filled from req_state when missing
    assert enriched["mm_features"] == req_state.mm_features
    # req_id should always be attached
    assert enriched["req_id"] == req_id


def test_maybe_attach_mimo_audio_req_infos_no_req_state_returns_input():
    runner = _make_runner_for_mimo()
    req_id = "missing"
    req_state = None
    req_infos = {"k": "v"}

    result = OmniGPUModelRunner._maybe_attach_mimo_audio_req_infos(runner, req_state, req_infos, req_id)

    # When no req_state, helper should be a no-op.
    assert result is req_infos


def test_update_states_keeps_buffer_for_unscheduled_requests(monkeypatch):
    import vllm_omni.worker.gpu_model_runner as mod

    monkeypatch.setattr(mod, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True))

    runner = _make_runner_for_update_states(("r1", "r2"))
    runner._get_valid_sampled_token_count = lambda: []
    scheduler_output = _make_scheduler_output(scheduled_req_ids=("r2",))

    OmniGPUModelRunner._update_states(runner, scheduler_output)

    assert "r1" in runner.model_intermediate_buffer
    assert runner.model_intermediate_buffer["r1"]["persisted"] == "r1"
    assert "r1" not in runner.input_batch.req_id_to_index
    assert "r1" in runner.requests


def test_update_states_clears_buffer_for_finished_requests(monkeypatch):
    import vllm_omni.worker.gpu_model_runner as mod

    monkeypatch.setattr(mod, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True))

    runner = _make_runner_for_update_states(("r1", "r2"))
    runner._get_valid_sampled_token_count = lambda: []
    scheduler_output = _make_scheduler_output(scheduled_req_ids=("r2",), finished_req_ids=("r1",))

    OmniGPUModelRunner._update_states(runner, scheduler_output)

    assert "r1" not in runner.model_intermediate_buffer
    assert "r1" not in runner.requests
    assert "r1" not in runner.input_batch.req_id_to_index


def test_update_states_seeds_initial_model_buffer(monkeypatch):
    import vllm_omni.worker.gpu_model_runner as mod

    monkeypatch.setattr(mod, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True))

    runner = _make_runner_for_update_states(())
    runner._get_valid_sampled_token_count = lambda: []
    new_req = SimpleNamespace(
        req_id="r1",
        prompt_token_ids=[101],
        prompt_embeds=None,
        mm_features=None,
        sampling_params=None,
        pooling_params=None,
        block_ids=([0],),
        num_computed_tokens=0,
        lora_request=None,
        initial_model_buffer={"speaker": ["alice"]},
    )
    scheduler_output = _make_scheduler_output()
    scheduler_output.scheduled_new_reqs = [new_req]

    OmniGPUModelRunner._update_states(runner, scheduler_output)

    assert runner.model_intermediate_buffer["r1"] == {"speaker": ["alice"]}
    assert runner.input_batch.req_ids == ["r1"]


def test_sync_local_stage_payloads_merges_without_scheduler_relay():
    runner = _make_runner(req_ids=("r1",), hidden_size=4)
    runner._local_stage_payload_cache = {
        "r1": {
            "new_tensor": torch.tensor([2.0]),
            "new_list": [3, 4],
        }
    }
    runner.model_intermediate_buffer["r1"] = {
        "persisted_tensor": torch.tensor([1.0]),
        "persisted_list": [1, 2],
    }

    OmniGPUModelRunner._sync_local_stage_payloads(runner)

    assert "r1" in runner.model_intermediate_buffer
    merged = runner.model_intermediate_buffer["r1"]
    assert torch.allclose(merged["persisted_tensor"], torch.tensor([1.0]))
    assert merged["persisted_list"] == [1, 2]
    assert torch.allclose(merged["new_tensor"], torch.tensor([2.0]))
    assert merged["new_list"] == [3, 4]
    assert runner._local_stage_payload_cache == {}


def test_sync_local_stage_payloads_skips_unknown_req_id():
    runner = _make_runner(req_ids=("r1",), hidden_size=4)
    runner._local_stage_payload_cache = {
        "unknown_req": {
            "new_tensor": torch.tensor([9.0]),
        }
    }
    runner._request_ids_mapping = {"unknown_req": "ext-unknown"}
    runner._request_payload = {
        "ext-unknown": {"stale": True},
        "unknown_req": {"stale": True},
    }
    runner._local_request_metadata = {"unknown_req": {"left_context_size": 7}}
    runner._pending_load_reqs = {}
    runner._finished_load_reqs = {"unknown_req"}
    runner._chunk_ready_req_ids = {"unknown_req"}
    runner._chunk_finished_req_ids = {"unknown_req"}
    runner._chunk_stream_completed = {"unknown_req"}
    runner._batch_recv_results = {"unknown_req": {"engine_inputs": True}}
    runner._stage_recv_req_ids = {"unknown_req"}
    runner._get_req_chunk = {"unknown_req": 3}
    runner.model_intermediate_buffer["unknown_req"] = {"stale": True}

    OmniGPUModelRunner._sync_local_stage_payloads(runner)

    assert "unknown_req" not in runner.model_intermediate_buffer
    assert runner._request_payload == {}
    assert runner._request_ids_mapping == {}
    assert runner._local_request_metadata == {}
    assert runner._pending_load_reqs == {}
    assert runner._finished_load_reqs == set()
    assert runner._chunk_ready_req_ids == set()
    assert runner._chunk_finished_req_ids == set()
    assert runner._chunk_stream_completed == set()
    assert runner._batch_recv_results == {}
    assert runner._stage_recv_req_ids == set()
    assert runner._get_req_chunk == {}
    assert runner._local_stage_payload_cache == {}


def test_sync_local_stage_payloads_keeps_pending_recv_request():
    runner = _make_runner(req_ids=(), hidden_size=4)
    runner.requests = {}
    runner._local_stage_payload_cache = {
        "req-1": {
            "thinker_prefill_embeddings": torch.ones(1, 1),
        }
    }
    runner._pending_load_reqs = {"req-1": object()}
    runner.drop_inactive_request_runtime_state = MagicMock()

    OmniGPUModelRunner._sync_local_stage_payloads(runner)

    runner.drop_inactive_request_runtime_state.assert_not_called()
    assert "req-1" in runner._pending_load_reqs
    assert torch.allclose(
        runner.model_intermediate_buffer["req-1"]["thinker_prefill_embeddings"],
        torch.ones(1, 1),
    )
    assert runner._local_stage_payload_cache == {}


def test_generation_execute_model_idle_path_keeps_connector_signals(monkeypatch):
    import vllm_omni.worker.gpu_generation_model_runner as mod

    runner = object.__new__(GPUGenerationModelRunner)
    runner.execute_model_state = None
    runner._idle_log_counter = 0
    runner._stage_id = 2
    runner.register_chunk_recv = MagicMock()
    runner.recv_stage_inputs = MagicMock()
    runner.cleanup_finished_request = MagicMock()
    runner.synchronize_input_prep = lambda: _noop_forward_context()
    runner._update_states = MagicMock()
    runner.prune_inactive_requests = MagicMock()
    runner.requests = {}
    runner.model_config = SimpleNamespace(async_chunk=False)
    runner.vllm_config = SimpleNamespace(model_config=SimpleNamespace(enable_return_routed_experts=False))

    expected = SimpleNamespace(source="empty-with-signals")
    runner._empty_output_with_connector_signals = MagicMock(return_value=expected)

    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=0,
        pending_chunk_registrations=[],
        pending_input_registrations=[],
        finished_req_ids=set(),
        preempted_req_ids=[],
    )

    monkeypatch.setattr(mod, "record_function_or_nullcontext", _noop_forward_context)
    monkeypatch.setattr(mod, "has_kv_transfer_group", lambda: False)
    monkeypatch.setattr(mod, "has_ec_transfer", lambda: False)

    output = GPUGenerationModelRunner.execute_model(runner, scheduler_output)

    assert output is expected
    runner._empty_output_with_connector_signals.assert_called_once_with()


def test_ar_execute_model_no_forward_attaches_connector_signals(monkeypatch):
    import vllm_omni.worker.gpu_ar_model_runner as mod

    runner = object.__new__(GPUARModelRunner)
    runner.execute_model_state = None
    runner._ar_log_counter = 0
    runner._stage_id = 1
    runner._pending_batch_send = {}
    runner._async_chunk = False
    runner.requests = {}
    runner.kv_caches = {}
    runner.register_chunk_recv = MagicMock()
    runner.recv_stage_inputs = MagicMock()
    runner.cleanup_finished_request = MagicMock()
    runner.mark_kv_transfer = MagicMock()
    runner.drain_pending_kv_transfers = MagicMock(return_value={})
    runner.send_kv_cache = MagicMock(return_value=[])
    runner.ack_kv_transfers = MagicMock()
    runner.synchronize_input_prep = lambda: _noop_forward_context()
    runner._update_states = MagicMock()
    runner.prune_inactive_requests = MagicMock()
    runner.flush_batch_outputs = MagicMock()
    runner.parallel_config = SimpleNamespace(
        distributed_executor_backend="",
        data_parallel_size=1,
    )
    runner.cache_config = SimpleNamespace(
        block_size=16,
        cache_dtype="float16",
        kv_sharing_fast_prefill=False,
    )
    runner.vllm_config = SimpleNamespace(model_config=SimpleNamespace(enable_return_routed_experts=False))
    runner._resolve_transfer_request_id = lambda req_id: req_id

    raw_result = SimpleNamespace(source="kv-no-forward")
    wrapped_result = SimpleNamespace(source="attached")
    runner.kv_connector_no_forward = MagicMock(return_value=raw_result)
    runner.attach_omni_connector_output = MagicMock(return_value=wrapped_result)

    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=0,
        pending_chunk_registrations=[],
        pending_input_registrations=[],
        finished_req_ids=set(),
        finished_requests_needing_kv_transfer={},
        preempted_req_ids=[],
    )

    monkeypatch.setattr(mod, "record_function_or_nullcontext", _noop_forward_context)
    monkeypatch.setattr(mod, "has_ec_transfer", lambda: False)
    monkeypatch.setattr(mod, "has_kv_transfer_group", lambda: True)

    output = GPUARModelRunner.execute_model(runner, scheduler_output)

    assert output is wrapped_result
    runner.kv_connector_no_forward.assert_called_once_with(
        scheduler_output,
        runner.vllm_config,
    )
    runner.attach_omni_connector_output.assert_called_once_with(raw_result)
