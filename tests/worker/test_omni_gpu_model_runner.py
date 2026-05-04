from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner
from vllm_omni.worker.gpu_generation_model_runner import GPUGenerationModelRunner
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner
from vllm_omni.worker.omni_connector_model_runner_mixin import OmniConnectorModelRunnerMixin
from vllm_omni.worker.payload_span import (
    CACHED_THINKER_DECODE_EMBEDDINGS_KEY,
    CACHED_THINKER_DECODE_TOKEN_END_KEY,
    CACHED_THINKER_DECODE_TOKEN_START_KEY,
    THINKER_DECODE_EMBEDDINGS_KEY,
    THINKER_DECODE_TOKEN_END_KEY,
    THINKER_DECODE_TOKEN_START_KEY,
    THINKER_OUTPUT_TOKEN_IDS_KEY,
    cache_thinker_decode_span,
    resolve_thinker_decode_step,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_runner_mro_pins_connector_mixin_before_omni_gpu_runner():
    assert GPUARModelRunner.__mro__[1] is OmniConnectorModelRunnerMixin
    assert GPUARModelRunner.__mro__[2] is OmniGPUModelRunner
    assert GPUGenerationModelRunner.__mro__[1] is OmniConnectorModelRunnerMixin
    assert GPUGenerationModelRunner.__mro__[2] is OmniGPUModelRunner


@pytest.mark.parametrize(
    ("runner_cls", "worker_type"),
    [
        (GPUARModelRunner, "ar"),
        (GPUGenerationModelRunner, "generation"),
    ],
)
def test_runner_constructor_smoke_preserves_base_init_and_connector_setup(monkeypatch, runner_cls, worker_type):
    calls = []

    def _fake_base_init(self, *args, **kwargs):
        calls.append(("base_init", args, kwargs))
        model_config = SimpleNamespace(
            async_chunk=True,
            worker_type=worker_type,
            stage_id=3,
            hf_text_config=SimpleNamespace(hidden_size=16),
        )
        self.max_num_tokens = 32
        self.dtype = torch.float16
        self.model_config = model_config
        self.vllm_config = SimpleNamespace(model_config=model_config)

    def _fake_init_omni_connectors(self, *, vllm_config, model_config, kv_transfer_manager=None):
        calls.append(("init_omni_connectors", vllm_config, model_config, kv_transfer_manager))
        self._test_connector_kv_manager = kv_transfer_manager
        self._test_connector_stage_id = model_config.stage_id

    monkeypatch.setattr(OmniGPUModelRunner, "__init__", _fake_base_init)
    monkeypatch.setattr(OmniConnectorModelRunnerMixin, "init_omni_connectors", _fake_init_omni_connectors)

    if runner_cls is GPUARModelRunner:
        import vllm_omni.worker.gpu_ar_model_runner as ar_mod

        kv_manager = object()

        monkeypatch.setattr(
            GPUARModelRunner,
            "_make_buffer",
            lambda self, *size, dtype, numpy=True: (size, dtype, numpy),
        )
        monkeypatch.setattr(
            ar_mod.OmniKVTransferManager,
            "from_vllm_config",
            staticmethod(
                lambda vllm_config, model_config: calls.append(("build_kv_manager", vllm_config, model_config))
                or kv_manager
            ),
        )

    runner = runner_cls("sentinel", demo=True)

    assert calls[0] == ("base_init", ("sentinel",), {"demo": True})
    if runner_cls is GPUARModelRunner:
        assert calls[1][0] == "build_kv_manager"
    else:
        assert calls[1][0] == "init_omni_connectors"
    assert calls[-1][0] == "init_omni_connectors"
    assert runner._test_connector_stage_id == 3

    if runner_cls is GPUARModelRunner:
        assert runner.hidden_size == 16
        assert runner.kv_transfer_manager is runner._test_connector_kv_manager
        assert runner.input_ids == ((32,), torch.int32, True)
        assert runner.inputs_embeds == ((32, 16), torch.float16, False)
    else:
        assert runner._test_connector_kv_manager is None


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

    def __init__(self, prompt_token_ids=None):
        self.prompt_token_ids = prompt_token_ids


class MiMoAudioForConditionalGeneration(torch.nn.Module):
    """Dummy model whose class name must exactly match the production check."""

    def __init__(self):
        super().__init__()

    # No real forward needed for these tests.


class DummyTalkerMTP(torch.nn.Module):
    """A fake talker_mtp module for deterministic CPU testing."""

    def forward(
        self,
        req_input_ids,
        req_embeds,
        last_talker_hidden,
        text_step,
        do_sample=None,
        temperature=None,
        top_k=None,
        top_p=None,
    ):
        # Deterministic behavior:
        # - output embeds = input embeds + 1
        # - output codes = [[0], [1], ...]
        bsz = req_embeds.shape[0]
        new_embeds = req_embeds + 1.0
        codes = torch.arange(bsz, dtype=torch.int64).view(bsz, 1)
        return new_embeds, codes


class CaptureTalkerMTP(torch.nn.Module):
    """A fake talker_mtp module that records sampling kwargs."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(
        self,
        req_input_ids,
        req_embeds,
        last_talker_hidden,
        text_step,
        do_sample=None,
        temperature=None,
        top_k=None,
        top_p=None,
    ):
        self.calls.append(
            {
                "do_sample": do_sample,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
            }
        )
        codes = torch.zeros((req_embeds.shape[0], 1), dtype=torch.int64)
        return req_embeds, codes


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
    runner.vllm_config = SimpleNamespace(model_config=SimpleNamespace())

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
    runner.use_async_spec_decode = False
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


def test_talker_mtp_forward_passes_qwen3_tts_subtalker_sampling_params_to_talker(monkeypatch):
    import vllm_omni.worker.gpu_model_runner as mod

    monkeypatch.setattr(mod, "set_forward_context", _noop_forward_context)

    runner = _make_runner(req_ids=("r1",), hidden_size=4)
    runner.talker_mtp = CaptureTalkerMTP()
    runner.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            subtalker_sampling_params={
                "do_sample": False,
                "temperature": 0.2,
                "top_k": 9,
                "top_p": 0.55,
            }
        )
    )

    def fake_determine(self, num_tokens, num_reqs, num_scheduled_tokens_np, max_num_scheduled_tokens, use_cascade_attn):
        batch_desc = SimpleNamespace(num_tokens=int(num_tokens))
        return (False, batch_desc, None, None, None)

    monkeypatch.setattr(runner, "_determine_batch_execution_and_padding", fake_determine.__get__(runner, type(runner)))

    inputs_embeds = torch.zeros((2, 4), dtype=torch.float32)
    OmniGPUModelRunner._talker_mtp_forward(runner, ["r1"], inputs_embeds)

    assert runner.talker_mtp.calls == [
        {
            "do_sample": False,
            "temperature": 0.2,
            "top_k": 9,
            "top_p": 0.55,
        }
    ]


def test_update_intermediate_buffer_writes_to_buffer_and_setattr(monkeypatch):
    """Validate that _update_intermediate_buffer writes to model_intermediate_buffer
    (forward path) and mirrors to additional_information_cpu setattr (backward compat)."""
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


def test_update_intermediate_buffer_generation_defers_codec_payload_ahead_of_prompt():
    runner = _make_runner(req_ids=("r1",), hidden_size=4)
    runner._model_mode = "generation"
    runner.requests["r1"].prompt_token_ids = [1, 2]

    OmniGPUModelRunner._update_intermediate_buffer(
        runner,
        "r1",
        {"code_predictor_codes": [10, 11, 12], "finished": True},
    )

    assert runner.model_intermediate_buffer == {}


def test_update_intermediate_buffer_generation_merges_codec_payload_after_prompt_catches_up():
    runner = _make_runner(req_ids=("r1",), hidden_size=4)
    runner._model_mode = "generation"
    payload = {"code_predictor_codes": [10, 11, 12], "finished": True}
    runner.requests["r1"].prompt_token_ids = [1, 2]

    OmniGPUModelRunner._update_intermediate_buffer(runner, "r1", payload)
    assert runner.model_intermediate_buffer == {}

    runner.requests["r1"].prompt_token_ids = [1, 2, 3]
    OmniGPUModelRunner._update_intermediate_buffer(runner, "r1", payload)

    assert runner.model_intermediate_buffer["r1"] == payload


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


def test_cache_thinker_decode_span_merges_incoming_span():
    payload = {
        CACHED_THINKER_DECODE_EMBEDDINGS_KEY: torch.tensor([[10.0], [11.0]], dtype=torch.float32),
        CACHED_THINKER_DECODE_TOKEN_START_KEY: 0,
        CACHED_THINKER_DECODE_TOKEN_END_KEY: 2,
        THINKER_DECODE_EMBEDDINGS_KEY: torch.tensor([[12.0], [13.0]], dtype=torch.float32),
        THINKER_DECODE_TOKEN_START_KEY: 2,
        THINKER_DECODE_TOKEN_END_KEY: 4,
        "num_processed_tokens": 2,
    }
    update = {}

    cache_thinker_decode_span(payload, update, device=torch.device("cpu"), dtype=torch.float32)

    assert update[THINKER_DECODE_EMBEDDINGS_KEY] is None
    assert update[CACHED_THINKER_DECODE_TOKEN_START_KEY] == 0
    assert update[CACHED_THINKER_DECODE_TOKEN_END_KEY] == 4
    assert torch.allclose(
        update[CACHED_THINKER_DECODE_EMBEDDINGS_KEY],
        torch.tensor([[10.0], [11.0], [12.0], [13.0]], dtype=torch.float32),
    )


def test_resolve_thinker_decode_step_replaces_stale_cached_span():
    payload = {
        CACHED_THINKER_DECODE_EMBEDDINGS_KEY: torch.tensor([[1.0], [2.0]], dtype=torch.float32),
        CACHED_THINKER_DECODE_TOKEN_START_KEY: 0,
        CACHED_THINKER_DECODE_TOKEN_END_KEY: 2,
        THINKER_DECODE_EMBEDDINGS_KEY: torch.tensor([[73.0], [74.0]], dtype=torch.float32),
        THINKER_DECODE_TOKEN_START_KEY: 73,
        THINKER_DECODE_TOKEN_END_KEY: 75,
        THINKER_OUTPUT_TOKEN_IDS_KEY: list(range(76)),
        "num_processed_tokens": 73,
    }
    update = {}

    step_state = resolve_thinker_decode_step(payload, update, device=torch.device("cpu"), dtype=torch.float32)

    assert torch.allclose(step_state.thinker_embed, torch.tensor([73.0], dtype=torch.float32))
    assert step_state.start_index == 73
    assert step_state.available_end == 75
    assert step_state.legacy_decode_end == 75
    assert update[THINKER_DECODE_EMBEDDINGS_KEY] is None
    assert update[CACHED_THINKER_DECODE_TOKEN_START_KEY] == 73
    assert update[CACHED_THINKER_DECODE_TOKEN_END_KEY] == 75


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


def test_update_states_preserves_existing_model_buffer_when_seeding_initial_buffer(monkeypatch):
    import vllm_omni.worker.gpu_model_runner as mod

    monkeypatch.setattr(mod, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True))

    runner = _make_runner_for_update_states(())
    runner._get_valid_sampled_token_count = lambda: []
    runner.model_intermediate_buffer["r1"] = {
        "thinker_prefill_embeddings": torch.tensor([1.0]),
        "thinker_hidden_states": torch.tensor([2.0]),
    }
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
        initial_model_buffer={"global_request_id": ["g1"]},
    )
    scheduler_output = _make_scheduler_output()
    scheduler_output.scheduled_new_reqs = [new_req]

    OmniGPUModelRunner._update_states(runner, scheduler_output)

    assert torch.equal(runner.model_intermediate_buffer["r1"]["thinker_prefill_embeddings"], torch.tensor([1.0]))
    assert torch.equal(runner.model_intermediate_buffer["r1"]["thinker_hidden_states"], torch.tensor([2.0]))
    assert runner.model_intermediate_buffer["r1"]["global_request_id"] == ["g1"]


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
    runner._send_side_request_payload = {
        "ext-unknown": {"stale": True},
        "unknown_req": {"stale": True},
    }
    runner._local_request_metadata = {}
    runner._pending_load_reqs = {}
    runner._finished_load_reqs = {"unknown_req"}
    runner._chunk_ready_req_ids = {"unknown_req"}
    runner._chunk_finished_req_ids = {"unknown_req"}
    runner._chunk_stream_completed = {"unknown_req"}
    runner._full_payload_pending_broadcast_req_ids = set()
    runner._stage_recv_req_ids = set()
    runner._get_req_chunk = {"unknown_req": 3}
    runner.model_intermediate_buffer["unknown_req"] = {"stale": True}

    OmniGPUModelRunner._sync_local_stage_payloads(runner)

    assert "unknown_req" not in runner.model_intermediate_buffer
    assert runner._send_side_request_payload == {}
    assert runner._request_ids_mapping == {}
    assert runner._local_request_metadata == {}
    assert runner._pending_load_reqs == {}
    assert runner._finished_load_reqs == set()
    assert runner._chunk_ready_req_ids == set()
    assert runner._chunk_finished_req_ids == set()
    assert runner._chunk_stream_completed == set()
    assert runner._full_payload_pending_broadcast_req_ids == set()
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


def test_sync_local_stage_payloads_keeps_recently_received_full_payload_state():
    runner = _make_runner(req_ids=(), hidden_size=4)
    runner.requests = {}
    runner._pending_load_reqs = {}
    runner._local_stage_payload_cache = {
        "req-1": {
            "thinker_prefill_embeddings": torch.ones(1, 1),
        }
    }
    runner._local_request_metadata = {"req-1": {"next_stage_prompt_len": 3}}
    runner._stage_recv_req_ids = {"req-1"}
    runner.drop_inactive_request_runtime_state = MagicMock()

    OmniGPUModelRunner._sync_local_stage_payloads(runner)

    runner.drop_inactive_request_runtime_state.assert_not_called()
    assert runner._stage_recv_req_ids == {"req-1"}
    assert runner._local_request_metadata == {"req-1": {"next_stage_prompt_len": 3}}
    assert torch.allclose(
        runner.model_intermediate_buffer["req-1"]["thinker_prefill_embeddings"],
        torch.ones(1, 1),
    )
    assert runner._local_stage_payload_cache == {}


def test_sync_local_stage_payloads_preserves_full_payload_pending_broadcast_cache():
    runner = _make_runner(req_ids=(), hidden_size=4)
    runner.requests = {}
    runner._pending_load_reqs = {}
    runner._local_stage_payload_cache = {
        "req-1": {
            "thinker_prefill_embeddings": torch.ones(1, 1),
        }
    }
    runner._local_request_metadata = {"req-1": {"next_stage_prompt_len": 3}}
    runner._stage_recv_req_ids = {"req-1"}
    runner._full_payload_pending_broadcast_req_ids = {"req-1"}
    runner.drop_inactive_request_runtime_state = MagicMock()

    OmniGPUModelRunner._sync_local_stage_payloads(runner)

    runner.drop_inactive_request_runtime_state.assert_not_called()
    assert runner._stage_recv_req_ids == {"req-1"}
    assert runner._full_payload_pending_broadcast_req_ids == {"req-1"}
    assert runner._local_request_metadata == {"req-1": {"next_stage_prompt_len": 3}}
    assert runner.model_intermediate_buffer == {}
    assert torch.allclose(
        runner._local_stage_payload_cache["req-1"]["thinker_prefill_embeddings"],
        torch.ones(1, 1),
    )


def test_sync_local_stage_payloads_generation_uses_net_new_codec_length():
    runner = _make_runner(req_ids=("r1",), hidden_size=4)
    runner._model_mode = "generation"
    runner.requests["r1"].prompt_token_ids = list(range(8))
    runner._local_stage_payload_cache = {
        "r1": {
            "code_predictor_codes": list(range(10)),
            "left_context_size": 2,
            "code_num_quantizers": 2,
            "finished": True,
        }
    }

    OmniGPUModelRunner._sync_local_stage_payloads(runner)

    assert runner._local_stage_payload_cache == {}
    queue = runner.model_intermediate_buffer["r1"][OmniGPUModelRunner._GENERATION_CODEC_PAYLOAD_QUEUE_KEY]
    assert len(queue) == 1
    assert queue[0]["code_predictor_codes"] == list(range(10))


def test_sync_local_stage_payloads_generation_queues_codec_payloads():
    runner = _make_runner(req_ids=("r1",), hidden_size=4)
    runner._model_mode = "generation"
    runner.requests["r1"].prompt_token_ids = list(range(8))
    runner._local_stage_payload_cache = {
        "r1": {
            "code_predictor_codes": list(range(10)),
            "left_context_size": 2,
            "code_num_quantizers": 2,
            "finished": False,
        }
    }

    OmniGPUModelRunner._sync_local_stage_payloads(runner)

    runner._local_stage_payload_cache = {
        "r1": {
            "code_predictor_codes": list(range(10, 20)),
            "left_context_size": 2,
            "code_num_quantizers": 2,
            "finished": True,
        }
    }

    OmniGPUModelRunner._sync_local_stage_payloads(runner)

    queue = runner.model_intermediate_buffer["r1"][OmniGPUModelRunner._GENERATION_CODEC_PAYLOAD_QUEUE_KEY]
    assert len(queue) == 2
    assert queue[0]["code_predictor_codes"] == list(range(10))
    assert queue[1]["code_predictor_codes"] == list(range(10, 20))


def test_sync_local_stage_payloads_generation_clears_send_side_payload_state():
    runner = _make_runner(req_ids=("r1",), hidden_size=4)
    runner._model_mode = "generation"
    runner.requests["r1"].prompt_token_ids = list(range(10))
    runner._request_ids_mapping = {"r1": "ext-r1"}
    runner._send_side_request_payload = {
        "ext-r1": {"stale": True},
        "r1": {"stale": True},
    }
    runner._local_stage_payload_cache = {
        "r1": {
            "code_predictor_codes": list(range(10)),
            "left_context_size": 0,
            "code_num_quantizers": 2,
            "finished": True,
        }
    }

    OmniGPUModelRunner._sync_local_stage_payloads(runner)

    assert runner._send_side_request_payload == {}
    queue = runner.model_intermediate_buffer["r1"][OmniGPUModelRunner._GENERATION_CODEC_PAYLOAD_QUEUE_KEY]
    assert len(queue) == 1
    assert queue[0]["code_predictor_codes"] == list(range(10))


def test_sync_local_stage_payloads_generation_does_not_queue_non_streaming_codec_payloads():
    runner = _make_runner(req_ids=("r1",), hidden_size=4)
    runner._model_mode = "generation"
    runner.requests["r1"].prompt_token_ids = list(range(10))
    runner._local_stage_payload_cache = {
        "r1": {
            "code_predictor_codes": list(range(10)),
            "finished": True,
        }
    }

    OmniGPUModelRunner._sync_local_stage_payloads(runner)

    assert OmniGPUModelRunner._GENERATION_CODEC_PAYLOAD_QUEUE_KEY not in runner.model_intermediate_buffer["r1"]
    assert runner.model_intermediate_buffer["r1"]["code_predictor_codes"] == list(range(10))
    assert runner.model_intermediate_buffer["r1"]["finished"] is True


def test_sync_local_stage_payloads_generation_rearms_wakeup_when_delayed_payload_becomes_admissible():
    runner = _make_runner(req_ids=("r1",), hidden_size=4)
    runner._model_mode = "generation"
    runner.requests["r1"].prompt_token_ids = []
    runner._finished_load_reqs = set()
    runner._local_stage_payload_cache = {
        "r1": {
            "code_predictor_codes": list(range(10)),
            "left_context_size": 1,
            "code_num_quantizers": 2,
            "finished": False,
        }
    }

    OmniGPUModelRunner._sync_local_stage_payloads(runner)

    assert "r1" in runner._local_stage_payload_cache
    assert runner._finished_load_reqs == set()

    runner.requests["r1"].prompt_token_ids = list(range(8))

    OmniGPUModelRunner._sync_local_stage_payloads(runner)

    assert runner._local_stage_payload_cache == {}
    assert runner._finished_load_reqs == {"r1"}
    queue = runner.model_intermediate_buffer["r1"][OmniGPUModelRunner._GENERATION_CODEC_PAYLOAD_QUEUE_KEY]
    assert len(queue) == 1
    assert queue[0]["code_predictor_codes"] == list(range(10))


def test_sync_local_stage_payloads_generation_uses_announced_prompt_len_when_runner_state_is_stale():
    runner = _make_runner(req_ids=("r1",), hidden_size=4)
    runner._model_mode = "generation"
    runner.requests["r1"].prompt_token_ids = []
    runner._generation_next_stage_prompt_len = {"r1": 8}
    runner._finished_load_reqs = set()
    runner._local_stage_payload_cache = {
        "r1": {
            "code_predictor_codes": list(range(10)),
            "left_context_size": 1,
            "code_num_quantizers": 2,
            "finished": False,
        }
    }

    OmniGPUModelRunner._sync_local_stage_payloads(runner)

    assert runner._local_stage_payload_cache == {}
    assert runner._finished_load_reqs == {"r1"}
    queue = runner.model_intermediate_buffer["r1"][OmniGPUModelRunner._GENERATION_CODEC_PAYLOAD_QUEUE_KEY]
    assert len(queue) == 1
    assert queue[0]["code_predictor_codes"] == list(range(10))


def test_gather_model_intermediate_buffer_clears_transient_generation_codec_payloads():
    runner = _make_runner(req_ids=("r1",), hidden_size=4)
    runner._model_mode = "generation"
    runner.requests["r1"].output_token_ids = [1, 2, 3]
    runner.model_intermediate_buffer["r1"] = {
        "code_predictor_codes": [10, 11, 12, 13],
        "left_context_size": 1,
        "code_num_quantizers": 2,
        "finished": True,
        "persisted": [99],
    }

    gathered = OmniGPUModelRunner._gather_model_intermediate_buffer(runner)

    assert gathered == [
        {
            "code_predictor_codes": [10, 11, 12, 13],
            "left_context_size": 1,
            "code_num_quantizers": 2,
            "finished": True,
            "persisted": [99],
            "generated_len": 3,
        }
    ]
    assert runner.model_intermediate_buffer["r1"] == {"persisted": [99]}


def test_gather_model_intermediate_buffer_pops_generation_codec_queue_one_payload_per_step():
    runner = _make_runner(req_ids=("r1",), hidden_size=4)
    runner._model_mode = "generation"
    runner._finished_load_reqs = set()
    runner._chunk_finished_req_ids = set()
    runner._chunk_stream_completed = set()
    runner._generation_terminal_chunk_pending_req_ids = {"r1"}
    runner.requests["r1"].output_token_ids = [1, 2, 3]
    runner.model_intermediate_buffer["r1"] = {
        OmniGPUModelRunner._GENERATION_CODEC_PAYLOAD_QUEUE_KEY: [
            {
                "code_predictor_codes": [10, 11, 12, 13],
                "left_context_size": 1,
                "code_num_quantizers": 2,
                "finished": False,
            },
            {
                "code_predictor_codes": [20, 21, 22, 23],
                "left_context_size": 1,
                "code_num_quantizers": 2,
                "finished": True,
            },
        ],
        "persisted": [99],
    }

    gathered = OmniGPUModelRunner._gather_model_intermediate_buffer(runner)
    assert gathered == [
        {
            "code_predictor_codes": [10, 11, 12, 13],
            "left_context_size": 1,
            "code_num_quantizers": 2,
            "finished": False,
            "persisted": [99],
            "generated_len": 3,
        }
    ]
    assert runner.model_intermediate_buffer["r1"] == {
        OmniGPUModelRunner._GENERATION_CODEC_PAYLOAD_QUEUE_KEY: [
            {
                "code_predictor_codes": [20, 21, 22, 23],
                "left_context_size": 1,
                "code_num_quantizers": 2,
                "finished": True,
            }
        ],
        "persisted": [99],
    }
    assert runner._finished_load_reqs == {"r1"}
    assert runner._chunk_finished_req_ids == set()

    gathered = OmniGPUModelRunner._gather_model_intermediate_buffer(runner)
    assert gathered == [
        {
            "code_predictor_codes": [20, 21, 22, 23],
            "left_context_size": 1,
            "code_num_quantizers": 2,
            "finished": True,
            "persisted": [99],
            "generated_len": 3,
        }
    ]
    assert runner.model_intermediate_buffer["r1"] == {"persisted": [99]}
    assert runner._chunk_finished_req_ids == {"r1"}
    assert runner._chunk_stream_completed == {"r1"}
    assert runner._generation_terminal_chunk_pending_req_ids == set()


def test_generation_execute_model_idle_path_keeps_connector_signals(monkeypatch):
    import vllm_omni.worker.gpu_generation_model_runner as mod

    runner = object.__new__(GPUGenerationModelRunner)
    runner.execute_model_state = None
    runner._idle_log_counter = 0
    runner._stage_id = 2
    runner.register_chunk_recv = MagicMock()
    runner.recv_full_payload_inputs = MagicMock()
    runner.cleanup_finished_request = MagicMock()
    runner._pending_full_payload_send = {}
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
    runner._pending_full_payload_send = {}
    runner._async_chunk = False
    runner.requests = {}
    runner.kv_caches = {}
    runner.model_intermediate_buffer = {}
    runner._local_stage_payload_cache = {}
    runner._local_request_metadata = {}
    runner._send_side_request_payload = {}
    runner._send_side_request_snapshot = {}
    runner._code_prompt_token_ids = {}
    runner._request_ids_mapping = {}
    runner.register_chunk_recv = MagicMock()
    runner.recv_full_payload_inputs = MagicMock()
    runner.cleanup_finished_request = MagicMock()
    runner.mark_kv_transfer = MagicMock()
    runner.drain_pending_kv_transfers = MagicMock(return_value={})
    runner.send_kv_cache = MagicMock(return_value=[])
    runner.ack_kv_transfers = MagicMock()
    runner.synchronize_input_prep = lambda: _noop_forward_context()
    runner._update_states = MagicMock()
    runner.prune_inactive_requests = MagicMock()
    runner.flush_full_payload_outputs = MagicMock()
    runner.routed_experts_initialized = False
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
