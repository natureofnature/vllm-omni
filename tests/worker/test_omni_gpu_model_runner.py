from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner, _filter_mrope_kwargs_for_model
from vllm_omni.worker.omni_connector_model_runner_mixin import OmniConnectorModelRunnerMixin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class DummyBuffer:
    """A minimal buffer wrapper that exposes the `.gpu` attribute."""

    def __init__(self, t: torch.Tensor):
        self.gpu = t


class DummyInputBatch:
    """A minimal input batch that only provides `req_ids`."""

    def __init__(self, req_ids):
        self.req_ids = req_ids
        self.req_id_to_index = {r: i for i, r in enumerate(req_ids)}


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
        generator=None,
    ):
        self.calls.append(
            {
                "batch_size": int(req_embeds.shape[0]),
                "do_sample": do_sample,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "generator": generator,
            }
        )
        codes = torch.zeros((req_embeds.shape[0], 1), dtype=torch.int64)
        return req_embeds, codes


class StrictMRoPEModel:
    def get_mrope_input_positions(self, input_tokens, mm_features):
        raise NotImplementedError


class FlexibleMRoPEModel:
    def get_mrope_input_positions(self, input_tokens, mm_features=None, **kwargs):
        raise NotImplementedError


@contextmanager
def _noop_forward_context(*args, **kwargs):
    """A no-op context manager to replace vLLM forward context in CPU tests."""
    yield


def test_filter_mrope_kwargs_for_strict_model_signature():
    kwargs = {
        "mm_features": ["audio"],
        "hf_config": object(),
        "image_grid_thw": [],
    }

    assert _filter_mrope_kwargs_for_model(StrictMRoPEModel(), kwargs) == {
        "mm_features": ["audio"],
    }


def test_filter_mrope_kwargs_preserves_flexible_model_kwargs():
    kwargs = {
        "mm_features": ["video"],
        "hf_config": object(),
        "video_grid_thw": [[1, 2, 3]],
    }

    assert _filter_mrope_kwargs_for_model(FlexibleMRoPEModel(), kwargs) is kwargs


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
    runner.model = SimpleNamespace(talker_mtp_output_key=("codes", "audio"))
    runner.vllm_config = SimpleNamespace(model_config=SimpleNamespace())

    # Provide a minimal implementation that returns the expected 4-tuple.
    def _determine_batch_execution_and_padding(**kwargs):
        return None, object(), None, None, None

    runner._determine_batch_execution_and_padding = _determine_batch_execution_and_padding

    # Use the real merge method from OmniGPUModelRunner.
    return runner


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

    # Attach a single request state with mm_features and additional_information_cpu.
    req_state = DummyReqState()
    req_state.mm_features = ["mm_feature_obj"]
    req_state.additional_information_cpu = {"some_key": "some_value"}

    runner.requests = {req_id: req_state}

    return runner


def test_talker_mtp_forward_cpu_updates_inputs_and_info(monkeypatch):
    # `_talker_mtp_forward` calls `current_omni_platform.set_forward_context`,
    # which would otherwise dispatch to the real device implementation.
    import vllm_omni.worker.gpu_model_runner as mod  # Must be the same module that defines OmniGPUModelRunner

    monkeypatch.setattr(mod.current_omni_platform, "set_forward_context", _noop_forward_context)

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

    # Validate per-request additional_information_cpu was updated
    info_r1 = runner.requests["r1"].additional_information_cpu
    info_r2 = runner.requests["r2"].additional_information_cpu
    assert int(info_r1["codes"]["audio"][0, 0]) == 0
    assert int(info_r2["codes"]["audio"][0, 0]) == 1


def test_talker_mtp_forward_cpu_empty_batch_noop(monkeypatch):
    import vllm_omni.worker.gpu_model_runner as mod

    monkeypatch.setattr(mod.current_omni_platform, "set_forward_context", _noop_forward_context)

    runner = _make_runner(req_ids=("r1",), hidden_size=4)

    inputs_embeds = torch.randn((2, 4))
    before = inputs_embeds.clone()

    OmniGPUModelRunner._talker_mtp_forward(runner, [], inputs_embeds)

    # Ensure no changes were made
    assert torch.allclose(inputs_embeds, before)


def test_talker_mtp_forward_ignores_default_sampling_seed_without_request_marker(monkeypatch):
    import vllm_omni.worker.gpu_model_runner as mod

    monkeypatch.setattr(mod.current_omni_platform, "set_forward_context", _noop_forward_context)

    runner = _make_runner(req_ids=("r1",), hidden_size=4)
    runner.requests["r1"].sampling_params = SimpleNamespace(seed=42)
    runner.talker_mtp = CaptureTalkerMTP()
    runner.vllm_config = SimpleNamespace(model_config=SimpleNamespace(subtalker_sampling_params={}))

    def fake_determine(self, num_tokens, num_reqs, num_scheduled_tokens_np, max_num_scheduled_tokens, use_cascade_attn):
        batch_desc = SimpleNamespace(num_tokens=int(num_tokens))
        return (False, batch_desc, None, None, None)

    monkeypatch.setattr(runner, "_determine_batch_execution_and_padding", fake_determine.__get__(runner, type(runner)))

    inputs_embeds = torch.zeros((2, 4), dtype=torch.float32)
    OmniGPUModelRunner._talker_mtp_forward(runner, ["r1"], inputs_embeds)

    assert runner.talker_mtp.calls[0]["generator"] is None


def test_talker_mtp_forward_passes_qwen3_tts_subtalker_sampling_params_to_talker(monkeypatch):
    import vllm_omni.worker.gpu_model_runner as mod

    monkeypatch.setattr(mod.current_omni_platform, "set_forward_context", _noop_forward_context)

    runner = _make_runner(req_ids=("r1",), hidden_size=4)
    runner.requests["r1"].sampling_params = SimpleNamespace(
        seed=42,
        extra_args={"tts_local_seed": 42},
    )
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
            "batch_size": 1,
            "do_sample": False,
            "temperature": 0.2,
            "top_k": 9,
            "top_p": 0.55,
            "generator": runner.talker_mtp.calls[0]["generator"],
        }
    ]
    assert runner.talker_mtp.calls[0]["generator"] is not None


def test_talker_mtp_forward_keeps_explicit_seeded_requests_scalar(monkeypatch):
    import vllm_omni.worker.gpu_model_runner as mod

    monkeypatch.setattr(mod.current_omni_platform, "set_forward_context", _noop_forward_context)

    runner = _make_runner(req_ids=("r1", "r2"), hidden_size=4)
    runner.requests["r1"].sampling_params = SimpleNamespace(
        seed=11,
        extra_args={"tts_local_seed": 11},
    )
    runner.requests["r2"].sampling_params = SimpleNamespace(
        seed=22,
        extra_args={"tts_local_seed": 22},
    )
    runner.talker_mtp = CaptureTalkerMTP()
    runner.vllm_config = SimpleNamespace(model_config=SimpleNamespace(subtalker_sampling_params={}))

    def fake_determine(self, num_tokens, num_reqs, num_scheduled_tokens_np, max_num_scheduled_tokens, use_cascade_attn):
        batch_desc = SimpleNamespace(num_tokens=int(num_tokens))
        return (False, batch_desc, None, None, None)

    monkeypatch.setattr(runner, "_determine_batch_execution_and_padding", fake_determine.__get__(runner, type(runner)))

    runner.talker_mtp_input_ids.gpu[:] = torch.tensor([101, 202], dtype=torch.int64)
    runner.talker_mtp_inputs_embeds.gpu[0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    runner.talker_mtp_inputs_embeds.gpu[1] = torch.tensor([10.0, 20.0, 30.0, 40.0])
    saved_input_ids = runner.talker_mtp_input_ids.gpu.clone()
    saved_embeds = runner.talker_mtp_inputs_embeds.gpu.clone()

    inputs_embeds = torch.zeros((6, 4), dtype=torch.float32)
    OmniGPUModelRunner._talker_mtp_forward(runner, ["r1", "r2"], inputs_embeds)

    assert [call["batch_size"] for call in runner.talker_mtp.calls] == [1, 1]
    assert all(call["generator"] is not None for call in runner.talker_mtp.calls)
    assert runner.talker_mtp.calls[0]["generator"] is not runner.talker_mtp.calls[1]["generator"]
    assert torch.equal(runner.talker_mtp_input_ids.gpu, saved_input_ids)
    assert torch.equal(runner.talker_mtp_inputs_embeds.gpu, saved_embeds)


def test_update_intermediate_buffer_writes_to_buffer_and_setattr(monkeypatch):
    """Validate that _update_intermediate_buffer writes to model_intermediate_buffer
    (forward path) and mirrors to additional_information_cpu setattr (backward compat)."""
    import vllm_omni.worker.gpu_model_runner as mod

    monkeypatch.setattr(mod.current_omni_platform, "set_forward_context", _noop_forward_context)

    runner = _make_runner(req_ids=("r1",), hidden_size=4)

    update = {"my_tensor": torch.tensor([1.0, 2.0]), "my_list": [3, 4]}
    OmniGPUModelRunner._update_intermediate_buffer(runner, "r1", update)

    # Forward: buffer is populated
    assert "r1" in runner.model_intermediate_buffer
    buf = runner.model_intermediate_buffer["r1"]
    assert torch.allclose(buf["my_tensor"], torch.tensor([1.0, 2.0]))
    assert buf["my_list"] == [3, 4]

    # Backward compat: setattr is also populated
    info_cpu = runner.requests["r1"].additional_information_cpu
    assert torch.allclose(info_cpu["my_tensor"], torch.tensor([1.0, 2.0]))
    assert info_cpu["my_list"] == [3, 4]


def test_update_intermediate_buffer_accumulates():
    """Validate that successive merges accumulate keys in the buffer."""
    runner = _make_runner(req_ids=("r1",), hidden_size=4)

    OmniGPUModelRunner._update_intermediate_buffer(runner, "r1", {"a": torch.tensor([1.0])})
    OmniGPUModelRunner._update_intermediate_buffer(runner, "r1", {"b": torch.tensor([2.0])})

    buf = runner.model_intermediate_buffer["r1"]
    assert "a" in buf and "b" in buf
    assert torch.allclose(buf["a"], torch.tensor([1.0]))
    assert torch.allclose(buf["b"], torch.tensor([2.0]))


def test_update_intermediate_buffer_preserves_prefill_hidden_across_decode_update():
    runner = _make_runner(req_ids=("r1",), hidden_size=4)
    prefill_hidden = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    decode_hidden = torch.tensor([[99.0, 100.0]])
    decode_a = torch.tensor([[1.0, 2.0]])
    decode_b = torch.tensor([[3.0, 4.0]])

    OmniGPUModelRunner._update_intermediate_buffer(
        runner,
        "r1",
        {
            "embed": {"prefill": torch.zeros(3, 2)},
            "hidden_states": {"output": prefill_hidden.clone()},
        },
    )
    OmniGPUModelRunner._update_intermediate_buffer(
        runner,
        "r1",
        {
            "embed": {"decode": decode_a},
            "hidden_states": {"output": decode_hidden},
        },
    )
    OmniGPUModelRunner._update_intermediate_buffer(
        runner,
        "r1",
        {"embed": {"decode": torch.cat([decode_a, decode_b], dim=0)}},
    )

    buf = runner.model_intermediate_buffer["r1"]
    assert torch.equal(buf["hidden_states"]["output"], prefill_hidden)
    assert torch.equal(buf["embed"]["decode"], torch.cat([decode_a, decode_b], dim=0))


def test_update_intermediate_buffer_resets_previous_segment_prefill_state():
    runner = _make_runner(req_ids=("r1",), hidden_size=4)
    old_prefill = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    old_hidden = old_prefill + 100
    new_prefill = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    new_hidden = new_prefill + 10

    runner.model_intermediate_buffer["r1"] = {
        "omni_final_stage_id": 2,
        "embed": {
            "prefill": old_prefill,
            "decode": torch.tensor([[9.0, 9.0]]),
        },
        "hidden_states": {"output": old_hidden},
        "ids": {"output": [99]},
        "meta": {"is_segment_finished": torch.tensor(True)},
        "model_specific_segment_state": {"stale": True},
    }

    OmniGPUModelRunner._update_intermediate_buffer(
        runner,
        "r1",
        {
            "embed": {"prefill": new_prefill},
            "hidden_states": {"output": new_hidden},
            "ids": {"prompt": [1, 2]},
            "meta": {"finished": torch.tensor(False)},
        },
    )

    buf = runner.model_intermediate_buffer["r1"]
    assert buf["omni_final_stage_id"] == 2
    assert torch.equal(buf["embed"]["prefill"], new_prefill)
    assert "decode" not in buf["embed"]
    assert torch.equal(buf["hidden_states"]["output"], new_hidden)
    assert buf["ids"] == {"prompt": [1, 2]}
    assert "model_specific_segment_state" not in buf
    assert buf["meta"]["finished"].item() is False


def test_update_intermediate_buffer_replaces_nonprefix_shorter_prefill_state():
    runner = _make_runner(req_ids=("r1",), hidden_size=4)
    old_prefill = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    old_hidden = old_prefill + 100
    new_prefill = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    new_hidden = new_prefill + 10

    OmniGPUModelRunner._update_intermediate_buffer(
        runner,
        "r1",
        {
            "embed": {"prefill": old_prefill},
            "hidden_states": {"output": old_hidden},
        },
    )
    OmniGPUModelRunner._update_intermediate_buffer(
        runner,
        "r1",
        {
            "embed": {"prefill": new_prefill},
            "hidden_states": {"output": new_hidden},
        },
    )

    buf = runner.model_intermediate_buffer["r1"]
    assert torch.equal(buf["embed"]["prefill"], new_prefill)
    assert torch.equal(buf["hidden_states"]["output"], new_hidden)


def test_streaming_segment_reset_does_not_require_finished_marker():
    runner = _make_runner(req_ids=("r1",), hidden_size=4)
    old_decode = torch.tensor([[9.0, 9.0]])
    old_cached_decode = torch.tensor([[8.0, 8.0]])
    new_prefill = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    new_decode = torch.tensor([[5.0, 6.0]])
    new_cached_decode = torch.tensor([[7.0, 8.0]])
    new_hidden = new_prefill + 10

    runner.model_intermediate_buffer["r1"] = {
        "omni_final_stage_id": 2,
        "embed": {
            "prefill": torch.zeros(2, 2),
            "decode": old_decode,
            "cached_decode": old_cached_decode,
        },
        "hidden_states": {"output": torch.ones(2, 2)},
        "ids": {"output": [99]},
        "meta": {"is_segment_finished": torch.tensor(False)},
        "model_specific_segment_state": {"stale": True},
    }

    OmniGPUModelRunner._update_streaming_input_additional_info(runner, "r1")
    OmniGPUModelRunner._update_intermediate_buffer(
        runner,
        "r1",
        {
            "embed": {
                "prefill": new_prefill,
                "decode": new_decode,
                "cached_decode": new_cached_decode,
            },
            "hidden_states": {"output": new_hidden},
            "ids": {"prompt": [1, 2]},
        },
    )

    buf = runner.model_intermediate_buffer["r1"]
    assert buf["omni_final_stage_id"] == 2
    assert torch.equal(buf["embed"]["prefill"], new_prefill)
    assert torch.equal(buf["embed"]["decode"], new_decode)
    assert torch.equal(buf["embed"]["cached_decode"], new_cached_decode)
    assert torch.equal(buf["hidden_states"]["output"], new_hidden)
    assert buf["ids"] == {"prompt": [1, 2]}
    assert "model_specific_segment_state" not in buf
    assert buf["meta"] == {"num_processed_tokens": 0, "resumable": True}
    assert "r1" not in runner._segment_runtime_reset_req_ids


def test_streaming_input_update_marks_cached_request_resumable():
    runner = _make_runner(req_ids=("r1",), hidden_size=4)

    OmniGPUModelRunner._update_streaming_input_additional_info(runner, "r1")

    assert runner.requests["r1"].resumable is True


def test_streaming_input_update_clears_previous_segment_runtime():
    runner = _make_runner(req_ids=("r1",), hidden_size=4)
    reset_calls = []
    runner.reset_segment_send_watermark = reset_calls.append
    runner.model_intermediate_buffer["r1"] = {
        "omni_final_stage_id": 2,
        "embed": {"prefill": torch.ones(2, 2), "decode": torch.ones(1, 2)},
        "hidden_states": {"output": torch.ones(2, 2)},
        "ids": {"prompt": [1, 2], "output": [3]},
        "meta": {
            "num_processed_tokens": 81,
            "is_segment_finished": torch.tensor(False),
        },
        "model_specific_segment_state": {"stale": True},
    }

    OmniGPUModelRunner._update_streaming_input_additional_info(runner, "r1")

    buf = runner.model_intermediate_buffer["r1"]
    assert buf["omni_final_stage_id"] == 2
    assert set(buf) == {"omni_final_stage_id", "meta"}
    assert buf["meta"] == {"num_processed_tokens": 0, "resumable": True}
    assert runner.requests["r1"].additional_information_cpu is buf
    assert runner.requests["r1"].resumable is True
    assert reset_calls == ["r1"]
    assert runner._segment_send_watermark_reset_req_ids == {"r1"}


def test_segment_send_watermark_reset_uses_resolved_transfer_id():
    runner = _make_runner(req_ids=("r1",), hidden_size=4)
    reset_calls = []
    runner.reset_segment_send_watermark = reset_calls.append
    runner._segment_send_watermark_reset_req_ids = {"r1"}

    OmniGPUModelRunner._reset_segment_send_watermark_for_transfer(runner, "r1", "ext-1")

    assert reset_calls == ["r1", "ext-1"]
    assert runner._segment_send_watermark_reset_req_ids == set()


def test_segment_prefill_reset_marks_send_watermark_reset():
    runner = _make_runner(req_ids=("r1",), hidden_size=4)
    reset_calls = []
    runner.reset_segment_send_watermark = reset_calls.append
    runner.model_intermediate_buffer["r1"] = {
        "omni_final_stage_id": 2,
        "embed": {"decode": torch.ones(1, 2)},
        "meta": {"is_segment_finished": torch.tensor(True)},
    }

    OmniGPUModelRunner._update_intermediate_buffer(
        runner,
        "r1",
        {"embed": {"prefill": torch.zeros(1, 2)}},
    )

    assert reset_calls == ["r1"]
    assert runner._segment_send_watermark_reset_req_ids == {"r1"}


def test_store_request_additional_information_marks_resumable_from_meta():
    runner = _make_runner(req_ids=("r1",), hidden_size=4)
    info = {"meta": {"resumable": torch.tensor(True)}, "embed": {"prefill": torch.ones(1, 2)}}

    OmniGPUModelRunner._store_request_additional_information(runner, "r1", info)

    assert runner.model_intermediate_buffer["r1"] is info
    assert runner.requests["r1"].additional_information_cpu is info
    assert runner.requests["r1"].resumable is True


def test_sync_local_stage_payloads_preserves_new_decode_after_segment_reset():
    runner = _make_runner(req_ids=("r1",), hidden_size=4)
    old_decode = torch.tensor([[9.0, 9.0]])
    new_prefill = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    new_decode = torch.tensor([[5.0, 6.0]])
    new_cached_decode = torch.tensor([[7.0, 8.0]])
    new_hidden = new_prefill + 10

    runner.model_intermediate_buffer["r1"] = {
        "omni_final_stage_id": 2,
        "embed": {"decode": old_decode},
        "meta": {"is_segment_finished": torch.tensor(False)},
    }
    runner._local_stage_payload_cache = {
        "r1": {
            "embed": {
                "prefill": new_prefill,
                "decode": new_decode,
                "cached_decode": new_cached_decode,
            },
            "hidden_states": {"output": new_hidden},
        }
    }
    runner._full_payload_pending_broadcast_req_ids = set()
    runner._lock = None
    runner._work_available = None

    OmniGPUModelRunner._update_streaming_input_additional_info(runner, "r1")
    OmniGPUModelRunner._sync_local_stage_payloads(runner)

    buf = runner.model_intermediate_buffer["r1"]
    assert runner._local_stage_payload_cache == {}
    assert torch.equal(buf["embed"]["prefill"], new_prefill)
    assert torch.equal(buf["embed"]["decode"], new_decode)
    assert torch.equal(buf["embed"]["cached_decode"], new_cached_decode)
    assert torch.equal(buf["hidden_states"]["output"], new_hidden)
    assert buf["meta"] == {"num_processed_tokens": 0, "resumable": True}


def test_build_model_kwargs_keeps_preprocess_stage_mtp_codes_until_forward():
    runner = _make_runner(req_ids=("r1",), hidden_size=4)
    runner.requests["r1"].output_token_ids = []
    runner.model = SimpleNamespace(has_preprocess=True)
    runner.model_config = SimpleNamespace(has_sampling_extra_args=False)
    runner._omni_num_scheduled_tokens_np = None
    runner._omni_query_start_loc_model_kwarg = False
    runner._local_stage_terminal_payload_cache = {}
    runner._full_payload_pending_broadcast_req_ids = set()
    runner._lock = None
    runner._work_available = None

    codes = torch.tensor([[1, 2, 3, 4]])
    runner.model_intermediate_buffer["r1"] = {"codes": {"audio": codes}}
    runner._local_stage_payload_cache = {"r1": {"embed": {"prefill": torch.ones(1, 4)}}}

    kwargs = OmniGPUModelRunner._build_model_kwargs_extra(runner, sync_local_stage_payloads=False)

    buf = runner.model_intermediate_buffer["r1"]
    assert torch.equal(buf["codes"]["audio"], codes)
    assert "embed" not in buf
    assert runner._local_stage_payload_cache
    assert kwargs["model_intermediate_buffer"] == [buf]


def test_build_model_kwargs_syncs_local_payload_for_no_preprocess_stage():
    runner = _make_runner(req_ids=("r1",), hidden_size=4)
    runner.requests["r1"].output_token_ids = []
    runner.model = SimpleNamespace(has_preprocess=False)
    runner.model_config = SimpleNamespace(has_sampling_extra_args=False)
    runner._omni_num_scheduled_tokens_np = None
    runner._omni_query_start_loc_model_kwarg = False
    runner._local_stage_terminal_payload_cache = {}
    runner._full_payload_pending_broadcast_req_ids = set()
    runner._lock = None
    runner._work_available = None

    payload = {"codes": {"audio": torch.tensor([[7, 8]])}}
    runner._local_stage_payload_cache = {"r1": payload}

    kwargs = OmniGPUModelRunner._build_model_kwargs_extra(runner)

    assert runner._local_stage_payload_cache == {}
    assert torch.equal(runner.model_intermediate_buffer["r1"]["codes"]["audio"], payload["codes"]["audio"])
    assert kwargs["model_intermediate_buffer"] == [runner.model_intermediate_buffer["r1"]]


def _enable_qwen3_talker_async_chunk(runner):
    runner.vllm_config.model_config = SimpleNamespace(
        model_arch="Qwen3OmniMoeForConditionalGeneration",
        model_stage="talker",
        async_chunk=True,
        stage_connector_config={"name": "SharedMemoryConnector"},
    )


def test_build_omni_mm_payload_keeps_multi_token_hidden_state_span():
    runner = object.__new__(GPUARModelRunner)
    hidden = torch.arange(12, dtype=torch.float32).reshape(6, 2)

    payload = GPUARModelRunner._build_omni_mm_payload(
        runner,
        combined_multimodal_outputs=None,
        mm_cpu={"hidden_states.layer_0": hidden},
        rid="r1",
        idx=0,
        start=0,
        end=6,
        audio_sparse_output=False,
        sparse_mm_index={},
        seq_len=6,
        build_flat_payload=True,
    )

    assert torch.equal(payload["hidden_states.layer_0"], hidden)


def test_build_omni_mm_payload_slices_padded_qwen3_hidden_state_span():
    runner = object.__new__(GPUARModelRunner)
    runner.model_config = SimpleNamespace(
        model_arch="Qwen3OmniMoeForConditionalGeneration",
        model_stage="thinker",
        async_chunk=True,
    )
    hidden = torch.arange(16, dtype=torch.float32).reshape(8, 2)

    payload = GPUARModelRunner._build_omni_mm_payload(
        runner,
        combined_multimodal_outputs=None,
        mm_cpu={"hidden_states.layer_0": hidden},
        rid="r1",
        idx=1,
        start=5,
        end=6,
        audio_sparse_output=False,
        sparse_mm_index={},
        seq_len=7,
        build_flat_payload=True,
    )

    assert torch.equal(payload["hidden_states.layer_0"], hidden[5:6])


def test_build_omni_mm_payload_keeps_presliced_qwen3_hidden_state_span():
    runner = object.__new__(GPUARModelRunner)
    runner.model_config = SimpleNamespace(
        model_arch="Qwen3OmniMoeForConditionalGeneration",
        model_stage="thinker",
        async_chunk=True,
    )
    hidden = torch.arange(6, dtype=torch.float32).reshape(3, 2)

    payload = GPUARModelRunner._build_omni_mm_payload(
        runner,
        combined_multimodal_outputs=None,
        mm_cpu={"hidden_states.layer_0": hidden},
        rid="r1",
        idx=1,
        start=5,
        end=8,
        audio_sparse_output=False,
        sparse_mm_index={},
        seq_len=10,
        build_flat_payload=True,
    )

    assert torch.equal(payload["hidden_states.layer_0"], hidden)
    assert payload["hidden_states.layer_0"] is not hidden


def test_sync_local_stage_payloads_defers_reused_streaming_segment_until_terminal_eos():
    runner = _make_runner(req_ids=("r1",), hidden_size=4)
    _enable_qwen3_talker_async_chunk(runner)
    reset_calls = []
    runner.reset_segment_send_watermark = reset_calls.append

    old_decode = torch.tensor([[9.0, 9.0]])
    new_prefill = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    new_decode = torch.tensor([[5.0, 6.0]])
    new_hidden = new_prefill + 10
    tts_bos = torch.tensor([[0.1, 0.2]])
    tts_eos = torch.tensor([[0.3, 0.4]])
    tts_pad = torch.tensor([[0.5, 0.6]])

    runner.model_intermediate_buffer["r1"] = {
        "omni_final_stage_id": 2,
        "embed": {"decode": old_decode, "tts_bos": tts_bos, "tts_eos": tts_eos, "tts_pad": tts_pad},
        "hidden_states": {"output": torch.ones(2, 2)},
        "ids": {"output": [99]},
        "meta": {
            "is_segment_finished": torch.tensor(False),
            "num_processed_tokens": 81,
            "prefill_consumed_text_tokens": 1,
        },
        "model_specific_segment_state": {"stale": True},
    }
    runner._local_stage_payload_cache = {
        "r1": {
            "embed": {"prefill": new_prefill, "decode": new_decode},
            "hidden_states": {"output": new_hidden},
            "ids": {"prompt": [1, 2]},
        }
    }
    runner._local_stage_terminal_payload_cache = {}
    runner._full_payload_pending_broadcast_req_ids = set()
    runner._finished_load_reqs = set()
    runner._lock = None
    runner._work_available = None

    OmniGPUModelRunner._sync_local_stage_payloads(runner)

    assert runner._local_stage_payload_cache["r1"]["embed"]["prefill"] is new_prefill
    assert runner._finished_load_reqs == set()
    assert reset_calls == []
    assert torch.equal(runner.model_intermediate_buffer["r1"]["embed"]["decode"], old_decode)

    runner._local_stage_terminal_payload_cache = {"r1": {"meta": {"is_segment_finished": torch.tensor(True)}}}
    OmniGPUModelRunner._sync_local_stage_payloads(runner, terminal_only=True)

    buf = runner.model_intermediate_buffer["r1"]
    assert runner._local_stage_payload_cache["r1"]["embed"]["prefill"] is new_prefill
    assert bool(buf["meta"]["is_segment_finished"].item()) is True
    assert torch.equal(buf["embed"]["decode"], old_decode)
    assert runner._finished_load_reqs == set()
    assert reset_calls == []
    assert OmniGPUModelRunner._select_ready_local_stage_payload_req_ids(runner, {"r1"}) == set()

    buf["meta"]["eos_emitted"] = torch.tensor(True)
    assert OmniGPUModelRunner._select_ready_local_stage_payload_req_ids(runner, {"r1"}) == {"r1"}
    OmniGPUModelRunner._sync_local_stage_payloads(runner, set())
    assert "r1" in runner._local_stage_payload_cache

    OmniGPUModelRunner._update_streaming_input_additional_info(runner, "r1")
    OmniGPUModelRunner._sync_local_stage_payloads(runner, {"r1"})

    buf = runner.model_intermediate_buffer["r1"]
    assert runner._local_stage_payload_cache == {}
    assert runner._local_stage_terminal_payload_cache == {}
    assert reset_calls == ["r1"]
    assert runner._segment_send_watermark_reset_req_ids == {"r1"}
    assert buf["omni_final_stage_id"] == 2
    assert torch.equal(buf["embed"]["prefill"], new_prefill)
    assert torch.equal(buf["embed"]["decode"], new_decode)
    assert torch.equal(buf["embed"]["tts_bos"], tts_bos)
    assert torch.equal(buf["embed"]["tts_eos"], tts_eos)
    assert torch.equal(buf["embed"]["tts_pad"], tts_pad)
    assert torch.equal(buf["hidden_states"]["output"], new_hidden)
    assert buf["ids"] == {"prompt": [1, 2]}
    assert buf["meta"] == {"num_processed_tokens": 0, "resumable": True}
    assert "model_specific_segment_state" not in buf


def test_sync_local_stage_payloads_reset_drops_stale_decode_progress_meta():
    runner = _make_runner(req_ids=("r1",), hidden_size=4)
    _enable_qwen3_talker_async_chunk(runner)
    runner.reset_segment_send_watermark = lambda req_id: None

    new_prefill = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    new_decode = torch.tensor([[5.0, 6.0]])
    runner.model_intermediate_buffer["r1"] = {
        "embed": {"decode": torch.tensor([[9.0, 9.0]])},
        "meta": {
            "is_segment_finished": torch.tensor(False),
            "decode_flag": True,
            "num_processed_tokens": 8,
            "prefill_consumed_text_tokens": 1,
            "eos_emitted": torch.tensor(True),
        },
    }
    runner._local_stage_payload_cache = {
        "r1": {
            "embed": {"prefill": new_prefill, "decode": new_decode},
            "meta": {
                "decode_flag": True,
                "num_processed_tokens": 1,
                "prefill_consumed_text_tokens": 1,
                "eos_emitted": True,
            },
        }
    }
    runner._full_payload_pending_broadcast_req_ids = set()
    runner._lock = None
    runner._work_available = None

    OmniGPUModelRunner._update_streaming_input_additional_info(runner, "r1")
    OmniGPUModelRunner._sync_local_stage_payloads(runner, {"r1"})

    meta = runner.model_intermediate_buffer["r1"]["meta"]
    assert "decode_flag" not in meta
    assert "eos_emitted" not in meta
    assert meta["num_processed_tokens"] == 0
    assert meta["prefill_consumed_text_tokens"] == 1
    assert meta["resumable"] is True
    assert torch.equal(runner.model_intermediate_buffer["r1"]["embed"]["decode"], new_decode)


def test_sync_local_stage_payloads_request_finish_prefill_keeps_decode_progress():
    runner = _make_runner(req_ids=("r1",), hidden_size=4)
    _enable_qwen3_talker_async_chunk(runner)
    reset_calls = []
    runner.reset_segment_send_watermark = reset_calls.append

    existing_decode = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    final_prefill = torch.tensor([[9.0, 9.0]])
    final_cached_decode = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    runner.model_intermediate_buffer["r1"] = {
        "embed": {"decode": existing_decode},
        "meta": {
            "num_processed_tokens": 2,
            "prefill_consumed_text_tokens": 1,
        },
    }
    runner._local_stage_payload_cache = {
        "r1": {
            "embed": {
                "prefill": final_prefill,
                "cached_decode": final_cached_decode,
            },
            "meta": {"finished": torch.tensor(True)},
        }
    }
    runner._full_payload_pending_broadcast_req_ids = set()
    runner._lock = None
    runner._work_available = None

    OmniGPUModelRunner._sync_local_stage_payloads(runner)

    buf = runner.model_intermediate_buffer["r1"]
    assert reset_calls == []
    assert getattr(runner, "_segment_send_watermark_reset_req_ids", set()) == set()
    assert torch.equal(buf["embed"]["prefill"], final_prefill)
    assert torch.equal(buf["embed"]["decode"], existing_decode)
    assert torch.equal(buf["embed"]["cached_decode"], final_cached_decode)
    assert buf["meta"]["num_processed_tokens"] == 2
    assert buf["meta"]["prefill_consumed_text_tokens"] == 1
    assert bool(buf["meta"]["finished"].item()) is True


def test_sync_local_stage_payloads_terminal_cache_keeps_forward_progress():
    runner = _make_runner(req_ids=("r1",), hidden_size=4)
    _enable_qwen3_talker_async_chunk(runner)

    final_cached_decode = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )
    runner.model_intermediate_buffer["r1"] = {
        "embed": {"cached_decode": final_cached_decode[:2]},
        "meta": {
            "decode_flag": True,
            "num_processed_tokens": 3,
            "prefill_consumed_text_tokens": 1,
        },
    }
    runner._local_stage_payload_cache = {
        "r1": {
            "embed": {
                "prefill": torch.ones((2, 2)),
                "cached_decode": final_cached_decode,
            },
            "meta": {
                "decode_flag": True,
                "num_processed_tokens": torch.tensor(1),
                "prefill_consumed_text_tokens": 1,
                "finished": torch.tensor(True),
                "is_segment_finished": torch.tensor(True),
            },
        }
    }
    runner._full_payload_pending_broadcast_req_ids = set()
    runner._lock = None
    runner._work_available = None

    OmniGPUModelRunner._sync_local_stage_payloads(runner)

    buf = runner.model_intermediate_buffer["r1"]
    assert torch.equal(buf["embed"]["cached_decode"], final_cached_decode)
    assert buf["meta"]["num_processed_tokens"] == 3
    assert buf["meta"]["prefill_consumed_text_tokens"] == 1
    assert bool(buf["meta"]["finished"].item()) is True
    assert bool(buf["meta"]["is_segment_finished"].item()) is True


def test_sync_local_stage_payloads_final_prefill_does_not_reset_after_segment_marker():
    runner = _make_runner(req_ids=("r1",), hidden_size=4)
    _enable_qwen3_talker_async_chunk(runner)
    reset_calls = []
    runner.reset_segment_send_watermark = reset_calls.append

    existing_decode = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    final_prefill = torch.tensor([[9.0, 9.0]])
    final_cached_decode = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )
    runner.model_intermediate_buffer["r1"] = {
        "embed": {"decode": existing_decode},
        "meta": {
            "is_segment_finished": torch.tensor(True),
            "decode_flag": True,
            "num_processed_tokens": 2,
            "prefill_consumed_text_tokens": 1,
        },
    }
    runner._local_stage_payload_cache = {
        "r1": {
            "embed": {
                "prefill": final_prefill,
                "cached_decode": final_cached_decode,
            },
            "meta": {
                "finished": torch.tensor(True),
                "is_segment_finished": torch.tensor(True),
            },
        }
    }
    runner._full_payload_pending_broadcast_req_ids = set()
    runner._lock = None
    runner._work_available = None

    OmniGPUModelRunner._sync_local_stage_payloads(runner)

    buf = runner.model_intermediate_buffer["r1"]
    assert reset_calls == []
    assert getattr(runner, "_segment_send_watermark_reset_req_ids", set()) == set()
    assert torch.equal(buf["embed"]["prefill"], final_prefill)
    assert torch.equal(buf["embed"]["decode"], existing_decode)
    assert torch.equal(buf["embed"]["cached_decode"], final_cached_decode)
    assert buf["meta"]["decode_flag"] is True
    assert buf["meta"]["num_processed_tokens"] == 2
    assert buf["meta"]["prefill_consumed_text_tokens"] == 1
    assert bool(buf["meta"]["finished"].item()) is True
    assert bool(buf["meta"]["is_segment_finished"].item()) is True


def test_sync_local_stage_payloads_prefill_does_not_reset_without_async_chunk():
    runner = _make_runner(req_ids=("r1",), hidden_size=4)
    reset_calls = []
    runner.reset_segment_send_watermark = reset_calls.append

    old_decode = torch.tensor([[9.0, 9.0]])
    new_prefill = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    runner.model_intermediate_buffer["r1"] = {
        "embed": {"decode": old_decode},
        "meta": {"is_segment_finished": torch.tensor(False), "num_processed_tokens": 81},
    }
    runner._local_stage_payload_cache = {"r1": {"embed": {"prefill": new_prefill}}}
    runner._full_payload_pending_broadcast_req_ids = set()
    runner._lock = None
    runner._work_available = None

    OmniGPUModelRunner._sync_local_stage_payloads(runner)

    assert reset_calls == []
    assert getattr(runner, "_segment_send_watermark_reset_req_ids", set()) == set()


def test_sync_local_stage_payloads_decode_only_keeps_current_segment_state():
    runner = _make_runner(req_ids=("r1",), hidden_size=4)
    _enable_qwen3_talker_async_chunk(runner)
    runner.requests["r1"].resumable = True
    reset_calls = []
    runner.reset_segment_send_watermark = reset_calls.append

    old_decode = torch.tensor([[1.0, 2.0]])
    new_decode = torch.tensor([[3.0, 4.0]])
    runner.model_intermediate_buffer["r1"] = {
        "embed": {"decode": old_decode},
        "meta": {"num_processed_tokens": 1},
    }
    runner._local_stage_payload_cache = {"r1": {"embed": {"decode": new_decode}}}
    runner._full_payload_pending_broadcast_req_ids = set()
    runner._lock = None
    runner._work_available = None

    OmniGPUModelRunner._sync_local_stage_payloads(runner)

    buf = runner.model_intermediate_buffer["r1"]
    assert reset_calls == []
    assert getattr(runner, "_segment_send_watermark_reset_req_ids", set()) == set()
    assert torch.equal(buf["embed"]["decode"], torch.cat([old_decode, new_decode], dim=0))
    assert buf["meta"] == {"num_processed_tokens": 1}


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


def test_maybe_run_batch_preprocess_calls_model_hook():
    runner = object.__new__(OmniGPUModelRunner)
    runner.model_intermediate_buffer = {"r1": {"text": ["hello"]}}
    calls = []

    class DummyModel:
        def preprocess_batch(self, *, req_ids, model_intermediate_buffer, device):
            calls.append((req_ids, model_intermediate_buffer, device))

    runner.model = DummyModel()

    OmniGPUModelRunner._maybe_run_batch_preprocess(runner, ["r1"], torch.device("cpu"))

    assert calls == [(["r1"], runner.model_intermediate_buffer, torch.device("cpu"))]


def test_maybe_run_batch_preprocess_skips_missing_hook():
    runner = object.__new__(OmniGPUModelRunner)
    runner.model_intermediate_buffer = {}
    runner.model = object()

    OmniGPUModelRunner._maybe_run_batch_preprocess(runner, ["r1"], torch.device("cpu"))


def _make_full_payload_accumulation_runner(
    model_arch="Qwen3OmniMoeForConditionalGeneration",
    model_stage="talker",
    async_chunk=False,
    final_output=False,
    custom_process_next_stage_input_func="module.full_payload",
):
    runner = object.__new__(OmniConnectorModelRunnerMixin)
    runner.model_config = SimpleNamespace(
        model_arch=model_arch,
        model_stage=model_stage,
        async_chunk=async_chunk,
        final_output=final_output,
        custom_process_next_stage_input_func=custom_process_next_stage_input_func,
    )
    runner._custom_process_func = object()
    runner._pending_full_payload_send = {}
    runner._stage_id = 1
    # Non-None sentinel: the gate short-circuits to False when no connector
    # is configured at all (terminal stages in pipelines with no connector).
    runner._omni_connector = object()
    return runner


def test_accumulate_full_payload_output_preserves_aligned_all_zero_qwen3_omni_codec_rows():
    runner = _make_full_payload_accumulation_runner()
    request = SimpleNamespace(output_token_ids=[0, 1])
    codes = torch.zeros((2, 3), dtype=torch.long)

    OmniConnectorModelRunnerMixin.accumulate_full_payload_output(runner, "r1", {"codes.audio": codes}, request)

    stored, _ = OmniConnectorModelRunnerMixin._materialize_full_payload_entry(runner._pending_full_payload_send["r1"])
    assert torch.equal(stored["codes.audio"], codes)


def test_accumulate_full_payload_output_keeps_misaligned_all_zero_qwen3_omni_codec_rows():
    # After removing the sender-side zero filter, the full-payload accumulator keeps every
    # codec row including misaligned all-zero rows. The downstream consumer
    # (_extract_qwen3_full_payload_codec_rows) is the authoritative crop and
    # filters by output_token_ids.
    runner = _make_full_payload_accumulation_runner()
    request = SimpleNamespace(output_token_ids=[0, 1])
    codes = torch.zeros((1, 3), dtype=torch.long)

    OmniConnectorModelRunnerMixin.accumulate_full_payload_output(runner, "r1", {"codes.audio": codes}, request)

    stored, _ = OmniConnectorModelRunnerMixin._materialize_full_payload_entry(runner._pending_full_payload_send["r1"])
    assert "codes.audio" in stored
    assert torch.equal(stored["codes.audio"], codes)


def test_accumulate_full_payload_output_preserves_incremental_aligned_all_zero_qwen3_omni_codec_rows():
    runner = _make_full_payload_accumulation_runner()
    request = SimpleNamespace(output_token_ids=[0, 1])
    runner._pending_full_payload_send["r1"] = (
        {"codes.audio": torch.ones((1, 3), dtype=torch.long)},
        request,
    )
    codes = torch.zeros((1, 3), dtype=torch.long)

    OmniConnectorModelRunnerMixin.accumulate_full_payload_output(runner, "r1", {"codes.audio": codes}, request)

    stored, _ = OmniConnectorModelRunnerMixin._materialize_full_payload_entry(runner._pending_full_payload_send["r1"])
    assert stored["codes.audio"].shape == (2, 3)
    assert torch.equal(stored["codes.audio"][1], torch.zeros(3, dtype=torch.long))


def test_accumulate_full_payload_output_keeps_all_zero_qwen3_omni_prefill_placeholder():
    # Prefill placeholder rows (output_token_ids empty) are no longer dropped
    # at the sender. The consumer-side crop trims them off using
    # output_token_ids, so the end-to-end semantics are unchanged.
    runner = _make_full_payload_accumulation_runner()
    request = SimpleNamespace(output_token_ids=[])
    codes = torch.zeros((2, 3), dtype=torch.long)

    OmniConnectorModelRunnerMixin.accumulate_full_payload_output(runner, "r1", {"codes.audio": codes}, request)

    stored, _ = OmniConnectorModelRunnerMixin._materialize_full_payload_entry(runner._pending_full_payload_send["r1"])
    assert "codes.audio" in stored
    assert torch.equal(stored["codes.audio"], codes)


def test_full_payload_output_accumulation_hook_matrix():
    """Producer-side gate: fires iff an explicit next-stage payload hook is loaded.

    A derived `*_full_payload` helper from `custom_process_input_func` is not
    enough: terminal/input-only consumer stages must not enqueue orphan
    downstream payloads.
    """
    # Thinker / talker producer stages: explicit next-stage payload hook -> gate fires.
    assert _make_full_payload_accumulation_runner(model_stage="thinker")._should_accumulate_full_payload_output()
    assert _make_full_payload_accumulation_runner(model_stage="talker")._should_accumulate_full_payload_output()

    # Qwen3 thinker is both a text final-output stage and a downstream
    # talker producer; the explicit next-stage hook is the producer signal.
    assert _make_full_payload_accumulation_runner(
        model_stage="thinker", final_output=True
    )._should_accumulate_full_payload_output()

    # Terminal stage without an explicit producer hook must not accumulate/send.
    runner = _make_full_payload_accumulation_runner(
        model_stage="code2wav", final_output=True, custom_process_next_stage_input_func=None
    )
    assert not runner._should_accumulate_full_payload_output()

    # Input-only consumer stage without an explicit producer hook must not
    # accumulate/send just because a same-module *_full_payload helper exists.
    runner = _make_full_payload_accumulation_runner(
        model_stage="token2audio",
        custom_process_next_stage_input_func=None,
    )
    assert not runner._should_accumulate_full_payload_output()

    # async_chunk mode -> gate off.
    assert not _make_full_payload_accumulation_runner(
        model_stage="talker", async_chunk=True
    )._should_accumulate_full_payload_output()

    # Non-qwen3 arches: gate is arch-agnostic, but if the fixture's arch
    # does not configure a connector payload builder, its runtime
    # `_custom_process_func` is None.  Emulate that.
    runner = _make_full_payload_accumulation_runner(model_arch="Qwen3TTSForConditionalGeneration")
    runner._custom_process_func = None
    runner._should_accumulate_full_payload_output_cached = None
    assert not runner._should_accumulate_full_payload_output()
    runner = _make_full_payload_accumulation_runner(model_arch="Qwen2_5OmniForConditionalGeneration")
    runner._custom_process_func = None
    runner._should_accumulate_full_payload_output_cached = None
    assert not runner._should_accumulate_full_payload_output()


def test_sync_local_stage_payloads_retains_payload_until_request_is_active():
    runner = object.__new__(OmniGPUModelRunner)
    payload = {"codes": {"audio": [1, 2, 3]}}
    runner._local_stage_payload_cache = {"late": payload}
    runner._full_payload_pending_broadcast_req_ids = set()
    runner.requests = {}
    runner.model_intermediate_buffer = {}

    OmniGPUModelRunner._sync_local_stage_payloads(runner)

    assert runner._local_stage_payload_cache == {"late": payload}
    assert runner.model_intermediate_buffer == {}

    runner.requests = {"late": DummyReqState()}
    OmniGPUModelRunner._sync_local_stage_payloads(runner)

    assert runner._local_stage_payload_cache == {}
    assert runner.model_intermediate_buffer["late"] == payload
    assert runner.requests["late"].additional_information_cpu == payload


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
