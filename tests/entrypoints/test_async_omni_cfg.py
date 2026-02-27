import asyncio

import pytest
from vllm import SamplingParams

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.stage_input_processors.bagel import ExpandedPrompt

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeStage:
    def __init__(self, stage_id, stage_type, prompt_expand_func=None):
        self.stage_id = stage_id
        self.stage_type = stage_type
        self.prompt_expand_func = prompt_expand_func
        self.submitted = []
        self.final_output = False
        self.final_output_type = None

    def submit(self, task):
        self.submitted.append(task)

    def set_engine_outputs(self, outputs):
        self.engine_outputs = outputs


def test_async_omni_generate_skips_cfg_companion_for_diffusion(monkeypatch):
    def expand(prompt, _sampling_params):
        return [
            ExpandedPrompt(
                prompt={"prompt": "<|im_start|><|im_end|>", "modalities": ["image"]},
                role="cfg_text",
                request_id_suffix="__cfg_text",
            )
        ]

    async def fake_process(
        self,
        request_id,
        req_state,
        metrics,
        final_stage_id_for_e2e,
        sampling_params_list,
        prompt,
        cfg_request_ids,
    ):
        assert cfg_request_ids == {}
        if False:
            yield request_id, req_state, metrics, final_stage_id_for_e2e, sampling_params_list, prompt

    monkeypatch.setattr(AsyncOmni, "_run_output_handler", lambda self: None)
    monkeypatch.setattr(AsyncOmni, "_process_sequential_results", fake_process)

    omni = AsyncOmni.__new__(AsyncOmni)
    omni._pause_cond = asyncio.Condition()
    omni._paused = False
    omni.request_states = {}
    omni.output_handler = None
    omni.stage_list = [FakeStage(0, "llm", prompt_expand_func=expand), FakeStage(1, "diffusion")]
    omni.default_sampling_params_list = [SamplingParams(), OmniDiffusionSamplingParams()]
    omni.log_stats = False
    omni.output_modalities = ["image"]
    omni.async_chunk = False

    async def run_generate():
        async for _ in omni.generate(
            prompt={"prompt": "city", "modalities": ["image"]},
            request_id="req-1",
            sampling_params_list=[SamplingParams(), OmniDiffusionSamplingParams()],
            output_modalities=["image"],
        ):
            pass

    asyncio.run(run_generate())

    assert [task["request_id"] for task in omni.stage_list[0].submitted] == ["req-1"]


def test_prepare_downstream_sampling_params_attaches_cfg_ids_for_diffusion_stage():
    omni = AsyncOmni.__new__(AsyncOmni)
    omni.stage_list = [FakeStage(0, "llm"), FakeStage(1, "diffusion")]

    original_params = OmniDiffusionSamplingParams()
    prepared_params = omni._prepare_downstream_sampling_params(
        request_id="req-1",
        stage_id=1,
        sampling_params=original_params,
        cfg_request_ids={"cfg_text": "req-1__cfg_text"},
    )

    assert prepared_params is not original_params
    assert prepared_params.cfg_kv_request_ids == {"cfg_text": "req-1__cfg_text"}
    assert getattr(original_params, "cfg_kv_request_ids", None) in (None, {})


def test_prepare_downstream_sampling_params_leaves_non_diffusion_stage_unchanged():
    omni = AsyncOmni.__new__(AsyncOmni)
    omni.stage_list = [FakeStage(0, "llm"), FakeStage(1, "llm")]

    original_params = SamplingParams()
    prepared_params = omni._prepare_downstream_sampling_params(
        request_id="req-1",
        stage_id=1,
        sampling_params=original_params,
        cfg_request_ids={"cfg_text": "req-1__cfg_text"},
    )

    assert prepared_params is not original_params
    assert not hasattr(prepared_params, "cfg_kv_request_ids")


def test_process_async_results_only_seeds_until_final_stage():
    omni = AsyncOmni.__new__(AsyncOmni)
    omni.stage_list = [FakeStage(0, "llm"), FakeStage(1, "llm"), FakeStage(2, "llm")]
    metrics = _FakeMetrics(num_stages=3)
    stage0_output = type("Stage0Output", (), {"prompt_token_ids": [1, 2, 3]})()

    def fake_process_single_result(result, stage, stage_id, metrics):
        if stage_id == 0:
            return stage0_output, True, None
        return result, True, None

    omni._process_single_result = fake_process_single_result

    async def run_test():
        req_state = type("ReqState", (), {})()
        req_state.stage_queues = {0: asyncio.Queue(), 1: asyncio.Queue()}
        await req_state.stage_queues[0].put({"stage": 0})
        await req_state.stage_queues[1].put({"stage": 1})
        async for _ in omni._process_async_results(
            request_id="req-1",
            prompt={"prompt": "hello"},
            sampling_params_list=[SamplingParams(), SamplingParams(), SamplingParams()],
            cfg_request_ids={},
            req_state=req_state,
            metrics=metrics,
            final_stage_id_for_e2e=1,
        ):
            pass

    asyncio.run(run_test())

    assert len(omni.stage_list[1].submitted) == 1
    assert omni.stage_list[2].submitted == []


def test_process_sequential_results_only_seeds_until_final_stage():
    omni = AsyncOmni.__new__(AsyncOmni)
    omni.stage_list = [FakeStage(0, "llm"), FakeStage(1, "llm"), FakeStage(2, "llm")]
    metrics = _FakeMetrics(num_stages=3)
    stage0_output = type("Stage0Output", (), {"prompt_token_ids": [1, 2, 3]})()

    async def run_test():
        req_state = type("ReqState", (), {})()
        req_state.stage_id = 0

        class _Queue:
            def __init__(self):
                self._items = iter([{"stage": 0}, {"stage": 1}])

            async def get(self):
                return next(self._items)

        req_state.queue = _Queue()

        def fake_process_single_result(result, stage, stage_id, metrics):
            if stage_id == 0:
                req_state.stage_id = 1
                return stage0_output, True, None
            return [result], True, None

        omni._process_single_result = fake_process_single_result

        async for _ in omni._process_sequential_results(
            request_id="req-1",
            req_state=req_state,
            metrics=metrics,
            final_stage_id_for_e2e=1,
            sampling_params_list=[SamplingParams(), SamplingParams(), SamplingParams()],
            prompt={"prompt": "hello"},
            cfg_request_ids={},
        ):
            pass

    asyncio.run(run_test())

    assert len(omni.stage_list[1].submitted) == 1
    assert omni.stage_list[2].submitted == []


class _FakeMetrics:
    def __init__(self, num_stages):
        self.stage_first_ts = [None] * num_stages


class _CfgStage:
    def __init__(self, stage_id, stage_type):
        self.stage_id = stage_id
        self.stage_type = stage_type
        self.submitted = []

    def submit(self, task):
        self.submitted.append(task)


def test_cfg_companion_tracker_seeds_downstream_diffusion_request():
    from vllm_omni.entrypoints.cfg_companion_tracker import CfgCompanionTracker

    tracker = CfgCompanionTracker(prompt_expand_func=None, stage0_sampling_params=SamplingParams())
    tracker._companion_map["req-1"] = {"cfg_text": "req-1__cfg_text"}

    stage_list = [_CfgStage(0, "llm"), _CfgStage(1, "diffusion")]
    sampling_params_list = [SamplingParams(), OmniDiffusionSamplingParams()]
    request_id_to_prompt = {"req-1": {"prompt": "city", "modalities": ["image"]}}
    final_stage_id_to_prompt = {"req-1": 1}
    metrics = _FakeMetrics(num_stages=2)
    remaining_by_stage = [0, 0]

    forwarded = tracker.forward_parent_with_cfg(
        req_id="req-1",
        parent_result={
            "stage_id": 0,
            "engine_outputs": [type("Out", (), {"prompt_token_ids": [1, 2, 3]})()],
        },
        stage_list=stage_list,
        sampling_params_list=sampling_params_list,
        request_id_to_prompt=request_id_to_prompt,
        final_stage_id_to_prompt=final_stage_id_to_prompt,
        metrics=metrics,
        remaining_by_stage=remaining_by_stage,
    )

    assert forwarded is True
    assert remaining_by_stage == [0, 1]
    assert metrics.stage_first_ts[1] is not None
    assert len(stage_list[1].submitted) == 1

    task = stage_list[1].submitted[0]
    assert task["request_id"] == "req-1"
    assert task["from_connector"] is False
    assert task["engine_inputs"]["prompt_token_ids"]
    assert task["engine_inputs"]["multi_modal_data"] is None
    assert isinstance(task["sampling_params"], OmniDiffusionSamplingParams)
    assert task["sampling_params"].cfg_kv_request_ids == {"cfg_text": "req-1__cfg_text"}
