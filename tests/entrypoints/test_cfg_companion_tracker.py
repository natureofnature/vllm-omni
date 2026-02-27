import time
from types import SimpleNamespace

import pytest
from vllm import SamplingParams

from vllm_omni.entrypoints.cfg_companion_tracker import CfgCompanionTracker
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def dummy_expand_func(prompt, sp0):
    if prompt == "expand_me":
        return [SimpleNamespace(prompt={"prompt": "neg"}, role="cfg_text", request_id_suffix="__cfg_text")]
    return []


@pytest.fixture
def tracker():
    sp0 = SimpleNamespace()
    return CfgCompanionTracker(prompt_expand_func=dummy_expand_func, stage0_sampling_params=sp0, timeout_s=0.1)


def test_companion_tracker_initialization(tracker):
    assert not tracker.is_active
    assert tracker.num_companions == 0


def test_expand_prompts_registers_companions(tracker):
    request_id_to_prompt = {"req1": "expand_me", "req2": "do_not_expand"}

    pairs = tracker.expand_prompts(request_id_to_prompt)

    assert len(pairs) == 1
    companion_id, prompt = pairs[0]
    assert companion_id == "req1__cfg_text"
    assert prompt == {"prompt": "neg"}

    assert tracker.is_active
    assert tracker.num_companions == 1
    assert tracker.is_companion("req1__cfg_text")
    assert not tracker.is_companion("req2__cfg_text")
    assert tracker.has_companions("req1")
    assert not tracker.has_companions("req2")

    comp_map = tracker.get_companion_request_ids("req1")
    assert comp_map == {"cfg_text": "req1__cfg_text"}


def test_companion_lifecycle_success(tracker):
    request_id_to_prompt = {"req1": "expand_me"}
    tracker.expand_prompts(request_id_to_prompt)

    # Defer parent
    engine_outputs = {"out": 123}
    tracker.defer_parent("req1", engine_outputs, stage_id=0)

    # Initially not done
    assert not tracker.all_companions_done("req1")

    # Companion completes
    parent_id = tracker.on_companion_completed("req1__cfg_text")

    # Parent should be returned since all companions are done and it is pending
    assert parent_id == "req1"
    assert tracker.all_companions_done("req1")

    # Pop pending parent
    popped = tracker.pop_pending_parent("req1")
    assert popped is not None
    assert popped["engine_outputs"] == engine_outputs
    assert popped["stage_id"] == 0


def test_companion_lifecycle_failure(tracker):
    request_id_to_prompt = {"req1": "expand_me"}
    tracker.expand_prompts(request_id_to_prompt)

    tracker.defer_parent("req1", {"out": 123}, stage_id=0)

    # Companion fails
    parent_id, aborted = tracker.on_companion_error("req1__cfg_text")

    assert parent_id == "req1"
    assert aborted is True
    assert tracker.is_parent_failed("req1")

    # Parent should be removed from pending list
    assert tracker.pop_pending_parent("req1") is None

    # Consume failure
    tracker.consume_parent_failure("req1")
    assert not tracker.is_parent_failed("req1")


def test_companion_lifecycle_timeout(tracker):
    request_id_to_prompt = {"req1": "expand_me"}
    tracker.expand_prompts(request_id_to_prompt)

    tracker.defer_parent("req1", {"out": 123}, stage_id=0)

    # Initially no timeouts
    timeouts = tracker.check_timeouts()
    assert len(timeouts) == 0

    # Wait for timeout
    time.sleep(0.15)

    # Check timeouts again
    timeouts = tracker.check_timeouts()
    assert len(timeouts) == 1
    assert timeouts[0] == "req1"

    # Should be removed from pending
    assert tracker.pop_pending_parent("req1") is None


class _CfgStage:
    def __init__(self, stage_id, stage_type):
        self.stage_id = stage_id
        self.stage_type = stage_type
        self.submitted = []

    def submit(self, task):
        self.submitted.append(task)


class _FakeMetrics:
    def __init__(self, num_stages):
        self.stage_first_ts = [None] * num_stages


def test_forward_parent_with_cfg_seeds_downstream_diffusion_request():
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
