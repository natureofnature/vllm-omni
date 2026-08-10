# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.stage_bypass import (
    OMNI_SKIP_STAGES_KEY,
    build_empty_asr_aura_chunk_payload,
    make_mock_text_stage_output,
    should_skip_stage,
    should_skip_stage_from_info,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_should_skip_stage_reads_additional_information_flag():
    assert should_skip_stage({}, stage_id=0) is False
    assert should_skip_stage({"additional_information": {OMNI_SKIP_STAGES_KEY: [0]}}, stage_id=0) is True
    assert should_skip_stage({"additional_information": {OMNI_SKIP_STAGES_KEY: [0]}}, stage_id=1) is False
    assert should_skip_stage({"additional_information": {OMNI_SKIP_STAGES_KEY: [0, 1]}}, stage_id=1) is True
    assert should_skip_stage({"additional_information": {OMNI_SKIP_STAGES_KEY: []}}, stage_id=0) is False


def test_should_skip_stage_from_info_accepts_string_ids():
    assert should_skip_stage_from_info({OMNI_SKIP_STAGES_KEY: ["0"]}, stage_id=0) is True
    assert should_skip_stage_from_info({OMNI_SKIP_STAGES_KEY: 0}, stage_id=0) is True


def test_make_mock_text_stage_output():
    mock = make_mock_text_stage_output("req-1", text="")
    assert mock.request_id == "req-1"
    assert mock.finished is True
    assert mock.outputs[0].text == ""
    assert mock.outputs[0].finished is True


def test_build_empty_asr_aura_chunk_payload():
    info = {OMNI_SKIP_STAGES_KEY: [0], "aura_session_id": "s1"}
    payload = build_empty_asr_aura_chunk_payload(info)
    assert payload["aura_asr_transcript"] == ""
    assert payload["additional_information"]["aura_session_id"] == "s1"
    meta = payload["meta"]
    assert bool(meta["finished"].item()) is True
    assert bool(meta["is_segment_finished"].item()) is True
    assert meta["finished"].dtype == torch.bool
