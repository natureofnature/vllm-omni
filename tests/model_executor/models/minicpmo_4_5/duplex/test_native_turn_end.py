# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.models.minicpmo_4_5.duplex.data_plane import _native_decision
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.plugin import (
    MiniCPMO45DuplexPlugin,
    _stage0_stop_token_ids,
)
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.policy import MiniCPMO45DuplexPolicy

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

TOKEN_IDS = {
    "listen_token_id": 3,
    "speak_token_id": 4,
    "chunk_eos_token_id": 8,
    "chunk_tts_eos_token_id": 9,
    "turn_eos_token_id": 10,
}


def test_native_stage0_stop_ids_leave_turn_eos_for_a_followup_forward():
    tokens = {MiniCPMO45DuplexPolicy.SPECIAL_TOKEN_FIELDS[key]: value for key, value in TOKEN_IDS.items()}
    tokenizer = SimpleNamespace(unk_token_id=-1, convert_tokens_to_ids=lambda token: tokens.get(token, -1))

    assert _stage0_stop_token_ids(tokenizer) == [8, 9, 3]


@pytest.mark.parametrize(
    ("history", "direct_listen"),
    [
        ([3], True),
        ([4, 21, 10, 3], False),
        ([3, 3, 4, 10, 3], False),
        ([4, 21, 10, 77, 3], False),
        ([4, 21, 10, 3, 3], True),
        ([4, 21, 10, 8, 3], True),
        ([4, 21, 3], True),  # explicit interruption before turn end is unchanged
    ],
)
def test_final_listen_routes_only_the_current_unit(history, direct_listen):
    completion = SimpleNamespace(token_ids=[3], cumulative_token_ids=history, stop_reason=3)
    metadata = {"meta": TOKEN_IDS}
    output = SimpleNamespace(outputs=[completion], multimodal_output=metadata)
    plugin = MiniCPMO45DuplexPlugin(lambda *args: "")

    decision = plugin.decide_output(
        stage_id=0,
        final_stage_id=2,
        segment_finished=True,
        segment_token_ids=(3,),
        segment_output_metadata=metadata,
        output=output,
    )

    assert (decision is not None) is direct_listen
    assert _native_decision(completion, metadata, token_ids=[3], finished=True) == ("listen" if direct_listen else None)
