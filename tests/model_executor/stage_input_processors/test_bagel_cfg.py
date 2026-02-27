# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm_omni.model_executor.stage_input_processors.bagel import (
    collect_cfg_kv_caches,
    expand_cfg_prompts,
)


def test_expand_cfg_prompts_uses_empty_negative_prompt_by_default():
    prompt = {"prompt": "city", "modalities": ["image"]}
    sampling_params = SimpleNamespace(extra_args={})

    expanded = expand_cfg_prompts(prompt, sampling_params)

    assert len(expanded) == 1
    assert expanded[0].role == "cfg_text"
    assert expanded[0].request_id_suffix == "__cfg_text"
    assert expanded[0].prompt == {"prompt": "<|im_start|><|im_end|>", "modalities": ["image"]}


def test_expand_cfg_prompts_respects_explicit_negative_prompt():
    prompt = {"prompt": "city", "modalities": ["image"], "negative_prompt": "fog"}
    sampling_params = SimpleNamespace(extra_args={})

    expanded = expand_cfg_prompts(prompt, sampling_params)

    assert expanded[0].prompt == {"prompt": "fog", "modalities": ["image"]}


def test_collect_cfg_kv_caches_uses_role_payload_map():
    cfg_role_payloads = {
        "cfg_text": (
            {
                "layer_blocks": {"k_cache": [1, 2], "v_cache": [3, 4]},
                "metadata": {"seq_len": 7},
            },
            128,
        )
    }

    collected = collect_cfg_kv_caches("req-1", cfg_role_payloads)

    assert collected["cfg_text_past_key_values"].k_cache == [1, 2]
    assert collected["cfg_text_past_key_values"].v_cache == [3, 4]
    assert collected["cfg_text_kv_metadata"] == {"seq_len": 7}
