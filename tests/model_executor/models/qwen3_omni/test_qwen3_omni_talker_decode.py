# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import (
    Qwen3OmniMoeForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_minimal_omni() -> Qwen3OmniMoeForConditionalGeneration:
    model = Qwen3OmniMoeForConditionalGeneration.__new__(Qwen3OmniMoeForConditionalGeneration)
    model.talker = SimpleNamespace(text_projection=lambda x: x + 10)
    model.tts_pad_embed = torch.full((2,), -1.0)
    model.tts_eos_embed = torch.full((2,), -2.0)
    return model


def _make_minimal_talker() -> Qwen3OmniMoeForConditionalGeneration:
    model = _make_minimal_omni()
    model.model_stage = "talker"
    return model


@pytest.mark.parametrize(
    ("cached_rows", "decode_rows", "num_processed_tokens", "expected"),
    [
        pytest.param(
            [[1.0, 2.0], [3.0, 4.0]],
            None,
            1,
            [11.0, 12.0],
            id="cached-only",
        ),
        pytest.param(
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0]],
            3,
            [15.0, 16.0],
            id="append-current",
        ),
        pytest.param(
            [[1.0, 2.0]],
            [[1.0, 2.0], [3.0, 4.0]],
            2,
            [13.0, 14.0],
            id="cache-prefix-of-current",
        ),
        pytest.param(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[1.0, 2.0], [3.0, 4.0]],
            3,
            [15.0, 16.0],
            id="current-prefix-of-cache",
        ),
        pytest.param(
            [[1.0, 2.0], [3.0, 4.0]],
            [[3.0, 4.0], [5.0, 6.0]],
            3,
            [15.0, 16.0],
            id="partial-overlap",
        ),
    ],
)
def test_async_chunk_decode_merges_handoff_rows(
    cached_rows: list[list[float]],
    decode_rows: list[list[float]] | None,
    num_processed_tokens: int,
    expected: list[float],
) -> None:
    model = _make_minimal_omni()
    embed = {"cached_decode": torch.tensor(cached_rows)}
    if decode_rows is not None:
        embed["decode"] = torch.tensor(decode_rows)
    payload = {
        "embed": embed,
        "meta": {
            "num_processed_tokens": num_processed_tokens,
            "prefill_consumed_text_tokens": 1,
        },
    }
    update: dict = {}

    out = model._thinker_decode_to_talker_decode(payload, torch.device("cpu"), update)

    assert torch.equal(out, torch.tensor(expected))
    assert update["_advance_num_processed_tokens"] is True


def test_async_chunk_decode_clears_prior_eos_when_new_decode_arrives() -> None:
    model = _make_minimal_omni()
    payload = {
        "embed": {"decode": torch.tensor([[7.0, 8.0]])},
        "meta": {
            "num_processed_tokens": 1,
            "prefill_consumed_text_tokens": 1,
            "eos_emitted": True,
        },
    }
    update: dict = {}

    out = model._thinker_decode_to_talker_decode(payload, torch.device("cpu"), update)

    assert torch.equal(out, torch.tensor([17.0, 18.0]))
    assert update["_advance_num_processed_tokens"] is True
    assert update["meta"]["eos_emitted"] is False


def test_async_chunk_decode_segment_terminal_emits_eos_after_cached_decode() -> None:
    model = _make_minimal_omni()
    payload = {
        "embed": {"cached_decode": torch.tensor([[1.0, 2.0]])},
        "meta": {
            "num_processed_tokens": 2,
            "prefill_consumed_text_tokens": 1,
            "is_segment_finished": True,
        },
    }
    update: dict = {}

    out = model._thinker_decode_to_talker_decode(payload, torch.device("cpu"), update)

    assert torch.equal(out, model.tts_eos_embed)
    assert update["_advance_num_processed_tokens"] is False
    assert update["meta"]["eos_emitted"] is True


def test_async_chunk_decode_uses_resumable_base_when_handoff_base_missing():
    model = _make_minimal_omni()
    payload = {
        "embed": {"cached_decode": torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
        "meta": {
            "decode_flag": True,
            "num_processed_tokens": 1,
            "resumable": True,
        },
    }
    update: dict = {}

    out = model._thinker_decode_to_talker_decode(payload, torch.device("cpu"), update)

    assert torch.equal(out, torch.tensor([11.0, 12.0]))
    assert update["_advance_num_processed_tokens"] is True


def test_talker_decode_recovers_streaming_reset_handoff_boundary() -> None:
    model = _make_minimal_omni()
    seen: dict = {}

    def fake_decode(input_ids, input_embeds, update_dict, payload):
        seen["num_processed_tokens"] = payload["meta"]["num_processed_tokens"]
        return torch.zeros(2), torch.tensor(0), update_dict

    model.talker_preprocess_decode = fake_decode
    payload = {"meta": {"resumable": True}}

    _, _, update = model.talker_preprocess(
        torch.tensor([0]),
        torch.ones(1, 2),
        **payload,
    )

    assert seen["num_processed_tokens"] == 1
    assert update["meta"]["decode_flag"] is True
    assert update["meta"]["prefill_consumed_text_tokens"] == 1


def test_talker_make_omni_output_preserves_dense_audio_codes() -> None:
    model = _make_minimal_talker()
    hidden = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    codes_0 = torch.tensor([[1, 2], [3, 4]])
    codes_1 = torch.tensor([[5, 6]])

    out = model.make_omni_output(
        hidden,
        model_intermediate_buffer=[
            {"request_id": "req-0", "codes": {"audio": codes_0}},
            {"request_id": "req-1", "codes": {"audio": codes_1}},
        ],
    )

    assert torch.equal(out.text_hidden_states, hidden)
    assert torch.equal(
        out.multimodal_outputs["codes"]["audio"],
        torch.cat([codes_0, codes_1], dim=0),
    )
    assert "meta" not in out.multimodal_outputs


def test_talker_make_omni_output_skips_control_step_without_codes() -> None:
    model = _make_minimal_talker()
    hidden = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    out = model.make_omni_output(
        hidden,
        model_intermediate_buffer=[{"request_id": "req-0"}, {"request_id": "req-1"}],
    )

    assert torch.equal(out.text_hidden_states, hidden)
    assert out.multimodal_outputs == {
        "codes": {"audio": []},
        "meta": {"req_id": [], "sparse_audio": ["1"]},
    }


def test_talker_make_omni_output_routes_partial_sparse_audio_codes() -> None:
    model = _make_minimal_talker()
    hidden = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    codes = torch.tensor([[7, 8]])

    out = model.make_omni_output(
        hidden,
        model_intermediate_buffer=[
            {"request_id": "req-0", "codes": {"audio": codes}},
            {"request_id": "req-1"},
        ],
    )

    assert torch.equal(out.text_hidden_states, hidden)
    assert out.multimodal_outputs["meta"] == {
        "req_id": ["req-0"],
        "sparse_audio": ["1"],
    }
    assert out.multimodal_outputs["codes"]["audio"] == [codes]
