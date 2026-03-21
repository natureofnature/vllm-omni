# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import Qwen3OmniMoeForConditionalGeneration
from vllm_omni.payload_span import (
    CACHED_THINKER_DECODE_EMBEDDINGS_KEY,
    CACHED_THINKER_DECODE_TOKEN_END_KEY,
    CACHED_THINKER_DECODE_TOKEN_START_KEY,
    THINKER_DECODE_EMBEDDINGS_KEY,
    THINKER_DECODE_TOKEN_END_KEY,
    THINKER_DECODE_TOKEN_START_KEY,
)


class _DummyTalker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.text_projection = torch.nn.Identity()
        self.register_buffer("_dummy", torch.zeros(1))


def _model() -> Qwen3OmniMoeForConditionalGeneration:
    model = Qwen3OmniMoeForConditionalGeneration.__new__(Qwen3OmniMoeForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.talker = _DummyTalker()
    model.tts_pad_embed = torch.tensor([[-1.0, -1.0]], dtype=torch.float32)
    model.tts_eos_embed = torch.tensor([[9.0, 9.0]], dtype=torch.float32)
    return model


def test_cache_preserves_absolute_decode_span_metadata():
    model = _model()
    update_dict = {}

    model._talker_cache_thinker_decode_embeds(
        {
            THINKER_DECODE_EMBEDDINGS_KEY: torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
            THINKER_DECODE_TOKEN_START_KEY: 100,
            THINKER_DECODE_TOKEN_END_KEY: 102,
        },
        update_dict,
    )

    assert update_dict[CACHED_THINKER_DECODE_TOKEN_START_KEY] == 100
    assert update_dict[CACHED_THINKER_DECODE_TOKEN_END_KEY] == 102
    assert torch.equal(
        update_dict[CACHED_THINKER_DECODE_EMBEDDINGS_KEY].to(torch.float32),
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
    )


def test_decode_reads_cached_span_by_absolute_row_index():
    model = _model()
    update_dict = {}

    out = model._thinker_decode_to_talker_decode(
        {
            CACHED_THINKER_DECODE_EMBEDDINGS_KEY: torch.tensor(
                [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]], dtype=torch.float32
            ),
            CACHED_THINKER_DECODE_TOKEN_START_KEY: 100,
            CACHED_THINKER_DECODE_TOKEN_END_KEY: 103,
            "num_processed_tokens": 101,
            "thinker_output_token_ids": list(range(104)),
            "finished": False,
        },
        torch.device("cpu"),
        update_dict,
    )

    assert torch.equal(out, torch.tensor([20.0, 21.0], dtype=torch.float32))
    assert update_dict["_advance_num_processed_tokens"] is True


def test_decode_emits_eos_when_finished_and_span_consumed(caplog):
    model = _model()
    update_dict = {}

    with caplog.at_level("WARNING"):
        out = model._thinker_decode_to_talker_decode(
            {
                CACHED_THINKER_DECODE_EMBEDDINGS_KEY: torch.tensor([[10.0, 11.0]], dtype=torch.float32),
                CACHED_THINKER_DECODE_TOKEN_START_KEY: 100,
                CACHED_THINKER_DECODE_TOKEN_END_KEY: 101,
                "num_processed_tokens": 101,
                "thinker_output_token_ids": list(range(104)),
                "finished": True,
            },
            torch.device("cpu"),
            update_dict,
        )

    assert torch.equal(out, model.tts_eos_embed)
    assert update_dict["finished_flag"] is True
    assert update_dict["_advance_num_processed_tokens"] is True
    assert "Talker decode reached available decode boundary" not in caplog.text


def test_decode_finished_flag_pad_is_quiet_at_terminal_boundary(caplog):
    model = _model()
    update_dict = {}

    with caplog.at_level("WARNING"):
        out = model._thinker_decode_to_talker_decode(
            {
                CACHED_THINKER_DECODE_EMBEDDINGS_KEY: torch.tensor([[10.0, 11.0]], dtype=torch.float32),
                CACHED_THINKER_DECODE_TOKEN_START_KEY: 100,
                CACHED_THINKER_DECODE_TOKEN_END_KEY: 102,
                "num_processed_tokens": 103,
                "thinker_output_token_ids": list(range(104)),
                "finished": True,
                "finished_flag": True,
            },
            torch.device("cpu"),
            update_dict,
        )

    assert torch.equal(out, model.tts_pad_embed)
    assert update_dict["_advance_num_processed_tokens"] is False
    assert "Talker decode reached available decode boundary" not in caplog.text


def test_decode_replaces_stale_cached_span_with_newer_incoming_span():
    model = _model()
    update_dict = {}

    out = model._thinker_decode_to_talker_decode(
        {
            CACHED_THINKER_DECODE_EMBEDDINGS_KEY: torch.tensor(
                [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]], dtype=torch.float32
            ),
            CACHED_THINKER_DECODE_TOKEN_START_KEY: 0,
            CACHED_THINKER_DECODE_TOKEN_END_KEY: 5,
            THINKER_DECODE_EMBEDDINGS_KEY: torch.tensor([[59.0, 59.5], [60.0, 60.5]], dtype=torch.float32),
            THINKER_DECODE_TOKEN_START_KEY: 59,
            THINKER_DECODE_TOKEN_END_KEY: 61,
            "num_processed_tokens": 59,
            "thinker_output_token_ids": list(range(104)),
            "finished": False,
        },
        torch.device("cpu"),
        update_dict,
    )

    assert torch.equal(out, torch.tensor([59.0, 59.5], dtype=torch.float32))
    assert update_dict[CACHED_THINKER_DECODE_TOKEN_START_KEY] == 59
    assert update_dict[CACHED_THINKER_DECODE_TOKEN_END_KEY] == 61


def test_cache_preserves_future_noncontiguous_span_when_cursor_still_needs_cached_rows():
    model = _model()
    update_dict = {}

    model._talker_cache_thinker_decode_embeds(
        {
            CACHED_THINKER_DECODE_EMBEDDINGS_KEY: torch.arange(188, dtype=torch.float32).reshape(94, 2),
            CACHED_THINKER_DECODE_TOKEN_START_KEY: 0,
            CACHED_THINKER_DECODE_TOKEN_END_KEY: 94,
            THINKER_DECODE_EMBEDDINGS_KEY: torch.tensor([[95.0, 95.5]], dtype=torch.float32),
            THINKER_DECODE_TOKEN_START_KEY: 95,
            THINKER_DECODE_TOKEN_END_KEY: 96,
            "num_processed_tokens": 68,
        },
        update_dict,
    )

    assert THINKER_DECODE_EMBEDDINGS_KEY not in update_dict
    assert CACHED_THINKER_DECODE_EMBEDDINGS_KEY not in update_dict
    assert CACHED_THINKER_DECODE_TOKEN_START_KEY not in update_dict
    assert CACHED_THINKER_DECODE_TOKEN_END_KEY not in update_dict


def test_decode_keeps_existing_cached_span_when_future_span_is_not_yet_stale():
    model = _model()
    update_dict = {}
    cached = torch.arange(188, dtype=torch.float32).reshape(94, 2)

    out = model._thinker_decode_to_talker_decode(
        {
            CACHED_THINKER_DECODE_EMBEDDINGS_KEY: cached.clone(),
            CACHED_THINKER_DECODE_TOKEN_START_KEY: 0,
            CACHED_THINKER_DECODE_TOKEN_END_KEY: 94,
            THINKER_DECODE_EMBEDDINGS_KEY: torch.tensor([[95.0, 95.5]], dtype=torch.float32),
            THINKER_DECODE_TOKEN_START_KEY: 95,
            THINKER_DECODE_TOKEN_END_KEY: 96,
            "num_processed_tokens": 68,
            "thinker_output_token_ids": list(range(104)),
            "finished": False,
        },
        torch.device("cpu"),
        update_dict,
    )

    assert torch.equal(out, cached[68])
    assert update_dict[CACHED_THINKER_DECODE_TOKEN_START_KEY] == 0
    assert update_dict[CACHED_THINKER_DECODE_TOKEN_END_KEY] == 94
    assert THINKER_DECODE_EMBEDDINGS_KEY not in update_dict
