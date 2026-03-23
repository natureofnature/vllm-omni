# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from types import SimpleNamespace

import torch

from vllm_omni.model_executor.stage_input_processors.qwen3_omni import (
    talker2code2wav,
    talker2code2wav_async_chunk,
    thinker2talker,
    thinker2talker_async_chunk,
)

_FRAME = list(range(1, 17))


def _req(
    rid: str,
    *,
    finished: bool,
    all_token_ids=None,
    prompt_token_ids=None,
    output_token_ids=None,
):
    return SimpleNamespace(
        external_req_id=rid,
        all_token_ids=[] if all_token_ids is None else all_token_ids,
        prompt_token_ids=[] if prompt_token_ids is None else prompt_token_ids,
        output_token_ids=[] if output_token_ids is None else output_token_ids,
        is_finished=lambda: finished,
    )


def _thinker_tm():
    return SimpleNamespace(
        request_payload={},
        put_req_chunk=defaultdict(int),
    )


def _tm(*, chunk_frames=25, left_context=25):
    return SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        put_req_chunk=defaultdict(int),
        connector=SimpleNamespace(
            config={
                "extra": {
                    "codec_chunk_frames": chunk_frames,
                    "codec_left_context_frames": left_context,
                }
            }
        ),
    )


def test_flushes_tail_with_left_context_on_finish_sentinel():
    tm = _tm(chunk_frames=25, left_context=25)
    rid = "rid-tail"
    tm.code_prompt_token_ids[rid] = [_FRAME[:] for _ in range(30)]

    payload = talker2code2wav_async_chunk(
        transfer_manager=tm,
        pooling_output={},
        request=_req(rid, finished=True),
    )

    assert payload is not None
    assert payload["finished"].item() is True
    assert payload["left_context_size"] == 25
    assert len(payload["code_predictor_codes"]) == 16 * 30


def test_uses_request_finish_state_without_explicit_kwarg():
    tm = _tm(chunk_frames=25, left_context=25)
    rid = "rid-finished"
    tm.code_prompt_token_ids[rid] = [_FRAME[:] for _ in range(24)]

    payload = talker2code2wav_async_chunk(
        transfer_manager=tm,
        pooling_output={"code_predictor_codes": torch.tensor([_FRAME], dtype=torch.long)},
        request=_req(rid, finished=True),
    )

    assert payload is not None
    assert payload["finished"].item() is True
    assert payload["left_context_size"] == 0
    assert len(payload["code_predictor_codes"]) == 16 * 25


def test_thinker2talker_uses_request_finish_state_without_explicit_kwarg():
    tm = _thinker_tm()
    rid = "rid-thinker-finished"

    payload = thinker2talker_async_chunk(
        transfer_manager=tm,
        pooling_output={
            "0": torch.ones((1, 2), dtype=torch.float32),
            "24": torch.ones((1, 2), dtype=torch.float32),
            "tts_bos_embed": torch.ones((1, 2), dtype=torch.float32),
            "tts_eos_embed": torch.ones((1, 2), dtype=torch.float32),
            "tts_pad_embed": torch.ones((1, 2), dtype=torch.float32),
        },
        request=_req(
            rid,
            finished=True,
            all_token_ids=[1, 2],
            prompt_token_ids=[1],
            output_token_ids=[2],
        ),
    )

    assert payload is not None
    assert payload["finished"].item() is True
    assert rid not in tm.request_payload


def test_async_chunk_emits_only_last_valid_codec_frame():
    tm = _tm(chunk_frames=1, left_context=1)

    payload = talker2code2wav_async_chunk(
        transfer_manager=tm,
        pooling_output={
            "code_predictor_codes": torch.tensor(
                [
                    [0] * 16,
                    [2150] * 16,
                    _FRAME,
                ],
                dtype=torch.long,
            )
        },
        request=_req("rid-valid-frame", finished=False),
    )

    assert payload is not None
    assert payload["finished"].item() is False
    assert payload["left_context_size"] == 0
    assert payload["code_predictor_codes"] == _FRAME
    assert tm.code_prompt_token_ids["rid-valid-frame"] == [_FRAME]


def test_async_chunk_drops_invalid_codec_frames():
    tm = _tm(chunk_frames=1, left_context=1)

    payload = talker2code2wav_async_chunk(
        transfer_manager=tm,
        pooling_output={"code_predictor_codes": torch.tensor([[2150] * 16], dtype=torch.long)},
        request=_req("rid-invalid-frame", finished=False),
    )

    assert payload is None
    assert tm.code_prompt_token_ids["rid-invalid-frame"] == []


def test_thinker2talker_reads_request_level_multimodal_output():
    completion = SimpleNamespace(token_ids=[7, 8])
    stage_output = SimpleNamespace(
        prompt_token_ids=[1, 2],
        outputs=[completion],
        multimodal_output={
            "0": torch.ones((2, 3), dtype=torch.float32),
            "24": torch.full((2, 3), 2.0, dtype=torch.float32),
            "tts_bos_embed": torch.full((1, 3), 3.0, dtype=torch.float32),
            "tts_eos_embed": torch.full((1, 3), 4.0, dtype=torch.float32),
            "tts_pad_embed": torch.full((1, 3), 5.0, dtype=torch.float32),
        },
    )
    stage = SimpleNamespace(engine_outputs=[stage_output])

    prompts = thinker2talker(stage_list=[stage], engine_input_source=[0])

    assert len(prompts) == 1
    info = prompts[0]["additional_information"]
    assert torch.equal(info["thinker_prefill_embeddings"].cpu(), stage_output.multimodal_output["0"])
    assert torch.equal(info["thinker_hidden_states"].cpu(), stage_output.multimodal_output["24"])
    assert prompts[0]["prompt_token_ids"] == [0] * len(prompts[0]["prompt_token_ids"])


def test_talker2code2wav_reads_request_level_multimodal_output():
    completion = SimpleNamespace(token_ids=[10, 11, 12])
    mm_output = {
        "code_predictor_codes": torch.tensor(
            [
                list(range(1, 17)),
                list(range(17, 33)),
            ],
            dtype=torch.long,
        )
    }
    stage_output = SimpleNamespace(outputs=[completion], multimodal_output=mm_output)
    stage_list = [SimpleNamespace(engine_outputs=None), SimpleNamespace(engine_outputs=[stage_output])]

    prompts = talker2code2wav(stage_list=stage_list, engine_input_source=[1])

    assert len(prompts) == 1
    assert prompts[0]["prompt_token_ids"] == [
        1,
        17,
        2,
        18,
        3,
        19,
        4,
        20,
        5,
        21,
        6,
        22,
        7,
        23,
        8,
        24,
        9,
        25,
        10,
        26,
        11,
        27,
        12,
        28,
        13,
        29,
        14,
        30,
        15,
        31,
        16,
        32,
    ]
