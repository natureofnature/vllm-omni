# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Qwen3-Omni streaming thinker→talker / talker→codec helpers (PR #2581)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import vllm_omni.model_executor.stage_input_processors.qwen3_omni as q3

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _streaming_context() -> SimpleNamespace:
    return SimpleNamespace(bridge_states={})


def test_get_streaming_talker_tokens_first_segment(_streaming_context: SimpleNamespace) -> None:
    inc_p, inc_o, merged, thinker_in = q3._get_streaming_talker_tokens(
        "r1",
        [1, 2],
        [10, 11],
        streaming_context=_streaming_context,
    )
    assert inc_p == [1, 2]
    assert inc_o == [10, 11]
    assert merged == [1, 2, 10, 11]
    assert thinker_in == [1, 2]


def test_get_streaming_talker_tokens_second_segment_accumulates(_streaming_context: SimpleNamespace) -> None:
    q3._get_streaming_talker_tokens("r2", [1, 2], [10, 11], streaming_context=_streaming_context)
    inc_p, inc_o, merged, thinker_in = q3._get_streaming_talker_tokens(
        "r2",
        [1, 2, 3, 4],
        [10, 11, 12, 13],
        streaming_context=_streaming_context,
    )
    assert inc_p == [3, 4]
    assert inc_o == [12, 13]
    assert merged == [1, 2, 10, 3, 4, 12, 13]
    assert thinker_in == [1, 2, 10, 3, 4]


def test_get_streaming_talker_tokens_new_prompt_len_snapshot_truncates(
    _streaming_context: SimpleNamespace,
) -> None:
    inc_p, inc_o, merged, thinker_in = q3._get_streaming_talker_tokens(
        "r3",
        [1, 2, 3, 4, 5, 6],
        [10],
        new_prompt_len_snapshot=2,
        streaming_context=_streaming_context,
    )
    assert inc_p == [1, 2, 3, 4]
    assert inc_o == [10]
    assert merged == [1, 2, 3, 4, 10]
    assert thinker_in == [1, 2, 3, 4]


def test_get_streaming_talker_tokens_clear_state(_streaming_context: SimpleNamespace) -> None:
    q3._get_streaming_talker_tokens("r4", [1], [2], streaming_context=_streaming_context, clear_state=True)
    state = q3._get_qwen3_streaming_state("r4", _streaming_context).thinker2talker
    assert state.last_prompt_len == 0
    assert state.last_output_len == 0
    assert state.merged_sequences == []


def test_get_streaming_codec_delta_len_increments_and_finishes(_streaming_context: SimpleNamespace) -> None:
    d1 = q3._get_streaming_codec_delta_len(5, "c1", SimpleNamespace(finished=False), _streaming_context)
    assert d1 == 5
    d2 = q3._get_streaming_codec_delta_len(8, "c1", SimpleNamespace(finished=False), _streaming_context)
    assert d2 == 2
    # After d2, stored cursor is cur_seq_len + 1 == 9; next delta uses new cur_seq_len - 9.
    d3 = q3._get_streaming_codec_delta_len(10, "c1", SimpleNamespace(finished=True), _streaming_context)
    assert d3 == 1
    state = q3._get_qwen3_streaming_state("c1", _streaming_context)
    assert state.talker2code2wav_last_seq_len == 0


def test_talker2code2wav_full_payload_filters_by_output_token_ids() -> None:
    request = SimpleNamespace(
        request_id="codec",
        output_token_ids=[4197, 1, 2, 4198, -1, 2048],
    )
    rows = torch.tensor(
        [
            [100, 101, 102],
            [10, 11, 12],
            [20, 21, 22],
            [30, 31, 32],
            [40, 41, 42],
            [50, 51, 52],
        ],
        dtype=torch.long,
    )

    payload = q3.talker2code2wav_full_payload(None, {"codes.audio": rows}, request)

    assert payload is not None
    assert payload["codes"]["audio"] == [10, 20, 11, 21, 12, 22]
    assert payload["code_predictor_codes"] == payload["codes"]["audio"]


def test_talker2code2wav_full_payload_drops_count_matched_terminal_row() -> None:
    request = SimpleNamespace(
        request_id="codec_terminal_row",
        output_token_ids=[0, 4198],
    )
    rows = torch.tensor(
        [
            [10, 11, 12],
        ],
        dtype=torch.long,
    )

    payload = q3.talker2code2wav_full_payload(None, {"codes.audio": rows}, request)

    assert payload is None


def test_talker2code2wav_full_payload_drops_rows_aligned_to_non_codec_ids() -> None:
    request = SimpleNamespace(
        request_id="codec_invalid_ids",
        output_token_ids=[4197, 0, 4198, 4196, -1, 2048],
    )
    rows = torch.tensor(
        [
            [91, 92, 93],
            [0, 0, 0],
            [81, 82, 83],
            [71, 72, 73],
            [61, 62, 63],
            [51, 52, 53],
        ],
        dtype=torch.long,
    )

    payload = q3.talker2code2wav_full_payload(None, {"codes.audio": rows}, request)

    assert payload is not None
    assert payload["codes"]["audio"] == [0, 0, 0]
    assert payload["code_predictor_codes"] == payload["codes"]["audio"]


def test_talker2code2wav_full_payload_keeps_all_zero_codec_rows() -> None:
    request = SimpleNamespace(
        request_id="codec_zero",
        output_token_ids=[0, 1],
    )
    rows = torch.tensor(
        [
            [0, 0, 0],
            [7, 8, 9],
        ],
        dtype=torch.long,
    )

    payload = q3.talker2code2wav_full_payload(None, {"codes.audio": rows}, request)

    assert payload is not None
    assert payload["codes"]["audio"] == [0, 7, 0, 8, 0, 9]
    assert payload["code_predictor_codes"] == payload["codes"]["audio"]


def test_thinker2talker_full_payload_packs_complete_tensors() -> None:
    """Standard max_tokens finish path: rows == target → no trim."""
    request = SimpleNamespace(
        request_id="thinker",
        prompt_token_ids=[151644, 872],
        output_token_ids=[3],
        all_token_ids=[151644, 872, 3],
    )
    pooling_output = {
        "hidden_states.layer_0": torch.ones(3, 2),
        "hidden_states.layer_24": torch.full((3, 2), 2.0),
        "embed.tts_bos": torch.zeros(1, 2),
    }

    payload = q3.thinker2talker_full_payload(None, pooling_output, request)

    assert payload is not None
    assert payload["ids"]["all"] == [151644, 872, 3]
    assert payload["embed"]["prefill"].device.type == "cpu"
    assert payload["hidden_states"]["output"].device.type == "cpu"
    assert payload["next_stage_prompt_len"] > 0
    # Lock down the no-trim invariant for rows == target.
    assert payload["embed"]["prefill"].shape[0] == 3
    assert payload["hidden_states"]["output"].shape[0] == 3


def test_thinker2talker_full_payload_trims_excess_stop_token_row() -> None:
    """Excess-rows path: rows == target + 1 → trim trailing row."""
    request = SimpleNamespace(
        request_id="thinker-excess",
        prompt_token_ids=[151644, 872],
        output_token_ids=[3],
        all_token_ids=[151644, 872, 3],
        sampling_params=None,
        status=None,
    )
    pooling_output = {
        "hidden_states.layer_0": torch.ones(4, 2),
        "hidden_states.layer_24": torch.full((4, 2), 2.0),
        "embed.tts_bos": torch.zeros(1, 2),
    }

    payload = q3.thinker2talker_full_payload(None, pooling_output, request)

    assert payload is not None
    assert payload["embed"]["prefill"].shape[0] == 3
    assert payload["hidden_states"]["output"].shape[0] == 3


def test_thinker2talker_full_payload_drops_stop_emission_row_when_finished_stopped() -> None:
    """FINISHED_STOPPED: drop 1 extra row even when rows == target.

    vLLM appends the stop-token to output_token_ids before check_stop, so
    len(all_token_ids) includes the stop slot AND the accumulator has the
    stop emission's forward row.  Both counts equal P+O (here 3).  Talker
    target should be P+O-1 (=2), not P+O.  Without the extra drop the
    stop emission's hidden state leaks into talker prefill (fba23325
    spurious-phoneme regression).
    """
    request = SimpleNamespace(
        request_id="thinker-stop-finished",
        prompt_token_ids=[151644, 872],
        output_token_ids=[3],
        all_token_ids=[151644, 872, 3],
        sampling_params=None,
        status=SimpleNamespace(name="FINISHED_STOPPED"),
    )
    pooling_output = {
        "hidden_states.layer_0": torch.ones(3, 2),
        "hidden_states.layer_24": torch.full((3, 2), 2.0),
        "embed.tts_bos": torch.zeros(1, 2),
    }

    payload = q3.thinker2talker_full_payload(None, pooling_output, request)

    assert payload is not None
    assert payload["embed"]["prefill"].shape[0] == 2
    assert payload["hidden_states"]["output"].shape[0] == 2


def test_thinker2talker_full_payload_drops_stop_emission_via_eos_fallback() -> None:
    """Stop-detection fallback: last token in sampling_params.eos_token_id."""
    EOS = 151645
    request = SimpleNamespace(
        request_id="thinker-stop-fallback",
        prompt_token_ids=[151644, 872],
        output_token_ids=[3, EOS],
        all_token_ids=[151644, 872, 3, EOS],
        sampling_params=SimpleNamespace(eos_token_id=EOS, stop_token_ids=None),
        status=None,
    )
    pooling_output = {
        "hidden_states.layer_0": torch.ones(4, 2),
        "hidden_states.layer_24": torch.full((4, 2), 2.0),
        "embed.tts_bos": torch.zeros(1, 2),
    }

    payload = q3.thinker2talker_full_payload(None, pooling_output, request)

    assert payload is not None
    assert payload["embed"]["prefill"].shape[0] == 3
    assert payload["hidden_states"]["output"].shape[0] == 3


def test_thinker2talker_full_payload_no_drop_when_finished_length_capped() -> None:
    """FINISHED_LENGTH_CAPPED (max_tokens): no extra drop; BK 9702 regression guard."""
    request = SimpleNamespace(
        request_id="thinker-length-capped",
        prompt_token_ids=[151644, 872],
        output_token_ids=[3],
        all_token_ids=[151644, 872, 3],
        sampling_params=SimpleNamespace(eos_token_id=999, stop_token_ids=None),
        status=SimpleNamespace(name="FINISHED_LENGTH_CAPPED"),
    )
    pooling_output = {
        "hidden_states.layer_0": torch.ones(3, 2),
        "hidden_states.layer_24": torch.full((3, 2), 2.0),
        "embed.tts_bos": torch.zeros(1, 2),
    }

    payload = q3.thinker2talker_full_payload(None, pooling_output, request)

    assert payload is not None
    assert payload["embed"]["prefill"].shape[0] == 3
    assert payload["hidden_states"]["output"].shape[0] == 3


def test_thinker2talker_full_payload_drops_via_private_eos_field() -> None:
    """Worker-side sampling_params where the public `eos_token_id` property is
    None but the private `_eos_token_id` / `_all_stop_token_ids` carry the
    primary EOS (the msgspec-deserialization shape on the worker boundary).

    The fallback must read the private fields to detect the stop.
    """
    EOS = 151643
    request = SimpleNamespace(
        request_id="thinker-private-eos",
        prompt_token_ids=[151644, 872],
        output_token_ids=[3, EOS],
        all_token_ids=[151644, 872, 3, EOS],
        # Public `eos_token_id` looks empty; only the private fields carry it.
        sampling_params=SimpleNamespace(
            eos_token_id=None,
            stop_token_ids=None,
            ignore_eos=False,
            _eos_token_id=EOS,
            _all_stop_token_ids={EOS},
        ),
        status=None,
    )
    pooling_output = {
        "hidden_states.layer_0": torch.ones(4, 2),
        "hidden_states.layer_24": torch.full((4, 2), 2.0),
        "embed.tts_bos": torch.zeros(1, 2),
    }

    payload = q3.thinker2talker_full_payload(None, pooling_output, request)

    assert payload is not None
    assert payload["embed"]["prefill"].shape[0] == 3
    assert payload["hidden_states"]["output"].shape[0] == 3


def test_thinker2talker_full_payload_drops_via_all_stop_token_ids() -> None:
    """Secondary EOS only in `_all_stop_token_ids` (not in `_eos_token_id`):
    multi-EOS Qwen3 case where the model finished on a secondary EOS.
    """
    SECONDARY_EOS = 151645
    request = SimpleNamespace(
        request_id="thinker-secondary-eos",
        prompt_token_ids=[151644, 872],
        output_token_ids=[3, SECONDARY_EOS],
        all_token_ids=[151644, 872, 3, SECONDARY_EOS],
        sampling_params=SimpleNamespace(
            eos_token_id=151643,  # primary, not the one we hit
            stop_token_ids=None,
            ignore_eos=False,
            _eos_token_id=151643,
            _all_stop_token_ids={151643, SECONDARY_EOS},
        ),
        status=None,
    )
    pooling_output = {
        "hidden_states.layer_0": torch.ones(4, 2),
        "hidden_states.layer_24": torch.full((4, 2), 2.0),
        "embed.tts_bos": torch.zeros(1, 2),
    }

    payload = q3.thinker2talker_full_payload(None, pooling_output, request)

    assert payload is not None
    assert payload["embed"]["prefill"].shape[0] == 3
    assert payload["hidden_states"]["output"].shape[0] == 3


def test_thinker2talker_full_payload_no_drop_when_ignore_eos_and_trailing_eos() -> None:
    """ignore_eos=True + length-capped + last token == EOS: no drop.

    Production worker uses CachedRequestState (no `.status` field), so
    the status path doesn't catch this case; we rely on the
    `sampling_params.ignore_eos` flag in the fallback to suppress the
    EOS-as-stop heuristic.
    """
    EOS = 151645
    request = SimpleNamespace(
        request_id="thinker-ignore-eos-trailing-eos",
        prompt_token_ids=[151644, 872],
        output_token_ids=[3, EOS],
        all_token_ids=[151644, 872, 3, EOS],
        sampling_params=SimpleNamespace(eos_token_id=EOS, stop_token_ids=None, ignore_eos=True),
        status=None,  # production worker state has no status
    )
    pooling_output = {
        "hidden_states.layer_0": torch.ones(4, 2),
        "hidden_states.layer_24": torch.full((4, 2), 2.0),
        "embed.tts_bos": torch.zeros(1, 2),
    }

    payload = q3.thinker2talker_full_payload(None, pooling_output, request)

    assert payload is not None
    assert payload["embed"]["prefill"].shape[0] == 4
    assert payload["hidden_states"]["output"].shape[0] == 4


def test_thinker2talker_full_payload_no_drop_when_length_capped_with_trailing_eos() -> None:
    """FINISHED_LENGTH_CAPPED + last token == EOS coincidence: no drop.

    Status path takes precedence over last-token heuristic.  Without
    this guard the fallback would incorrectly drop a row when a length-capped
    request happens to end on the EOS token id.
    """
    EOS = 151645
    request = SimpleNamespace(
        request_id="thinker-len-cap-trailing-eos",
        prompt_token_ids=[151644, 872],
        output_token_ids=[3, EOS],
        all_token_ids=[151644, 872, 3, EOS],
        sampling_params=SimpleNamespace(eos_token_id=EOS, stop_token_ids=None),
        status=SimpleNamespace(name="FINISHED_LENGTH_CAPPED"),
    )
    pooling_output = {
        "hidden_states.layer_0": torch.ones(4, 2),
        "hidden_states.layer_24": torch.full((4, 2), 2.0),
        "embed.tts_bos": torch.zeros(1, 2),
    }

    payload = q3.thinker2talker_full_payload(None, pooling_output, request)

    assert payload is not None
    assert payload["embed"]["prefill"].shape[0] == 4
    assert payload["hidden_states"]["output"].shape[0] == 4


def test_thinker2talker_full_payload_preserves_under_capture() -> None:
    """Under-capture path: rows < target → no trim, safe degrade."""
    request = SimpleNamespace(
        request_id="thinker-undercap",
        prompt_token_ids=[151644, 872],
        output_token_ids=[3],
        all_token_ids=[151644, 872, 3],
    )
    pooling_output = {
        "hidden_states.layer_0": torch.ones(2, 2),
        "hidden_states.layer_24": torch.full((2, 2), 2.0),
        "embed.tts_bos": torch.zeros(1, 2),
    }

    payload = q3.thinker2talker_full_payload(None, pooling_output, request)

    assert payload is not None
    assert payload["embed"]["prefill"].shape[0] == 2
    assert payload["hidden_states"]["output"].shape[0] == 2


def test_accumulator_replaces_keys_in_replace_set() -> None:
    """REPLACE-key semantics: subsequent emissions of the same key replace, not append."""
    from vllm_omni.worker.omni_connector_model_runner_mixin import OmniConnectorModelRunnerMixin

    class _StubMixin(OmniConnectorModelRunnerMixin):
        def __init__(self):
            self._pending_full_payload_send = {}
            self._full_payload_replace_keys_cached = frozenset({"model_outputs"})

    stub = _StubMixin()
    stub.accumulate_full_payload_output(
        "req1",
        {
            "model_outputs": torch.tensor([[1.0, 2.0]]),
            "hidden_states.output": torch.tensor([[10.0]]),
        },
        request=None,
    )
    stub.accumulate_full_payload_output(
        "req1",
        {
            "model_outputs": torch.tensor([[3.0, 4.0]]),
            "hidden_states.output": torch.tensor([[20.0]]),
        },
        request=None,
    )
    output, _ = stub._materialize_full_payload_entry(stub._pending_full_payload_send["req1"])
    # model_outputs REPLACED (second value only):
    assert torch.equal(output["model_outputs"], torch.tensor([[3.0, 4.0]]))
    # hidden_states.output CONCATENATED:
    assert torch.equal(output["hidden_states.output"], torch.tensor([[10.0], [20.0]]))


def test_accumulator_concat_default_when_no_replace_keys() -> None:
    """Default semantics: 2-D+ tensors concat across emissions when not in replace_keys."""
    from vllm_omni.worker.omni_connector_model_runner_mixin import OmniConnectorModelRunnerMixin

    class _StubMixin(OmniConnectorModelRunnerMixin):
        def __init__(self):
            self._pending_full_payload_send = {}
            self._full_payload_replace_keys_cached = frozenset()

    stub = _StubMixin()
    stub.accumulate_full_payload_output(
        "req1",
        {"embed.prefill": torch.tensor([[1.0]])},
        request=None,
    )
    stub.accumulate_full_payload_output(
        "req1",
        {"embed.prefill": torch.tensor([[2.0]])},
        request=None,
    )
    output, _ = stub._materialize_full_payload_entry(stub._pending_full_payload_send["req1"])
    assert torch.equal(output["embed.prefill"], torch.tensor([[1.0], [2.0]]))


def test_covo_audio_llm2code2wav_token_only_smoke() -> None:
    """Smoke: covo_audio token-only builder marks `_is_sync_input`
    and returns placeholder prompts sized to audio_codes count."""
    from vllm_omni.model_executor.stage_input_processors.covo_audio import (
        llm2code2wav_token_only,
    )

    assert getattr(llm2code2wav_token_only, "_is_sync_input", False) is True

    # source_outputs is a list of objects with .outputs[0].token_ids
    from vllm_omni.model_executor.models.covo_audio.config_covo_audio import COVO_AUDIO_TOKEN_INDEX

    class _Out:
        def __init__(self, tids):
            self.token_ids = tids

    class _Wrapper:
        def __init__(self, tids):
            self.outputs = [_Out(tids)]

    # 3 codec tokens + 2 non-codec
    src = [_Wrapper([COVO_AUDIO_TOKEN_INDEX + 0, COVO_AUDIO_TOKEN_INDEX + 1, COVO_AUDIO_TOKEN_INDEX + 2, 100, 200])]
    out = llm2code2wav_token_only(src)
    assert len(out) == 1
    assert len(out[0]["prompt_token_ids"]) == 3
    assert out[0]["additional_information"] is None


def test_covo_audio_llm2code2wav_full_payload_smoke() -> None:
    """Smoke: covo_audio producer-side packer returns audio_codes + finished."""
    from types import SimpleNamespace

    from vllm_omni.model_executor.models.covo_audio.config_covo_audio import COVO_AUDIO_TOKEN_INDEX
    from vllm_omni.model_executor.stage_input_processors.covo_audio import (
        llm2code2wav_full_payload,
    )

    req = SimpleNamespace(
        output_token_ids=[COVO_AUDIO_TOKEN_INDEX + 5, COVO_AUDIO_TOKEN_INDEX + 6, 99],
    )
    payload = llm2code2wav_full_payload(None, {}, req)
    assert payload is not None
    assert payload["codes"]["audio"] == [5, 6]
    assert payload["meta"]["finished"].item() is True


def test_dynin_omni_token_only_smoke() -> None:
    """Smoke: dynin_omni token-only builders mark _is_sync_input and return placeholders."""
    from vllm_omni.model_executor.stage_input_processors.dynin_omni import (
        token2image_to_token2audio_token_only,
        token2text_to_token2image_token_only,
    )

    assert getattr(token2text_to_token2image_token_only, "_is_sync_input", False) is True
    assert getattr(token2image_to_token2audio_token_only, "_is_sync_input", False) is True

    class _Out:
        def __init__(self, tids, mm=None):
            self.token_ids = tids
            self.multimodal_output = mm

    class _Wrapper:
        def __init__(self, tids, mm=None):
            self.outputs = [_Out(tids, mm)]
            self.request_id = "r0"

    class _Stage:
        def __init__(self, outs):
            self.engine_outputs = outs

    src = [_Wrapper([10, 11, 12])]
    out = token2text_to_token2image_token_only([_Stage(src)], [0])
    assert len(out) == 1
    assert len(out[0]["prompt_token_ids"]) == 3
    assert out[0]["additional_information"] is None


def test_dynin_omni_full_payload_smoke() -> None:
    """Smoke: dynin_omni producer-side packer returns token_ids + finished."""
    from types import SimpleNamespace

    from vllm_omni.model_executor.stage_input_processors.dynin_omni import (
        token2text_to_token2image_full_payload,
    )

    pooling = {"token_ids": [1, 2, 3]}
    req = SimpleNamespace(output_token_ids=[], additional_information={"speaker": ["alice"]})
    payload = token2text_to_token2image_full_payload(None, pooling, req)
    assert payload is not None
    assert payload["code_predictor_codes"] == [1, 2, 3]
    assert payload["finished"].item() is True
    # additional_information carried forward as list-wrapped (speaker)
    assert payload.get("speaker") == ["alice"]


def test_qwen2_5_omni_talker2code2wav_token_only_smoke() -> None:
    """Smoke: qwen2_5_omni talker→code2wav token_only marker + boundary strip."""
    from vllm_omni.model_executor.stage_input_processors.qwen2_5_omni import (
        TALKER_CODEC_END_TOKEN_ID,
        TALKER_CODEC_START_TOKEN_ID,
        talker2code2wav_token_only,
    )

    assert getattr(talker2code2wav_token_only, "_is_sync_input", False) is True

    class _Out:
        def __init__(self, tids):
            self.cumulative_token_ids = tids

    class _Wrap:
        def __init__(self, tids):
            self.outputs = [_Out(tids)]

    # 3 inner codes wrapped by START + END
    src = [_Wrap([TALKER_CODEC_START_TOKEN_ID, 10, 11, 12, TALKER_CODEC_END_TOKEN_ID])]
    out = talker2code2wav_token_only(src)
    assert len(out) == 1
    assert len(out[0]["prompt_token_ids"]) == 3
    assert out[0]["additional_information"] is None


def test_qwen2_5_omni_talker2code2wav_full_payload_smoke() -> None:
    """Smoke: qwen2_5_omni producer-side packer strips boundaries."""
    from types import SimpleNamespace

    from vllm_omni.model_executor.stage_input_processors.qwen2_5_omni import (
        TALKER_CODEC_END_TOKEN_ID,
        TALKER_CODEC_START_TOKEN_ID,
        talker2code2wav_full_payload,
    )

    req = SimpleNamespace(
        output_token_ids=[TALKER_CODEC_START_TOKEN_ID, 5, 6, 7, TALKER_CODEC_END_TOKEN_ID],
    )
    payload = talker2code2wav_full_payload(None, {}, req)
    assert payload is not None
    assert payload["codes"]["audio"] == [5, 6, 7]
    assert payload["meta"]["finished"].item() is True


def test_mimo_audio_llm2code2wav_token_only_smoke() -> None:
    """Smoke: mimo_audio token-only builder marks _is_sync_input + sizes prompt."""
    import torch

    from vllm_omni.model_executor.stage_input_processors.mimo_audio import (
        llm2code2wav_token_only,
    )

    assert getattr(llm2code2wav_token_only, "_is_sync_input", False) is True

    class _Out:
        def __init__(self, mm):
            self.multimodal_output = mm

    class _Wrap:
        def __init__(self, mm):
            self.outputs = [_Out(mm)]

    # 3 batch rows of [1, 8, 4]: prepend_and_flatten_colmajor → 3*1*4*9 = 108
    codes = torch.arange(96, dtype=torch.long).reshape(3, 1, 8, 4)
    codes = codes.clamp(min=1)  # ensure nonzero so zero-row filter doesn't drop them
    src = [_Wrap({"codes": {"audio": codes}})]
    out = llm2code2wav_token_only(src)
    assert len(out) == 1
    assert len(out[0]["prompt_token_ids"]) == 108
    assert out[0]["additional_information"] is None


def test_mimo_audio_llm2code2wav_full_payload_smoke() -> None:
    """Smoke: mimo_audio producer-side packer reads flat codes.audio + flattens."""
    from types import SimpleNamespace

    import torch

    from vllm_omni.model_executor.stage_input_processors.mimo_audio import (
        TALKER_CODEC_PAD_TOKEN_ID,
        llm2code2wav_full_payload,
    )

    # Simulate accumulator output: 2 steps of [1, 1, 8, 4] CONCAT'd → [2, 1, 8, 4]
    audio = torch.arange(2 * 1 * 8 * 4, dtype=torch.long).reshape(2, 1, 8, 4)
    audio = audio.clamp(min=1)  # avoid zero-row drop
    pooling_output = {"codes.audio": audio}
    req = SimpleNamespace(output_token_ids=[])
    payload = llm2code2wav_full_payload(None, pooling_output, req)
    assert payload is not None
    assert "codes" in payload and "audio" in payload["codes"]
    # Flattened length = numel + B*4 (per-batch pad_vec prepended by prepend_and_flatten_colmajor)
    batch_size = int(audio.shape[0])
    assert len(payload["codes"]["audio"]) == audio.numel() + batch_size * 4
    # prepend_and_flatten_colmajor: PAD appears at column start in col-major flatten.
    # For shape [B=2, 1, 9, 4], each column has 1 PAD then 8 codec vals → PAD at indices 0, 9, 18, 27.
    out = payload["codes"]["audio"]
    assert out[0] == TALKER_CODEC_PAD_TOKEN_ID
    assert out[9] == TALKER_CODEC_PAD_TOKEN_ID
    assert payload["meta"]["finished"].item() is True


def test_mimo_audio_full_payload_nested_fallback() -> None:
    """Back-compat: full_payload still works if runtime returns nested codes.audio."""
    from types import SimpleNamespace

    import torch

    from vllm_omni.model_executor.stage_input_processors.mimo_audio import (
        llm2code2wav_full_payload,
    )

    audio = torch.arange(1 * 1 * 8 * 4, dtype=torch.long).reshape(1, 1, 8, 4)
    audio = audio.clamp(min=1)
    pooling_output = {"codes": {"audio": audio}}  # nested, not flat
    req = SimpleNamespace(output_token_ids=[])
    payload = llm2code2wav_full_payload(None, pooling_output, req)
    assert payload is not None
    assert len(payload["codes"]["audio"]) == audio.numel() + int(audio.shape[0]) * 4


def test_qwen3_tts_talker2code2wav_token_only_smoke() -> None:
    """Smoke: qwen3_tts token-only marks _is_sync_input + sizes placeholder."""
    import torch

    from vllm_omni.model_executor.stage_input_processors.qwen3_tts import (
        talker2code2wav_token_only,
    )

    assert getattr(talker2code2wav_token_only, "_is_sync_input", False) is True

    class _Out:
        def __init__(self, mm, tids):
            self.multimodal_output = mm
            self.cumulative_token_ids = tids

    class _Wrap:
        def __init__(self, mm, tids):
            self.outputs = [_Out(mm, tids)]
            self.finished = True

    # 3 valid codec frames Q=16; non-zero & under codebook size
    audio = torch.arange(3 * 16, dtype=torch.long).reshape(3, 16) + 1
    mm = {"codes": {"audio": audio}}
    src = [_Wrap(mm, list(range(10)))]  # seq_len = 9; 3 < 9, no trim
    out = talker2code2wav_token_only(src)
    assert len(out) == 1
    # Codebook-major flat: 16 * 3 = 48
    assert len(out[0]["prompt_token_ids"]) == 48


def test_qwen3_tts_talker2code2wav_full_payload_smoke() -> None:
    """Smoke: qwen3_tts full_payload reads flat codes.audio + flattens col-major."""
    from types import SimpleNamespace

    import torch

    from vllm_omni.model_executor.stage_input_processors.qwen3_tts import (
        talker2code2wav_full_payload,
    )

    # 3 valid codec frames [3, 16] CONCAT'd from per-step emits via flatten
    audio = torch.arange(3 * 16, dtype=torch.long).reshape(3, 16) + 1
    pooling_output = {"codes.audio": audio}
    req = SimpleNamespace(output_token_ids=list(range(10)))  # seq_len = 9
    payload = talker2code2wav_full_payload(None, pooling_output, req)
    assert payload is not None
    assert "codes" in payload and "audio" in payload["codes"]
    # codebook-major: shape [3, 16] -> [16, 3] -> flatten = 48 entries
    assert len(payload["codes"]["audio"]) == 48
    assert payload["meta"]["finished"].item() is True


def test_qwen3_tts_full_payload_with_ref_code() -> None:
    """Smoke: ref_code prepended via codes.ref + meta.ref_code_len from flat path."""
    from types import SimpleNamespace

    import torch

    from vllm_omni.model_executor.stage_input_processors.qwen3_tts import (
        talker2code2wav_full_payload,
    )

    # Audio: 3 frames [3, 16]
    audio = torch.arange(3 * 16, dtype=torch.long).reshape(3, 16) + 1
    # Ref code: 2 frames [2, 16] (already 2-D)
    ref = torch.arange(2 * 16, dtype=torch.long).reshape(2, 16) + 100
    pooling_output = {
        "codes.audio": audio,
        "codes.ref": [ref],
        "meta.ref_code_len": torch.tensor([2], dtype=torch.int32),
    }
    req = SimpleNamespace(output_token_ids=list(range(10)))
    payload = talker2code2wav_full_payload(None, pooling_output, req)
    assert payload is not None
    # Total frames = 2 (ref) + 3 (audio) = 5; codebook-major: 16 * 5 = 80
    assert len(payload["codes"]["audio"]) == 80


def test_qwen3_tts_full_payload_nested_fallback() -> None:
    """Back-compat: full_payload works if pooler returns un-flattened nested dict."""
    from types import SimpleNamespace

    import torch

    from vllm_omni.model_executor.stage_input_processors.qwen3_tts import (
        talker2code2wav_full_payload,
    )

    audio = torch.arange(2 * 16, dtype=torch.long).reshape(2, 16) + 1
    pooling_output = {"codes": {"audio": audio}}  # nested, not flat
    req = SimpleNamespace(output_token_ids=list(range(10)))
    payload = talker2code2wav_full_payload(None, pooling_output, req)
    assert payload is not None
    assert len(payload["codes"]["audio"]) == 32  # 16 * 2


def test_cosyvoice3_text2flow_token_only_smoke() -> None:
    """Smoke: cosyvoice3 token-only marks _is_sync_input + carries ids.prompt only."""
    from vllm_omni.model_executor.stage_input_processors.cosyvoice3 import (
        text2flow_token_only,
    )

    assert getattr(text2flow_token_only, "_is_sync_input", False) is True

    class _Out:
        def __init__(self, tids):
            self.cumulative_token_ids = tids
            self.multimodal_output = {}

    class _Wrap:
        def __init__(self, output_tids, prompt_tids):
            self.outputs = [_Out(output_tids)]
            self.prompt_token_ids = prompt_tids
            self.finished = True

    # multimodal_output has embed.* + we expect token_only to preserve it (Phase 4 #90 follow-up).
    import torch

    embed = {"speech_token": torch.zeros(2, 4)}
    src = [_Wrap(output_tids=[10, 20, 30], prompt_tids=[1, 2, 3, 4])]
    src[0].outputs[0].multimodal_output = {"embed": embed}
    out = text2flow_token_only(src)
    assert len(out) == 1
    # prompt_token_ids is the talker's cumulative_token_ids (real codec tokens, not zeros).
    assert out[0]["prompt_token_ids"] == [10, 20, 30]
    # additional_information carries ids.prompt PLUS the original multimodal_output (embed.* still inline).
    # Heavy embed.* removal pending the model_intermediate_buffer plumbing on the code2wav side.
    assert out[0]["additional_information"]["ids"]["prompt"] == [1, 2, 3, 4]
    assert "embed" in out[0]["additional_information"]


def test_cosyvoice3_text2flow_full_payload_smoke() -> None:
    """Smoke: cosyvoice3 producer-side reads flat embed.* keys."""
    from types import SimpleNamespace

    import torch

    from vllm_omni.model_executor.stage_input_processors.cosyvoice3 import (
        text2flow_full_payload,
    )

    speech_token = torch.randn(4, 8)
    speech_feat = torch.randn(4, 16)
    embedding = torch.randn(1, 32)
    pooling_output = {
        "embed.speech_token": speech_token,
        "embed.speech_feat": speech_feat,
        "embed.embedding": embedding,
    }
    req = SimpleNamespace(external_req_id="r-1")
    payload = text2flow_full_payload(None, pooling_output, req)
    assert payload is not None
    assert "embed" in payload
    assert torch.equal(payload["embed"]["speech_token"], speech_token)
    assert torch.equal(payload["embed"]["speech_feat"], speech_feat)
    assert torch.equal(payload["embed"]["embedding"], embedding)
    assert payload["meta"]["finished"].item() is True


def test_cosyvoice3_text2flow_full_payload_nested_fallback() -> None:
    """Back-compat: full_payload works if pooler returns un-flattened nested embed."""
    from types import SimpleNamespace

    import torch

    from vllm_omni.model_executor.stage_input_processors.cosyvoice3 import (
        text2flow_full_payload,
    )

    speech_token = torch.randn(3, 8)
    pooling_output = {"embed": {"speech_token": speech_token}}  # nested, not flat
    req = SimpleNamespace(external_req_id="r-2")
    payload = text2flow_full_payload(None, pooling_output, req)
    assert payload is not None
    assert "speech_token" in payload["embed"]
    assert torch.equal(payload["embed"]["speech_token"], speech_token)


def test_cosyvoice3_full_payload_replace_keys_present() -> None:
    """Confirm _FULL_PAYLOAD_REPLACE_KEYS lists the three embed.* keys."""
    from vllm_omni.model_executor.stage_input_processors.cosyvoice3 import (
        _FULL_PAYLOAD_REPLACE_KEYS,
    )

    assert _FULL_PAYLOAD_REPLACE_KEYS == frozenset({"embed.speech_token", "embed.speech_feat", "embed.embedding"})


def test_ming_flash_omni_thinker2talker_token_only_smoke() -> None:
    """Smoke: ming_flash_omni token-only marks _is_sync_input + carries voice metadata."""
    from vllm_omni.model_executor.stage_input_processors.ming_flash_omni import (
        thinker2talker_token_only,
    )

    assert getattr(thinker2talker_token_only, "_is_sync_input", False) is True

    class _Out:
        def __init__(self, text):
            self.text = text

    class _Wrap:
        def __init__(self, text):
            self.outputs = [_Out(text)]

    class _Prompt:
        def __init__(self, info):
            self.additional_information = info

    src = [_Wrap("hello world")]
    prompt = _Prompt({"voice_name": "ZH_FEMALE", "prompt_text": "ref text"})
    out = thinker2talker_token_only(src, prompt=prompt)
    assert len(out) == 1
    assert out[0]["prompt_token_ids"] == [0]  # talker self-tokenizes; dummy id
    info = out[0]["additional_information"]
    assert info["text"] == "hello world"
    assert info["voice_name"] == "ZH_FEMALE"
    assert info["prompt_text"] == "ref text"
    assert info["ming_task"] == "omni"


def test_ming_flash_omni_thinker2talker_full_payload_noop() -> None:
    """thinker2talker_full_payload returns None — no heavy tensor migration."""
    from vllm_omni.model_executor.stage_input_processors.ming_flash_omni import (
        thinker2talker_full_payload,
    )

    payload = thinker2talker_full_payload(None, {"anything": "ignored"}, None)
    assert payload is None


def test_qwen2_5_omni_thinker2talker_token_only_smoke() -> None:
    """Smoke: qwen2_5_omni thinker token-only marks _is_sync_input + ports legacy body."""
    import torch

    from vllm_omni.model_executor.stage_input_processors.qwen2_5_omni import (
        TALKER_CODEC_END_TOKEN_ID,
        TALKER_CODEC_PAD_TOKEN_ID,
        TALKER_CODEC_START_TOKEN_ID,
        thinker2talker_token_only,
    )

    assert getattr(thinker2talker_token_only, "_is_sync_input", False) is True

    class _Out:
        def __init__(self, ctids, mm):
            self.cumulative_token_ids = ctids
            self.multimodal_output = mm

    class _Wrap:
        def __init__(self, prompt_tids, ctids, mm, rid):
            self.outputs = [_Out(ctids, mm)]
            self.prompt_token_ids = prompt_tids
            self.request_id = rid

    class _Prompt(dict):
        pass

    # Latent shaped [prompt_len + decode_len, hidden] = [5 + 3, 8]
    latent = torch.randn(8, 8)
    src = [_Wrap(prompt_tids=[1, 2, 3, 4, 5], ctids=[10, 20, 30], mm={"latent": latent}, rid="r-1")]
    prompt = [_Prompt(multi_modal_data=None)]
    out = thinker2talker_token_only(src, prompt=prompt)
    assert len(out) == 1
    # Talker prompt = START + PAD*prompt_len + END
    expected_prompt_len = 1 + len([1, 2, 3, 4, 5]) + 1
    assert len(out[0]["prompt_token_ids"]) == expected_prompt_len
    assert out[0]["prompt_token_ids"][0] == TALKER_CODEC_START_TOKEN_ID
    assert out[0]["prompt_token_ids"][-1] == TALKER_CODEC_END_TOKEN_ID
    assert all(t == TALKER_CODEC_PAD_TOKEN_ID for t in out[0]["prompt_token_ids"][1:-1])


def test_qwen2_5_omni_thinker2talker_full_payload_noop() -> None:
    """thinker2talker_full_payload returns None — no heavy tensor migration today."""
    from vllm_omni.model_executor.stage_input_processors.qwen2_5_omni import (
        thinker2talker_full_payload,
    )

    payload = thinker2talker_full_payload(None, {"any": "thing"}, None)
    assert payload is None
