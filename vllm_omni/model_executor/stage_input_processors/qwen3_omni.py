# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The Qwen team.
"""Stage input processor for Qwen3 Omni MoE: Thinker → Talker transition."""

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.inputs import TextPrompt
from vllm.platforms import current_platform

from vllm_omni.data_entry_keys import OmniPayload
from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.stage_input_processors.tts_utils import (
    extract_language_from_prompt,
    extract_language_from_request,
    extract_speaker_from_prompt,
    extract_speaker_from_request,
)

logger = logging.getLogger(__name__)

# Pooling output layer keys: "0" = word embedding, "24" = accept_hidden_layer
_EMBED_LAYER_KEY = "0"
_HIDDEN_LAYER_KEY = "24"
_QWEN3_CODEC_CODEBOOK_SIZE = 2048
_QWEN3_CODEC_PAD_TOKEN_ID = 4196
_QWEN3_CODEC_BOS_TOKEN_ID = 4197
_QWEN3_CODEC_EOS_TOKEN_ID = 4198


def _layer_tensor(layers: dict[Any, Any], key: str) -> torch.Tensor | None:
    """Fetch layer tensor with tolerant key lookup (str/int)."""
    if not isinstance(layers, dict):
        return None
    key_int = int(key)
    val = layers.get(key_int)
    if val is None:
        val = layers.get(key)
    return val if isinstance(val, torch.Tensor) else None


def _compute_talker_prompt_ids_length(info: OmniPayload, device: torch.device | str = "cuda") -> int:
    im_start_token_id = 151644
    system_token_id = 8948
    user_token_id = 872
    assistant_token_id = 77091

    ids = info.get("ids", {})
    thinker_sequences = torch.tensor(ids["all"], dtype=torch.long, device=device).unsqueeze(0)  # [1, T]

    input_ids = torch.tensor(ids["prompt"], dtype=torch.long, device=device).unsqueeze(0)  # [1, T]

    im_start_indexes = torch.cat(
        [
            torch.nonzero(input_ids[0] == im_start_token_id).squeeze(1),
            torch.tensor([thinker_sequences.shape[-1]], device=input_ids.device, dtype=input_ids.dtype),
        ],
        dim=0,
    )

    sum_user_len = 0
    assistant_len = 0
    for i in range(len(im_start_indexes) - 1):
        s = int(im_start_indexes[i].item())
        e = int(im_start_indexes[i + 1].item())
        role = int(input_ids[0, s + 1].item())
        if role == system_token_id:
            continue
        elif role == user_token_id:
            sum_user_len += e - s
        elif role == assistant_token_id and i == len(im_start_indexes) - 2:
            assistant_len += 9  # 3 + 4 + 1 + 1
        else:
            pass

    return sum_user_len + assistant_len


# =========================
# Common helpers
# =========================


def _ensure_list(x):
    """Convert ConstantList / tensor-like to Python list."""
    if hasattr(x, "_x"):
        return list(x._x)
    elif not isinstance(x, list):
        return x
    return list(x)


def _as_tensor_or_none(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
        return value[0].detach().cpu()
    return None


def _is_valid_qwen3_codec_token_id(token_id: Any) -> bool:
    try:
        token_id = int(token_id)
    except (TypeError, ValueError):
        return False
    return 0 <= token_id < _QWEN3_CODEC_CODEBOOK_SIZE


def should_accumulate_qwen3_omni_full_payload_output(
    model_config: Any,
    custom_process_func: Any,
) -> bool:
    """Return whether Qwen3-Omni should accumulate full-payload outputs."""
    return (
        custom_process_func is not None
        and not getattr(model_config, "async_chunk", False)
        and getattr(model_config, "model_arch", None) == "Qwen3OmniMoeForConditionalGeneration"
        and getattr(model_config, "model_stage", None) in {"thinker", "talker"}
    )


def _qwen3_full_payload_valid_codec_output_count(request: Any) -> int:
    output_token_ids = getattr(request, "output_token_ids", None)
    if output_token_ids is None:
        output_token_ids = getattr(request, "_output_token_ids", None)
    if output_token_ids is None:
        return 0
    return sum(1 for token_id in output_token_ids if _is_valid_qwen3_codec_token_id(token_id))


def _qwen3_full_payload_codec_tensor_rows(value: Any) -> int | None:
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        return None
    return int(value.shape[0])


def should_keep_qwen3_omni_all_zero_full_payload_tensor(
    key: str,
    value: Any,
    request: Any,
    existing_output: dict[str, Any] | None,
    model_config: Any,
    custom_process_func: Any,
) -> bool:
    """Return whether a Qwen3-Omni all-zero codec tensor is real output."""
    if (
        not should_accumulate_qwen3_omni_full_payload_output(model_config, custom_process_func)
        or getattr(model_config, "model_stage", None) != "talker"
        or key not in {"codes.audio", "code_predictor_codes"}
    ):
        return False
    rows = _qwen3_full_payload_codec_tensor_rows(value)
    if rows is None or rows <= 0:
        return False
    previous_rows = 0
    if isinstance(existing_output, dict):
        previous_rows = _qwen3_full_payload_codec_tensor_rows(existing_output.get(key)) or 0
    return previous_rows + rows == _qwen3_full_payload_valid_codec_output_count(request)


def _extract_qwen3_full_payload_codec_rows(
    code_predictor_codes: torch.Tensor,
    output_token_ids: list[int],
) -> tuple[torch.Tensor, dict[str, int]]:
    """Filter full-payload codec rows by the authoritative output ids."""
    if code_predictor_codes.ndim != 2 or code_predictor_codes.numel() == 0:
        return code_predictor_codes, {
            "raw_rows": int(code_predictor_codes.shape[0]) if code_predictor_codes.ndim > 0 else 0,
            "aligned_rows": 0,
            "valid_rows": 0,
            "trailing_placeholder_count": 0,
        }

    trailing_placeholder_count = 0
    while (
        trailing_placeholder_count < len(output_token_ids) and output_token_ids[-1 - trailing_placeholder_count] == -1
    ):
        trailing_placeholder_count += 1

    aligned_len = min(int(code_predictor_codes.shape[0]), len(output_token_ids))
    if aligned_len <= 0:
        return code_predictor_codes[:0], {
            "raw_rows": int(code_predictor_codes.shape[0]),
            "aligned_rows": 0,
            "valid_rows": 0,
            "trailing_placeholder_count": trailing_placeholder_count,
        }

    aligned_rows = code_predictor_codes[-aligned_len:]
    aligned_token_ids = output_token_ids[-aligned_len:]
    aligned_token_mask = torch.tensor(
        [_is_valid_qwen3_codec_token_id(token_id) for token_id in aligned_token_ids],
        dtype=torch.bool,
        device=aligned_rows.device,
    )
    row_valid_mask = (aligned_rows.max(dim=1).values < _QWEN3_CODEC_CODEBOOK_SIZE) & (
        aligned_rows.min(dim=1).values >= 0
    )
    filtered_rows = aligned_rows[aligned_token_mask & row_valid_mask]
    if filtered_rows.numel() == 0:
        filtered_rows = aligned_rows[:0]
    return filtered_rows, {
        "raw_rows": int(code_predictor_codes.shape[0]),
        "aligned_rows": aligned_len,
        "valid_rows": int(filtered_rows.shape[0]) if filtered_rows.ndim > 0 else 0,
        "trailing_placeholder_count": trailing_placeholder_count,
    }


# =========================
# PD disaggregation helpers
# =========================


def _get_prefill_multimodal_output(
    request_id: str,
    streaming_context: Any | None,
) -> dict[str, Any] | None:
    bridge_states = getattr(streaming_context, "bridge_states", None)
    if not isinstance(bridge_states, dict):
        return None
    by_req = bridge_states.get("pd_prefill_multimodal_output_by_req")
    if not isinstance(by_req, dict):
        return None
    prefill_mm = by_req.get(request_id)
    return prefill_mm if isinstance(prefill_mm, dict) else None


def _merge_pd_embeddings(
    decode_emb: torch.Tensor,
    decode_hid: torch.Tensor,
    prefill_mm: dict[str, Any],
    device: torch.device,
    expected_total: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge prefill prompt embeddings with decode generated embeddings.

    In PD mode the prefill engine processes the prompt and the decode engine
    generates tokens starting from position 1.  This function concatenates
    them, removing the overlapping token(s):

        merged = prefill[:P] + decode[overlap:]

    where overlap = P + D - expected_total.
    """
    try:
        p_layers = prefill_mm.get("hidden_states", {}).get("layers", {})
        p_emb = p_layers[int(_EMBED_LAYER_KEY)].detach().to(device=device, dtype=torch.float)
        p_hid = p_layers[int(_HIDDEN_LAYER_KEY)].detach().to(device=device, dtype=torch.float)
    except (KeyError, AttributeError, TypeError) as exc:
        available_keys = list(prefill_mm.keys()) if isinstance(prefill_mm, dict) else type(prefill_mm).__name__
        logger.error(
            "_merge_pd_embeddings: failed to extract prefill embeddings (%s). "
            "Expected keys %r and %r, got: %s. "
            "Falling back to decode-only embeddings – talker user-segment will be degraded.",
            exc,
            _EMBED_LAYER_KEY,
            _HIDDEN_LAYER_KEY,
            available_keys,
        )
        return decode_emb, decode_hid

    if p_emb.shape[0] == 0 or decode_emb.shape[0] == 0:
        return decode_emb, decode_hid

    raw_total = p_emb.shape[0] + decode_emb.shape[0]
    overlap = max(0, raw_total - expected_total) if expected_total is not None else 0

    merged_emb = torch.cat([p_emb, decode_emb[overlap:]], dim=0)
    merged_hid = torch.cat([p_hid, decode_hid[overlap:]], dim=0)
    return merged_emb, merged_hid


def _resolve_tts_token_embedding(
    key: str,
    *,
    thinker_mm: dict[str, Any],
    prefill_mm: dict[str, Any] | None,
    device: torch.device,
) -> torch.Tensor | None:
    """Return TTS BOS/EOS/PAD embedding tensors for the talker projection path.

    Values are taken from the current thinker (decode) ``multimodal_output``; in
    PD mode, missing keys may be filled from the paired prefill stage output.
    """
    val = thinker_mm.get("embed", {}).get(key)
    if val is None and prefill_mm is not None:
        val = prefill_mm.get("embed", {}).get(key)
    return val.detach().to(device=device, dtype=torch.float) if val is not None else None


# =========================
# Streaming input helpers
# =========================


@dataclass
class _Thinker2TalkerStreamingState:
    last_prompt_len: int = 0
    last_output_len: int = 0
    merged_sequences: list[int] = field(default_factory=list)


@dataclass
class _Qwen3OmniStreamingState:
    thinker2talker: _Thinker2TalkerStreamingState = field(default_factory=_Thinker2TalkerStreamingState)
    talker2code2wav_last_seq_len: int = 0


def _get_qwen3_streaming_state(
    request_id: str,
    streaming_context: Any | None,
) -> _Qwen3OmniStreamingState:
    bridge_states = getattr(streaming_context, "bridge_states", None)
    per_model_state = bridge_states.setdefault("qwen3_omni", {})
    state = per_model_state.get(request_id)
    if state is None:
        state = _Qwen3OmniStreamingState()
        per_model_state[request_id] = state
    return state


def _get_streaming_talker_tokens(
    request_id: str,
    prompt_token_ids: list[int],
    output_token_ids: list[int],
    new_prompt_len_snapshot: int | None = None,
    streaming_context: Any | None = None,
    *,
    clear_state: bool = False,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Return streaming token slices and merged token views for thinker->talker.
       e.g. For the second streaming input request:
       merged_sequences: [input_prompt 1, output_tokens 1[:-1], input_prompt 2, output_tokens 2]
      thinker_input_ids: [input_prompt 1, output_tokens 1[:-1], input_prompt 2]
    Returns:
        inc_prompt: prompt token delta for this segment.
        inc_output: output token delta for this segment.
        merged_sequences: full thinker_sequences to send downstream.
        thinker_input_ids: full thinker_input_ids paired with merged_sequences.
    """
    state = _get_qwen3_streaming_state(request_id, streaming_context).thinker2talker
    if new_prompt_len_snapshot:
        prompt_token_ids = prompt_token_ids[:-new_prompt_len_snapshot]
    cur_prompt_len = len(prompt_token_ids)
    cur_output_len = len(output_token_ids)

    inc_prompt = prompt_token_ids[state.last_prompt_len :]
    inc_output = output_token_ids[state.last_output_len :]
    delta_sequences = inc_prompt + inc_output
    cached_sequences = state.merged_sequences

    merged_sequences = cached_sequences + delta_sequences
    thinker_input_ids = cached_sequences + inc_prompt

    # Persist history for next segment. Drop the latest sampled token to keep
    # thinker_input_ids / thinker_sequences alignment with next-step append.
    cached_sequences.extend(delta_sequences[:-1])

    state.last_prompt_len = cur_prompt_len
    state.last_output_len = cur_output_len

    if clear_state:
        state.last_prompt_len = 0
        state.last_output_len = 0
        state.merged_sequences.clear()

    return inc_prompt, inc_output, merged_sequences, thinker_input_ids


def _get_streaming_codec_delta_len(
    cur_seq_len: int,
    request_id: str,
    talker_output: Any,
    streaming_context: Any | None = None,
) -> int:
    """Return newly added seq_len for talker->code2wav in streaming mode."""
    state = _get_qwen3_streaming_state(request_id, streaming_context)
    prev_seq_len = state.talker2code2wav_last_seq_len
    seq_len = cur_seq_len - prev_seq_len
    state.talker2code2wav_last_seq_len = cur_seq_len + 1
    if bool(getattr(talker_output, "finished", False)):
        # Final segment: clear history to avoid cross-session carry-over.
        state.talker2code2wav_last_seq_len = 0
    return seq_len


# =========================
# Thinker -> Talker
# =========================


def thinker2talker_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: OmniEngineCoreRequest,
    is_finished: bool = False,
) -> list[dict[str, Any]]:
    """
    Process thinker outputs to create talker inputs.
    1. thinker's text generation outputs (token IDs + hidden states)
    2. Split hidden states into: prompt embeddings + generated embeddings
    3. Package for talker with additional information
    """

    request_id = request.external_req_id
    chunk_id = transfer_manager.put_req_chunk[request_id]
    if not isinstance(pooling_output, dict):
        logger.debug("thinker2talker_async_chunk: skip non-dict pooling_output for req=%s", request_id)
        return None

    thinker_hs = pooling_output.get("hidden_states", {})
    thinker_layers = thinker_hs.get("layers", {}) if isinstance(thinker_hs, dict) else {}
    thinker_embed = pooling_output.get("embed", {}) if isinstance(pooling_output.get("embed", {}), dict) else {}
    thinker_emb = _layer_tensor(thinker_layers, _EMBED_LAYER_KEY)
    thinker_hid = _layer_tensor(thinker_layers, _HIDDEN_LAYER_KEY)
    if thinker_emb is None or thinker_hid is None:
        logger.debug(
            "thinker2talker_async_chunk: missing thinker layers for req=%s (embed=%s hidden=%s)",
            request_id,
            thinker_emb is not None,
            thinker_hid is not None,
        )
        return None

    if chunk_id == 0:
        all_token_ids = request.all_token_ids  # prefill + decode
        prompt_token_ids = request.prompt_token_ids
        # Convert ConstantList to regular list for OmniSerializer serialization
        all_token_ids = _ensure_list(all_token_ids)
        prompt_token_ids = _ensure_list(prompt_token_ids)
        payload: OmniPayload = {
            "embed": {
                "prefill": thinker_emb.detach().cpu(),
                # Provide thinker-side TTS token embeddings for talker projection
                "tts_bos": thinker_embed.get("tts_bos").detach().cpu()
                if isinstance(thinker_embed.get("tts_bos"), torch.Tensor)
                else None,
                "tts_eos": thinker_embed.get("tts_eos").detach().cpu()
                if isinstance(thinker_embed.get("tts_eos"), torch.Tensor)
                else None,
                "tts_pad": thinker_embed.get("tts_pad").detach().cpu()
                if isinstance(thinker_embed.get("tts_pad"), torch.Tensor)
                else None,
            },
            "hidden_states": {"output": thinker_hid.detach().cpu()},
            "ids": {"all": all_token_ids, "prompt": prompt_token_ids},
            "meta": {"finished": torch.tensor(is_finished, dtype=torch.bool)},
        }
        talker_additional_info = payload
        speaker = extract_speaker_from_request(request)
        if speaker is not None:
            talker_additional_info["speaker"] = speaker
        language = extract_language_from_request(request)
        if language is not None:
            talker_additional_info["language"] = language
        if transfer_manager.request_payload.get(request_id) is None:
            if not is_finished:
                transfer_manager.request_payload[request_id] = talker_additional_info
                return None
        else:
            save_payload = transfer_manager.request_payload.pop(request_id)
            talker_additional_info["embed"]["prefill"] = torch.cat(
                (
                    save_payload.get("embed", {}).get("prefill"),
                    talker_additional_info.get("embed", {}).get("prefill"),
                ),
                dim=0,
            )
            talker_additional_info["hidden_states"]["output"] = torch.cat(
                (
                    save_payload.get("hidden_states", {}).get("output"),
                    talker_additional_info.get("hidden_states", {}).get("output"),
                ),
                dim=0,
            )
    else:
        output_token_ids = request.output_token_ids
        # Convert ConstantList to regular list for OmniSerializer serialization
        output_token_ids = _ensure_list(output_token_ids)

        talker_additional_info: OmniPayload = {
            "meta": {"finished": torch.tensor(is_finished, dtype=torch.bool)},
        }
        speaker = extract_speaker_from_request(request)
        if speaker is not None:
            talker_additional_info["speaker"] = speaker
        language = extract_language_from_request(request)
        if language is not None:
            talker_additional_info["language"] = language

        if output_token_ids:
            talker_additional_info["meta"]["override_keys"] = [("embed", "decode"), ("ids", "output")]
            talker_additional_info["embed"] = {"decode": thinker_emb.detach().cpu()}
            talker_additional_info["ids"] = {"output": output_token_ids}
        else:
            # When prefilling a chunked thinker, thinker_hidden_states needs to be updated.
            talker_additional_info["embed"] = {"prefill": thinker_emb.detach().cpu()}
            talker_additional_info["hidden_states"] = {"output": thinker_hid.detach().cpu()}
    return talker_additional_info


def thinker2talker_full_payload(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: OmniEngineCoreRequest,
) -> dict[str, Any] | None:
    """Pack complete thinker output for the non-async connector path."""
    if not isinstance(pooling_output, dict):
        return None

    layers = {
        0: pooling_output.get("hidden_states.layer_0"),
        24: pooling_output.get("hidden_states.layer_24"),
    }
    thinker_emb = _layer_tensor(layers, _EMBED_LAYER_KEY)
    thinker_hid = _layer_tensor(layers, _HIDDEN_LAYER_KEY)
    if thinker_emb is None:
        hidden = pooling_output.get("hidden")
        thinker_emb = hidden if isinstance(hidden, torch.Tensor) else None
    if thinker_emb is None or thinker_hid is None:
        logger.debug(
            "thinker2talker_full_payload: missing thinker tensors for req=%s (embed=%s hidden=%s)",
            getattr(request, "request_id", None),
            thinker_emb is not None,
            thinker_hid is not None,
        )
        return None

    prompt_token_ids = _ensure_list(getattr(request, "prompt_token_ids", []) or [])
    all_token_ids = _ensure_list(getattr(request, "all_token_ids", None) or [])
    if not all_token_ids:
        output_token_ids = _ensure_list(getattr(request, "output_token_ids", []) or [])
        all_token_ids = list(prompt_token_ids) + list(output_token_ids)

    # Match legacy thinker2talker convention: slice last (total-1) rows.
    # Talker's _thinker_to_talker_prefill computes the last assistant segment
    # boundary as target_len = len(ids.all); embed.prefill / hidden_states.output
    # must be 1 row shorter so slicing [im_start:target_len] clips to the
    # assistant segment exactly as base 4a24a517 does. Otherwise the assistant
    # segment grabs an extra (output-token) row, trailing_text becomes [2,1024]
    # instead of [1,1024], and talker over-generates codec frames.
    new_seq_length = max(0, len(all_token_ids) - 1)
    if isinstance(thinker_emb, torch.Tensor) and thinker_emb.shape[0] >= new_seq_length and new_seq_length > 0:
        thinker_emb_prefill = thinker_emb[-new_seq_length:]
    else:
        thinker_emb_prefill = thinker_emb
    if isinstance(thinker_hid, torch.Tensor) and thinker_hid.shape[0] >= new_seq_length and new_seq_length > 0:
        thinker_hid_prefill = thinker_hid[-new_seq_length:]
    else:
        thinker_hid_prefill = thinker_hid

    payload: OmniPayload = {
        "embed": {
            "prefill": thinker_emb_prefill.detach().cpu(),
            "tts_bos": _as_tensor_or_none(pooling_output.get("embed.tts_bos")),
            "tts_eos": _as_tensor_or_none(pooling_output.get("embed.tts_eos")),
            "tts_pad": _as_tensor_or_none(pooling_output.get("embed.tts_pad")),
        },
        "hidden_states": {"output": thinker_hid_prefill.detach().cpu()},
        "ids": {"all": list(all_token_ids), "prompt": list(prompt_token_ids)},
        "meta": {"finished": torch.tensor(True, dtype=torch.bool)},
    }
    payload["next_stage_prompt_len"] = _compute_talker_prompt_ids_length(payload, device="cpu")
    speaker = extract_speaker_from_request(request)
    if speaker is not None:
        payload["speaker"] = speaker
    language = extract_language_from_request(request)
    if language is not None:
        payload["language"] = language
    return payload


def thinker2talker(
    source_outputs: list[Any],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> list[OmniTokensPrompt]:
    """
    Process thinker outputs to create talker inputs.

    Workflow:
    1. Extract thinker's text generation outputs (token IDs + hidden states)
    2. Split hidden states into: prompt embeddings + generated embeddings
    3. Package for talker with additional information

    In PD disaggregation mode, merges prefill-stage prompt embeddings with
    decode-stage generated embeddings before handing off to the talker.

    Args:
        prompt: Original prompt data
        requires_multimodal_data: Whether multimodal data is required

    Returns:
        List of OmniTokensPrompt for talker stage
    """
    thinker_outputs = source_outputs
    talker_inputs: list[OmniTokensPrompt] = []

    device = torch.device(current_platform.device_type)

    # Process each thinker output
    for i, thinker_output in enumerate(thinker_outputs):
        output = thinker_output.outputs[0]
        req_id = str(getattr(thinker_output, "request_id", f"idx-{i}"))
        prompt_token_ids = _ensure_list(thinker_output.prompt_token_ids)
        output_ids = _ensure_list(output.cumulative_token_ids)
        is_streaming_session = bool(getattr(streaming_context, "enabled", False))
        if is_streaming_session:
            prompt_token_ids, output_ids, thinker_sequences, thinker_input_ids = _get_streaming_talker_tokens(
                req_id,
                prompt_token_ids,
                output_ids,
                getattr(streaming_context, "new_prompt_len_snapshot", None),
                streaming_context,
                clear_state=bool(getattr(thinker_output, "finished", False)),
            )
        else:
            thinker_sequences = prompt_token_ids + output_ids
            thinker_input_ids = prompt_token_ids
        new_seq_length = len(prompt_token_ids + output_ids) - 1
        thinker_mm_raw = getattr(output, "multimodal_output", None)
        if not isinstance(thinker_mm_raw, dict):
            logger.debug("thinker2talker: skip req=%s due to empty multimodal_output", req_id)
            continue
        thinker_mm: OmniPayload = thinker_mm_raw
        mm_hs = thinker_mm.get("hidden_states", {})
        mm_layers = mm_hs.get("layers", {}) if isinstance(mm_hs, dict) else {}
        emb_layer = _layer_tensor(mm_layers, _EMBED_LAYER_KEY)
        hid_layer = _layer_tensor(mm_layers, _HIDDEN_LAYER_KEY)
        if emb_layer is None or hid_layer is None:
            logger.debug("thinker2talker: skip req=%s due to missing hidden-state layers", req_id)
            continue
        thinker_emb = emb_layer.detach().to(device=device, dtype=torch.float)[-new_seq_length:]
        thinker_hid = hid_layer.detach().to(device=device, dtype=torch.float)[-new_seq_length:]

        prefill_mm: dict[str, Any] | None = None
        prefill_mm = _get_prefill_multimodal_output(req_id, streaming_context)

        if prefill_mm is not None:
            expected_total = len(prompt_token_ids) + len(output_ids)
            try:
                thinker_emb, thinker_hid = _merge_pd_embeddings(
                    thinker_emb, thinker_hid, prefill_mm, device, expected_total=expected_total
                )
            except Exception as exc:
                logger.warning("[PD] Could not merge prefill embeddings: %s", exc)

        payload: OmniPayload = {
            "embed": {
                "prefill": thinker_emb,
                "tts_bos": _resolve_tts_token_embedding(
                    "tts_bos", thinker_mm=thinker_mm, prefill_mm=prefill_mm, device=device
                ),
                "tts_eos": _resolve_tts_token_embedding(
                    "tts_eos", thinker_mm=thinker_mm, prefill_mm=prefill_mm, device=device
                ),
                "tts_pad": _resolve_tts_token_embedding(
                    "tts_pad", thinker_mm=thinker_mm, prefill_mm=prefill_mm, device=device
                ),
            },
            "hidden_states": {
                "output": thinker_hid,
            },
            "ids": {
                "all": thinker_sequences,
                "prompt": thinker_input_ids,
            },
        }
        info = payload
        speaker = extract_speaker_from_prompt(prompt, index=i)
        if speaker is not None:
            info["speaker"] = speaker
        language = extract_language_from_prompt(prompt, index=i)
        if language is not None:
            info["language"] = language

        prompt_len = _compute_talker_prompt_ids_length(payload, device=device)

        talker_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0] * prompt_len,
                additional_information=info,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return talker_inputs


# =========================
# Talker -> Code2Wav
# =========================


def talker2code2wav_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: OmniEngineCoreRequest,
    is_finished: bool = False,
):
    """
    Pooling version.
    """
    if not isinstance(pooling_output, dict):
        return None
    talker_codes = pooling_output.get("codes", {})
    if not isinstance(talker_codes, dict):
        return None
    code_predictor_codes = talker_codes.get("audio")
    if code_predictor_codes is None:
        return None

    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size_config = int(cfg.get("codec_chunk_frames", 25))
    left_context_size_config = int(cfg.get("codec_left_context_frames", 25))

    if code_predictor_codes is None:
        return None
    if isinstance(code_predictor_codes, torch.Tensor):
        if code_predictor_codes.numel() == 0:
            return None
    elif hasattr(code_predictor_codes, "__len__"):
        if len(code_predictor_codes) == 0:
            return None

    if isinstance(code_predictor_codes, torch.Tensor):
        if not code_predictor_codes.any():
            return None
    else:
        code_tensor = torch.tensor(code_predictor_codes, dtype=torch.long)
        if not code_tensor.any():
            return None

    codec_codes = code_predictor_codes.to(torch.long).transpose(0, 1).cpu().to(torch.long).reshape(-1).tolist()
    if sum(codec_codes) == 0:
        return None

    request_id = request.external_req_id
    transfer_manager.code_prompt_token_ids[request_id].append(codec_codes)
    length = len(transfer_manager.code_prompt_token_ids[request_id])

    chunk_length = length % chunk_size_config
    if chunk_length != 0 and not is_finished:
        return None

    context_length = chunk_length if chunk_length != 0 else chunk_size_config
    # ensure left context does not exceed available length
    left_context_size = max(0, min(length - context_length, left_context_size_config))
    end_index = min(length, left_context_size + context_length)

    codes = (
        torch.tensor(transfer_manager.code_prompt_token_ids[request_id][-end_index:])
        .transpose(0, 1)
        .reshape(-1)
        .tolist()
    )

    return {
        "codes": {"audio": codes},
        "meta": {"left_context_size": left_context_size, "finished": torch.tensor(is_finished, dtype=torch.bool)},
    }


def talker2code2wav_full_payload(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: OmniEngineCoreRequest,
) -> dict[str, Any] | None:
    """Pack complete talker codec output for the non-async connector path."""
    if not isinstance(pooling_output, dict):
        return None
    code_predictor_codes = pooling_output.get("codes.audio")
    if code_predictor_codes is None:
        codes = pooling_output.get("codes")
        if isinstance(codes, dict):
            code_predictor_codes = codes.get("audio")
    if code_predictor_codes is None:
        return None
    if not isinstance(code_predictor_codes, torch.Tensor):
        code_predictor_codes = torch.as_tensor(code_predictor_codes)
    if code_predictor_codes.numel() == 0:
        return None

    output_token_ids = _ensure_list(getattr(request, "output_token_ids", []) or [])
    raw_shape = tuple(code_predictor_codes.shape)
    code_predictor_codes, codec_stats = _extract_qwen3_full_payload_codec_rows(
        code_predictor_codes.to(torch.long),
        list(output_token_ids),
    )
    if code_predictor_codes.numel() == 0:
        return None

    codec_codes = code_predictor_codes.transpose(0, 1).cpu().reshape(-1).tolist()
    logger.debug(
        "talker2code2wav_full_payload: raw_shape=%s output_ids_len=%s aligned_rows=%s "
        "valid_rows=%s placeholders=%s flattened_len=%s pad4196=%s bos4197=%s eos4198=%s",
        raw_shape,
        len(output_token_ids),
        codec_stats["aligned_rows"],
        codec_stats["valid_rows"],
        codec_stats["trailing_placeholder_count"],
        len(codec_codes),
        sum(1 for tid in output_token_ids if tid == _QWEN3_CODEC_PAD_TOKEN_ID),
        sum(1 for tid in output_token_ids if tid == _QWEN3_CODEC_BOS_TOKEN_ID),
        sum(1 for tid in output_token_ids if tid == _QWEN3_CODEC_EOS_TOKEN_ID),
    )
    return {
        "codes": {"audio": codec_codes},
        "code_predictor_codes": codec_codes,
        "meta": {"finished": torch.tensor(True, dtype=torch.bool)},
    }


def talker2code2wav(
    source_outputs: list[Any],
    _prompt: OmniTokensPrompt | TextPrompt | None = None,
    _requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> list[OmniTokensPrompt]:
    """
    Process talker outputs to create code2wav inputs.

    Workflow:
    1. Extract talker's codec code outputs (8-layer RVQ codes)
    2. Flatten codes for code2wav input
    3. Package for code2wav stage

    Args:
    Returns:
        List of OmniTokensPrompt for code2wav stage
    """
    talker_outputs = source_outputs
    code2wav_inputs: list[OmniTokensPrompt] = []
    # Process each talker output
    for i, talker_output in enumerate(talker_outputs):
        output = talker_output.outputs[0]
        req_id = str(getattr(talker_output, "request_id", f"idx-{i}"))
        cur_seq_len = len(output.cumulative_token_ids) - 1
        seq_len = cur_seq_len
        is_streaming_session = bool(getattr(streaming_context, "enabled", False))
        if is_streaming_session:
            seq_len = _get_streaming_codec_delta_len(cur_seq_len, req_id, talker_output, streaming_context)
        mm_raw = getattr(output, "multimodal_output", None)
        if not isinstance(mm_raw, dict):
            logger.debug("talker2code2wav: skip req=%s due to empty multimodal_output", req_id)
            continue
        mm: OmniPayload = mm_raw
        if "codes" not in mm or not isinstance(mm.get("codes"), dict) or "audio" not in mm["codes"]:
            logger.debug("talker2code2wav: skip req=%s due to missing codes.audio", req_id)
            continue
        # Extract codec codes from talker output
        # Expected shape: [8, seq_len] (8-layer RVQ codes)
        codec_codes = (
            mm["codes"]["audio"][-seq_len:].to(torch.long).transpose(0, 1).cpu().to(torch.long).reshape(-1).tolist()
        )  # 16, seq_len
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=codec_codes,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return code2wav_inputs
