# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The Qwen team.
"""Stage input processor for Qwen3 Omni MoE: Thinker → Talker transition."""

from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.inputs import TextPrompt
from vllm.logger import init_logger
from vllm.platforms import current_platform

from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.stage_input_processors._common import ensure_list, validate_stage_inputs
from vllm_omni.model_executor.stage_input_processors.tts_utils import (
    extract_language_from_prompt,
    extract_language_from_request,
    extract_speaker_from_prompt,
    extract_speaker_from_request,
)
from vllm_omni.worker.payload_span import (
    THINKER_DECODE_EMBEDDINGS_KEY,
    THINKER_DECODE_TOKEN_END_KEY,
    THINKER_DECODE_TOKEN_START_KEY,
    THINKER_OUTPUT_TOKEN_IDS_KEY,
)

logger = init_logger(__name__)

# Pooling output layer keys: "0" = word embedding, "24" = accept_hidden_layer
_EMBED_LAYER_KEY = "0"
_HIDDEN_LAYER_KEY = "24"


def _compute_talker_prompt_ids_length(info, device: torch.device | str = "cuda") -> int:
    im_start_token_id = 151644
    system_token_id = 8948
    user_token_id = 872
    assistant_token_id = 77091

    thinker_sequences = torch.tensor(info["thinker_sequences"], dtype=torch.long, device=device).unsqueeze(0)  # [1, T]

    input_ids = torch.tensor(info["thinker_input_ids"], dtype=torch.long, device=device).unsqueeze(0)  # [1, T]

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


_QWEN3_CODEC_CODEBOOK_SIZE = 2048
_QWEN3_CODEC_PAD_TOKEN_ID = 4196
_QWEN3_CODEC_BOS_TOKEN_ID = 4197
_QWEN3_CODEC_EOS_TOKEN_ID = 4198


def _request_finished(request: Any) -> bool:
    finished_fn = getattr(request, "is_finished", None)
    if callable(finished_fn):
        try:
            return bool(finished_fn())
        except Exception:
            logger.debug("request.is_finished() failed", exc_info=True)
    return bool(getattr(request, "finished", False))


def _extract_last_valid_qwen3_codec_frame(code_predictor_codes: Any) -> list[int] | None:
    """Return the last valid codec frame for async Qwen3 code2wav handoff."""
    if not isinstance(code_predictor_codes, torch.Tensor) or code_predictor_codes.numel() == 0:
        return None

    code_frames = code_predictor_codes.to(torch.long).cpu()
    if code_frames.ndim == 1:
        frame = code_frames.reshape(-1)
        if frame.numel() == 0 or not bool(frame.any().item()):
            return None
        if int(frame.max().item()) >= _QWEN3_CODEC_CODEBOOK_SIZE:
            return None
        return frame.tolist()
    if code_frames.ndim != 2:
        raise ValueError(f"Invalid code_predictor_codes shape for Qwen3-Omni async_chunk: {tuple(code_frames.shape)}")

    valid_mask = code_frames.any(dim=1) & (code_frames.max(dim=1).values < _QWEN3_CODEC_CODEBOOK_SIZE)
    if not bool(valid_mask.any().item()):
        return None
    return code_frames[valid_mask][-1].reshape(-1).tolist()


def _get_qwen3_full_payload_codec_seq_len(output_token_ids: list[int]) -> tuple[int, int, int]:
    """Return the expected codec frame count for Qwen3 full-payload flushes."""
    if not output_token_ids:
        return 0, 0, 0

    trailing_placeholder_count = 0
    while (
        trailing_placeholder_count < len(output_token_ids) and output_token_ids[-1 - trailing_placeholder_count] == -1
    ):
        trailing_placeholder_count += 1

    effective_output_ids = (
        output_token_ids[:-trailing_placeholder_count] if trailing_placeholder_count > 0 else output_token_ids
    )
    if effective_output_ids and effective_output_ids[-1] == _QWEN3_CODEC_EOS_TOKEN_ID:
        effective_output_ids = effective_output_ids[:-1]

    valid_codec_len = sum(1 for tid in effective_output_ids if 0 <= tid < _QWEN3_CODEC_CODEBOOK_SIZE)
    seq_len = valid_codec_len + (trailing_placeholder_count if valid_codec_len > 0 else 0)
    return seq_len, trailing_placeholder_count, valid_codec_len


# =========================
# PD disaggregation helpers
# =========================


def _get_prefill_stage(stage_list: list[Any], source_stage_id: int) -> Any | None:
    if source_stage_id <= 0:
        return None
    source_stage = stage_list[source_stage_id]
    if not getattr(source_stage, "is_decode_only", False):
        return None
    prev_stage = stage_list[source_stage_id - 1]
    if getattr(prev_stage, "is_prefill_only", False) and prev_stage.engine_outputs is not None:
        return prev_stage
    return None


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
        p_emb = prefill_mm[_EMBED_LAYER_KEY].detach().to(device=device, dtype=torch.float)
        p_hid = prefill_mm[_HIDDEN_LAYER_KEY].detach().to(device=device, dtype=torch.float)
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


def _get_prefill_multimodal_output(prefill_stage: Any, output_index: int) -> dict[str, Any] | None:
    """Return multimodal_output dict from the PD prefill stage for a given batch index."""
    try:
        prefill_eos = prefill_stage.engine_outputs
        prefill_eo = prefill_eos[min(output_index, len(prefill_eos) - 1)]
        return prefill_eo.outputs[0].multimodal_output
    except Exception:
        return None


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
    val = thinker_mm.get(key)
    if val is None and prefill_mm is not None:
        val = prefill_mm.get(key)
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
    request_finished = _request_finished(request)
    effective_finished = bool(is_finished or request_finished)

    # Finish sentinel: empty pooling_output with an explicit terminal signal.
    # The terminal signal may arrive before the local request state flips to
    # finished, so honor the merged effective flag rather than request-local
    # state alone.
    if effective_finished and (pooling_output is None or pooling_output.get("0") is None):
        decode_embed_offsets = getattr(transfer_manager, "thinker_decode_embed_offsets", None)
        if isinstance(decode_embed_offsets, dict):
            decode_embed_offsets.pop(request_id, None)
        return {
            "finished": torch.tensor(True, dtype=torch.bool),
        }

    if chunk_id == 0:
        prompt_token_ids = ensure_list(getattr(request, "prompt_token_ids", None) or [])
        all_token_ids = getattr(request, "all_token_ids", None)
        if all_token_ids is None:
            output_token_ids = ensure_list(getattr(request, "output_token_ids", None) or [])
            all_token_ids = prompt_token_ids + output_token_ids
        else:
            all_token_ids = ensure_list(all_token_ids)
        talker_additional_info = {
            "thinker_prefill_embeddings": pooling_output.get(_EMBED_LAYER_KEY).detach().cpu(),
            "thinker_hidden_states": pooling_output.get(_HIDDEN_LAYER_KEY).detach().cpu(),
            "thinker_sequences": all_token_ids,
            "thinker_input_ids": prompt_token_ids,
            "tts_bos_embed": pooling_output.get("tts_bos_embed").detach().cpu(),
            "tts_eos_embed": pooling_output.get("tts_eos_embed").detach().cpu(),
            "tts_pad_embed": pooling_output.get("tts_pad_embed").detach().cpu(),
            "finished": torch.tensor(effective_finished, dtype=torch.bool),
        }
        speaker = extract_speaker_from_request(request)
        if speaker is not None:
            talker_additional_info["speaker"] = speaker
        language = extract_language_from_request(request)
        if language is not None:
            talker_additional_info["language"] = language
        if transfer_manager.request_payload.get(request_id) is None:
            if not effective_finished:
                transfer_manager.request_payload[request_id] = talker_additional_info
                return None
        else:
            save_payload = transfer_manager.request_payload.pop(request_id)
            talker_additional_info["thinker_prefill_embeddings"] = torch.cat(
                (
                    save_payload.get("thinker_prefill_embeddings"),
                    talker_additional_info.get("thinker_prefill_embeddings"),
                ),
                dim=0,
            )
            talker_additional_info["thinker_hidden_states"] = torch.cat(
                (
                    save_payload.get("thinker_hidden_states"),
                    talker_additional_info.get("thinker_hidden_states"),
                ),
                dim=0,
            )
        talker_additional_info["next_stage_prompt_len"] = _compute_talker_prompt_ids_length(
            talker_additional_info,
            device="cpu",
        )
    else:
        output_token_ids = request.output_token_ids
        output_token_ids = ensure_list(output_token_ids)

        talker_additional_info = {
            "finished": torch.tensor(effective_finished, dtype=torch.bool),
        }
        speaker = extract_speaker_from_request(request)
        if speaker is not None:
            talker_additional_info["speaker"] = speaker
        language = extract_language_from_request(request)
        if language is not None:
            talker_additional_info["language"] = language

        if output_token_ids:
            decode_embeddings = pooling_output.get("0").detach().cpu()
            decode_embed_offsets = getattr(transfer_manager, "thinker_decode_embed_offsets", None)
            if decode_embed_offsets is None:
                decode_embed_offsets = {}
                transfer_manager.thinker_decode_embed_offsets = decode_embed_offsets
            if chunk_id == 1:
                decode_embed_offsets[request_id] = 0

            talker_additional_info["override_keys"] = [
                THINKER_DECODE_EMBEDDINGS_KEY,
                THINKER_OUTPUT_TOKEN_IDS_KEY,
                THINKER_DECODE_TOKEN_START_KEY,
                THINKER_DECODE_TOKEN_END_KEY,
            ]
            talker_additional_info[THINKER_DECODE_EMBEDDINGS_KEY] = decode_embeddings
            talker_additional_info[THINKER_OUTPUT_TOKEN_IDS_KEY] = output_token_ids

            decode_rows = int(decode_embeddings.shape[0]) if decode_embeddings.ndim > 1 else 1
            token_start = int(decode_embed_offsets.get(request_id, 0))
            token_end = token_start + decode_rows
            decode_embed_offsets[request_id] = token_end
            talker_additional_info[THINKER_DECODE_TOKEN_START_KEY] = token_start
            talker_additional_info[THINKER_DECODE_TOKEN_END_KEY] = token_end
        else:
            talker_additional_info["thinker_prefill_embeddings"] = pooling_output.get("0").detach().cpu()
            talker_additional_info["thinker_hidden_states"] = pooling_output.get("24").detach().cpu()

    return talker_additional_info


def thinker2talker_full_payload(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: OmniEngineCoreRequest,
) -> dict[str, Any] | None:
    """full_payload_mode thinker->talker processor (async-style signature).

    Unlike ``thinker2talker_async_chunk`` which is called per-step,
    this is called once when the thinker finishes processing a request.
    It packs the complete thinker outputs into a single payload with
    ``finished=True``.
    """
    embeddings = pooling_output.get("0")
    hidden_states = pooling_output.get("24")
    tts_bos = pooling_output.get("tts_bos_embed")
    tts_eos = pooling_output.get("tts_eos_embed")
    tts_pad = pooling_output.get("tts_pad_embed")
    if any(v is None for v in (embeddings, hidden_states, tts_bos, tts_eos, tts_pad)):
        return None

    if hasattr(request, "all_token_ids"):
        all_token_ids = ensure_list(request.all_token_ids)
    else:
        prompt_ids = ensure_list(getattr(request, "prompt_token_ids", []) or [])
        output_ids = ensure_list(getattr(request, "output_token_ids", []) or [])
        all_token_ids = prompt_ids + output_ids
    prompt_token_ids = ensure_list(getattr(request, "prompt_token_ids", []) or [])

    # Compute next_stage_prompt_len using the full thinker_sequences
    # This is the correct place to compute it because we have all the data
    info = {
        "thinker_sequences": all_token_ids,
        "thinker_input_ids": prompt_token_ids,
    }
    next_stage_prompt_len = _compute_talker_prompt_ids_length(info, device="cpu")

    payload = {
        "thinker_prefill_embeddings": embeddings.detach().cpu(),
        "thinker_hidden_states": hidden_states.detach().cpu(),
        "thinker_sequences": all_token_ids,
        "thinker_input_ids": prompt_token_ids,
        "tts_bos_embed": tts_bos.detach().cpu(),
        "tts_eos_embed": tts_eos.detach().cpu(),
        "tts_pad_embed": tts_pad.detach().cpu(),
        "next_stage_prompt_len": next_stage_prompt_len,
        "finished": torch.tensor(True, dtype=torch.bool),
    }
    speaker = extract_speaker_from_request(request)
    if speaker is not None:
        payload["speaker"] = speaker
    language = extract_language_from_request(request)
    if language is not None:
        payload["language"] = language
    return payload


def thinker2talker(
    stage_list: list[Any],
    engine_input_source: list[int],
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
        stage_list: List of stage objects
        engine_input_source: Source stage IDs (typically [0] for thinker)
        prompt: Original prompt data
        requires_multimodal_data: Whether multimodal data is required

    Returns:
        List of OmniTokensPrompt for talker stage
    """
    thinker_outputs = validate_stage_inputs(stage_list, engine_input_source)
    talker_inputs: list[OmniTokensPrompt] = []

    device = torch.device(current_platform.device_type)

    # PD disaggregation: look up the preceding prefill stage (if any)
    source_stage_id = engine_input_source[0]
    prefill_stage = _get_prefill_stage(stage_list, source_stage_id)

    # Process each thinker output
    for i, thinker_output in enumerate(thinker_outputs):
        output = thinker_output.outputs[0]
        req_id = str(getattr(thinker_output, "request_id", f"idx-{i}"))
        prompt_token_ids = ensure_list(thinker_output.prompt_token_ids)
        output_ids = ensure_list(output.token_ids)
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
        thinker_mm = output.multimodal_output
        thinker_emb = thinker_mm[_EMBED_LAYER_KEY].detach().to(device=device, dtype=torch.float)[-new_seq_length:]
        thinker_hid = thinker_mm[_HIDDEN_LAYER_KEY].detach().to(device=device, dtype=torch.float)[-new_seq_length:]

        prefill_mm: dict[str, Any] | None = None
        if prefill_stage is not None:
            prefill_mm = _get_prefill_multimodal_output(prefill_stage, i)

        if prefill_mm is not None:
            expected_total = len(prompt_token_ids) + len(output_ids)
            try:
                thinker_emb, thinker_hid = _merge_pd_embeddings(
                    thinker_emb, thinker_hid, prefill_mm, device, expected_total=expected_total
                )
            except Exception as exc:
                logger.warning("[PD] Could not merge prefill embeddings: %s", exc)

        info = {
            "thinker_prefill_embeddings": thinker_emb,
            "thinker_hidden_states": thinker_hid,
            "thinker_sequences": thinker_sequences,
            "thinker_input_ids": thinker_input_ids,
            "tts_bos_embed": _resolve_tts_token_embedding(
                "tts_bos_embed", thinker_mm=thinker_mm, prefill_mm=prefill_mm, device=device
            ),
            "tts_eos_embed": _resolve_tts_token_embedding(
                "tts_eos_embed", thinker_mm=thinker_mm, prefill_mm=prefill_mm, device=device
            ),
            "tts_pad_embed": _resolve_tts_token_embedding(
                "tts_pad_embed", thinker_mm=thinker_mm, prefill_mm=prefill_mm, device=device
            ),
        }
        speaker = extract_speaker_from_prompt(prompt, index=i)
        if speaker is not None:
            info["speaker"] = speaker
        language = extract_language_from_prompt(prompt, index=i)
        if language is not None:
            info["language"] = language

        prompt_len = _compute_talker_prompt_ids_length(info, device=device)

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

    request_finished = _request_finished(request)
    effective_finished = bool(is_finished or request_finished)

    # The finish sentinel can arrive before request.is_finished() flips on the
    # local request object. Flush any cached codec codes as soon as this chunk
    # is explicitly marked finished.
    if effective_finished and "code_predictor_codes" not in pooling_output:
        request_id = request.external_req_id
        accumulated = transfer_manager.code_prompt_token_ids.get(request_id)
        if accumulated and len(accumulated) > 0:
            connector = getattr(transfer_manager, "connector", None)
            raw_cfg = getattr(connector, "config", {}) or {}
            cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
            chunk_size = int(cfg.get("codec_chunk_frames", 25))
            left_context_size = int(cfg.get("codec_left_context_frames", 25))
            length = len(accumulated)
            chunk_length = length % chunk_size
            context_length = chunk_length if chunk_length != 0 else chunk_size
            left_context_size = max(0, min(length - context_length, left_context_size))
            end_index = min(length, left_context_size + context_length)
            return {
                "code_predictor_codes": (torch.tensor(accumulated[-end_index:]).transpose(0, 1).reshape(-1).tolist()),
                "left_context_size": left_context_size,
                "finished": torch.tensor(True, dtype=torch.bool),
            }
        return {
            "finished": torch.tensor(True, dtype=torch.bool),
        }

    if "code_predictor_codes" not in pooling_output:
        return None

    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size_config = int(cfg.get("codec_chunk_frames", 25))
    left_context_size_config = int(cfg.get("codec_left_context_frames", 25))

    frame = _extract_last_valid_qwen3_codec_frame(pooling_output.get("code_predictor_codes"))
    if frame is None:
        return None

    request_id = request.external_req_id
    transfer_manager.code_prompt_token_ids[request_id].append(frame)
    length = len(transfer_manager.code_prompt_token_ids[request_id])
    chunk_length = length % chunk_size_config
    if chunk_length != 0 and not effective_finished:
        return None

    context_length = chunk_length if chunk_length != 0 else chunk_size_config
    left_context_size = max(0, min(length - context_length, left_context_size_config))
    end_index = min(length, left_context_size + context_length)

    codes = (
        torch.tensor(transfer_manager.code_prompt_token_ids[request_id][-end_index:])
        .transpose(0, 1)
        .reshape(-1)
        .tolist()
    )

    info = {
        "code_predictor_codes": codes,
        "left_context_size": left_context_size,
        "finished": torch.tensor(effective_finished, dtype=torch.bool),
    }
    return info


def talker2code2wav_full_payload(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: OmniEngineCoreRequest,
) -> dict[str, Any] | None:
    """full_payload_mode talker->code2wav processor (async-style signature).

    Called once when the talker finishes processing a request.
    Match the legacy no-async path by trimming to the valid generated
    codec frames instead of flattening the entire predictor buffer.
    """
    if "code_predictor_codes" not in pooling_output:
        return None

    code_predictor_codes = pooling_output["code_predictor_codes"]
    if code_predictor_codes is None:
        return None
    if not isinstance(code_predictor_codes, torch.Tensor):
        code_predictor_codes = torch.as_tensor(code_predictor_codes)
    if code_predictor_codes.numel() == 0:
        return None

    output_token_ids = ensure_list(getattr(request, "output_token_ids", []) or [])
    raw_shape = tuple(code_predictor_codes.shape)
    seq_len, trailing_placeholder_count, valid_codec_len = _get_qwen3_full_payload_codec_seq_len(output_token_ids)
    if code_predictor_codes.ndim > 0:
        seq_len = min(seq_len, int(code_predictor_codes.shape[0]))
    if seq_len > 0:
        code_predictor_codes = code_predictor_codes[-seq_len:]
    if code_predictor_codes.numel() == 0:
        return None

    codec_codes = code_predictor_codes.to(torch.long).transpose(0, 1).cpu().reshape(-1).tolist()
    pad_4196 = sum(1 for tid in output_token_ids if tid == _QWEN3_CODEC_PAD_TOKEN_ID)
    bos_4197 = sum(1 for tid in output_token_ids if tid == _QWEN3_CODEC_BOS_TOKEN_ID)
    eos_4198 = sum(1 for tid in output_token_ids if tid == _QWEN3_CODEC_EOS_TOKEN_ID)
    logger.debug(
        "talker2code2wav_full_payload: raw_shape=%s output_ids_len=%s seq_len=%s "
        "valid_codec_len=%s placeholders=%s flattened_len=%s first16=%s last16_tokens=%s "
        "pad4196=%s bos4197=%s eos4198=%s",
        raw_shape,
        len(output_token_ids),
        seq_len,
        valid_codec_len,
        trailing_placeholder_count,
        len(codec_codes),
        codec_codes[:16],
        output_token_ids[-16:],
        pad_4196,
        bos_4197,
        eos_4198,
    )

    return {
        "code_predictor_codes": codec_codes,
        "finished": torch.tensor(True, dtype=torch.bool),
    }


# Backward-compatible aliases for configs that still reference the old suffix.
thinker2talker_batch = thinker2talker_full_payload
talker2code2wav_batch = talker2code2wav_full_payload


def talker2code2wav(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> list[OmniTokensPrompt]:
    """
    Process talker outputs to create code2wav inputs.

    Workflow:
    1. Extract talker's codec code outputs (8-layer RVQ codes)
    2. Flatten codes for code2wav input
    3. Package for code2wav stage

    Args:
        stage_list: List of stage objects
        engine_input_source: Source stage IDs (typically [1] for talker)
        prompt: Original prompt data
        requires_multimodal_data: Whether multimodal data is required

    Returns:
        List of OmniTokensPrompt for code2wav stage
    """
    talker_outputs = validate_stage_inputs(stage_list, engine_input_source)
    code2wav_inputs: list[OmniTokensPrompt] = []
    # Process each talker output
    for i, talker_output in enumerate(talker_outputs):
        output = talker_output.outputs[0]
        req_id = str(getattr(talker_output, "request_id", f"idx-{i}"))
        cur_seq_len = len(output.token_ids) - 1
        seq_len = cur_seq_len
        is_streaming_session = bool(getattr(streaming_context, "enabled", False))
        if is_streaming_session:
            seq_len = _get_streaming_codec_delta_len(cur_seq_len, req_id, talker_output, streaming_context)
        # Extract codec codes from talker output
        # Expected shape: [8, seq_len] (8-layer RVQ codes)
        multimodal_output = getattr(output, "multimodal_output", {}) or {}
        if "code_predictor_codes" not in multimodal_output:
            logger.warning(
                "talker2code2wav missing code_predictor_codes for req=%s keys=%s",
                req_id,
                list(multimodal_output.keys())
                if isinstance(multimodal_output, dict)
                else type(multimodal_output).__name__,
            )
            continue
        codec_codes = (
            multimodal_output["code_predictor_codes"][-seq_len:]
            .to(torch.long)
            .transpose(0, 1)
            .cpu()
            .to(torch.long)
            .reshape(-1)
            .tolist()
        )  # 16, seq_len
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=codec_codes,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return code2wav_inputs
