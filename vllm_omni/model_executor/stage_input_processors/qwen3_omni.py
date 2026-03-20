# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The Qwen team.
"""Stage input processor for Qwen3 Omni MoE: Thinker → Talker transition."""

from typing import Any

import torch
from vllm.inputs import TextPrompt
from vllm.platforms import current_platform

from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.payload_span import (
    THINKER_DECODE_EMBEDDINGS_KEY,
    THINKER_DECODE_TOKEN_END_KEY,
    THINKER_DECODE_TOKEN_START_KEY,
    THINKER_OUTPUT_TOKEN_IDS_KEY,
)


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


def _validate_stage_inputs(stage_list, engine_input_source):
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    stage_id = engine_input_source[0]
    if stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {stage_id}")

    stage = stage_list[stage_id]
    if stage.engine_outputs is None:
        raise RuntimeError(f"Stage {stage_id} has no outputs yet")

    return stage.engine_outputs


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
    request_finished = bool(request.is_finished())
    effective_finished = bool(is_finished or request_finished)

    # Finish sentinel: empty pooling_output with is_finished()=True
    # sent when the engine-core marks the request as completed one cycle
    # after the last real token was sampled.
    if request_finished and pooling_output.get("0") is None:
        return {
            "finished": torch.tensor(True, dtype=torch.bool),
        }

    if chunk_id == 0:
        all_token_ids = request.all_token_ids
        prompt_token_ids = request.prompt_token_ids
        all_token_ids = _ensure_list(all_token_ids)
        prompt_token_ids = _ensure_list(prompt_token_ids)
        talker_additional_info = {
            "thinker_prefill_embeddings": pooling_output.get("0").detach().cpu(),
            "thinker_hidden_states": pooling_output.get("24").detach().cpu(),
            "thinker_sequences": all_token_ids,
            "thinker_input_ids": prompt_token_ids,
            "tts_bos_embed": pooling_output.get("tts_bos_embed").detach().cpu(),
            "tts_eos_embed": pooling_output.get("tts_eos_embed").detach().cpu(),
            "tts_pad_embed": pooling_output.get("tts_pad_embed").detach().cpu(),
            "finished": torch.tensor(effective_finished, dtype=torch.bool),
        }
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
        output_token_ids = _ensure_list(output_token_ids)

        talker_additional_info = {
            "finished": torch.tensor(effective_finished, dtype=torch.bool),
        }
        if output_token_ids:
            decode_embeddings = pooling_output.get("0").detach().cpu()
            talker_additional_info["override_keys"] = [
                THINKER_DECODE_EMBEDDINGS_KEY,
                THINKER_OUTPUT_TOKEN_IDS_KEY,
                THINKER_DECODE_TOKEN_START_KEY,
                THINKER_DECODE_TOKEN_END_KEY,
            ]
            talker_additional_info[THINKER_DECODE_EMBEDDINGS_KEY] = decode_embeddings
            talker_additional_info[THINKER_OUTPUT_TOKEN_IDS_KEY] = output_token_ids
            token_end = len(output_token_ids)
            token_start = token_end - int(decode_embeddings.shape[0])
            if token_start >= 0:
                talker_additional_info[THINKER_DECODE_TOKEN_START_KEY] = token_start
                talker_additional_info[THINKER_DECODE_TOKEN_END_KEY] = token_end
        else:
            talker_additional_info["thinker_prefill_embeddings"] = pooling_output.get("0").detach().cpu()
            talker_additional_info["thinker_hidden_states"] = pooling_output.get("24").detach().cpu()
    return talker_additional_info


def thinker2talker_batch(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: OmniEngineCoreRequest,
) -> dict[str, Any] | None:
    """Batch-mode thinker->talker processor (async-style signature).

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
        all_token_ids = _ensure_list(request.all_token_ids)
    else:
        prompt_ids = _ensure_list(getattr(request, "prompt_token_ids", []) or [])
        output_ids = _ensure_list(getattr(request, "output_token_ids", []) or [])
        all_token_ids = prompt_ids + output_ids
    prompt_token_ids = _ensure_list(getattr(request, "prompt_token_ids", []) or [])

    # Compute next_stage_prompt_len using the full thinker_sequences
    # This is the correct place to compute it because we have all the data
    info = {
        "thinker_sequences": all_token_ids,
        "thinker_input_ids": prompt_token_ids,
    }
    next_stage_prompt_len = _compute_talker_prompt_ids_length(info, device="cpu")

    return {
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


def thinker2talker(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """
    Process thinker outputs to create talker inputs.

    Workflow:
    1. Extract thinker's text generation outputs (token IDs + hidden states)
    2. Split hidden states into: prompt embeddings + generated embeddings
    3. Package for talker with additional information

    Args:
        stage_list: List of stage objects
        engine_input_source: Source stage IDs (typically [0] for thinker)
        prompt: Original prompt data
        requires_multimodal_data: Whether multimodal data is required

    Returns:
        List of OmniTokensPrompt for talker stage
    """
    thinker_outputs = _validate_stage_inputs(stage_list, engine_input_source)
    talker_inputs: list[OmniTokensPrompt] = []

    device = torch.device(current_platform.device_type)

    # Process each thinker output
    for thinker_output in thinker_outputs:
        output = thinker_output.outputs[0]

        info = {
            "thinker_prefill_embeddings": output.multimodal_output["0"].detach().to(device=device, dtype=torch.float),
            "thinker_hidden_states": output.multimodal_output["24"].detach().to(device=device, dtype=torch.float),
            "thinker_sequences": (
                thinker_output.prompt_token_ids + output.token_ids
            ),  # the thinker_sequences is the whole ids
            "thinker_input_ids": thinker_output.prompt_token_ids,
            # Provide thinker-side TTS token embeddings for talker projection
            "tts_bos_embed": output.multimodal_output["tts_bos_embed"].detach().to(device=device, dtype=torch.float),
            "tts_eos_embed": output.multimodal_output["tts_eos_embed"].detach().to(device=device, dtype=torch.float),
            "tts_pad_embed": output.multimodal_output["tts_pad_embed"].detach().to(device=device, dtype=torch.float),
        }

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

    request_finished = bool(request.is_finished())
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

    code_predictor_codes = pooling_output["code_predictor_codes"]

    if code_predictor_codes is None:
        return None
    if isinstance(code_predictor_codes, torch.Tensor):
        if code_predictor_codes.numel() == 0:
            return None
    elif hasattr(code_predictor_codes, "__len__"):
        if len(code_predictor_codes) == 0:
            return None

    code_frames = code_predictor_codes.to(torch.long).cpu()
    if code_frames.ndim == 1:
        new_frames = [code_frames.reshape(-1).tolist()]
    elif code_frames.ndim == 2:
        # Qwen3-Omni talker already emits [frames, codebooks] here.
        new_frames = code_frames.tolist()
    else:
        raise ValueError(f"Invalid code_predictor_codes shape for Qwen3-Omni async_chunk: {tuple(code_frames.shape)}")
    if len(new_frames) == 0:
        return None

    request_id = request.external_req_id
    transfer_manager.code_prompt_token_ids[request_id].extend(new_frames)
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


def talker2code2wav_batch(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: OmniEngineCoreRequest,
) -> dict[str, Any] | None:
    """Batch-mode talker->code2wav processor (async-style signature).

    Called once when the talker finishes processing a request.
    Extracts codec codes from pooling_output and sends the complete
    set with ``finished=True``.
    """
    if "code_predictor_codes" not in pooling_output:
        return None

    code_predictor_codes = pooling_output["code_predictor_codes"]
    if code_predictor_codes is None:
        return None
    if isinstance(code_predictor_codes, torch.Tensor):
        if code_predictor_codes.numel() == 0:
            return None
    elif hasattr(code_predictor_codes, "__len__") and len(code_predictor_codes) == 0:
        return None

    # Codec code 0 is valid; only empty payloads should be skipped.
    codec_codes = code_predictor_codes.to(torch.long).transpose(0, 1).cpu().to(torch.long).reshape(-1).tolist()

    return {
        "code_predictor_codes": codec_codes,
        "finished": torch.tensor(True, dtype=torch.bool),
    }


def talker2code2wav(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
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
    talker_outputs = _validate_stage_inputs(stage_list, engine_input_source)
    code2wav_inputs: list[OmniTokensPrompt] = []
    # Process each talker output
    for talker_output in talker_outputs:
        output = talker_output.outputs[0]
        seq_len = len(output.token_ids) - 1
        # Extract codec codes from talker output
        # Expected shape: [8, seq_len] (8-layer RVQ codes)
        codec_codes = (
            output.multimodal_output["code_predictor_codes"][-seq_len:]
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
