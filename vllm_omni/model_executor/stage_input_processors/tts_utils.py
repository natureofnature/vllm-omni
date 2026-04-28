# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The Qwen team.
"""Shared TTS utility functions for speaker and language extraction.

These utilities are model-agnostic and can be used by any TTS model stage
processor (qwen3_omni, qwen2_5_omni, qwen3_tts, etc.).
"""

from typing import Any

import torch

QWEN3_TTS_TALKER_GPU_RESIDENT_BUFFER_KEYS = frozenset(
    {"audio_codes", "last_talker_hidden", "tts_pad_embed", "tailing_text_hidden"}
)


def build_qwen3_tts_talker_multimodal_outputs(
    info_dicts: list[dict[str, Any]] | None,
) -> tuple[int, dict[str, Any]]:
    """Aggregate per-request talker outputs into one multimodal payload."""
    if not info_dicts:
        return 0, {}

    audio_codes_list: list[torch.Tensor] = []
    ref_code_len_list: list[torch.Tensor] = []
    ref_code_tensor: torch.Tensor | None = None
    codec_streaming_list: list[torch.Tensor] = []

    for info in info_dicts:
        if not isinstance(info, dict):
            continue
        ac = info.get("audio_codes")
        if isinstance(ac, torch.Tensor):
            audio_codes_list.append(ac)
            cs = info.get("codec_streaming")
            if isinstance(cs, bool):
                codec_streaming_list.append(
                    torch.full((int(ac.shape[0]),), int(cs), dtype=torch.int8, device=ac.device)
                )
        ref_code = info.get("ref_code")
        if isinstance(ref_code, torch.Tensor) and ref_code.numel() > 0:
            ref_code_tensor = ref_code
        ref_len = info.get("ref_code_len")
        if ref_len is None:
            continue
        if isinstance(ref_len, torch.Tensor):
            if ref_len.numel() == 0:
                raise ValueError("ref_code_len is an empty tensor")
            ref_len_val = int(ref_len.reshape(-1)[-1].item())
        elif isinstance(ref_len, list):
            if len(ref_len) != 1:
                raise ValueError(f"ref_code_len must be scalar or 1-element list, got len={len(ref_len)}")
            ref_len_val = int(ref_len[0])
        else:
            ref_len_val = int(ref_len)
        if isinstance(ac, torch.Tensor):
            ref_code_len_list.append(torch.full((int(ac.shape[0]),), ref_len_val, dtype=torch.int32, device=ac.device))

    if not audio_codes_list:
        return 0, {}

    audio_codes = torch.cat(audio_codes_list, dim=0)
    span_len = int(audio_codes.shape[0])
    mm: dict[str, Any] = {"audio_codes": audio_codes}
    if ref_code_len_list:
        mm["ref_code_len"] = torch.cat(ref_code_len_list, dim=0)[:span_len]
    if ref_code_tensor is not None:
        mm["ref_code"] = [ref_code_tensor]
    if codec_streaming_list:
        mm["codec_streaming"] = torch.cat(codec_streaming_list, dim=0)[:span_len]
    return span_len, mm


# =============================================================================
# Speaker helpers
# =============================================================================


def extract_speaker_from_runtime_info(
    runtime_additional_information: list[dict[str, Any]] | None,
) -> str | None:
    """Extract speaker from per-request runtime info dicts.

    Iterates through the list of per-request info dicts and returns the first
    non-empty speaker string found, normalized to lowercase.

    Args:
        runtime_additional_information: List of per-request additional info
            dicts, as passed to the model's forward() method.

    Returns:
        The speaker string (lowercase, stripped), or None if not present.
    """
    if not runtime_additional_information:
        return None
    for info in runtime_additional_information:
        vt = info.get("speaker")
        if vt is None:
            continue
        if isinstance(vt, (list, tuple)) and len(vt) > 0:
            vt = vt[0]
        if isinstance(vt, str) and vt.strip():
            return vt.lower().strip()
        if vt is not None:
            return str(vt).lower().strip()
    return None


def extract_speaker_from_request(request: Any) -> str | None:
    """Extract speaker from a request's additional_information field.

    Reads from the structured ``additional_information.entries["speaker"]``
    field used by the engine serialization layer.

    Args:
        request: An OmniEngineCoreRequest (or compatible object) with an
            ``additional_information`` attribute.

    Returns:
        The speaker string (lowercase, stripped), or None if not present.
    """
    additional_information = getattr(request, "additional_information", None)
    if additional_information is None:
        return None
    entries = getattr(additional_information, "entries", None)
    if not isinstance(entries, dict):
        return None
    entry = entries.get("speaker")
    if entry is None:
        return None
    list_data = getattr(entry, "list_data", None)
    if isinstance(list_data, list) and list_data:
        val = list_data[0]
        return val.lower().strip() if isinstance(val, str) else str(val).lower().strip()
    return None


def extract_speaker_from_prompt(
    prompt: Any,
    index: int = 0,
) -> list[str] | None:
    """Extract speaker from a prompt's additional_information dict.

    Used in non-async stage processors where the prompt is an
    OmniTokensPrompt / TextPrompt dict (or a list of them).

    Args:
        prompt: A single prompt dict, or a list of prompt dicts.
        index: Which element to pick when prompt is a list.

    Returns:
        The speaker as a list (for serialization compatibility), or None.
    """
    if prompt is None:
        return None
    p = prompt[index] if isinstance(prompt, list) and index < len(prompt) else prompt
    if p is None:
        return None
    add_info = p.get("additional_information")
    if not isinstance(add_info, dict):
        return None
    speaker = add_info.get("speaker")
    if isinstance(speaker, list) and speaker:
        return speaker
    return None


# =============================================================================
# Language helpers
# =============================================================================


def extract_language_from_runtime_info(
    runtime_additional_information: list[dict[str, Any]] | None,
) -> str | None:
    """Extract language from per-request runtime info dicts.
    Args:
        runtime_additional_information: List of per-request additional info
            dicts, as passed to the model's forward() method.

    Returns:
        The language string (e.g. "Chinese", "English", "Auto"), or None.
    """
    if not runtime_additional_information:
        return None
    for info in runtime_additional_information:
        lang = info.get("language")
        if lang is None:
            continue
        if isinstance(lang, (list, tuple)) and len(lang) > 0:
            return lang
        if isinstance(lang, str) and lang.strip():
            return [lang.strip()]
    return None


def extract_language_from_request(request: Any) -> str | None:
    """Extract language from a request's additional_information field.

    Args:
        request: An OmniEngineCoreRequest (or compatible object) with an
            ``additional_information`` attribute.

    Returns:
        The language string, or None if not present.
    """
    additional_information = getattr(request, "additional_information", None)
    if additional_information is None:
        return None
    entries = getattr(additional_information, "entries", None)
    if not isinstance(entries, dict):
        return None
    entry = entries.get("language")
    if entry is None:
        return None
    list_data = getattr(entry, "list_data", None)
    if isinstance(list_data, list) and list_data:
        return list_data
    return None


def extract_language_from_prompt(
    prompt: Any,
    index: int = 0,
) -> list[str] | None:
    """Extract language from a prompt's additional_information dict.
    Args:
        prompt: A single prompt dict, or a list of prompt dicts.
        index: Which element to pick when prompt is a list.

    Returns:
        The language as a list (for serialization compatibility), or None.
    """
    if prompt is None:
        return None
    p = prompt[index] if isinstance(prompt, list) and index < len(prompt) else prompt
    if p is None:
        return None
    add_info = p.get("additional_information")
    if not isinstance(add_info, dict):
        return None
    language = add_info.get("language")
    if isinstance(language, list) and language:
        return language
    return None
