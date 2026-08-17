# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage processors for the AURA Omni pipeline."""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from vllm.logger import init_logger
from vllm.tokenizers import cached_tokenizer_from_config

from vllm_omni.engine.serialization import deserialize_additional_information
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (
    PRECOMPUTED_TEXT_IDS_KEY,
)
from vllm_omni.model_executor.stage_input_processors.aura_session_history import (
    DEFAULT_AURA_SYSTEM_PROMPT,
    SILENT_TEXT,
    SessionHistory,
    commit_session_turn,
    get_or_create_session_history,
    get_session_history,
    is_effectively_silent,
    record_pending_turn,
)

QWEN_IM_START_ID = 151644
QWEN_IM_END_ID = 151645
QWEN_ASSISTANT_ID = 77091
AURA_SILENT_TOKEN_IDS = [151669]
QWEN_TEXT_SILENT_TOKEN_IDS = [27, 91, 68658, 91, 29]
QWEN_TEXT_SILENT_PREFIX_TOKEN_IDS = [
    [27],
    [27, 91],
    [27, 91, 34804],
    [27, 91, 34804, 91],
]

logger = init_logger(__name__)


def _aura_log_turn_prompt_enabled() -> bool:
    return os.environ.get("VLLM_AURA_LOG_TURN_PROMPT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


QWEN_NEWLINE_ID = 198
QWEN_ASSISTANT_PREFIX_IDS = [QWEN_IM_START_ID, QWEN_ASSISTANT_ID, QWEN_NEWLINE_ID]
QWEN_ASSISTANT_SUFFIX_IDS = [
    QWEN_IM_END_ID,
    QWEN_NEWLINE_ID,
    QWEN_IM_START_ID,
    QWEN_ASSISTANT_ID,
    QWEN_NEWLINE_ID,
]
DEFAULT_QWEN3_TTS_REF_AUDIO = "vllm-omni/tests/assets/qwen3_tts/clone_2.wav"
DEFAULT_QWEN3_TTS_REF_TEXT = (
    "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
)
# Used only to size Talker prompt_token_ids placeholders (must match build_prompt_embeds).
DEFAULT_QWEN3_TTS_TOKENIZER = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

_AURA_TTS_INFO_KEYS = (
    "tts_task_type",
    "tts_language",
    "tts_instruct",
    "tts_max_new_tokens",
    "tts_ref_audio",
    "tts_ref_text",
    "tts_x_vector_only_mode",
    "tts_speaker",
    "tts_non_streaming_mode",
    "tts_ref_code_length",
    "tts_pass_token_ids",
)

# Lazy cache: (path, tokenizer, codec_language_id, spk_is_dialect)
_qwen3_tts_prompt_len_cache: dict[str, Any] | None = None


def _resolve_qwen3_tts_tokenizer_path(additional_info: dict[str, Any] | None = None) -> str:
    """Resolve Qwen3-TTS tokenizer / config path for prompt_len parity."""
    if additional_info:
        for key in ("tts_tokenizer", "tts_model", "qwen3_tts_model"):
            raw = _first_value(additional_info.get(key), None)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
    for env_key in ("VLLM_AURA_TTS_TOKENIZER", "VLLM_AURA_TTS_MODEL"):
        env = os.environ.get(env_key, "").strip()
        if env:
            return env
    # Prefer a local CustomVoice snapshot when present (demo / shared model cache).
    for candidate in (
        "/workspace/models/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice/snapshots/0c0e3051f131929182e2c023b9537f8b1c68adfe",
        str(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice"),
    ):
        if Path(candidate).is_dir() and (Path(candidate) / "tokenizer_config.json").is_file():
            return candidate
        # Hub layout may be snapshots/<hash>/
        snaps = Path(candidate) / "snapshots"
        if snaps.is_dir():
            for snap in sorted(snaps.iterdir()):
                if (snap / "tokenizer_config.json").is_file():
                    return str(snap)
    return DEFAULT_QWEN3_TTS_TOKENIZER


def _load_qwen3_tts_prompt_len_tools(
    additional_info: dict[str, Any] | None = None,
) -> tuple[Any, Mapping[str, int] | None, Mapping[str, object] | None] | None:
    """Load tokenizer + talker dialect maps for official prompt_len estimate."""
    global _qwen3_tts_prompt_len_cache
    path = _resolve_qwen3_tts_tokenizer_path(additional_info)
    if (
        isinstance(_qwen3_tts_prompt_len_cache, dict)
        and _qwen3_tts_prompt_len_cache.get("path") == path
        and _qwen3_tts_prompt_len_cache.get("tokenizer") is not None
    ):
        return (
            _qwen3_tts_prompt_len_cache["tokenizer"],
            _qwen3_tts_prompt_len_cache.get("codec_language_id"),
            _qwen3_tts_prompt_len_cache.get("spk_is_dialect"),
        )
    try:
        from transformers import AutoConfig, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            path,
            trust_remote_code=True,
            padding_side="left",
        )
        codec_language_id = None
        spk_is_dialect = None
        try:
            hf_config = AutoConfig.from_pretrained(path, trust_remote_code=True)
            talker_config = getattr(hf_config, "talker_config", None) or hf_config
            codec_language_id = getattr(talker_config, "codec_language_id", None)
            spk_is_dialect = getattr(talker_config, "spk_is_dialect", None)
        except Exception as cfg_err:
            # Snapshot JSON fallback when Transformers rejects qwen3_tts arch.
            cfg_path = Path(path) / "config.json"
            if cfg_path.is_file():
                raw = json.loads(cfg_path.read_text(encoding="utf-8"))
                talker = raw.get("talker_config") if isinstance(raw.get("talker_config"), dict) else raw
                codec_language_id = talker.get("codec_language_id")
                spk_is_dialect = talker.get("spk_is_dialect")
            else:
                logger.warning("Qwen3-TTS talker_config unavailable for prompt_len (%s): %s", path, cfg_err)
        _qwen3_tts_prompt_len_cache = {
            "path": path,
            "tokenizer": tokenizer,
            "codec_language_id": codec_language_id,
            "spk_is_dialect": spk_is_dialect,
        }
        return tokenizer, codec_language_id, spk_is_dialect
    except Exception as e:
        logger.warning("Failed to load Qwen3-TTS tokenizer for prompt_len (%s): %s", path, e)
        return None


def _estimate_tts_prompt_len_official(
    tts_info: dict[str, Any],
    *,
    task_type: str,
    additional_info: dict[str, Any] | None = None,
) -> int | None:
    """Match standalone speech API: real BPE + estimate_prompt_len_from_additional_information."""
    tools = _load_qwen3_tts_prompt_len_tools(additional_info)
    if tools is None:
        return None
    tokenizer, codec_language_id, spk_is_dialect = tools
    try:
        from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (
            Qwen3TTSPromptEmbedsBuilder,
        )

        return int(
            Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information(
                additional_information=tts_info,
                task_type=task_type,
                tokenize_prompt=lambda t: tokenizer(t, padding=False)["input_ids"],
                codec_language_id=codec_language_id if isinstance(codec_language_id, dict) else None,
                spk_is_dialect=spk_is_dialect if isinstance(spk_is_dialect, dict) else None,
            )
        )
    except Exception as e:
        logger.warning("Official Qwen3-TTS prompt_len estimate failed; falling back to heuristic: %s", e)
        return None


def default_qwen3_tts_ref_audio_path() -> str:
    """Return absolute path to the bundled ``clone_2.wav`` reference asset."""
    bundled = Path(__file__).resolve().parents[3] / "tests" / "assets" / "qwen3_tts" / "clone_2.wav"
    if bundled.is_file():
        return str(bundled)
    return DEFAULT_QWEN3_TTS_REF_AUDIO


def default_aura_tts_additional_information() -> dict[str, Any]:
    """Default Qwen3-TTS fields for AURA ``additional_information``.

    Matches Base checkpoint deployments (e.g. ``aura_omni_gpu23.yaml``).
    For CustomVoice checkpoints, set ``tts_task_type`` / ``tts_speaker`` explicitly.
    """
    return {
        "tts_task_type": "Base",
        "tts_language": "Chinese",
        "tts_instruct": "",
        "tts_ref_audio": default_qwen3_tts_ref_audio_path(),
        "tts_ref_text": DEFAULT_QWEN3_TTS_REF_TEXT,
        "tts_pass_token_ids": False,
    }


def aura_tts_additional_information_from_session(
    *,
    task_type: str | None = None,
    language: str | None = None,
    speaker: str | None = None,
    ref_audio: str | None = None,
    ref_text: str | None = None,
    instruct: str | None = None,
    max_new_tokens: int | None = None,
    pass_token_ids: bool | None = None,
) -> dict[str, Any]:
    """Merge WebSocket ``session.config`` TTS fields into ``additional_information``."""
    info = default_aura_tts_additional_information()
    if isinstance(task_type, str) and task_type.strip():
        info["tts_task_type"] = task_type.strip()
    if isinstance(language, str) and language.strip():
        info["tts_language"] = language.strip()
    if isinstance(instruct, str):
        info["tts_instruct"] = instruct
    if isinstance(speaker, str) and speaker.strip():
        info["tts_speaker"] = _normalize_qwen3_tts_speaker(speaker.strip())
    if isinstance(ref_audio, str) and ref_audio.strip():
        info["tts_ref_audio"] = ref_audio.strip()
    if isinstance(ref_text, str) and ref_text.strip():
        info["tts_ref_text"] = ref_text.strip()
    if max_new_tokens is not None:
        info["tts_max_new_tokens"] = int(max_new_tokens)
    if pass_token_ids is not None:
        info["tts_pass_token_ids"] = bool(pass_token_ids)
    if info.get("tts_task_type") == "CustomVoice":
        info.pop("tts_ref_audio", None)
        info.pop("tts_ref_text", None)
    return info


# Nested ``np.ndarray`` inside ``additional_information.scalar_data`` is msgpack-
# encoded as ``(dtype, shape, buffer)`` but not decoded back when typed as Any
# (and large buffers use aux indices that are gone after decode). Pack pixels as
# plain ``{marker, dtype, shape, data: bytes}`` before EngineCore transport.
AURA_VIDEO_WIRE_MARKER = "__aura_video_ndarray__"


def pack_aura_video_ndarray(video_array: np.ndarray) -> dict[str, Any]:
    """Pack a video ndarray into a msgspec-safe dict for ``additional_information``."""
    arr = np.ascontiguousarray(np.asarray(video_array))
    return {
        AURA_VIDEO_WIRE_MARKER: True,
        "dtype": arr.dtype.str,
        "shape": list(arr.shape),
        "data": arr.tobytes(),
    }


def unpack_aura_video_ndarray(payload: Any) -> np.ndarray | None:
    """Restore a video ndarray packed by :func:`pack_aura_video_ndarray`."""
    if not isinstance(payload, dict) or not payload.get(AURA_VIDEO_WIRE_MARKER):
        return None
    shape = payload.get("shape")
    data = payload.get("data")
    dtype = payload.get("dtype")
    if not isinstance(shape, (list, tuple)) or data is None or dtype is None:
        return None
    try:
        buffer = data if isinstance(data, (bytes, bytearray, memoryview)) else bytes(data)
        return np.frombuffer(buffer, dtype=np.dtype(dtype)).reshape(tuple(int(x) for x in shape)).copy()
    except (TypeError, ValueError, BufferError):
        return None


def frames_to_video_tuple(
    frames: list[np.ndarray],
    *,
    fps: float,
    max_frames: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Stack per-turn frames into a ``(ndarray, metadata)`` video tuple for AURA."""
    if not frames:
        raise ValueError("At least one frame is required to build video_tuple")

    selected = list(frames[-max_frames:])
    if len(selected) == 1:
        all_frames = np.stack([selected[0], selected[0]], axis=0)
    else:
        all_frames = np.stack(selected, axis=0)

    if all_frames.shape[0] < 2:
        all_frames = np.concatenate([all_frames, all_frames], axis=0)[:2]
    elif all_frames.shape[0] > max_frames:
        all_frames = all_frames[-max_frames:]

    metadata = {
        "fps": fps,
        "duration": all_frames.shape[0] / fps,
        "total_num_frames": int(all_frames.shape[0]),
        "frames_indices": list(range(all_frames.shape[0])),
        "video_backend": "opencv",
        "do_sample_frames": False,
    }
    return all_frames, metadata


def build_aura_streaming_turn_additional_information(
    *,
    session_id: str,
    video_array: np.ndarray,
    video_metadata: dict[str, Any],
    system_prompt: str,
    skip_asr: bool,
    include_tts: bool,
    tts_task_type: str | None = None,
    tts_language: str | None = None,
    tts_speaker: str | None = None,
    tts_ref_audio: str | None = None,
    tts_ref_text: str | None = None,
    tts_instruct: str | None = None,
    tts_max_new_tokens: int | None = None,
    tts_pass_token_ids: bool | None = None,
    max_rounds: int | None = None,
    num_rounds_keep: int | None = None,
    pruning_enabled: bool | None = None,
    max_context_qas: int | None = None,
    max_1qna_rounds: int | None = None,
) -> dict[str, Any]:
    """Build ``additional_information`` for one AURA streaming inference turn."""
    # Pack pixels before EngineCore msgpack: nested ndarray under scalar_data
    # does not survive decode (see ``pack_aura_video_ndarray``).
    packed_video = pack_aura_video_ndarray(video_array)
    additional_information: dict[str, Any] = {
        "aura_session_id": session_id,
        "deferred_multi_modal_data": {
            "video": [(packed_video, dict(video_metadata))],
        },
        "aura_system_prompt": [system_prompt],
        "omni_skip_stages": [0] if skip_asr else [],
    }
    # Pass client/API history knobs so Stage-1 does not fall back to its
    # shorter constructor defaults (max_rounds=20).
    if max_rounds is not None:
        additional_information["aura_max_rounds"] = [int(max_rounds)]
    if num_rounds_keep is not None:
        additional_information["aura_num_rounds_keep"] = [int(num_rounds_keep)]
    if pruning_enabled is not None:
        additional_information["aura_pruning_enabled"] = [bool(pruning_enabled)]
    if max_context_qas is not None:
        additional_information["aura_max_context_qas"] = [int(max_context_qas)]
    if max_1qna_rounds is not None:
        additional_information["aura_max_1qna_rounds"] = [int(max_1qna_rounds)]
    if include_tts:
        additional_information.update(
            aura_tts_additional_information_from_session(
                task_type=tts_task_type,
                language=tts_language,
                speaker=tts_speaker,
                ref_audio=tts_ref_audio,
                ref_text=tts_ref_text,
                instruct=tts_instruct,
                max_new_tokens=tts_max_new_tokens,
                pass_token_ids=tts_pass_token_ids,
            )
        )
    return additional_information


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _as_prompt_dict(prompt_item: Any) -> dict[str, Any]:
    return prompt_item if isinstance(prompt_item, dict) else {}


def _first_value(value: Any, default: Any = None) -> Any:
    if isinstance(value, list):
        return value[0] if value else default
    return default if value is None else value


def _first_bool(value: Any, default: bool = False) -> bool:
    value = _first_value(value, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _estimate_tts_max_new_tokens(text: str, content_ids: list[int], explicit: Any = None) -> int:
    explicit = _first_value(explicit, None)
    if explicit is not None:
        try:
            return max(1, int(explicit))
        except (TypeError, ValueError):
            pass

    # Qwen3-TTS emits one layer-0 codec token per 12 Hz audio frame. Cap must
    # stay close to real speech length: when CustomVoice EOS is late, the model
    # fills until this budget (e.g. 8-char「好，我在这儿呢。」hit 61/62 → ~5s babble).
    #
    # Observed healthy rate ≈ 3–3.5 frames/char. Prefer spoken char count over
    # AURA content_ids (subword count can inflate text-mode caps).
    spoken_chars = sum(1 for ch in text if not ch.isspace())
    if spoken_chars > 0:
        basis = spoken_chars
    else:
        basis = max(1, len(content_ids))
    return min(1024, max(16, int(math.ceil(basis * 3.5 + 10))))


def _normalize_qwen3_tts_speaker(speaker: Any) -> Any:
    if not isinstance(speaker, str):
        return speaker
    speaker = speaker.strip()
    if not speaker:
        return speaker
    if "_" in speaker:
        return speaker
    return speaker[0].upper() + speaker[1:].lower()


def _extract_output(source_output: Any) -> Any:
    outputs = getattr(source_output, "outputs", None)
    if isinstance(outputs, list) and outputs:
        return outputs[0]
    return source_output


def _extract_text(source_output: Any) -> str:
    output = _extract_output(source_output)
    cumulative_text = getattr(output, "cumulative_text", None)
    if isinstance(cumulative_text, str) and cumulative_text:
        return cumulative_text
    text = getattr(output, "text", None)
    if isinstance(text, str):
        return text
    mm = getattr(output, "multimodal_output", None)
    if isinstance(mm, dict):
        for key in ("text", "transcript", "asr_text"):
            value = mm.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, list) and value and isinstance(value[0], str):
                return value[0]
    return ""


def _clean_asr_transcript(text: str) -> str:
    """Strip Qwen3-ASR wrappers and leaked chat-template special tokens."""
    if not isinstance(text, str):
        return ""
    cleaned = text.strip()
    if "<asr_text>" in cleaned:
        cleaned = cleaned.split("<asr_text>", 1)[-1]
    cleaned = re.sub(r"^language\s+[\w-]+\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<\|im_(?:start|end)\|>", "", cleaned)
    cleaned = re.sub(r"<\|im_start\|>(?:system|user|assistant)", "", cleaned)
    return cleaned.strip()


def _extract_token_ids(source_output: Any) -> list[int]:
    output = _extract_output(source_output)
    token_ids = getattr(output, "cumulative_token_ids", None)
    if isinstance(token_ids, list):
        return [int(token_id) for token_id in token_ids if isinstance(token_id, int)]
    return []


def _trim_aura_response_token_ids(token_ids: list[int]) -> list[int]:
    ids = list(token_ids)
    if ids[: len(QWEN_ASSISTANT_PREFIX_IDS)] == QWEN_ASSISTANT_PREFIX_IDS:
        ids = ids[len(QWEN_ASSISTANT_PREFIX_IDS) :]
    if QWEN_IM_END_ID in ids:
        ids = ids[: ids.index(QWEN_IM_END_ID)]
    while ids and ids[-1] in {QWEN_IM_START_ID, QWEN_IM_END_ID, QWEN_NEWLINE_ID}:
        ids.pop()
    return ids


def _qwen3_tts_assistant_token_ids_from_aura(source_output: Any) -> list[int]:
    content_ids = _trim_aura_response_token_ids(_extract_token_ids(source_output))
    if not content_ids:
        return []
    return QWEN_ASSISTANT_PREFIX_IDS + content_ids + QWEN_ASSISTANT_SUFFIX_IDS


def _ensure_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    return [int(token_id) for token_id in list(value) if isinstance(token_id, int)]


def _is_silent_token_prefix(content_ids: list[int]) -> bool:
    if not content_ids:
        return False
    candidates = [
        AURA_SILENT_TOKEN_IDS,
        QWEN_TEXT_SILENT_TOKEN_IDS,
        *QWEN_TEXT_SILENT_PREFIX_TOKEN_IDS,
    ]
    return any(candidate[: len(content_ids)] == content_ids for candidate in candidates)


def _request_additional_info(request: Any) -> dict[str, Any]:
    def decode_info(raw_info: Any) -> dict[str, Any]:
        if isinstance(raw_info, dict):
            return raw_info
        info = deserialize_additional_information(raw_info)
        return info if isinstance(info, dict) else {}

    info = decode_info(getattr(request, "omni_stage_payload", None))
    current_info = decode_info(getattr(request, "additional_information", None))
    if current_info:
        info = {**info, **current_info}

    nested_info = info.get("additional_information") if isinstance(info, dict) else None
    if nested_info is not None:
        nested_info = decode_info(nested_info)
        if isinstance(nested_info, dict):
            info = {**nested_info, **info}

    return info


def _request_output_text(request: Any) -> str:
    output_text = getattr(request, "output_text", None)
    if isinstance(output_text, str):
        return output_text
    if isinstance(output_text, list) and output_text and isinstance(output_text[0], str):
        return output_text[0]
    return ""


def _clean_tts_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(text.split()).strip()


def _source_prompt_by_request_id(source_outputs: list[Any], prompt: Any) -> dict[str, dict[str, Any]]:
    prompts = _as_list(prompt)
    return {
        str(getattr(source_output, "request_id", idx)): _as_prompt_dict(prompt_item)
        for idx, (source_output, prompt_item) in enumerate(zip(source_outputs, prompts))
    }


AURA_VISION_PAD_TEXT = "<|vision_start|><|video_pad|><|vision_end|>"


def _vision_placeholder(multi_modal_data: dict[str, Any]) -> str:
    if "video" in multi_modal_data:
        return AURA_VISION_PAD_TEXT
    if "image" in multi_modal_data:
        return "<|vision_start|><|image_pad|><|vision_end|>"
    return ""


def _vision_multimodal_data(multi_modal_data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in multi_modal_data.items() if key in {"image", "video"}}


def _pending_vision_pad_text(
    transcript: str,
    video_tuple: tuple[Any, dict[str, Any]] | None,
    multi_modal_data: dict[str, Any],
) -> str | None:
    """Text vision pad for live placeholder video when ``video_tuple`` cannot be resolved."""
    if transcript.strip() or video_tuple is not None:
        return None
    if _vision_multimodal_data(multi_modal_data):
        return AURA_VISION_PAD_TEXT
    return None


def _aura_prompt(system_prompt: str, transcript: str, multi_modal_data: dict[str, Any]) -> str:
    vision = _vision_placeholder(multi_modal_data)
    query = transcript.strip()
    user_body = f"{vision}{query}" if query else vision
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_body}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_aura_input(
    transcript: str,
    additional_info: dict[str, Any],
    multi_modal_data: dict[str, Any],
    request_id: str,
    tokenizer: Any | None = None,
    *,
    requires_multimodal_data: bool = True,
    mm_processor_kwargs: Any = None,
) -> dict[str, Any]:
    """Build the AURA stage-1 input for both sync and async-chunk ASR output.

    Conversation history comes from ``get_or_create_session_history``
    (process-local session store). API ``AuraSessionState`` still owns frames /
    trigger / penalty.
    """
    session_id = additional_info.get("aura_session_id")
    system_prompt = _first_value(additional_info.get("aura_system_prompt"), DEFAULT_AURA_SYSTEM_PROMPT)
    transcript = _clean_asr_transcript(transcript)
    vision_data = _vision_multimodal_data(multi_modal_data)
    mm_uuids: dict[str, Any] | None = None

    if session_id:
        history = get_session_history(str(session_id))
        if history is None:
            history = get_or_create_session_history(
                str(session_id),
                system_prompt=str(system_prompt),
                max_rounds=int(_first_value(additional_info.get("aura_max_rounds"), 45)),
                num_rounds_keep=int(_first_value(additional_info.get("aura_num_rounds_keep"), 30)),
                pruning_enabled=_first_bool(additional_info.get("aura_pruning_enabled"), True),
                max_context_qas=int(_first_value(additional_info.get("aura_max_context_qas"), 10)),
                max_1qna_rounds=int(_first_value(additional_info.get("aura_max_1qna_rounds"), 4)),
            )
        video_tuple = _resolve_turn_video_tuple(additional_info, multi_modal_data)
        pending_kwargs: dict[str, Any] = {
            "deferred_mm": additional_info.get("deferred_multi_modal_data"),
            "aura_turn_video": additional_info.get("aura_turn_video"),
            "multi_modal_data": dict(multi_modal_data) if isinstance(multi_modal_data, dict) else None,
            "had_vision": False,
            "mm_uuid": None,
        }
        if video_tuple is None and history.current_rounds == 0 and _vision_multimodal_data(multi_modal_data):
            prompt = _aura_prompt(str(system_prompt), transcript, _vision_multimodal_data(multi_modal_data))
            vision_data = _vision_multimodal_data(multi_modal_data)
        else:
            vision_pad_text = _pending_vision_pad_text(transcript, video_tuple, multi_modal_data)
            vllm_inputs = history.preview_vllm_inputs(
                transcript,
                video_tuple=video_tuple,
                vision_pad_text=vision_pad_text,
            )
            prompt = vllm_inputs["prompt"]
            mm_uuids = vllm_inputs.get("multi_modal_uuids")
            pending_kwargs["mm_uuid"] = history._pending_mm_uuid
            if video_tuple is not None:
                vision_data = vllm_inputs.get("multi_modal_data", {})
            else:
                vision_data = _merge_vision_multimodal_data(
                    vllm_inputs.get("multi_modal_data", {}),
                    _vision_multimodal_data(multi_modal_data),
                )
        commit_video_tuple = (
            video_tuple
            or video_tuple_from_multi_modal_data(vision_data)
            or video_tuple_from_multi_modal_data(multi_modal_data)
        )
        pending_kwargs["had_vision"] = bool(
            any(v is not None for v in ((vision_data or {}).get("video") or []) if isinstance(vision_data, dict))
            or bool(video_tuple)
            or bool(_vision_multimodal_data(multi_modal_data))
        )
        record_pending_turn(
            str(session_id),
            request_id=request_id,
            transcript=transcript,
            video_tuple=commit_video_tuple,
            **pending_kwargs,
        )
    else:
        prompt = _aura_prompt(str(system_prompt), transcript, vision_data)

    if _aura_log_turn_prompt_enabled():
        logger.info(
            "AURA turn prompt request_id=%s transcript=%r: %s",
            request_id,
            transcript,
            _summarize_vllm_inputs({"prompt": prompt, "multi_modal_data": vision_data}),
        )

    additional_for_next = _copy_aura_tts_fields(additional_info)
    additional_for_next["aura_system_prompt"] = [str(system_prompt)]
    if session_id:
        additional_for_next["aura_session_id"] = session_id
    next_input: dict[str, Any] = {
        "prompt": prompt,
        "additional_information": additional_for_next,
    }
    if tokenizer is not None:
        prompt_token_ids = tokenizer.encode(prompt)
        next_input["prompt_token_ids"] = prompt_token_ids
        next_input["ids"] = {"prompt": prompt_token_ids}
    if requires_multimodal_data:
        next_input["multi_modal_data"] = vision_data
    if mm_uuids is not None:
        next_input["multi_modal_uuids"] = mm_uuids
    if mm_processor_kwargs is not None:
        next_input["mm_processor_kwargs"] = mm_processor_kwargs
    return next_input


def _commit_session_turn_if_present(
    additional_info: dict[str, Any],
    response_text: str,
    request_id: str | None = None,
) -> None:
    session_id = additional_info.get("aura_session_id")
    if session_id:
        commit_session_turn(str(_first_value(session_id)), response_text, request_id=request_id)


_AURA_STAGE_INPUT_PROCESSORS: dict[int, Any] = {}


def _get_aura_stage_input_processor(vllm_config: Any) -> Any:
    """Lazy InputProcessor for async_chunk Stage-1 (same as sync orchestrator)."""
    key = id(vllm_config)
    processor = _AURA_STAGE_INPUT_PROCESSORS.get(key)
    if processor is None:
        from vllm_omni.engine.stage_init_utils import build_stage0_input_processor

        processor = build_stage0_input_processor(vllm_config)
        _AURA_STAGE_INPUT_PROCESSORS[key] = processor
    return processor


def _expand_aura_async_chunk_with_input_processor(
    *,
    vllm_config: Any,
    request: Any,
    built: dict[str, Any],
    request_id: str,
) -> tuple[list[int], list[Any]]:
    """Mirror sync Stage-1: expand ``video_pad`` + attach ``mm_features``."""
    from vllm.sampling_params import SamplingParams

    processor = _get_aura_stage_input_processor(vllm_config)
    params = getattr(request, "sampling_params", None)
    if params is None:
        params = SamplingParams(max_tokens=1)
    prompt: dict[str, Any] = {
        "prompt": built["prompt"],
        "additional_information": built.get("additional_information"),
    }
    mm = built.get("multi_modal_data")
    if mm:
        prompt["multi_modal_data"] = mm
    mm_uuids = built.get("multi_modal_uuids")
    if mm_uuids is not None:
        prompt["multi_modal_uuids"] = mm_uuids
    mm_kwargs = built.get("mm_processor_kwargs")
    if mm_kwargs is not None:
        prompt["mm_processor_kwargs"] = mm_kwargs
    processed = processor.process_inputs(
        request_id=str(request_id),
        prompt=prompt,
        params=params,
        supported_tasks=("generate",),
        arrival_time=getattr(request, "arrival_time", None),
        resumable=bool(getattr(request, "resumable", False)),
    )
    prompt_token_ids = list(getattr(processed, "prompt_token_ids", None) or [])
    mm_features = list(getattr(processed, "mm_features", None) or [])
    return prompt_token_ids, mm_features


def _try_splice_pending_video_expand(
    *,
    hist: SessionHistory,
    built: dict[str, Any],
    transcript: str,
    vllm_config: Any,
    request: Any,
    request_id: str,
) -> tuple[list[int], list[Any], float] | None:
    """When committed history is unchanged, only process the pending video.

    History fingerprint stable across consecutive silent turns with the same
    committed videos. Returns ``(token_ids, mm_features, mini_process_ms)``.
    """
    cache = hist.get_expand_cache()
    if not cache:
        return None
    hist_uuids = hist.history_video_uuids()
    # Empty history + mini prompt (no system) corrupted splices and suppressed
    # spoken turns in short benches. Only splice with committed history videos.
    if not hist_uuids:
        return None
    if tuple(cache.get("hist_uuids") or ()) != hist_uuids:
        return None
    mm = built.get("multi_modal_data") or {}
    videos = mm.get("video") if isinstance(mm, dict) else None
    built_uuids = built.get("multi_modal_uuids")
    uuids = (built_uuids or {}).get("video") if isinstance(built_uuids, dict) else None
    if not isinstance(videos, list) or not isinstance(uuids, list):
        return None
    if len(videos) != len(uuids) or not videos:
        return None
    if tuple(str(u) for u in uuids[:-1]) != hist_uuids:
        return None
    pending_uuid = str(uuids[-1]) if uuids[-1] else None
    if not pending_uuid:
        return None
    old_ids = list(cache.get("prompt_token_ids") or [])
    old_feats = list(cache.get("mm_features") or [])
    if len(old_feats) != len(uuids) or not old_ids:
        return None
    if pending_uuid == cache.get("pending_uuid"):
        return old_ids, old_feats, 0.0

    import time as _time
    from dataclasses import replace

    from vllm.multimodal.inputs import PlaceholderRange

    text = (transcript or "").strip()
    mini_prompt = f"<|im_start|>user<|vision_start|><|video_pad|><|vision_end|>{text}<|im_end|><|im_start|>assistant"
    mini_built: dict[str, Any] = {
        "prompt": mini_prompt,
        "multi_modal_data": {"video": [videos[-1]]},
        "multi_modal_uuids": {"video": [pending_uuid]},
    }
    if built.get("mm_processor_kwargs") is not None:
        mini_built["mm_processor_kwargs"] = built.get("mm_processor_kwargs")
    _t0 = _time.perf_counter()
    try:
        new_ids, new_feats = _expand_aura_async_chunk_with_input_processor(
            vllm_config=vllm_config,
            request=request,
            built=mini_built,
            request_id=f"{request_id}:inc",
        )
    except Exception:  # noqa: BLE001
        return None
    mini_ms = (_time.perf_counter() - _t0) * 1000.0
    if not new_feats:
        return None
    new_f = new_feats[0]
    new_pos = getattr(new_f, "mm_position", None)
    old_last = old_feats[-1]
    old_pos = getattr(old_last, "mm_position", None)
    if new_pos is None or old_pos is None:
        return None
    new_off = int(new_pos.offset)
    new_len = int(new_pos.length)
    old_off = int(old_pos.offset)
    old_len = int(old_pos.length)
    if new_off < 0 or new_len <= 0 or old_off < 0 or old_len <= 0:
        return None
    if new_off + new_len > len(new_ids) or old_off + old_len > len(old_ids):
        return None
    video_toks = new_ids[new_off : new_off + new_len]
    spliced_ids = old_ids[:old_off] + video_toks + old_ids[old_off + old_len :]
    try:
        new_pos_adj = PlaceholderRange(
            offset=old_off,
            length=new_len,
            is_embed=getattr(new_pos, "is_embed", None),
        )
        new_f_adj = replace(new_f, mm_position=new_pos_adj)
    except Exception:  # noqa: BLE001
        return None
    spliced_feats = list(old_feats[:-1]) + [new_f_adj]
    return spliced_ids, spliced_feats, mini_ms


def resolve_aura_async_chunk_stage_payload(
    payload_data: dict[str, Any],
    request: Any,
    model_config: Any | None = None,
    vllm_config: Any | None = None,
) -> None:
    """Build the AURA prompt on the stage-1 worker from an ASR passthrough payload.

    When ``vllm_config`` is provided (async_chunk Stage-1 worker), multimodal
    expansion goes through ``InputProcessor.process_inputs`` — the same path
    sync uses — so ``prompt_token_ids`` include vision embeds and
    ``request.mm_features`` is populated. Pixel payloads are then stripped from
    ``payload_data`` so the scheduler does not re-IPC raw video through
    ``additional_information``.
    """
    if "aura_asr_transcript" not in payload_data:
        return

    additional_info = payload_data.get("additional_information")
    if not isinstance(additional_info, dict):
        additional_info = _request_additional_info(request)
    else:
        nested = additional_info.get("additional_information")
        if isinstance(nested, dict):
            additional_info = {**nested, **additional_info}

    if payload_data.get("aura_turn_video") is not None:
        additional_info = {**additional_info, "aura_turn_video": payload_data["aura_turn_video"]}

    multi_modal_data: dict[str, Any] = {}
    payload_mm = payload_data.get("multi_modal_data")
    if isinstance(payload_mm, dict):
        multi_modal_data.update(payload_mm)
    request_mm = getattr(request, "multi_modal_data", None)
    if isinstance(request_mm, dict):
        multi_modal_data.update(request_mm)
    deferred_mm = additional_info.get("deferred_multi_modal_data")
    if isinstance(deferred_mm, dict):
        multi_modal_data.update(deferred_mm)

    request_id = getattr(request, "external_req_id", None) or getattr(request, "request_id", None)
    try:
        # Prefer InputProcessor expansion (sync-aligned). Plain tokenizer.encode
        # leaves a single <|video_pad|> and never attaches mm_features.
        use_processor = vllm_config is not None
        tokenizer = None
        if not use_processor and model_config is not None:
            tokenizer = cached_tokenizer_from_config(model_config)
        built = build_aura_input(
            transcript=str(payload_data.get("aura_asr_transcript", "")),
            additional_info=additional_info,
            multi_modal_data=multi_modal_data,
            request_id=str(request_id),
            tokenizer=tokenizer,
            requires_multimodal_data=True,
            mm_processor_kwargs=payload_data.get("mm_processor_kwargs"),
        )
        payload_data.update(built)

        if use_processor:
            sid = additional_info.get("aura_session_id")
            hist = get_session_history(str(sid)) if sid else None
            transcript_s = str(payload_data.get("aura_asr_transcript", "") or "")
            spliced = None
            if hist is not None:
                spliced = _try_splice_pending_video_expand(
                    hist=hist,
                    built=built,
                    transcript=transcript_s,
                    vllm_config=vllm_config,
                    request=request,
                    request_id=str(request_id),
                )
            try:
                if spliced is not None:
                    prompt_token_ids, mm_features, _mini_ms = spliced
                else:
                    prompt_token_ids, mm_features = _expand_aura_async_chunk_with_input_processor(
                        vllm_config=vllm_config,
                        request=request,
                        built=built,
                        request_id=str(request_id),
                    )
            except (ValueError, AssertionError) as exc:
                # Cache miss / UUID-only path failure — rebuild with cold pixels.
                err = str(exc)
                if "Cache miss" not in err and "unreachable" not in err and "None" not in err:
                    raise
                if hist is not None:
                    hist._warm_mm_uuids.clear()
                    hist.clear_expand_cache()
                # Drop warm UUIDs and rebuild from SessionHistory pixels.
                if sid:
                    rebuilt = build_aura_input(
                        transcript=transcript_s,
                        additional_info=additional_info,
                        multi_modal_data=multi_modal_data,
                        request_id=str(request_id),
                        tokenizer=None,
                        requires_multimodal_data=True,
                        mm_processor_kwargs=payload_data.get("mm_processor_kwargs"),
                    )
                    built = rebuilt
                    payload_data.update(built)
                prompt_token_ids, mm_features = _expand_aura_async_chunk_with_input_processor(
                    vllm_config=vllm_config,
                    request=request,
                    built=built,
                    request_id=str(request_id),
                )
                logger.warning(
                    "AURA_MM_CACHE miss fallback to pixels request_id=%s err=%s",
                    request_id,
                    exc,
                )
            payload_data["prompt_token_ids"] = prompt_token_ids
            payload_data["ids"] = {"prompt": prompt_token_ids}
            request.mm_features = mm_features
            # Processor cache is warm for every UUID submitted this turn.
            video_uuids = (built.get("multi_modal_uuids") or {}).get("video") or []
            if hist is not None and video_uuids:
                hist.mark_mm_uuids_warm([str(u) for u in video_uuids if u])
                hist.save_expand_cache(
                    hist_uuids=hist.history_video_uuids(),
                    pending_uuid=str(video_uuids[-1]) if video_uuids[-1] else None,
                    prompt_token_ids=prompt_token_ids,
                    mm_features=mm_features,
                )
        else:
            prompt_token_ids = list(built.get("prompt_token_ids") or [])

        # Pixels live in mm_features after expansion; do not leave them on the
        # connector payload (it used to become request.additional_information).
        payload_data.pop("multi_modal_data", None)
        payload_data.pop("aura_turn_video", None)
        payload_data.pop("prompt", None)

        # Stage-local TTFT clock is set by chunk_transfer_adapter at chunk-ready
        # (before this resolve) so process_inputs is included, matching sync.
    except Exception:
        logger.exception(
            "Failed to resolve AURA async-chunk stage payload for request_id=%s session_id=%s",
            request_id,
            additional_info.get("aura_session_id"),
        )
        raise


def asr2aura(
    source_outputs: list[Any],
    prompt: Any = None,
    requires_multimodal_data: bool = True,
) -> list[dict[str, Any]]:
    """Build AURA Qwen3-VL prompts from ASR output and optional SessionHistory."""
    prompt_by_request_id = _source_prompt_by_request_id(source_outputs, prompt)
    next_inputs: list[dict[str, Any]] = []

    for idx, source_output in enumerate(source_outputs):
        src_prompt = prompt_by_request_id.get(str(getattr(source_output, "request_id", idx)), {})
        additional_info = src_prompt.get("additional_information") or {}
        request_id = str(getattr(source_output, "request_id", idx))
        multi_modal_data: dict[str, Any] = {}
        source_multi_modal_data = src_prompt.get("multi_modal_data") or {}
        if isinstance(source_multi_modal_data, dict):
            multi_modal_data.update(source_multi_modal_data)
        deferred_multi_modal_data = additional_info.get("deferred_multi_modal_data") or {}
        if isinstance(deferred_multi_modal_data, dict):
            multi_modal_data.update(deferred_multi_modal_data)
        next_inputs.append(
            build_aura_input(
                _extract_text(source_output),
                additional_info,
                multi_modal_data,
                request_id,
                requires_multimodal_data=requires_multimodal_data,
                mm_processor_kwargs=src_prompt.get("mm_processor_kwargs"),
            )
        )

    return next_inputs


def asr2aura_async_chunk(
    transfer_manager: Any,
    multimodal_output: Any | None = None,
    request: Any | None = None,
    is_finished: bool = False,
    **_: Any,
) -> dict[str, Any] | None:
    """Accumulate ASR text chunks and emit one complete AURA input at ASR finish."""
    del multimodal_output
    if request is None:
        raise ValueError("asr2aura_async_chunk requires request.")

    request_id = getattr(request, "external_req_id", None) or getattr(request, "request_id", None)
    finished = bool(is_finished or request.is_finished())
    request_payload = getattr(transfer_manager, "request_payload", None)
    if request_payload is None:
        request_payload = {}
        transfer_manager.request_payload = request_payload
    state = request_payload.setdefault(str(request_id), {})
    if not isinstance(state, dict):
        state = {}
        request_payload[str(request_id)] = state

    output_text = _request_output_text(request)
    if output_text:
        previous_text = str(state.get("asr_text", ""))
        cleaned_output_text = _clean_asr_transcript(output_text)
        state["asr_text"] = (
            cleaned_output_text
            if cleaned_output_text.startswith(previous_text)
            else _clean_asr_transcript(previous_text + output_text)
        )

    if not finished:
        return None

    additional_info = _request_additional_info(request)
    if not state.get("asr_text"):
        tokenizer = cached_tokenizer_from_config(transfer_manager.config)
        token_ids = _ensure_int_list(getattr(request, "output_token_ids", []) or [])
        if token_ids:
            state["asr_text"] = _clean_asr_transcript(tokenizer.decode(token_ids))

    multi_modal_data: dict[str, Any] = {}
    request_mm = getattr(request, "multi_modal_data", None)
    if isinstance(request_mm, dict):
        multi_modal_data.update(request_mm)
    deferred_mm = additional_info.get("deferred_multi_modal_data")
    if isinstance(deferred_mm, dict):
        multi_modal_data.update(deferred_mm)

    payload: dict[str, Any] = {
        "aura_asr_transcript": _clean_asr_transcript(str(state.get("asr_text", ""))),
        "additional_information": additional_info,
        "mm_processor_kwargs": getattr(request, "mm_processor_kwargs", None),
    }
    _attach_aura_turn_video_payload(payload, additional_info, multi_modal_data)
    return payload


def _coerce_video_frames_array(frames: Any) -> np.ndarray | None:
    """Coerce streaming turn frames into a uint8 ``[T, H, W, C]`` array."""
    if frames is None:
        return None
    try:
        video_array = np.asarray(frames, dtype=np.uint8)
        if video_array.ndim == 4:
            return video_array
        if video_array.ndim == 1 and video_array.dtype == object:
            return _coerce_video_frames_array(list(video_array))
    except (ValueError, TypeError):
        pass
    if not isinstance(frames, (list, tuple)) or not frames:
        return None
    try:
        arrays = [np.asarray(frame, dtype=np.uint8) for frame in frames]
    except (ValueError, TypeError):
        return None
    arrays = [frame for frame in arrays if frame.ndim == 3]
    if not arrays:
        return None
    target_shape = arrays[0].shape
    uniform_frames = [frame for frame in arrays if frame.shape == target_shape]
    if not uniform_frames:
        return None
    return np.stack(uniform_frames, axis=0)


def _attach_aura_turn_video_payload(
    payload: dict[str, Any],
    additional_info: dict[str, Any],
    multi_modal_data: dict[str, Any],
) -> None:
    """Best-effort JSON-serializable turn video for stage-1 resolve."""
    try:
        video_tuple = _resolve_turn_video_tuple(additional_info, multi_modal_data)
        if video_tuple is None:
            deferred = additional_info.get("deferred_multi_modal_data")
            if isinstance(deferred, dict):
                video_tuple = video_tuple_from_deferred_multi_modal(deferred)
        if video_tuple is None:
            return
        frames, metadata = video_tuple
        # Use pack_aura_video_ndarray (bytes), NOT frames.tolist(). Nested
        # Python ints from tolist() inflate Stage0→1 connector IPC by ~3× and
        # were a major contributor to the historic async ~500ms floor when the
        # payload was also reattached as request.additional_information.
        payload["aura_turn_video"] = {
            "frames": pack_aura_video_ndarray(np.asarray(frames)),
            "metadata": dict(metadata),
        }
    except Exception:
        logger.warning(
            "Failed to serialize aura_turn_video for async-chunk passthrough; "
            "stage 1 will fall back to deferred_multi_modal_data",
            exc_info=True,
        )


def _normalize_video_tuple(
    frames: Any,
    metadata: dict[str, Any] | None,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Return (uint8 ndarray [T,H,W,C], metadata) with at least two frames."""
    unpacked = unpack_aura_video_ndarray(frames)
    if unpacked is not None:
        frames = unpacked
    video_array = _coerce_video_frames_array(frames)
    if video_array is None or video_array.ndim != 4:
        return None
    meta = dict(metadata or {})
    if video_array.shape[0] < 2:
        video_array = np.concatenate([video_array, video_array], axis=0)[:2]
        meta = dict(meta)
        meta["total_num_frames"] = 2
        meta["duration"] = 2 / float(meta.get("fps", 2.0))
    return video_array, meta


def video_tuple_from_aura_turn_video(aura_turn_video: Any) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Legacy JSON-serializable turn video: ``{frames: list, metadata: dict}``."""
    if not isinstance(aura_turn_video, dict):
        return None
    return _normalize_video_tuple(aura_turn_video.get("frames"), aura_turn_video.get("metadata"))


def _is_frame_array(value: Any) -> bool:
    shape = getattr(value, "shape", None)
    return isinstance(shape, tuple) and len(shape) == 3


def _video_entry_to_tuple(entry: Any) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Best-effort conversion of one multimodal video entry to a normalized tuple."""
    if isinstance(entry, (tuple, list)) and len(entry) == 2:
        meta = entry[1] if isinstance(entry[1], dict) else {}
        return _normalize_video_tuple(entry[0], meta)
    if isinstance(entry, list) and entry and _is_frame_array(entry[0]):
        return _normalize_video_tuple(entry, {})
    if isinstance(entry, list) and entry:
        return _normalize_video_tuple(entry, {})
    if hasattr(entry, "shape") and len(entry.shape) == 4:
        return _normalize_video_tuple(entry, {})
    return None


def video_tuple_from_deferred_multi_modal(deferred: Any) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Read the first video entry from deferred multimodal data."""
    if not isinstance(deferred, dict):
        return None
    videos = deferred.get("video")
    if videos is None:
        return None

    # Top-level sibling layout: [frames, metadata]
    if (
        isinstance(videos, list)
        and len(videos) == 2
        and isinstance(videos[1], dict)
        and not isinstance(videos[0], dict)
    ):
        video_tuple = _normalize_video_tuple(videos[0], videos[1])
        if video_tuple is not None:
            return video_tuple

    # Entire clip as one ndarray: (T, H, W, C)
    if hasattr(videos, "shape") and len(videos.shape) == 4:
        return _normalize_video_tuple(videos, {})

    items = videos if isinstance(videos, list) else [videos]
    if len(items) >= 2 and all(_is_frame_array(item) for item in items):
        video_tuple = _normalize_video_tuple(items, {})
        if video_tuple is not None:
            return video_tuple

    for item in items:
        video_tuple = _video_entry_to_tuple(item)
        if video_tuple is not None:
            return video_tuple
    return None


def video_tuple_from_multi_modal_data(multi_modal_data: Any) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Read the first video tuple from a ``multi_modal_data`` dict."""
    if not isinstance(multi_modal_data, dict):
        return None
    for entry in multi_modal_data.get("video") or []:
        if isinstance(entry, dict) and "frames" in entry:
            video_tuple = video_tuple_from_aura_turn_video(entry)
            if video_tuple is not None:
                return video_tuple
        video_tuple = _video_entry_to_tuple(entry)
        if video_tuple is not None:
            return video_tuple
    return video_tuple_from_deferred_multi_modal(multi_modal_data)


def video_tuple_from_additional_info(additional_info: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Resolve per-turn video from ``deferred_multi_modal_data`` or legacy ``aura_turn_video``."""
    video_tuple = video_tuple_from_deferred_multi_modal(additional_info.get("deferred_multi_modal_data"))
    if video_tuple is not None:
        return video_tuple
    return video_tuple_from_aura_turn_video(additional_info.get("aura_turn_video"))


def _merge_vision_multimodal_data(
    preview_mm: dict[str, Any] | None,
    current_mm: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge preview history videos with the current turn's deferred video payload."""
    merged: dict[str, Any] = {}
    preview_videos = list((preview_mm or {}).get("video") or [])
    current_videos = list((current_mm or {}).get("video") or [])
    if preview_videos or current_videos:
        merged["video"] = preview_videos + current_videos
    preview_images = list((preview_mm or {}).get("image") or [])
    current_images = list((current_mm or {}).get("image") or [])
    if preview_images or current_images:
        merged["image"] = preview_images + current_images
    return merged


def _resolve_turn_video_tuple(
    additional_info: dict[str, Any],
    multi_modal_data: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]] | None:
    for source in (
        multi_modal_data,
        additional_info.get("deferred_multi_modal_data"),
        additional_info,
    ):
        video_tuple = video_tuple_from_deferred_multi_modal(source)
        if video_tuple is not None:
            return video_tuple
    return video_tuple_from_aura_turn_video(additional_info.get("aura_turn_video"))


def _copy_aura_tts_fields(additional_info: dict[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key in _AURA_TTS_INFO_KEYS:
        if key in additional_info:
            copied[key] = additional_info[key]
    return copied


def _summarize_vllm_inputs(vllm_inputs: dict[str, Any]) -> str:
    """JSON summary of AURA stage-1 prompt (text skeleton + video metadata, no pixels)."""
    videos = vllm_inputs.get("multi_modal_data", {}).get("video", [])
    video_info: list[dict[str, Any]] = []
    for vt in videos:
        if isinstance(vt, (tuple, list)) and len(vt) == 2 and not isinstance(vt[0], str):
            arr, meta = vt
        else:
            arr, meta = vt, {}
        unpacked = unpack_aura_video_ndarray(arr)
        if unpacked is not None:
            arr = unpacked
        if hasattr(arr, "shape"):
            shape = list(arr.shape)
            frames = int(arr.shape[0])
        elif isinstance(arr, list):
            frames = len(arr)
            if arr and hasattr(arr[0], "shape"):
                shape = [frames, *list(arr[0].shape)]
            else:
                shape = [frames]
        else:
            shape = None
            frames = 0
        video_info.append(
            {
                "frames": frames,
                "shape": shape,
                "fps": (meta or {}).get("fps"),
                "duration": (meta or {}).get("duration"),
            }
        )
    return json.dumps(
        {
            "prompt_text": vllm_inputs.get("prompt", ""),
            "videos": video_info,
        },
        ensure_ascii=False,
    )


def _estimate_ref_code_len_from_ref_audio(ref_audio: Any) -> int | None:
    """Estimate Qwen3-TTS ref_code length from a ref-audio payload.

    For Qwen3-TTS 12Hz models, code length is approximately:
        ceil(duration_seconds * 12.5)
    i.e. one codec frame per 1920 samples at 24kHz.
    """

    codec_frame_rate = 24000.0 / 1920.0

    # Unwrap common list wrappers.
    item = ref_audio
    while isinstance(item, list) and item:
        item = item[0]

    # Accept tuple/list like (wav, sr).
    if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[1], (int, float)):
        wav, sr = item
        sr_i = int(sr)
        if sr_i <= 0:
            return None
        if hasattr(wav, "__len__"):
            n_samples = len(wav)
        elif hasattr(wav, "shape"):
            shape = getattr(wav, "shape", None)
            if not shape:
                return None
            n_samples = shape[-1] if len(shape) > 1 else shape[0]
        else:
            return None
        if n_samples <= 0:
            return None
        return max(1, int(math.ceil((float(n_samples) / float(sr_i)) * codec_frame_rate)))

    # Accept file path (wav only).
    if isinstance(item, str) and item:
        audio_path = item
        if not os.path.isfile(audio_path) or not audio_path.lower().endswith(".wav"):
            return None
        try:
            info = sf.info(audio_path)
            n_frames = int(info.frames)
            sr = int(info.samplerate)
            if n_frames <= 0 or sr <= 0:
                return None
            return max(1, int(math.ceil((float(n_frames) / float(sr)) * codec_frame_rate)))
        except Exception:
            return None

    return None


def _approx_qwen_token_count(text: str) -> int:
    """Rough Qwen BPE length without loading a tokenizer.

    CJK / CJK punctuation ≈ 1 token each; contiguous non-CJK (incl. spaces)
    ≈ 1 token per 4 chars. Do **not** count spaces as their own tokens — that
    inflated English ``tts_instruct`` (~144 chars) from real ~35 to ~62 and
    zero-padded Talker prefill by ~100 (leading garbage / early cut).
    """
    if not text:
        return 0
    n = 0
    i = 0
    while i < len(text):
        code = ord(text[i])
        if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or 0x3000 <= code <= 0x303F or 0xFF00 <= code <= 0xFFEF:
            n += 1
            i += 1
            continue
        j = i + 1
        while j < len(text):
            cj = ord(text[j])
            if 0x4E00 <= cj <= 0x9FFF or 0x3400 <= cj <= 0x4DBF or 0x3000 <= cj <= 0x303F or 0xFF00 <= cj <= 0xFFEF:
                break
            j += 1
        n += max(1, (j - i + 3) // 4)
        i = j
    return n


def _estimate_instruct_prompt_tokens(instruct: str) -> int:
    """Token length of ``build_instruct_text(instruct)`` without a tokenizer."""
    body = instruct.strip() if isinstance(instruct, str) else ""
    if not body:
        return 0
    # <|im_start|>user\n ... <|im_end|>\n ≈ 5 special/template tokens + body.
    return 5 + _approx_qwen_token_count(body)


def _estimate_assistant_prompt_tokens(text: str) -> int:
    """Token length of ``build_assistant_text(text)`` without a tokenizer."""
    body = text if isinstance(text, str) else ""
    # Chat-template overhead (im_start/assistant/newlines/im_end) ≈ 8 tokens.
    return max(8, 8 + _approx_qwen_token_count(body))


def _estimate_tts_prompt_len_from_token_ids(
    token_ids: list[int],
    *,
    task_type: str = "Base",
    language: str = "Chinese",
    instruct: str = "",
    x_vector_only_mode: bool = False,
    non_streaming_mode: bool | None = None,
    ref_code_len: int | None = None,
) -> int:
    """Estimate Talker prefill length from prompt structure.

    This mirrors Qwen3-TTS prompt assembly at length level:
      prompt_len = instruct_len + role_len + codec_prefix_len + text/icl term

    ``instruct_len`` must be a *token* estimate. Using ``len(instruct)`` chars
    (e.g. 144 for the demo style string vs real ~40 tokens) zero-pads Talker
    prefill by ~100 and causes leading garbage / truncated speech.
    """

    # Official defaults: Base -> streaming, others -> non-streaming.
    if non_streaming_mode is None:
        non_streaming_mode = task_type in ("CustomVoice", "VoiceDesign")

    instruct_len = _estimate_instruct_prompt_tokens(instruct) if isinstance(instruct, str) else 0
    assistant_len = max(0, len(token_ids))

    # role_len = 3; codec_prefix_len = (prefill_len + speaker_len + 2) - 1
    # prefill_len = 4 when language_id exists else 3. Use non-auto language as
    # the language-id-present proxy.
    has_language_id = isinstance(language, str) and language.strip().lower() != "auto"
    prefill_len = 4 if has_language_id else 3
    speaker_len = 1 if task_type in ("CustomVoice", "Base") else 0
    base_len = instruct_len + 3 + (prefill_len + speaker_len + 2 - 1)
    if task_type in ("CustomVoice", "VoiceDesign"):
        if non_streaming_mode:
            prompt_len = base_len + max(0, assistant_len - 6)
        else:
            prompt_len = base_len + 1
        return int(prompt_len)

    if task_type == "Base":
        in_context_mode = not bool(x_vector_only_mode)
        if in_context_mode and ref_code_len is not None:
            codec_lens = 1 + int(ref_code_len)
            if non_streaming_mode:
                # Exact non-streaming ICL needs ref_ids token length; unavailable
                # in this processor. Keep a conservative upper estimate.
                prompt_len = base_len + codec_lens + max(0, assistant_len - 8) + 1
            else:
                # Streaming ICL exact length term: 1 + ref_code_len
                prompt_len = base_len + codec_lens
        else:
            # Base x-vector-only (or missing ref_code length) follows CV shape.
            if non_streaming_mode:
                prompt_len = base_len + max(0, assistant_len - 6)
            else:
                prompt_len = base_len + 1
        return int(prompt_len)

    # Defensive fallback for unknown task types.
    return int(base_len + max(assistant_len, 1))


def _aura2tts_empty_finished_payload() -> dict[str, Any]:
    """Finish-sentinel for silent AURA turns in async_chunk.

    Releases the prewarmed Talker wait gate without running synthesis,
    matching the empty-finished payload pattern used by Qwen3-TTS stages.
    """
    return {
        "prompt_token_ids": [],
        "meta": {"finished": torch.tensor(True, dtype=torch.bool)},
    }


# Native AURA sentence boundaries for incremental Stage1→TTS handoff.
_NATIVE_TTS_SENT_ENDS = frozenset("。！？；.!?;\n")
_NATIVE_TTS_COMMA_ENDS = frozenset("，,")
_NATIVE_TTS_MIN_CHARS = 10
# Hold mid-gen emits shorter than this (content chars) and merge into the next
# sentence. Prevents solo TTS of "有。" / "好的。" / "收到。" which over-generate
# filler, while still allowing normal short sentences like "今天天气很好。" to stream.
_NATIVE_TTS_MIN_EMIT_CHARS = 4


def _sentence_tts_enabled() -> bool:
    """Match Native: emit TTS per sentence while Stage1 still generates.

    Default on (2-card skip best / TTFP). Disable with ``VLLM_AURA_SENTENCE_TTS=0``.
    """
    raw = (os.environ.get("VLLM_AURA_SENTENCE_TTS") or "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _tts_content_char_count(text: str) -> int:
    """Count alphanumeric / CJK content chars (ignore punctuation/whitespace)."""
    return sum(1 for ch in text if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")


def _pop_native_tts_sentence(buf: str) -> tuple[str | None, str]:
    """Pop one Native-aligned sentence from ``buf``; return (sentence_or_None, rest)."""
    if not buf:
        return None, buf
    split_pos = -1
    for i, ch in enumerate(buf):
        if ch in _NATIVE_TTS_SENT_ENDS:
            split_pos = i + 1
            break
        if ch in _NATIVE_TTS_COMMA_ENDS and i + 1 >= _NATIVE_TTS_MIN_CHARS:
            split_pos = i + 1
            break
    if split_pos < 0:
        return None, buf
    sentence = buf[:split_pos]
    rest = buf[split_pos:]
    if not sentence.strip():
        return _pop_native_tts_sentence(rest)
    return sentence, rest


def _pop_emit_ready_tts_text(
    buf: str,
    min_chars: int = _NATIVE_TTS_MIN_EMIT_CHARS,
) -> tuple[str | None, str]:
    """Pop one or more sentences until content length >= ``min_chars``.

    Short fragments such as ``有。`` stay buffered and merge into the next
    completed sentence. Final flush (Stage1 finished) bypasses this helper.
    """
    parts: list[str] = []
    rest = buf
    while True:
        sentence, rest = _pop_native_tts_sentence(rest)
        if sentence is None:
            break
        parts.append(sentence)
        if _tts_content_char_count("".join(parts)) >= min_chars:
            return "".join(parts), rest
    if not parts:
        return None, buf
    # Not enough content yet: put complete-but-short sentences back and wait.
    return None, "".join(parts) + rest


def _tts_payload_from_talker_input(
    tts_input: OmniTokensPrompt,
    *,
    finished: bool,
) -> dict[str, Any]:
    payload = dict(tts_input["additional_information"])
    prompt_token_ids = list(tts_input["prompt_token_ids"])
    payload["prompt_token_ids"] = prompt_token_ids
    payload["meta"] = {
        "finished": torch.tensor(finished, dtype=torch.bool),
        "next_stage_prompt_len": len(prompt_token_ids),
    }
    return payload


def build_tts_talker_input(
    text: str,
    content_ids: list[int],
    additional_info: dict[str, Any],
    pass_token_ids: bool | None,
) -> OmniTokensPrompt | None:
    """Build the Qwen3-TTS Talker input shared by sync and async-chunk paths."""
    text = _clean_tts_text(text)
    content_ids = _trim_aura_response_token_ids(content_ids)
    if is_effectively_silent(text) or _is_silent_token_prefix(content_ids):
        return None

    task_type = _first_value(additional_info.get("tts_task_type"), "Base")
    language = _first_value(additional_info.get("tts_language"), "Chinese")
    instruct = _first_value(additional_info.get("tts_instruct"), "")
    x_vector_only_mode = _first_bool(additional_info.get("tts_x_vector_only_mode"), False)
    non_streaming_mode_raw = _first_value(additional_info.get("tts_non_streaming_mode"), None)
    non_streaming_mode = non_streaming_mode_raw if isinstance(non_streaming_mode_raw, bool) else None
    ref_code_len_raw = _first_value(additional_info.get("tts_ref_code_length"), None)
    ref_code_len = int(ref_code_len_raw) if isinstance(ref_code_len_raw, int) else None
    pass_token_ids = _first_bool(pass_token_ids, False)

    assistant_token_ids = QWEN_ASSISTANT_PREFIX_IDS + content_ids + QWEN_ASSISTANT_SUFFIX_IDS if content_ids else []
    ref_audio = None
    ref_text = None
    if task_type == "Base" and not x_vector_only_mode and ref_code_len is None:
        ref_audio = _first_value(additional_info.get("tts_ref_audio"), None)
        ref_code_len = _estimate_ref_code_len_from_ref_audio(ref_audio)

    tts_info = {
        "task_type": [task_type],
        "language": [language],
        "instruct": [instruct],
        "max_new_tokens": [
            _estimate_tts_max_new_tokens(
                text,
                content_ids,
                additional_info.get("tts_max_new_tokens"),
            )
        ],
    }
    if pass_token_ids and assistant_token_ids:
        tts_info[PRECOMPUTED_TEXT_IDS_KEY] = [assistant_token_ids]
        length_token_ids = assistant_token_ids
    else:
        tts_info["text"] = [text]
        # Text-mode TTS uses the Qwen3-TTS tokenizer on ``text``, not AURA
        # content_ids. Prefer a text-length placeholder so prompt_len matches
        # build_prompt_embeds (mismatched zeros → leading garbage / early cut).
        length_token_ids = [0] * _estimate_assistant_prompt_tokens(text)
    if ref_code_len is not None:
        tts_info["ref_code_length"] = [int(ref_code_len)]
    if isinstance(non_streaming_mode, bool):
        tts_info["non_streaming_mode"] = [non_streaming_mode]

    if task_type == "Base":
        ref_audio = ref_audio or _first_value(additional_info.get("tts_ref_audio"), None)
        ref_text = _first_value(additional_info.get("tts_ref_text"), None)
        if not ref_audio:
            ref_audio = default_qwen3_tts_ref_audio_path()
        if not ref_text:
            ref_text = DEFAULT_QWEN3_TTS_REF_TEXT
        tts_info["ref_audio"] = [ref_audio]
        tts_info["ref_text"] = [ref_text]
        tts_info["x_vector_only_mode"] = [x_vector_only_mode]
    elif task_type == "CustomVoice":
        tts_info["speaker"] = [_normalize_qwen3_tts_speaker(_first_value(additional_info.get("tts_speaker"), "Vivian"))]

    # Prefer official BPE prompt_len (same as serving_speech) so placeholder
    # length matches Talker build_prompt_embeds. Heuristic only as fallback.
    prompt_len = None
    session_id = _first_value(additional_info.get("aura_session_id"), None)
    if not session_id and assistant_token_ids:
        length_token_ids = assistant_token_ids
    if session_id and not (pass_token_ids and assistant_token_ids):
        prompt_len = _estimate_tts_prompt_len_official(
            tts_info,
            task_type=str(task_type),
            additional_info=additional_info,
        )
    if prompt_len is None:
        prompt_len = _estimate_tts_prompt_len_from_token_ids(
            length_token_ids,
            task_type=str(task_type),
            language=str(language),
            instruct=str(instruct),
            x_vector_only_mode=x_vector_only_mode,
            non_streaming_mode=non_streaming_mode,
            ref_code_len=ref_code_len,
        )

    logger.info(
        "[aura2tts] build talker input task=%s language=%s speaker=%s text_len=%d "
        "content_ids=%d pass_token_ids=%s prompt_len=%d max_new_tokens=%s ref_code_len=%s "
        "has_ref_audio=%s text_preview=%r",
        task_type,
        language,
        tts_info.get("speaker", [None])[0],
        len(text),
        len(content_ids),
        pass_token_ids,
        prompt_len,
        tts_info.get("max_new_tokens"),
        ref_code_len,
        bool(tts_info.get("ref_audio")),
        text[:120],
    )

    return OmniTokensPrompt(
        prompt_token_ids=[0] * prompt_len,
        additional_information=tts_info,
        multi_modal_data=None,
        mm_processor_kwargs=None,
    )


def aura2tts(
    source_outputs: list[Any],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Convert AURA text output into Qwen3-TTS Talker requests.

    Also commits the Stage-worker SessionHistory turn (sync path). Silent
    turns skip TTS but still need the history commit.
    """
    del requires_multimodal_data
    prompt_by_request_id = _source_prompt_by_request_id(source_outputs, prompt)
    next_inputs: list[OmniTokensPrompt] = []
    for idx, source_output in enumerate(source_outputs):
        text = _extract_text(source_output).strip()
        source_request_id = getattr(source_output, "request_id", None)
        src_prompt = prompt_by_request_id.get(str(source_request_id if source_request_id is not None else idx), {})
        additional_info = src_prompt.get("additional_information") or {}
        _commit_session_turn_if_present(
            additional_info,
            text or SILENT_TEXT,
            request_id=str(source_request_id) if source_request_id is not None else None,
        )
        pass_token_ids = _first_bool(additional_info.get("tts_pass_token_ids"), False)
        tts_input = build_tts_talker_input(
            text,
            _extract_token_ids(source_output),
            additional_info,
            pass_token_ids,
        )
        if tts_input is not None:
            next_inputs.append(tts_input)
    return next_inputs


def aura2tts_async_chunk(
    transfer_manager: Any,
    multimodal_output: Any | None = None,
    request: Any | None = None,
    is_finished: bool = False,
    **_: Any,
) -> dict[str, Any] | None:
    """Accumulate AURA output; emit TTS per sentence (Native) or once at finish.

    When ``VLLM_AURA_SENTENCE_TTS`` is on (default), complete sentences are
    handed to Stage2 while Stage1 is still generating. History commits once
    at Stage1 finish. Silent turns still emit an empty finished payload.
    """
    del multimodal_output
    if request is None:
        raise ValueError("aura2tts_async_chunk requires request.")

    content_ids = _trim_aura_response_token_ids(_ensure_int_list(getattr(request, "output_token_ids", []) or []))
    finished = bool(is_finished or request.is_finished())
    request_id = getattr(request, "external_req_id", None) or getattr(request, "request_id", None)
    additional_info = _request_additional_info(request)
    logger.info(
        "[aura2tts_async_chunk] req=%s is_finished_arg=%s request_finished=%s content_ids=%d output_text_len=%d",
        request_id,
        is_finished,
        finished,
        len(content_ids),
        len(_request_output_text(request) or ""),
    )
    if content_ids and _is_silent_token_prefix(content_ids):
        if not finished:
            logger.info(
                "[aura2tts_async_chunk] req=%s holding silent-prefix partial content_ids=%d",
                request_id,
                len(content_ids),
            )
            return None
        logger.info("[aura2tts_async_chunk] req=%s emitting silent finished payload", request_id)
        _commit_session_turn_if_present(additional_info, SILENT_TEXT, request_id=str(request_id))
        return _aura2tts_empty_finished_payload()

    request_payload = getattr(transfer_manager, "request_payload", None)
    if request_payload is None:
        request_payload = {}
        transfer_manager.request_payload = request_payload
    state = request_payload.setdefault(str(request_id), {})
    if not isinstance(state, dict):
        state = {}
        request_payload[str(request_id)] = state

    request_text = _clean_tts_text(_request_output_text(request))
    if content_ids:
        state["aura2tts_content_ids"] = content_ids
    if request_text:
        previous_text = str(state.get("aura2tts_text", ""))
        state["aura2tts_text"] = (
            request_text if request_text.startswith(previous_text) else _clean_tts_text(previous_text + request_text)
        )
    elif content_ids and _sentence_tts_enabled():
        # Streaming chunks may only expose token ids until finish; decode for
        # Native-style mid-generation sentence boundaries.
        try:
            tokenizer = cached_tokenizer_from_config(transfer_manager.config)
            decoded = _clean_tts_text(tokenizer.decode(content_ids))
            if decoded:
                state["aura2tts_text"] = decoded
        except Exception:
            logger.debug(
                "[aura2tts_async_chunk] req=%s token decode for sentence TTS failed",
                request_id,
                exc_info=True,
            )

    tts_metadata = _copy_aura_tts_fields(additional_info)
    if tts_metadata:
        state["aura2tts_tts_metadata"] = dict(tts_metadata)

    cached_tts_metadata = state.get("aura2tts_tts_metadata")
    if isinstance(cached_tts_metadata, dict):
        emit_info = {**cached_tts_metadata, **additional_info}
    else:
        emit_info = additional_info
    pass_token_ids = _first_bool(emit_info.get("tts_pass_token_ids"), False)
    sentence_tts = _sentence_tts_enabled()
    pending_buf = str(state.get("aura2tts_pending_sentence", ""))
    full_text = _clean_tts_text(str(state.get("aura2tts_text", ""))) or request_text
    # Append only newly seen text into the sentence buffer.
    emitted_prefix = str(state.get("aura2tts_emitted_prefix", ""))
    if full_text.startswith(emitted_prefix):
        new_tail = full_text[len(emitted_prefix) :]
    else:
        new_tail = full_text
        pending_buf = ""
        emitted_prefix = ""
    if new_tail:
        pending_buf = _clean_tts_text(pending_buf + new_tail)
        emitted_prefix = full_text
        state["aura2tts_emitted_prefix"] = emitted_prefix

    if sentence_tts and not finished:
        sentence, pending_buf = _pop_emit_ready_tts_text(pending_buf)
        state["aura2tts_pending_sentence"] = pending_buf
        if sentence is None:
            logger.info(
                "[aura2tts_async_chunk] req=%s waiting for sentence boundary pending_len=%d accumulated_text_len=%d",
                request_id,
                len(pending_buf),
                len(full_text),
            )
            return None
        # Mid-generation: text-mode TTS for the complete sentence only.
        tts_input = build_tts_talker_input(sentence, [], emit_info, pass_token_ids=False)
        if tts_input is None:
            return None
        state["aura2tts_sentence_emits"] = int(state.get("aura2tts_sentence_emits", 0)) + 1
        logger.info(
            "[aura2tts_async_chunk] req=%s emitting mid-gen sentence #%d text_len=%d preview=%r",
            request_id,
            state["aura2tts_sentence_emits"],
            len(sentence),
            sentence[:80],
        )
        return _tts_payload_from_talker_input(tts_input, finished=False)

    if not finished:
        state["aura2tts_pending_sentence"] = pending_buf
        logger.info(
            "[aura2tts_async_chunk] req=%s waiting for final AURA chunk accumulated_text_len=%d "
            "accumulated_content_ids=%d tts_metadata_keys=%s",
            request_id,
            len(str(state.get("aura2tts_text", ""))),
            len(state.get("aura2tts_content_ids", []) or []),
            sorted(tts_metadata.keys()),
        )
        return None

    # Finished: flush any remainder sentence buffer, then commit history once.
    content_ids = list(state.get("aura2tts_content_ids", content_ids) or [])
    if not content_ids and not full_text and not pending_buf:
        logger.info("[aura2tts_async_chunk] req=%s finished with no content ids; no TTS input", request_id)
        return None
    if _is_silent_token_prefix(content_ids) and not pending_buf:
        logger.info("[aura2tts_async_chunk] req=%s final content is silent; emitting finish payload", request_id)
        _commit_session_turn_if_present(additional_info, SILENT_TEXT, request_id=str(request_id))
        return _aura2tts_empty_finished_payload()

    request_text = full_text
    if not request_text and not pass_token_ids:
        try:
            tokenizer = cached_tokenizer_from_config(transfer_manager.config)
            request_text = _clean_tts_text(tokenizer.decode(content_ids))
        except Exception:
            logger.exception(
                "[aura2tts_async_chunk] req=%s failed to decode AURA token ids; falling back to token ids",
                getattr(request, "request_id", None),
            )

    if sentence_tts:
        n_emitted = int(state.get("aura2tts_sentence_emits", 0))
        flush_text = pending_buf.strip()
        state["aura2tts_pending_sentence"] = ""
        if n_emitted > 0:
            # Already streamed sentences mid-gen: flush remnant as text, then
            # optional empty finish if nothing left.
            _commit_session_turn_if_present(additional_info, request_text or SILENT_TEXT, request_id=str(request_id))
            if flush_text:
                tts_input = build_tts_talker_input(flush_text, [], emit_info, pass_token_ids=False)
                if tts_input is not None:
                    logger.info(
                        "[aura2tts_async_chunk] req=%s flushing final sentence remnant text_len=%d",
                        request_id,
                        len(flush_text),
                    )
                    return _tts_payload_from_talker_input(tts_input, finished=True)
            logger.info(
                "[aura2tts_async_chunk] req=%s finish after %d sentence emits; empty finish payload",
                request_id,
                n_emitted,
            )
            return _aura2tts_empty_finished_payload()
        # No mid-gen emits: keep classic single-shot finish (supports pass_token_ids).

    tts_input = build_tts_talker_input(
        request_text,
        content_ids,
        emit_info,
        pass_token_ids,
    )
    if tts_input is None:
        if is_effectively_silent(request_text) or _is_silent_token_prefix(content_ids):
            logger.info("[aura2tts_async_chunk] req=%s TTS input is silent; emitting finish payload", request_id)
            _commit_session_turn_if_present(additional_info, request_text or SILENT_TEXT, request_id=str(request_id))
            return _aura2tts_empty_finished_payload()
        logger.info(
            "[aura2tts_async_chunk] req=%s build_tts_talker_input returned None text_len=%d content_ids=%d",
            request_id,
            len(request_text),
            len(content_ids),
        )
        return None
    _commit_session_turn_if_present(additional_info, request_text or SILENT_TEXT, request_id=str(request_id))
    payload = _tts_payload_from_talker_input(tts_input, finished=True)
    assistant_token_ids = QWEN_ASSISTANT_PREFIX_IDS + content_ids + QWEN_ASSISTANT_SUFFIX_IDS
    if pass_token_ids and assistant_token_ids:
        payload[PRECOMPUTED_TEXT_IDS_KEY] = [assistant_token_ids]
        payload.pop("text", None)
    logger.info(
        "[aura2tts_async_chunk] req=%s emitting TTS payload prompt_len=%d text_len=%d "
        "content_ids=%d pass_token_ids=%s max_new_tokens=%s keys=%s",
        request_id,
        len(payload.get("prompt_token_ids", []) or []),
        len(request_text),
        len(content_ids),
        pass_token_ids,
        payload.get("max_new_tokens"),
        sorted(payload.keys()),
    )
    return payload
