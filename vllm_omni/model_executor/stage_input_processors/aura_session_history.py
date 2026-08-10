# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Session history with sliding-window context management for AURA streaming."""

from __future__ import annotations

import logging
import os
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SILENT_TEXT = "<|silent|>"
# Qwen3-VL / AURA (matches native AURA ContextManaged SILENT_TOKEN_ID for VL paths).
AURA_SILENT_TOKEN_ID = 151669
AURA_IM_END_TOKEN_ID = 151645
DEFAULT_AURA_SYSTEM_PROMPT = (
    "You are receiving a live video stream where the final frame is the present moment. "
    "Respond only when a response is needed based on the user's message or the visual context. "
    "Otherwise, output '<|silent|>' to signify silence. Respond in Chinese."
)

logger = logging.getLogger(__name__)


def _env_flag_on(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _session_history_diag_enabled() -> bool:
    """Gate debug logging for diagnosing session_history visibility across stages."""
    return _env_flag_on("VLLM_AURA_SESSION_HISTORY_DIAG")


def _count_videos_in_messages(messages: list[dict[str, Any]]) -> tuple[int, int]:
    """Return ``(n_videos, total_frames)`` for diagnostic logs."""
    n_videos = 0
    total_frames = 0
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "video":
                continue
            n_videos += 1
            video = item.get("video")
            arr = video[0] if isinstance(video, (tuple, list)) and video else video
            if hasattr(arr, "shape") and len(getattr(arr, "shape", ())) >= 1:
                total_frames += int(arr.shape[0])
            elif isinstance(arr, list):
                total_frames += len(arr)
    return n_videos, total_frames


# Same set used by CrossTurnPenalty to skip non-content tokens; extended with
# common CJK filler glyphs (e.g. ﹑) that models emit instead of <|silent|>.
AURA_PUNCT_CHARS = frozenset(".,!?;:，。！？；：、'\"()[]{}''…—–\n\t\r /-_@#$%^&*+=<>~`|\\（）【】《》﹑·")

__all__ = [
    "AURA_IM_END_TOKEN_ID",
    "AURA_PUNCT_CHARS",
    "AURA_SILENT_TOKEN_ID",
    "AuraSessionState",
    "SessionHistory",
    "DEFAULT_AURA_SYSTEM_PROMPT",
    "SILENT_TEXT",
    "aura_silent_stop_token_ids",
    "is_punctuation_only_text",
    "is_effectively_silent",
    "normalize_assistant_text",
    "should_stop_aura_silent_generation",
    "create_session_id",
    "create_streaming_session",
    "register_session",
    "register_session_state",
    "get_session_history",
    "get_or_create_session_history",
    "get_session_state",
    "unregister_session",
    "clear_all_sessions",
    "record_pending_turn",
    "commit_session_turn",
]


def aura_silent_stop_token_ids() -> tuple[int, ...]:
    """Token ids that end an AURA silent turn (aligned with native AURA bench)."""
    return (AURA_SILENT_TOKEN_ID, AURA_IM_END_TOKEN_ID)


def should_stop_aura_silent_generation(
    *,
    token_ids: list[int] | None = None,
    text: str | None = None,
) -> bool:
    """True when generation should stop like native AURA (first token silent or filler)."""
    ids = list(token_ids or [])
    if ids and ids[0] in aura_silent_stop_token_ids():
        return True
    if text is not None and is_effectively_silent(text):
        return True
    return False


def is_punctuation_only_text(text: str) -> bool:
    """True when stripped text is empty or only whitespace / AURA punctuation."""
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if not stripped:
        return True
    return all(ch.isspace() or ch in AURA_PUNCT_CHARS for ch in stripped)


def is_effectively_silent(text: str) -> bool:
    """Return True for empty, <|silent|>, or punctuation-only filler (e.g. \" ﹑\")."""
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if stripped == SILENT_TEXT:
        return True
    return is_punctuation_only_text(text)


def normalize_assistant_text(text: str) -> str:
    """Map degenerate non-answers to the canonical silent marker for history/TTS."""
    if is_effectively_silent(text):
        return SILENT_TEXT
    return text


class SessionHistory:
    """Two-tier conversation history for AURA streaming (sliding window + context)."""

    def __init__(
        self,
        max_rounds: int = 20,
        num_rounds_keep: int = 15,
        pruning_enabled: bool = True,
        max_context_qas: int = 10,
        max_1qna_rounds: int = 4,
        system_prompt: str | None = None,
    ) -> None:
        self.max_rounds = max_rounds
        self.num_rounds_keep = num_rounds_keep
        self.pruning_enabled = pruning_enabled
        self.max_context_qas = max_context_qas
        self.max_1qna_rounds = max_1qna_rounds
        self.current_rounds = 0

        self.system_prompt = system_prompt or DEFAULT_AURA_SYSTEM_PROMPT
        self._system_msg = {"role": "system", "content": self.system_prompt}
        self._context_history: list[list[dict[str, Any]]] = []
        self._sliding_window: list[dict[str, Any]] = []
        self.history: list[dict[str, Any]] = []
        # UUIDs whose mm processor outputs are known present in Stage-1
        # ``mm_processor_cache``. History turns with these ids can be submitted
        # as UUID-only (``multi_modal_data`` entry ``None``) so HF process_inputs
        # does not re-touch historical pixels.
        self._warm_mm_uuids: set[str] = set()
        self._pending_mm_uuid: str | None = None
        # Last successful Stage-1 expand (for pending-video splice across silent turns).
        self._expand_cache: dict[str, Any] | None = None
        self._rebuild_history()

    @staticmethod
    def new_mm_uuid() -> str:
        return f"aura-mm-{uuid.uuid4().hex}"

    def mark_mm_uuids_warm(self, mm_uuids: list[str] | tuple[str, ...] | set[str]) -> None:
        for item in mm_uuids:
            if item:
                self._warm_mm_uuids.add(str(item))

    def is_mm_uuid_warm(self, mm_uuid: str | None) -> bool:
        return bool(mm_uuid) and str(mm_uuid) in self._warm_mm_uuids

    def history_video_uuids(self) -> tuple[str, ...]:
        """Ordered ``aura_mm_uuid`` values already committed into ``history``."""
        out: list[str] = []
        for msg in self.history:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, dict) and item.get("type") == "video":
                    u = item.get("aura_mm_uuid")
                    if u:
                        out.append(str(u))
        return tuple(out)

    def clear_expand_cache(self) -> None:
        self._expand_cache = None

    def save_expand_cache(
        self,
        *,
        hist_uuids: tuple[str, ...],
        pending_uuid: str | None,
        prompt_token_ids: list[int],
        mm_features: list[Any],
    ) -> None:
        self._expand_cache = {
            "hist_uuids": hist_uuids,
            "pending_uuid": str(pending_uuid) if pending_uuid else None,
            "prompt_token_ids": list(prompt_token_ids),
            "mm_features": list(mm_features),
        }

    def get_expand_cache(self) -> dict[str, Any] | None:
        return self._expand_cache

    def _rebuild_history(self) -> None:
        self.history = [self._system_msg]
        for qa in self._context_history:
            self.history.extend(qa)
        self.history.extend(self._sliding_window)

    def _sw_round_count(self) -> int:
        return sum(1 for m in self._sliding_window if m["role"] == "user")

    @staticmethod
    def _extract_user_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text", "")
                    if text:
                        texts.append(text)
            return " ".join(texts)
        return ""

    @staticmethod
    def _is_silent_response(content: Any) -> bool:
        return isinstance(content, str) and is_effectively_silent(content)

    def _has_user_text(self, user_msg: dict[str, Any]) -> bool:
        content = user_msg.get("content", [])
        if isinstance(content, str):
            return bool(content.strip())
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text" and item.get("text", "").strip():
                    return True
        return False

    def _parse_sw_rounds(self) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
        rounds: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
        i = 0
        while i < len(self._sliding_window):
            msg = self._sliding_window[i]
            if msg["role"] == "user":
                user_msg = msg
                assistant_msg = None
                if i + 1 < len(self._sliding_window) and self._sliding_window[i + 1]["role"] == "assistant":
                    assistant_msg = self._sliding_window[i + 1]
                    i += 2
                else:
                    i += 1
                rounds.append((user_msg, assistant_msg))
            else:
                i += 1
        return rounds

    def _group_rounds_into_qas(
        self,
        rounds: list[tuple[dict[str, Any], dict[str, Any] | None]],
    ) -> list[list[tuple[dict[str, Any], dict[str, Any] | None]]]:
        groups: list[list[tuple[dict[str, Any], dict[str, Any] | None]]] = []
        current_group: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
        for user_msg, assistant_msg in rounds:
            if self._has_user_text(user_msg):
                if current_group:
                    groups.append(current_group)
                current_group = [(user_msg, assistant_msg)]
            else:
                current_group.append((user_msg, assistant_msg))
        if current_group:
            groups.append(current_group)
        return groups

    def _rewrite_qa_for_history(
        self,
        qa_rounds: list[tuple[dict[str, Any], dict[str, Any] | None]],
    ) -> list[dict[str, Any]] | None:
        rewritten: list[dict[str, Any]] = []
        for user_msg, assistant_msg in qa_rounds:
            if assistant_msg and self._is_silent_response(assistant_msg["content"]):
                continue
            user_text = self._extract_user_text(user_msg["content"])
            rewritten.append({"role": "user", "content": user_text})
            if assistant_msg:
                rewritten.append({"role": "assistant", "content": assistant_msg["content"]})
        return rewritten if rewritten else None

    @staticmethod
    def _qa_to_round_pairs(qa_messages: list[dict[str, Any]]) -> list[tuple[Any, Any | None]]:
        pairs: list[tuple[Any, Any | None]] = []
        i = 0
        while i < len(qa_messages):
            if qa_messages[i]["role"] == "user":
                user_content = qa_messages[i]["content"]
                assistant_content = None
                if i + 1 < len(qa_messages) and qa_messages[i + 1]["role"] == "assistant":
                    assistant_content = qa_messages[i + 1]["content"]
                    i += 2
                else:
                    i += 1
                pairs.append((user_content, assistant_content))
            else:
                i += 1
        return pairs

    @staticmethod
    def _count_qa_rounds(qa_messages: list[dict[str, Any]]) -> int:
        return sum(1 for m in qa_messages if m["role"] == "user")

    def _classify_qa(self, qa_messages: list[dict[str, Any]]) -> str | None:
        pairs = self._qa_to_round_pairs(qa_messages)
        n = len(pairs)
        if n == 0:
            return None
        first_has_text = bool(pairs[0][0] and str(pairs[0][0]).strip())
        if n == 1:
            return "basic" if first_has_text else "truncated"
        if n == 2 and first_has_text:
            return "1q1a"
        if n >= 3 and first_has_text:
            return "1qna"
        if not first_has_text:
            return "truncated"
        return None

    def _enforce_1qna_limit(self, qa_messages: list[dict[str, Any]]) -> None:
        while self._count_qa_rounds(qa_messages) > self.max_1qna_rounds:
            found = False
            i = 2
            while i < len(qa_messages):
                if qa_messages[i]["role"] == "user" and qa_messages[i]["content"] == "":
                    del qa_messages[i]
                    if i < len(qa_messages) and qa_messages[i]["role"] == "assistant":
                        del qa_messages[i]
                    found = True
                    break
                i += 1
            if not found:
                break

    def _merge_truncated_qa(self, truncated_messages: list[dict[str, Any]]) -> None:
        if self._context_history:
            last_qa = self._context_history[-1]
            last_qa.extend(truncated_messages)
            if self._count_qa_rounds(last_qa) > self.max_1qna_rounds:
                self._enforce_1qna_limit(last_qa)
        else:
            self._context_history.append(list(truncated_messages))

    def _prune_history(self) -> None:
        rounds = self._parse_sw_rounds()
        if len(rounds) <= self.max_rounds:
            return

        num_to_move = len(rounds) - self.num_rounds_keep
        if num_to_move <= 0:
            return
        rounds_to_move = rounds[:num_to_move]
        rounds_remaining = rounds[num_to_move:]

        for qa_rounds in self._group_rounds_into_qas(rounds_to_move):
            head_has_text = self._has_user_text(qa_rounds[0][0])
            rewritten = self._rewrite_qa_for_history(qa_rounds)
            if not rewritten:
                continue

            if head_has_text:
                qa_type = self._classify_qa(rewritten)
                if qa_type == "1qna":
                    self._enforce_1qna_limit(rewritten)
                self._context_history.append(rewritten)
            else:
                i = 0
                while i < len(rewritten):
                    if rewritten[i]["role"] == "user":
                        truncated = [rewritten[i]]
                        if i + 1 < len(rewritten) and rewritten[i + 1]["role"] == "assistant":
                            truncated.append(rewritten[i + 1])
                            i += 2
                        else:
                            i += 1
                        self._merge_truncated_qa(truncated)
                    else:
                        i += 1

        while len(self._context_history) > self.max_context_qas:
            self._context_history.pop(0)

        self._sliding_window = []
        for user_msg, assistant_msg in rounds_remaining:
            self._sliding_window.append(user_msg)
            if assistant_msg is not None:
                self._sliding_window.append(assistant_msg)

        self.current_rounds = self._sw_round_count()
        self._rebuild_history()
        self._resync_warm_mm_uuids_from_history()
        self.clear_expand_cache()

    def _resync_warm_mm_uuids_from_history(self) -> None:
        """Drop warm UUID entries that no longer appear in the sliding window."""
        live: set[str] = set()
        for msg in self._sliding_window:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, dict) and item.get("type") == "video":
                    uid = item.get("aura_mm_uuid")
                    if uid:
                        live.add(str(uid))
        if self._pending_mm_uuid:
            live.add(str(self._pending_mm_uuid))
        self._warm_mm_uuids.intersection_update(live)

    def add_user_message(
        self,
        text: str,
        images: list[Any] | None = None,
        video_tuple: tuple[Any, dict[str, Any]] | None = None,
        *,
        mm_uuid: str | None = None,
    ) -> None:
        content: list[dict[str, Any]] = []

        if video_tuple:
            video_item: dict[str, Any] = {"type": "video", "video": video_tuple}
            if mm_uuid:
                video_item["aura_mm_uuid"] = str(mm_uuid)
            content.append(video_item)
        elif images:
            content.extend([{"type": "image", "image": img} for img in images])

        if text:
            content.append({"type": "text", "text": text})
        elif not images and not video_tuple:
            return

        msg = {"role": "user", "content": content}
        self._sliding_window.append(msg)
        self.history.append(msg)
        self.current_rounds += 1

        if self.pruning_enabled and self._sw_round_count() > self.max_rounds:
            self._prune_history()

    def add_assistant_message(self, text: str) -> None:
        msg = {"role": "assistant", "content": normalize_assistant_text(text)}
        self._sliding_window.append(msg)
        self.history.append(msg)

    def preview_vllm_inputs(
        self,
        text: str = "",
        video_tuple: tuple[Any, dict[str, Any]] | None = None,
        images: list[Any] | None = None,
        *,
        vision_pad_text: str | None = None,
        mm_uuid: str | None = None,
    ) -> dict[str, Any]:
        """Build prompt/mm inputs for a pending user turn without mutating history."""
        pending_content: list[dict[str, Any]] = []
        pending_uuid = mm_uuid
        if video_tuple:
            if not pending_uuid:
                pending_uuid = self.new_mm_uuid()
            pending_content.append(
                {
                    "type": "video",
                    "video": video_tuple,
                    "aura_mm_uuid": pending_uuid,
                }
            )
        elif images:
            pending_content.extend([{"type": "image", "image": img} for img in images])
        if text:
            pending_content.append({"type": "text", "text": text})
        elif vision_pad_text and not pending_content:
            pending_content.append({"type": "text", "text": vision_pad_text})

        self._pending_mm_uuid = pending_uuid
        messages = list(self.history)
        if pending_content:
            messages.append({"role": "user", "content": pending_content})
        return self._messages_to_vllm_inputs(messages)

    def get_vllm_inputs(self) -> dict[str, Any]:
        return self._messages_to_vllm_inputs(self.history)

    def _messages_to_vllm_inputs(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        full_prompt = ""
        all_images: list[Any] = []
        all_videos: list[Any] = []
        video_uuids: list[str | None] = []
        n_uuid_only = 0
        n_pixel_videos = 0

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            full_prompt += f"<|im_start|>{role}"

            if isinstance(content, str):
                full_prompt += content
            elif isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        full_prompt += item.get("text", "")
                    elif item.get("type") == "image":
                        full_prompt += "<|vision_start|><|image_pad|><|vision_end|>"
                        all_images.append(item.get("image"))
                    elif item.get("type") == "video":
                        full_prompt += "<|vision_start|><|video_pad|><|vision_end|>"
                        mm_uuid = item.get("aura_mm_uuid")
                        mm_uuid_s = str(mm_uuid) if mm_uuid else None
                        # Always keep pixels in multi_modal_data: UUID-only (None)
                        # entries trip assertion in the HF video processor path.
                        # Warm UUIDs still skip HF via mm_processor_cache (hash=uuid).
                        all_videos.append(item.get("video"))
                        video_uuids.append(mm_uuid_s)
                        if mm_uuid_s and self.is_mm_uuid_warm(mm_uuid_s):
                            n_uuid_only += 1
                        else:
                            n_pixel_videos += 1

            full_prompt += "<|im_end|>"

        full_prompt += "<|im_start|>assistant"

        multi_modal_data: dict[str, Any] = {}
        if all_images:
            multi_modal_data["image"] = all_images
        if all_videos:
            multi_modal_data["video"] = all_videos

        out: dict[str, Any] = {
            "prompt": full_prompt,
            "multi_modal_data": multi_modal_data,
            "mm_uuid_only_videos": n_uuid_only,
            "mm_pixel_videos": n_pixel_videos,
        }
        if any(u is not None for u in video_uuids):
            # Parallel list required by vLLM when mixed None data + uuids.
            out["multi_modal_uuids"] = {"video": video_uuids}
        return out

    @staticmethod
    def _serialize_video_tuple(video_tuple: Any) -> dict[str, Any] | None:
        if not video_tuple or not isinstance(video_tuple, tuple) or len(video_tuple) != 2:
            return None
        array, metadata = video_tuple
        if array is None:
            return None
        if hasattr(array, "tolist"):
            frames = array.tolist()
        else:
            frames = array
        return {"frames": frames, "metadata": dict(metadata or {})}

    @staticmethod
    def _deserialize_video_tuple(payload: Any) -> tuple[np.ndarray, dict[str, Any]] | None:
        if not isinstance(payload, dict):
            return None
        frames = payload.get("frames")
        if frames is None:
            return None
        metadata = dict(payload.get("metadata") or {})
        return np.asarray(frames, dtype=np.uint8), metadata

    def _serialize_message_content(self, content: Any) -> Any:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return content
        serialized: list[Any] = []
        for item in content:
            if not isinstance(item, dict):
                serialized.append(item)
                continue
            item_type = item.get("type")
            if item_type == "video":
                video_payload = self._serialize_video_tuple(item.get("video"))
                out_item: dict[str, Any] = {"type": "video", "video": video_payload}
                if item.get("aura_mm_uuid"):
                    out_item["aura_mm_uuid"] = item["aura_mm_uuid"]
                serialized.append(out_item)
            elif item_type == "image":
                serialized.append({"type": "image", "image": "<image>"})
            else:
                serialized.append(item)
        return serialized

    def _deserialize_message_content(self, content: Any) -> Any:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return content
        deserialized: list[Any] = []
        for item in content:
            if not isinstance(item, dict):
                deserialized.append(item)
                continue
            item_type = item.get("type")
            if item_type == "video":
                video_tuple = self._deserialize_video_tuple(item.get("video"))
                out_item: dict[str, Any] = {"type": "video", "video": video_tuple}
                if item.get("aura_mm_uuid"):
                    out_item["aura_mm_uuid"] = item["aura_mm_uuid"]
                deserialized.append(out_item)
            elif item_type == "image":
                deserialized.append({"type": "image", "image": None})
            else:
                deserialized.append(item)
        return deserialized

    def _serialize_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "role": msg["role"],
                "content": self._serialize_message_content(msg.get("content")),
            }
            for msg in messages
        ]

    def _deserialize_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "role": msg["role"],
                "content": self._deserialize_message_content(msg.get("content")),
            }
            for msg in messages
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_prompt": self.system_prompt,
            "max_rounds": self.max_rounds,
            "num_rounds_keep": self.num_rounds_keep,
            "pruning_enabled": self.pruning_enabled,
            "max_context_qas": self.max_context_qas,
            "max_1qna_rounds": self.max_1qna_rounds,
            "current_rounds": self.current_rounds,
            "context_history": [self._serialize_messages(qa) for qa in self._context_history],
            "sliding_window": self._serialize_messages(self._sliding_window),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionHistory:
        history = cls(
            max_rounds=int(data.get("max_rounds", 20)),
            num_rounds_keep=int(data.get("num_rounds_keep", 15)),
            pruning_enabled=bool(data.get("pruning_enabled", True)),
            max_context_qas=int(data.get("max_context_qas", 10)),
            max_1qna_rounds=int(data.get("max_1qna_rounds", 4)),
            system_prompt=data.get("system_prompt"),
        )
        history.current_rounds = int(data.get("current_rounds", 0))
        history._context_history = [history._deserialize_messages(qa) for qa in data.get("context_history", [])]
        history._sliding_window = history._deserialize_messages(data.get("sliding_window", []))
        history._system_msg = {"role": "system", "content": history.system_prompt}
        history._rebuild_history()
        return history


@dataclass
class AuraSessionState:
    """Per-WebSocket session state for AURA streaming."""

    history: SessionHistory
    turn_frame_arrays: list[np.ndarray] = field(default_factory=list)
    session_id: str = ""
    cross_turn_penalty: Any = None
    pending_turn_video: dict[str, Any] | None = None
    # Frames present when ``pending_turn_video`` was frozen for the current
    # engine request. Later appends belong to the *next* turn and must survive
    # ``commit_turn`` so proactive visuals during TTS are not discarded.
    frozen_turn_frame_count: int = 0

    def append_turn_frame(self, frame: np.ndarray) -> None:
        self.turn_frame_arrays.append(np.asarray(frame))

    def freeze_turn_video(self, deferred_mm: dict[str, Any] | None) -> None:
        """Snapshot this turn's video and mark how many frames were included."""
        self.pending_turn_video = deferred_mm if isinstance(deferred_mm, dict) else None
        self.frozen_turn_frame_count = len(self.turn_frame_arrays)

    def commit_turn(
        self,
        *,
        response_text: str,
        request_id: str | None = None,
        video_fps: float = 2.0,
        max_frames_per_round: int = 16,
    ) -> None:
        """Finish an API-side turn: frames / penalty only.

        Dialogue ``SessionHistory`` lives in the stage-worker registry for both
        sync and async. Prefer ``aura2tts`` / ``aura2tts_async_chunk`` to commit
        pending turns; this method falls back to ``commit_session_turn`` when
        pending still exists (e.g. text-only sync that skips the TTS stage).
        """
        del video_fps, max_frames_per_round
        del request_id
        if self.session_id:
            commit_session_turn(self.session_id, response_text)

        self.pending_turn_video = None
        # Drop only frames that belonged to the frozen turn; keep frames that
        # arrived while generation/TTS was still running for the next trigger.
        keep_from = min(max(self.frozen_turn_frame_count, 0), len(self.turn_frame_arrays))
        if keep_from > 0:
            self.turn_frame_arrays = list(self.turn_frame_arrays[keep_from:])
        self.frozen_turn_frame_count = 0
        if self.cross_turn_penalty is not None:
            if is_effectively_silent(response_text):
                self.cross_turn_penalty.record(None)
            else:
                self.cross_turn_penalty.record(response_text)


def create_streaming_session(
    *,
    max_rounds: int = 45,
    num_rounds_keep: int = 30,
    pruning_enabled: bool = True,
    max_context_qas: int = 10,
    max_1qna_rounds: int = 4,
    system_prompt: str | None = None,
) -> AuraSessionState:
    """Create API-side WebSocket session state (frames / penalty / session_id).

    Prompt conversation history uses ``get_or_create_session_history``; the
    ``SessionHistory`` attached here is kept for registry compatibility only.
    """
    session_id = create_session_id()
    history = SessionHistory(
        max_rounds=max_rounds,
        num_rounds_keep=num_rounds_keep,
        pruning_enabled=pruning_enabled,
        max_context_qas=max_context_qas,
        max_1qna_rounds=max_1qna_rounds,
        system_prompt=system_prompt or DEFAULT_AURA_SYSTEM_PROMPT,
    )
    register_session(session_id, history)
    state = AuraSessionState(history=history, session_id=session_id)
    register_session_state(session_id, state)
    if _session_history_diag_enabled():
        logger.info(
            "AURA session_history create pid=%s session_id=%s",
            os.getpid(),
            session_id,
        )
    return state


# In-process registry: aura_session_id -> per-WebSocket session state.
_SESSION_LOCK = threading.Lock()
_SESSIONS: dict[str, SessionHistory] = {}
_SESSION_STATES: dict[str, AuraSessionState] = {}

# Prompt SessionHistory registry in the Orchestrator/head process. The current
# API and Orchestrator share a process, so disconnect cleanup reaches this
# registry. It remains process-local and is not durable across multiple heads.
_STAGE_WORKER_LOCK = threading.Lock()
_STAGE_WORKER_SESSIONS: dict[str, SessionHistory] = {}
_STAGE_PENDING_TURNS: dict[str, dict[str, Any]] = {}


def create_session_id() -> str:
    return f"aura-{uuid.uuid4().hex}"


def register_session(session_id: str, history: SessionHistory) -> None:
    with _SESSION_LOCK:
        _SESSIONS[session_id] = history


def register_session_state(session_id: str, state: AuraSessionState) -> None:
    with _SESSION_LOCK:
        _SESSION_STATES[session_id] = state
        _SESSIONS[session_id] = state.history


def get_session_state(session_id: str) -> AuraSessionState | None:
    with _SESSION_LOCK:
        return _SESSION_STATES.get(session_id)


def unregister_session(session_id: str) -> None:
    with _SESSION_LOCK:
        _SESSIONS.pop(session_id, None)
        _SESSION_STATES.pop(session_id, None)
    with _STAGE_WORKER_LOCK:
        _STAGE_WORKER_SESSIONS.pop(session_id, None)
        _STAGE_PENDING_TURNS.pop(session_id, None)


def clear_all_sessions() -> None:
    """Clear API session state and prompt SessionHistory registries (for tests)."""
    with _SESSION_LOCK:
        _SESSIONS.clear()
        _SESSION_STATES.clear()
    with _STAGE_WORKER_LOCK:
        _STAGE_WORKER_SESSIONS.clear()
        _STAGE_PENDING_TURNS.clear()


def get_or_create_session_history(
    session_id: str,
    *,
    system_prompt: str | None = None,
    max_rounds: int = 45,
    num_rounds_keep: int = 30,
    pruning_enabled: bool = True,
    max_context_qas: int = 10,
    max_1qna_rounds: int = 4,
) -> SessionHistory:
    """Return process-local prompt ``SessionHistory`` for this ``session_id``.

    Source of truth for both sync and async_chunk. Knobs must match client
    ``session.config`` / ``create_streaming_session`` (defaults 45/30, not
    ``SessionHistory``'s 20/15).
    """
    with _STAGE_WORKER_LOCK:
        history = _STAGE_WORKER_SESSIONS.get(session_id)
        if history is None:
            history = SessionHistory(
                max_rounds=max_rounds,
                num_rounds_keep=num_rounds_keep,
                pruning_enabled=pruning_enabled,
                max_context_qas=max_context_qas,
                max_1qna_rounds=max_1qna_rounds,
                system_prompt=system_prompt or DEFAULT_AURA_SYSTEM_PROMPT,
            )
            _STAGE_WORKER_SESSIONS[session_id] = history
            if _session_history_diag_enabled():
                logger.info(
                    "AURA session_history create pid=%s session_id=%s "
                    "max_rounds=%d num_rounds_keep=%d max_context_qas=%d",
                    os.getpid(),
                    session_id,
                    max_rounds,
                    num_rounds_keep,
                    max_context_qas,
                )
        return history


def get_session_history(session_id: str) -> SessionHistory | None:
    """Return prompt ``SessionHistory`` if it already exists for ``session_id``."""
    with _STAGE_WORKER_LOCK:
        return _STAGE_WORKER_SESSIONS.get(session_id)


def record_pending_turn(
    session_id: str,
    *,
    request_id: str,
    transcript: str,
    video_tuple: tuple[Any, dict[str, Any]] | None,
    deferred_mm: dict[str, Any] | None = None,
    aura_turn_video: Any = None,
    multi_modal_data: dict[str, Any] | None = None,
    had_vision: bool = False,
    mm_uuid: str | None = None,
) -> None:
    with _STAGE_WORKER_LOCK:
        _STAGE_PENDING_TURNS[session_id] = {
            "request_id": request_id,
            "transcript": transcript,
            "video_tuple": video_tuple,
            "deferred_mm": dict(deferred_mm) if isinstance(deferred_mm, dict) else None,
            "aura_turn_video": aura_turn_video,
            "multi_modal_data": dict(multi_modal_data) if isinstance(multi_modal_data, dict) else None,
            "had_vision": bool(had_vision),
            "mm_uuid": str(mm_uuid) if mm_uuid else None,
        }


def commit_session_turn(session_id: str, response_text: str) -> None:
    """Commit the finished user/assistant turn into prompt ``SessionHistory``.

    No-op when there is no pending turn (idempotent if ``aura2tts`` already
    committed and the API ``commit_turn`` fallback runs later).
    """
    with _STAGE_WORKER_LOCK:
        history = _STAGE_WORKER_SESSIONS.get(session_id)
        pending = _STAGE_PENDING_TURNS.pop(session_id, None)
    if history is None or pending is None:
        return
    transcript = str(pending.get("transcript", ""))
    video_tuple = pending.get("video_tuple")
    if video_tuple is None:
        from vllm_omni.model_executor.stage_input_processors.aura_omni import (
            _resolve_turn_video_tuple,
            video_tuple_from_multi_modal_data,
        )

        video_tuple = _resolve_turn_video_tuple(
            {
                "deferred_multi_modal_data": pending.get("deferred_mm"),
                "aura_turn_video": pending.get("aura_turn_video"),
            },
            pending.get("multi_modal_data") or {},
        )
        if video_tuple is None:
            video_tuple = video_tuple_from_multi_modal_data(pending.get("multi_modal_data"))
    normalized = normalize_assistant_text(response_text)
    mm_uuid = pending.get("mm_uuid") or history._pending_mm_uuid
    if video_tuple is not None or transcript:
        history.add_user_message(transcript, video_tuple=video_tuple, mm_uuid=mm_uuid)
    elif pending.get("had_vision"):
        from vllm_omni.model_executor.stage_input_processors.aura_omni import (
            AURA_VISION_PAD_TEXT,
        )

        history.add_user_message(AURA_VISION_PAD_TEXT)
    history.add_assistant_message(normalized)
    history._pending_mm_uuid = None
    # History fingerprint changed — invalidate pending-splice cache.
    history.clear_expand_cache()
    if _session_history_diag_enabled():
        n_vid, n_frames = _count_videos_in_messages(history.history)
        logger.info(
            "AURA session_history commit pid=%s session_id=%s rounds=%d "
            "history_videos=%d history_frames=%d video_mode=%s",
            os.getpid(),
            session_id,
            history.current_rounds,
            n_vid,
            n_frames,
            "all",
        )
