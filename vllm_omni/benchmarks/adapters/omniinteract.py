"""Model-neutral request configuration for the OmniInteract benchmark."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_SYSTEM_PROMPT = "You are a helpful multimodal assistant that understands video and audio."

_CONTENT_TYPES = frozenset({"audio", "video", "question"})
_CONFIG_KEYS = frozenset({"preset", "name", "content_order", "system_prompt", "extra_body"})
_PRESET_ALIASES = {
    "video": "video",
    "video_native": "video",
    "minicpm": "minicpmo_4_5",
    "minicpmo": "minicpmo_4_5",
    "minicpm-o-4.5": "minicpmo_4_5",
    "minicpmo_4_5": "minicpmo_4_5",
    "realtime": "minicpmo_4_5_realtime",
    "minicpmo_4_5_realtime": "minicpmo_4_5_realtime",
}


@dataclass(frozen=True)
class OmniInteractModelSpecialConfig:
    """Request-shaping fields that vary between served model protocols."""

    name: str
    content_order: tuple[str, ...]
    system_prompt: str
    extra_body: dict[str, Any]

    @property
    def requires_audio(self) -> bool:
        return "audio" in self.content_order

    @property
    def is_realtime(self) -> bool:
        return self.name.endswith("_realtime") or self.name == "minicpmo_4_5_realtime"


def _preset_data(name: str) -> dict[str, Any]:
    canonical_name = _PRESET_ALIASES.get(name.strip().lower())
    if canonical_name is None:
        supported = ", ".join(sorted(set(_PRESET_ALIASES.values())))
        raise ValueError(f"Unknown OmniInteract model preset {name!r}. Supported presets: {supported}.")

    common: dict[str, Any] = {
        "name": canonical_name,
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
    }
    if canonical_name == "video":
        return {
            **common,
            "content_order": ["video", "question"],
            "extra_body": {"mm_processor_kwargs": {"use_audio_in_video": True}},
        }
    if canonical_name in {"minicpmo_4_5", "minicpmo_4_5_realtime"}:
        return {
            **common,
            "content_order": ["audio", "video"],
            "extra_body": {
                "modalities": ["text", "audio"],
                "mm_processor_kwargs": {"use_audio_in_video": False},
                "chat_template_kwargs": {"use_tts_template": True},
            },
        }
    raise ValueError(f"Unsupported OmniInteract preset: {canonical_name}")


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_raw_config(spec: str | Mapping[str, Any] | None) -> dict[str, Any]:
    if spec is None or (isinstance(spec, str) and not spec.strip()):
        return {"preset": "video"}
    if isinstance(spec, Mapping):
        return deepcopy(dict(spec))

    value = spec.strip()
    if value.lower() in _PRESET_ALIASES:
        return {"preset": value}

    path = Path(value).expanduser()
    try:
        file_contents = path.read_text(encoding="utf-8")
    except OSError:
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "OmniInteract model special config must be a preset name, a JSON object, or a JSON file path."
            ) from exc
    else:
        try:
            loaded = json.loads(file_contents)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid OmniInteract model special config JSON file: {path}") from exc
    if not isinstance(loaded, dict):
        raise ValueError("OmniInteract model special config must decode to a JSON object.")
    return loaded


def load_model_special_config(
    spec: str | Mapping[str, Any] | OmniInteractModelSpecialConfig | None,
) -> OmniInteractModelSpecialConfig:
    """Load a built-in preset, inline JSON object, or JSON file."""
    if isinstance(spec, OmniInteractModelSpecialConfig):
        merged: dict[str, Any] = {
            "name": spec.name,
            "content_order": list(spec.content_order),
            "system_prompt": spec.system_prompt,
            "extra_body": deepcopy(spec.extra_body),
        }
    else:
        raw = _load_raw_config(spec)
        unknown_keys = sorted(set(raw) - _CONFIG_KEYS)
        if unknown_keys:
            raise ValueError(f"Unknown OmniInteract model special config fields: {unknown_keys}.")

        preset = raw.pop("preset", "video")
        if not isinstance(preset, str):
            raise ValueError("OmniInteract model special config 'preset' must be a string.")
        merged = _deep_merge(_preset_data(preset), raw)

    name = merged.get("name")
    system_prompt = merged.get("system_prompt")
    content_order = merged.get("content_order")
    extra_body = merged.get("extra_body")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("OmniInteract model special config 'name' must be a non-empty string.")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("OmniInteract model special config 'system_prompt' must be a non-empty string.")
    if not isinstance(content_order, list) or not content_order:
        raise ValueError("OmniInteract model special config 'content_order' must be a non-empty list.")
    if any(not isinstance(item, str) or item not in _CONTENT_TYPES for item in content_order):
        raise ValueError(
            f"OmniInteract model special config 'content_order' values must be in {sorted(_CONTENT_TYPES)}."
        )
    if len(set(content_order)) != len(content_order):
        raise ValueError("OmniInteract model special config 'content_order' cannot contain duplicates.")
    if "video" not in content_order:
        raise ValueError("OmniInteract model special config 'content_order' must include 'video'.")
    if not {"audio", "question"}.intersection(content_order):
        raise ValueError("OmniInteract model special config 'content_order' must include 'audio' or 'question'.")
    if not isinstance(extra_body, dict):
        raise ValueError("OmniInteract model special config 'extra_body' must be a JSON object.")

    return OmniInteractModelSpecialConfig(
        name=name.strip(),
        content_order=tuple(content_order),
        system_prompt=system_prompt.strip(),
        extra_body=deepcopy(extra_body),
    )
