"""Model-neutral request configuration for OmniInteract full-duplex benchmark."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_SYSTEM_PROMPT = "Streaming Omni Conversation."

_CONFIG_KEYS = frozenset({"preset", "name", "system_prompt", "extra_body"})
_PRESET_ALIASES = {
    "realtime": "minicpmo_4_5_realtime",
    "minicpm": "minicpmo_4_5_realtime",
    "minicpmo": "minicpmo_4_5_realtime",
    "minicpm-o-4.5": "minicpmo_4_5_realtime",
    "minicpmo_4_5": "minicpmo_4_5_realtime",
    "minicpmo_4_5_realtime": "minicpmo_4_5_realtime",
}


@dataclass(frozen=True)
class OmniInteractModelSpecialConfig:
    """Session-level duplex request fields."""

    name: str
    system_prompt: str
    extra_body: dict[str, Any]


def _preset_data(name: str) -> dict[str, Any]:
    canonical_name = _PRESET_ALIASES.get(name.strip().lower())
    if canonical_name is None:
        supported = ", ".join(sorted(set(_PRESET_ALIASES.values())))
        raise ValueError(f"Unknown OmniInteract model preset {name!r}. Supported presets: {supported}.")
    return {
        "name": canonical_name,
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "extra_body": {
            "modalities": ["text", "audio"],
            "chat_template_kwargs": {"use_tts_template": True},
        },
    }


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
        return {"preset": "minicpmo_4_5_realtime"}
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
    """Load a built-in realtime preset, inline JSON object, or JSON file."""
    if isinstance(spec, OmniInteractModelSpecialConfig):
        merged: dict[str, Any] = {
            "name": spec.name,
            "system_prompt": spec.system_prompt,
            "extra_body": deepcopy(spec.extra_body),
        }
    else:
        raw = _load_raw_config(spec)
        unknown_keys = sorted(set(raw) - _CONFIG_KEYS)
        if unknown_keys:
            raise ValueError(f"Unknown OmniInteract model special config fields: {unknown_keys}.")
        preset = raw.pop("preset", "minicpmo_4_5_realtime")
        if not isinstance(preset, str):
            raise ValueError("OmniInteract model special config 'preset' must be a string.")
        merged = _deep_merge(_preset_data(preset), raw)

    name = merged.get("name")
    system_prompt = merged.get("system_prompt")
    extra_body = merged.get("extra_body")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("OmniInteract model special config 'name' must be a non-empty string.")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("OmniInteract model special config 'system_prompt' must be a non-empty string.")
    if not isinstance(extra_body, dict):
        raise ValueError("OmniInteract model special config 'extra_body' must be a JSON object.")

    return OmniInteractModelSpecialConfig(
        name=name.strip(),
        system_prompt=system_prompt.strip(),
        extra_body=deepcopy(extra_body),
    )
