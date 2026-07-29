"""Tests for OmniInteract duplex request profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from vllm_omni.benchmarks.adapters.omniinteract import (
    OmniInteractModelSpecialConfig,
    load_model_special_config,
)
from vllm_omni.entrypoints.cli.benchmark.cli_args import add_omniinteract_cli_args

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_realtime_profile_is_default_and_supports_aliases():
    config = load_model_special_config("minicpmo_4_5")
    assert config.name == "minicpmo_4_5_realtime"
    assert config.system_prompt == "Streaming Omni Conversation."
    assert config.extra_body == {
        "modalities": ["text", "audio"],
        "chat_template_kwargs": {"use_tts_template": True},
    }

    aliased = load_model_special_config("realtime")
    assert aliased.name == "minicpmo_4_5_realtime"


def test_cli_accepts_unified_model_special_config():
    parser = argparse.ArgumentParser()
    add_omniinteract_cli_args(parser)

    args = parser.parse_args(["--omniinteract-model-special-config", "minicpmo_4_5_realtime"])

    assert args.omniinteract_model_special_config == "minicpmo_4_5_realtime"


def test_custom_profile_loads_from_json_file_and_deep_merges(tmp_path: Path):
    config_path = tmp_path / "model-special-config.json"
    config_path.write_text(
        json.dumps(
            {
                "preset": "minicpmo_4_5_realtime",
                "name": "custom-realtime",
                "extra_body": {
                    "chat_template_kwargs": {"custom_template_flag": True},
                    "custom_request_field": "value",
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_model_special_config(str(config_path))

    assert config.name == "custom-realtime"
    assert config.extra_body["chat_template_kwargs"] == {
        "use_tts_template": True,
        "custom_template_flag": True,
    }
    assert config.extra_body["custom_request_field"] == "value"


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ('{"preset": "unknown"}', "Unknown OmniInteract model preset"),
        ('{"preset": "minicpmo_4_5_realtime", "content_order": ["video"]}', "Unknown OmniInteract"),
        ('{"preset": "minicpmo_4_5_realtime", "extra_body": []}', "extra_body"),
    ],
)
def test_model_special_config_rejects_invalid_values(raw: str, match: str):
    with pytest.raises(ValueError, match=match):
        load_model_special_config(raw)


def test_typed_model_special_config_is_accepted():
    config = OmniInteractModelSpecialConfig(
        name="custom",
        system_prompt="Streaming Omni Conversation.",
        extra_body={"modalities": ["text", "audio"]},
    )
    loaded = load_model_special_config(config)
    assert loaded.name == "custom"
    assert loaded.extra_body == {"modalities": ["text", "audio"]}
