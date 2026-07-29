"""Tests for model-neutral OmniInteract request profiles."""

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


def test_minicpm_profile_uses_separate_audio_video_and_tts_template():
    config = load_model_special_config("minicpmo_4_5")

    assert config.content_order == ("audio", "video")
    assert config.requires_audio
    assert config.is_realtime is False
    assert config.extra_body == {
        "modalities": ["text", "audio"],
        "mm_processor_kwargs": {"use_audio_in_video": False},
        "chat_template_kwargs": {"use_tts_template": True},
    }


def test_realtime_profile_is_marked_as_realtime():
    config = load_model_special_config("minicpmo_4_5_realtime")

    assert config.is_realtime is True
    assert config.content_order == ("audio", "video")


def test_cli_accepts_unified_model_special_config():
    parser = argparse.ArgumentParser()
    add_omniinteract_cli_args(parser)

    args = parser.parse_args(["--omniinteract-model-special-config", "minicpmo_4_5"])

    assert args.omniinteract_model_special_config == "minicpmo_4_5"


def test_custom_profile_loads_from_json_file_and_deep_merges(tmp_path: Path):
    config_path = tmp_path / "model-special-config.json"
    config_path.write_text(
        json.dumps(
            {
                "preset": "video",
                "name": "custom-video",
                "extra_body": {
                    "mm_processor_kwargs": {"custom_processor_flag": True},
                    "custom_request_field": "value",
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_model_special_config(str(config_path))

    assert config.name == "custom-video"
    assert config.content_order == ("video", "question")
    assert config.extra_body["mm_processor_kwargs"] == {
        "use_audio_in_video": True,
        "custom_processor_flag": True,
    }
    assert config.extra_body["custom_request_field"] == "value"


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ('{"preset": "unknown"}', "Unknown OmniInteract model preset"),
        ('{"preset": "video", "content_order": ["video", "bad"]}', "content_order"),
        ('{"preset": "video", "content_order": ["video"]}', "audio.*question"),
        ('{"preset": "video", "extra_body": []}', "extra_body"),
    ],
)
def test_model_special_config_rejects_invalid_values(raw: str, match: str):
    with pytest.raises(ValueError, match=match):
        load_model_special_config(raw)


def test_typed_model_special_config_is_validated():
    config = OmniInteractModelSpecialConfig(
        name="invalid",
        content_order=("video", "unknown"),
        system_prompt="system",
        extra_body={},
    )

    with pytest.raises(ValueError, match="content_order"):
        load_model_special_config(config)
