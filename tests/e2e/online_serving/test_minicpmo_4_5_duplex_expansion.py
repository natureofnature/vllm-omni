# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Nightly lifecycle coverage for the MiniCPM-o 4.5 native-duplex API."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.e2e.online_serving.helpers.minicpmo_4_5_duplex import (
    SERVER_PARAMS,
    SOFT_INTERRUPT_SHA256,
    deploy_max_sessions,
    multi_session_args,
    realtime_url,
    resolve_ref_audio,
    validated_input_wav,
    validated_soft_interrupt_wav,
)
from tests.e2e.online_serving.run_minicpmo_realtime_duplex_multi_session import (
    run_lifecycle_probes,
)
from tests.e2e.online_serving.run_minicpmo_realtime_duplex_server_vad import run_server_vad_interrupt
from tests.e2e.online_serving.run_minicpmo_realtime_duplex_soft_interrupt import (
    run_soft_interrupt,
)
from tests.helpers.mark import hardware_test
from vllm_omni.benchmarks.data_modules.omniinteract_dataset import DEFAULT_OMNIINTERACT_REPO

pytestmark = [pytest.mark.full_model, pytest.mark.omni]

OMNIINTERACT_NIGHTLY_DATASET = f"{DEFAULT_OMNIINTERACT_REPO}@e195f75fe2666fcc5fe74f537ae49ca143a79969"


@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_duplex_admission_and_expiry_reaper(omni_server, tmp_path: Path) -> None:
    args = multi_session_args(
        omni_server=omni_server,
        input_wav=validated_input_wav(),
        ref_audio=resolve_ref_audio(),
        output_dir=tmp_path / "admission_expiry",
        response_required=False,
    )
    args.sessions = 1
    args.disconnect_session_index = None
    args.takeover_session_index = None
    args.expire_session_index = 0
    args.verify_admission_limit = deploy_max_sessions()
    result = asyncio.run(run_lifecycle_probes(args))

    assert result["ok"] is True, json.dumps(result, ensure_ascii=False, indent=2)
    assert result["expiry"]["ok"] is True
    assert result["expiry"]["error_code"] == "session_resume_expired"
    assert result["admission"]["ok"] is True
    assert result["admission"]["overflow_error_code"] == "resource_exhausted"


# CUDA-only for now: this contract asserts the model speaks while the user is
# still talking, which only holds when the duplex pipeline sustains real-time
# throughput. The current NPU stack runs several times slower than real time,
# so it never reaches a mid-stream decision point.
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_duplex_soft_interrupt(omni_server, tmp_path: Path) -> None:
    input_wav = validated_soft_interrupt_wav()
    result = asyncio.run(
        run_soft_interrupt(
            SimpleNamespace(
                url=realtime_url(omni_server),
                model=omni_server.model,
                input_wav=str(input_wav),
                ref_audio=str(resolve_ref_audio()),
                output_dir=str(tmp_path / "soft_interrupt"),
                summary_output=None,
                chunk_ms=200,
                timeout_s=180.0,
                require_audio=True,
                no_realtime_pacing=False,
                validation_mode="response-required",
                min_responses=2,
                min_audio_deltas_per_response=2,
                input_sha256=SOFT_INTERRUPT_SHA256,
                expect_followup_response_substring=None,
            )
        )
    )

    assert result["ok"] is True, json.dumps(result, ensure_ascii=False, indent=2)
    assert result["error_count"] == 0
    assert result["response_lifecycle_ok"] is True
    assert result["response_audio_contract_ok"] is True
    assert result["followup_response_transcript_ok"] is True


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_duplex_server_vad_hard_interrupt(omni_server) -> None:
    result = asyncio.run(
        run_server_vad_interrupt(
            SimpleNamespace(
                url=realtime_url(omni_server),
                model=omni_server.model,
                input_wav=str(validated_input_wav()),
                interrupt_wav=str(validated_soft_interrupt_wav()),
                ref_audio=str(resolve_ref_audio()),
                chunk_ms=200,
                timeout_s=180.0,
            )
        )
    )
    assert result["ok"] is True, json.dumps(result, ensure_ascii=False, indent=2)


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.skipif(
    os.environ.get("VLLM_OMNI_RUN_OMNIINTERACT_E2E") != "1",
    reason="enable the real-time OmniInteract dataset E2E",
)
@pytest.mark.parametrize("subset", ("1q1a", "1q1a_math", "1qna"))
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_minicpmo_4_5_omniinteract_nightly(
    omni_server,
    tmp_path: Path,
    subset: str,
) -> None:
    dataset_path = os.environ.get("OMNIINTERACT_ROOT", "").strip() or OMNIINTERACT_NIGHTLY_DATASET
    run_root = tmp_path / subset
    artifact_root = run_root / "artifacts"
    result_filename = "benchmark_result.json"
    vllm = shutil.which("vllm")
    assert vllm is not None, "Could not find `vllm` on PATH"
    subprocess.run(
        [
            vllm,
            "bench",
            "serve",
            "--omni",
            "--trust-remote-code",
            "--host",
            omni_server.host,
            "--port",
            str(omni_server.port),
            "--backend",
            "openai-realtime-duplex",
            "--endpoint",
            "/v1/realtime",
            "--model",
            omni_server.model,
            "--dataset-name",
            "omniinteract",
            "--dataset-path",
            dataset_path,
            "--omniinteract-subsets",
            subset,
            "--omniinteract-ref-audio",
            str(resolve_ref_audio()),
            "--omniinteract-output-dir",
            str(artifact_root),
            "--omniinteract-max-video-duration-s",
            "600",
            "--num-warmups",
            "0",
            "--num-prompts",
            "4",
            "--max-concurrency",
            "2",
            "--request-rate",
            "inf",
            "--disable-shuffle",
            "--save-result",
            "--result-dir",
            str(run_root),
            "--result-filename",
            result_filename,
        ],
        check=True,
    )

    benchmark_result = json.loads((run_root / result_filename).read_text())
    artifact_summary = benchmark_result["omniinteract"]
    assert artifact_summary["artifacts_complete"] is True
    assert (artifact_summary["total"], artifact_summary["success"], artifact_summary["failed"]) == (4, 4, 0)

    batch_summary = json.loads((artifact_root / "batch_summary.json").read_text())
    assert (batch_summary["total"], batch_summary["success"], batch_summary["failed"]) == (4, 4, 0)
    manifest_rows = [
        json.loads(row) for row in (artifact_root / "official_eval_manifest.jsonl").read_text().splitlines()
    ]
    assert len(manifest_rows) == batch_summary["eligible_for_official_eval"]
    assert {row["video"] for row in manifest_rows} == {
        result["video"] for result in batch_summary["results"] if result["eligible_for_official_eval"]
    }
    assert {result["subset"] for result in batch_summary["results"]} == {subset}
    assert len({result["video"] for result in batch_summary["results"]}) == 4

    for result in batch_summary["results"]:
        assert result["success"] is True
        assert result["input_audio_chunks"] > 0
        assert result["input_video_frames"] > 0
        sample_root = Path(result["output_dir"])
        assert all(
            (sample_root / name).is_file()
            for name in (".done", "output.wav", "wav_transcript.json", "events.json", "result.json")
        )
        with wave.open(str(sample_root / "output.wav"), "rb") as output_wav:
            assert (
                output_wav.getframerate(),
                output_wav.getnchannels(),
                output_wav.getsampwidth(),
                output_wav.getcomptype(),
            ) == (24_000, 1, 2, "NONE")
        transcript = json.loads((sample_root / "wav_transcript.json").read_text())
        assert isinstance(transcript["chunks"], list)
        assert transcript["timestamp_semantics"]
        published_result = json.loads((sample_root / "result.json").read_text())
        assert published_result["video"] == result["video"]
        assert published_result["success"] is True
        events = json.loads((sample_root / "events.json").read_text())
        created = {event["response"]["id"] for event in events if event.get("type") == "response.created"}
        done = {event["response"]["id"] for event in events if event.get("type") == "response.done"}
        assert created == done
        assert len(created) == result["responses"]
        if not result["eligible_for_official_eval"]:
            assert result["official_eval_ineligible_reasons"]
