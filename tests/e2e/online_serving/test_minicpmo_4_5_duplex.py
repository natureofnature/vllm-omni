# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CI coverage for the MiniCPM-o 4.5 native-duplex Realtime API."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
import wave
from pathlib import Path

import pytest
import websockets

from tests.e2e.accuracy.qwen3_omni.qwen3_omni_acc_bench_core import (
    find_vllm_cli,
    run_vllm_bench_subprocess,
)
from tests.e2e.online_serving.helpers.minicpmo_4_5_duplex import (
    SERVER_PARAMS,
    demo_args,
    multi_session_args,
    realtime_url,
    resolve_ref_audio,
    validated_input_wav,
)
from tests.e2e.online_serving.helpers.minicpmo_realtime_duplex_scenarios import (
    _ref_audio_data_url,
    run_demo,
)
from tests.e2e.online_serving.run_minicpmo_realtime_duplex_multi_session import (
    run_multi_session,
)
from tests.helpers.mark import hardware_test
from vllm_omni.benchmarks.data_modules.omniinteract_dataset import DEFAULT_OMNIINTERACT_REPO
from vllm_omni.experimental.fullduplex.client import build_realtime_url

pytestmark = pytest.mark.omni


def _assert_request_metrics(metrics: object, *, expected_count: int) -> None:
    assert isinstance(metrics, list)
    assert len(metrics) == expected_count
    for request_index, request in enumerate(metrics):
        assert isinstance(request["session_id"], str)
        assert request["request_index"] == request_index
        assert isinstance(request["response_id"], str)
        assert request["ttft_ms"] is not None and request["ttft_ms"] >= 0
        assert request["ttfp_ms"] >= 0
        assert request["rtf"] is not None and request["rtf"] >= 0
        assert request["audio_generation_ms"] >= 0
        assert request["audio_duration_ms"] > 0


def _assert_session_metrics(metrics: object, *, expected_count: int) -> None:
    assert isinstance(metrics, dict)
    assert isinstance(metrics["session_id"], str)
    assert metrics["audio_turn_count"] == expected_count
    assert metrics["mean_ttft_ms"] is not None and metrics["mean_ttft_ms"] >= 0
    assert metrics["mean_ttfp_ms"] is not None and metrics["mean_ttfp_ms"] >= 0
    assert metrics["mean_rtf"] is not None and metrics["mean_rtf"] >= 0


async def _receive_protocol_events(ws, required_types: set[str], *, timeout_s: float) -> list[dict[str, object]]:
    async def receive() -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        seen: set[str] = set()
        while not required_types.issubset(seen):
            raw = await ws.recv()
            if not isinstance(raw, str):
                continue
            event = json.loads(raw)
            if not isinstance(event, dict):
                continue
            events.append(event)
            event_type = event.get("type")
            if event_type == "error":
                raise AssertionError(f"WebSocket protocol smoke received an error: {event}")
            if isinstance(event_type, str):
                seen.add(event_type)
        return events

    return await asyncio.wait_for(receive(), timeout=timeout_s)


async def _run_protocol_smoke(*, url: str, model: str, ref_audio: Path) -> list[dict[str, object]]:
    session_id = f"duplex-ci-protocol-{uuid.uuid4().hex}"
    websocket_url = build_realtime_url(url, model, autostart=False, session_id=session_id)
    async with websockets.connect(websocket_url, max_size=64 * 1024 * 1024) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "session_id": session_id,
                        "model": model,
                        "modalities": ["audio", "text"],
                        "ref_audio": _ref_audio_data_url(str(ref_audio)),
                        "extra_body": {"minicpmo45_native_duplex": True},
                    },
                }
            )
        )
        events = await _receive_protocol_events(
            ws,
            {"session.created", "session.updated"},
            timeout_s=60,
        )
        await ws.send(json.dumps({"type": "session.close"}))
        events.extend(await _receive_protocol_events(ws, {"session.closed"}, timeout_s=60))
    return events


@pytest.mark.core_model
@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_duplex_websocket_protocol_smoke(omni_server) -> None:
    ref_audio = resolve_ref_audio()
    events = asyncio.run(
        _run_protocol_smoke(
            url=realtime_url(omni_server),
            model=omni_server.model,
            ref_audio=ref_audio,
        )
    )
    event_types = [event.get("type") for event in events]
    assert "session.created" in event_types
    assert "session.updated" in event_types
    assert event_types[-1] == "session.closed"


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_duplex_single_session_response_required(omni_server, tmp_path: Path) -> None:
    args = demo_args(
        omni_server=omni_server,
        input_wav=validated_input_wav(),
        ref_audio=resolve_ref_audio(),
        output_dir=tmp_path / "single_session",
    )
    args.turns = 2
    # Every turn replays the same active-speech window as the first one. The default
    # shorter follow-up window is a different mid-utterance slice, which the native
    # duplex model may legitimately answer with "listen" instead of a response.
    args.turn_duration_ms = [args.first_turn_ms] * args.turns
    result = asyncio.run(run_demo(args))
    assert result["ok"] is True
    assert result["audio_delta_count"] > 0
    assert result["done_count"] == 2
    assert result["error_count"] == 0
    assert result["all_audio_responses_have_transcript"] is True
    assert result["transcript_delta_done_ok"] is True
    _assert_request_metrics(result["request_metrics"], expected_count=2)
    _assert_session_metrics(result["session_metrics"], expected_count=2)


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_duplex_two_sessions_resume_and_takeover(omni_server, tmp_path: Path) -> None:
    result = asyncio.run(
        run_multi_session(
            multi_session_args(
                omni_server=omni_server,
                input_wav=validated_input_wav(),
                ref_audio=resolve_ref_audio(),
                output_dir=tmp_path / "multi_session",
                response_required=True,
            )
        )
    )
    assert result["ok"] is True
    assert result["session_count"] == 2
    assert result["resume"]["ok"] is True
    assert result["takeover"]["ok"] is True
    assert not result["failures"]
    assert all(session["audio_delta_count"] > 0 for session in result["sessions"])
    assert all(session["done_count"] == 1 for session in result["sessions"])
    assert all(session["error_count"] == 0 for session in result["sessions"])
    for session in result["sessions"]:
        _assert_request_metrics(session["request_metrics"], expected_count=1)
        _assert_session_metrics(session["session_metrics"], expected_count=1)


@pytest.mark.full_model
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
    dataset_path = os.environ.get("OMNIINTERACT_ROOT", "").strip() or DEFAULT_OMNIINTERACT_REPO
    run_root = tmp_path / subset
    artifact_root = run_root / "artifacts"
    result_filename = "benchmark_result.json"
    run_vllm_bench_subprocess(
        find_vllm_cli(),
        [
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
            "--omniinteract-require-response",
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
    )

    benchmark_result = json.loads((run_root / result_filename).read_text())
    artifact_summary = benchmark_result["omniinteract"]
    assert artifact_summary["artifacts_complete"] is True
    assert (artifact_summary["total"], artifact_summary["success"], artifact_summary["failed"]) == (4, 4, 0)

    batch_summary = json.loads((artifact_root / "batch_summary.json").read_text())
    assert (batch_summary["total"], batch_summary["success"], batch_summary["failed"]) == (4, 4, 0)
    assert batch_summary["eligible_for_official_eval"] + batch_summary["successful_but_ineligible"] == 4
    manifest_rows = (artifact_root / "official_eval_manifest.jsonl").read_text().splitlines()
    assert len(manifest_rows) == batch_summary["eligible_for_official_eval"]
    assert {result["subset"] for result in batch_summary["results"]} == {subset}
    assert len({result["video"] for result in batch_summary["results"]}) == 4

    for result in batch_summary["results"]:
        assert result["success"] is True
        assert result["input_audio_chunks"] > 0
        assert result["input_video_frames"] > 0
        assert result["responses"] > 0
        sample_root = Path(result["output_dir"])
        assert all(
            (sample_root / name).is_file()
            for name in (".done", "output.wav", "wav_transcript.json", "events.json", "result.json")
        )
        with wave.open(str(sample_root / "output.wav"), "rb") as output_wav:
            assert output_wav.getnframes() > 0
            assert (
                output_wav.getframerate(),
                output_wav.getnchannels(),
                output_wav.getsampwidth(),
                output_wav.getcomptype(),
            ) == (24_000, 1, 2, "NONE")
        assert json.loads((sample_root / "wav_transcript.json").read_text())["chunks"]
        if not result["eligible_for_official_eval"]:
            assert result["official_eval_ineligible_reasons"]
