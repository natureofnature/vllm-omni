# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import shutil
import subprocess
import tarfile
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
from vllm.benchmarks.lib.endpoint_request_func import RequestFuncInput

from vllm_omni.benchmarks import omniinteract as oi
from vllm_omni.benchmarks import serve as benchmark_serve
from vllm_omni.benchmarks.data_modules import omniinteract_dataset as data
from vllm_omni.benchmarks.patch import patch as benchmark_patch
from vllm_omni.entrypoints.cli.benchmark.cli_args import add_omniinteract_cli_args, preprocess_serve_args
from vllm_omni.entrypoints.cli.benchmark.serve import OmniBenchmarkServingSubcommand
from vllm_omni.experimental.fullduplex.client import RealtimeEventCollector, stream_pcm16_chunks
from vllm_omni.utils.tracking_parser import TrackingArgumentParser

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.benchmark]


def _case(tmp_path: Path, *, subset: str = "1q1a", name: str = "video.mp4") -> data.OmniInteractCase:
    video = tmp_path / name
    annotation = tmp_path / f"{Path(name).stem}.json"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.touch()
    annotation.write_text("{}")
    return data.OmniInteractCase(subset, name, video, annotation, "multi_turn")


def _collector(*events: tuple[dict[str, object], float]) -> RealtimeEventCollector:
    collector = RealtimeEventCollector()
    for event, received_at in events:
        collector.add(event, received_at_s=received_at)
    return collector


def _created(response_id: str) -> dict[str, object]:
    return {"type": "response.created", "response": {"id": response_id}}


def _done(response_id: str, status: str = "completed") -> dict[str, object]:
    return {"type": "response.done", "response": {"id": response_id, "status": status}}


def _listen(*, model_listen: bool = True, buffering: bool = False, reason: str | None = None) -> dict[str, object]:
    metadata: dict[str, object] = {
        "type": "response.listen",
        "model_listen": model_listen,
    }
    if buffering:
        metadata["buffering"] = True
    if reason is not None:
        metadata["reason"] = reason
    return {"type": "response.listen", "response": {"metadata": metadata}}


def _audio(response_id: str, frames: int = 2400, value: int = 1) -> dict[str, object]:
    return {
        "type": "response.audio.delta",
        "response_id": response_id,
        "format": "pcm16",
        "sample_rate_hz": 24_000,
        "delta": base64.b64encode(bytes((value, 0)) * frames).decode("ascii"),
    }


def _transcript(response_id: str, text: str = "hello") -> dict[str, object]:
    return {"type": "response.audio_transcript.delta", "response_id": response_id, "delta": text}


def _write_dataset(root: Path) -> Path:
    data_root = root / "data"
    for subset in ("1q1a", "1q1a_math"):
        subset_root = data_root / subset
        (subset_root / "videos").mkdir(parents=True)
        (subset_root / "annotations").mkdir()
        (subset_root / "videos" / f"{subset}.mp4").touch()
        (subset_root / "annotations" / f"{subset}.json").write_text("{}")
        (subset_root / "video_json_map.json").write_text(
            json.dumps(
                {
                    "entries": [
                        {
                            "video": f"videos/{subset}.mp4",
                            "annotation": f"annotations/{subset}.json",
                            "scene_type": "multi_turn",
                        }
                    ]
                }
            )
        )
    one_to_many = data_root / "1qna"
    (one_to_many / "videos_bench" / "nested").mkdir(parents=True)
    (one_to_many / "annotations" / "nested").mkdir(parents=True)
    (one_to_many / "videos_bench" / "nested" / "guide.mp4").touch()
    (one_to_many / "annotations" / "nested" / "guide.json").write_text("{}")
    return data_root


def test_discovers_all_official_layouts_without_oversampling(tmp_path: Path):
    _write_dataset(tmp_path)

    cases = data.discover_omniinteract_cases(
        tmp_path,
        data.OMNIINTERACT_SUBSETS,
        num_prompts=0,
        disable_shuffle=True,
    )

    assert [case.subset for case in cases] == ["1q1a", "1q1a_math", "1qna"]
    assert cases[-1].video_rel == "videos_bench/nested/guide.mp4"


def test_num_prompts_is_total_across_selected_subsets(tmp_path: Path):
    _write_dataset(tmp_path)

    cases = data.discover_omniinteract_cases(
        tmp_path,
        data.OMNIINTERACT_SUBSETS,
        num_prompts=2,
        disable_shuffle=True,
    )

    assert [case.subset for case in cases] == ["1q1a", "1q1a_math"]


def test_dataset_rejects_duplicate_subsets_and_clamps_missing_capacity(tmp_path: Path, caplog):
    root = _write_dataset(tmp_path)

    with pytest.raises(ValueError, match="must not contain duplicates"):
        data.discover_omniinteract_cases(root, ("1q1a", "1q1a"), num_prompts=1)
    cases = data.discover_omniinteract_cases(root, ("1q1a",), num_prompts=1000)

    assert len(cases) == 1
    assert "only 1 are available; using all cases" in caplog.text


def test_dataset_accepts_one_to_many_only_tree(tmp_path: Path):
    root = _write_dataset(tmp_path)
    (root / "1q1a").rename(tmp_path / "unused-1q1a")
    (root / "1q1a_math").rename(tmp_path / "unused-1q1a-math")

    cases = data.discover_omniinteract_cases(root, ("1qna",), num_prompts=0)

    assert [case.subset for case in cases] == ["1qna"]


def test_dataset_rejects_an_empty_requested_subset(tmp_path: Path):
    root = _write_dataset(tmp_path)
    (root / "1q1a" / "video_json_map.json").write_text(json.dumps({"entries": []}))

    with pytest.raises(ValueError, match="requested subset '1q1a'"):
        data.discover_omniinteract_cases(root, ("1q1a", "1qna"), num_prompts=0)


@pytest.mark.parametrize("field", ["video", "annotation"])
def test_dataset_mapping_rejects_path_escape(tmp_path: Path, field: str):
    root = tmp_path / "1q1a"
    root.mkdir()
    outside = tmp_path / "outside.mp4"
    outside.touch()
    row = {"video": "inside.mp4", "annotation": "inside.json"}
    (root / "inside.mp4").touch()
    (root / "inside.json").write_text("{}")
    row[field] = "../outside.mp4"
    (root / "video_json_map.json").write_text(json.dumps({"entries": [row]}))

    with pytest.raises(ValueError, match=f"Unsafe OmniInteract {field} path"):
        data.discover_omniinteract_cases(tmp_path, ("1q1a",), num_prompts=1)


def test_dataset_mapping_rejects_symlink_escape(tmp_path: Path):
    root = tmp_path / "1q1a"
    root.mkdir()
    outside = tmp_path / "outside.mp4"
    outside.touch()
    (root / "link.mp4").symlink_to(outside)
    (root / "inside.json").write_text("{}")
    (root / "video_json_map.json").write_text(
        json.dumps({"entries": [{"video": "link.mp4", "annotation": "inside.json"}]})
    )

    with pytest.raises(ValueError, match="Unsafe OmniInteract video path"):
        data.discover_omniinteract_cases(tmp_path, ("1q1a",), num_prompts=1)


def test_archive_rejects_parent_traversal(tmp_path: Path):
    payload = b"escape"
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as handle:
        member = tarfile.TarInfo("../escape.txt")
        member.size = len(payload)
        handle.addfile(member, io.BytesIO(payload))
    archive.seek(0)

    with tarfile.open(fileobj=archive, mode="r") as handle, pytest.raises(ValueError, match="Unsafe path"):
        data._safe_extract(handle, tmp_path / "extract")


def test_archive_extraction_is_atomically_published_across_shared_cache_users(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source_root = _write_dataset(tmp_path / "source")
    archive = tmp_path / "data.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(source_root, arcname="data")
    target = tmp_path / "cache" / "dataset"
    extraction_count = 0
    original_extract = data._safe_extract
    both_extracting = threading.Barrier(2)

    def slow_extract(handle: tarfile.TarFile, directory: Path) -> None:
        nonlocal extraction_count
        extraction_count += 1
        both_extracting.wait(timeout=1)
        original_extract(handle, directory)

    monkeypatch.setattr(data, "_safe_extract", slow_extract)
    with ThreadPoolExecutor(max_workers=2) as executor:
        roots = list(executor.map(lambda _: data._extract_archive(archive, target), range(2)))

    published_root = target / data._archive_fingerprint(archive) / "data"
    assert roots == [published_root, published_root]
    assert extraction_count == 2
    assert (roots[0] / "1q1a" / "video_json_map.json").is_file()
    assert not list(target.glob(".tmp-*"))


def test_archive_fingerprint_is_a_portable_directory_name(tmp_path: Path) -> None:
    archive = tmp_path / "data.tar.gz"
    archive.write_bytes(b"archive")

    fingerprint = data._archive_fingerprint(archive)

    assert fingerprint.replace("-", "").isdigit()


def test_downloads_dataset_archive_through_vllm_hf_filesystem(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source_root = _write_dataset(tmp_path / "source")
    archive = tmp_path / "data.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(source_root, arcname="data")
    downloads: list[str] = []

    class FakeFilesystem:
        def get_file(self, remote: str, local: str) -> None:
            downloads.append(remote)
            shutil.copyfile(archive, local)

    monkeypatch.setenv("HF_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(data, "hf_fs", FakeFilesystem)

    root = data.resolve_omniinteract_root(None, "owner/dataset")

    assert (root / "1q1a").is_dir()
    assert downloads == ["datasets/owner/dataset/data.tar.gz"]


def test_media_command_timeout_is_bounded(monkeypatch: pytest.MonkeyPatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("ffprobe", 3)

    monkeypatch.setattr(subprocess, "run", timeout)

    with pytest.raises(TimeoutError, match="timed out after 3s"):
        oi._run_media_command(["ffprobe"], timeout_s=3)


def test_prepare_media_bounds_frame_extraction(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    video = tmp_path / "video.mp4"
    video.touch()
    timeouts: list[float] = []

    def fake_media_command(command, *, timeout_s, text=False):
        timeouts.append(timeout_s)
        if command[0] == "ffprobe":
            return subprocess.CompletedProcess(command, 0, stdout="1.2", stderr="")
        if "-vn" in command:
            return subprocess.CompletedProcess(command, 0, stdout=bytes(32_000), stderr=b"")
        output = Path(command[-1].replace("%06d", "000001"))
        output.write_bytes(b"jpeg")
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(oi, "_run_media_command", fake_media_command)

    duration, pcm, frames = oi.prepare_media(video, 1.0, timeout_s=7)

    assert duration == 1.2
    assert len(pcm) == 64_000
    assert frames == [base64.b64encode(b"jpeg").decode("ascii"), None]
    assert timeouts == [7, 7, 7]


@pytest.mark.asyncio
async def test_case_requires_reference_audio(tmp_path: Path):
    with pytest.raises(ValueError, match="ref_audio is required"):
        await oi.run_omniinteract_case(
            _case(tmp_path),
            oi.OmniInteractBenchmarkConfig(),
            request_index=0,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["timeout_s", "media_timeout_s"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
async def test_case_rejects_non_finite_timeouts(tmp_path: Path, field: str, value: float):
    config = oi.OmniInteractBenchmarkConfig(ref_audio="unused.wav", **{field: value})

    with pytest.raises(ValueError, match=f"{field} must be finite and positive"):
        await oi.run_omniinteract_case(_case(tmp_path), config, request_index=0)


@pytest.mark.parametrize(
    "events, error",
    [
        ([(_created("duplicate"), 1.0), (_created("duplicate"), 1.1)], "duplicate response.created"),
        (
            [(_created("a"), 1.0), (_created("b"), 1.1), (_done("a"), 1.2), (_done("a"), 1.3)],
            "duplicate response.done",
        ),
        ([(_done("orphan"), 1.0)], "without response.created"),
        ([(_created("failed"), 1.0), (_done("failed", "failed"), 1.1)], "reports failure"),
        (
            [
                (_created("nested-failure"), 1.0),
                (
                    {
                        "type": "response.done",
                        "response": {"id": "nested-failure", "status_details": {"type": "failed"}},
                    },
                    1.1,
                ),
            ],
            "reports failure",
        ),
    ],
)
def test_response_ledger_rejects_identity_and_status_errors(events, error: str):
    with pytest.raises(ValueError, match=error):
        oi.response_ledger(_collector(*events))


def test_artifacts_clip_audio_and_transcript_to_official_video_horizon(tmp_path: Path):
    case = _case(tmp_path)
    collector = _collector(
        (_created("r1"), 1.0),
        (_audio("r1", frames=4_800), 1.9),
        (_transcript("r1"), 1.95),
        (_done("r1"), 2.2),
        (_created("r2"), 3.3),
        (_audio("r2"), 3.4),
        (_transcript("r2", "too late"), 3.45),
        (_done("r2"), 3.5),
    )
    result = oi.OmniInteractCaseResult("1q1a", str(case.video_path), "unused")

    summary = oi.write_success_artifacts(
        tmp_path / "output",
        case,
        collector,
        stream_start=1.0,
        video_duration_s=1.25,
        require_response=False,
        result=result,
    )

    output_dir = Path(summary["output_dir"])
    with wave.open(str(output_dir / "output.wav"), "rb") as handle:
        assert handle.getframerate() == 24_000
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getnframes() == 2 * handle.getframerate()
    transcript = json.loads((output_dir / "wav_transcript.json").read_text())
    assert transcript["text"] == "hello"
    assert transcript["chunks"] == [{"response_id": "r1", "text": "hello", "timestamp": [0.9, 1.1]}]
    events = json.loads((output_dir / "events.json").read_text())
    audio_event = next(event for event in events if event["type"] == "response.audio.delta")
    transcript_event = next(event for event in events if event["type"] == "response.audio_transcript.delta")
    assert "delta" not in audio_event
    assert transcript_event["delta"] == "hello"
    assert summary["audio_clipped_bytes"] == 4_800
    assert (output_dir / ".done").is_file()
    assert not (output_dir / ".failed.json").exists()


def test_overlapping_responses_are_serialized_without_overwrite(tmp_path: Path):
    case = _case(tmp_path)
    collector = _collector(
        (_created("r1"), 1.0),
        (_audio("r1", frames=4_800, value=1), 1.1),
        (_created("r2"), 1.11),
        (_audio("r2", frames=2_400, value=2), 1.15),
        (_done("r1"), 1.31),
        (_done("r2"), 1.41),
    )
    result = oi.OmniInteractCaseResult("1q1a", str(case.video_path), "unused")

    summary = oi.write_success_artifacts(
        tmp_path / "output",
        case,
        collector,
        stream_start=1.0,
        video_duration_s=1.0,
        require_response=False,
        result=result,
    )

    with wave.open(str(Path(summary["output_dir"]) / "output.wav"), "rb") as handle:
        pcm = handle.readframes(handle.getnframes())
    first_offset = round(0.1 * 24_000) * 2
    assert pcm[first_offset : first_offset + 9_600] == bytes((1, 0)) * 4_800
    assert pcm[first_offset + 9_600 : first_offset + 14_400] == bytes((2, 0)) * 2_400
    assert summary["audio_clipped_bytes"] == 0


def test_output_dir_flattens_backslash_traversal(tmp_path: Path):
    case = _case(tmp_path, name=r"..\..\escape/video.mp4")
    output_root = tmp_path / "output"

    output_dir = oi._output_dir(output_root, case)

    assert output_dir.resolve().is_relative_to(output_root.resolve())


def test_listen_only_is_valid_but_response_required_e2e_fails(tmp_path: Path):
    case = _case(tmp_path)
    collector = _collector((_listen(), 1.0))
    result = oi.OmniInteractCaseResult("1q1a", str(case.video_path), "unused")

    summary = oi.write_success_artifacts(
        tmp_path / "normal",
        case,
        collector,
        stream_start=1.0,
        video_duration_s=1.0,
        require_response=False,
        result=result,
    )
    output_dir = Path(summary["output_dir"])
    with wave.open(str(output_dir / "output.wav"), "rb") as handle:
        assert handle.getnframes() == handle.getframerate()
    assert json.loads((output_dir / "wav_transcript.json").read_text()) == {
        "text": "",
        "chunks": [],
        "timestamp_semantics": "serialized playback queue time relative to input streaming start",
    }

    with pytest.raises(ValueError, match="E2E requires a response"):
        oi.write_success_artifacts(
            tmp_path / "required",
            case,
            collector,
            stream_start=1.0,
            video_duration_s=1.0,
            require_response=True,
            result=oi.OmniInteractCaseResult("1q1a", str(case.video_path), "unused"),
        )


def test_artifacts_reject_non_official_output_rate(tmp_path: Path):
    case = _case(tmp_path)
    audio = _audio("r1")
    audio["sample_rate_hz"] = 16_000
    collector = _collector((_created("r1"), 1.0), (audio, 1.1), (_done("r1"), 1.2))

    with pytest.raises(ValueError, match="24000 Hz"):
        oi.write_success_artifacts(
            tmp_path / "output",
            case,
            collector,
            stream_start=1.0,
            video_duration_s=1.0,
            require_response=False,
            result=oi.OmniInteractCaseResult("1q1a", str(case.video_path), "unused"),
        )


@pytest.mark.parametrize(
    "collector",
    [
        _collector(
            (_created("audio-only"), 1.0),
            (_audio("audio-only"), 1.1),
            (_done("audio-only"), 1.2),
            (_created("text-only"), 1.3),
            (_transcript("text-only"), 1.4),
            (_done("text-only"), 1.5),
        ),
        _collector(
            (_created("silent"), 1.0),
            (_audio("silent", value=0), 1.1),
            (_transcript("silent"), 1.2),
            (_done("silent"), 1.3),
        ),
        _collector(
            (_created("cancelled"), 1.0),
            (_audio("cancelled"), 1.1),
            (_transcript("cancelled"), 1.2),
            (_done("cancelled", "cancelled"), 1.3),
        ),
    ],
)
def test_response_required_e2e_needs_one_complete_non_silent_response(
    tmp_path: Path,
    collector: RealtimeEventCollector,
):
    case = _case(tmp_path)

    with pytest.raises(ValueError, match="E2E requires a response"):
        oi.write_success_artifacts(
            tmp_path / "output",
            case,
            collector,
            stream_start=1.0,
            video_duration_s=1.0,
            require_response=True,
            result=oi.OmniInteractCaseResult("1q1a", str(case.video_path), "unused"),
        )


def test_artifacts_reject_orphan_transcript(tmp_path: Path):
    case = _case(tmp_path)
    collector = _collector((_transcript("orphan"), 1.0))

    with pytest.raises(ValueError, match="transcript has no matching response.created"):
        oi.write_success_artifacts(
            tmp_path / "output",
            case,
            collector,
            stream_start=1.0,
            video_duration_s=1.0,
            require_response=False,
            result=oi.OmniInteractCaseResult("1q1a", str(case.video_path), "unused"),
        )


def test_failure_artifact_removes_stale_success_files(tmp_path: Path):
    case = _case(tmp_path)
    output_dir = oi._output_dir(tmp_path / "output", case)
    output_dir.mkdir(parents=True)
    for name in oi.SUCCESS_ARTIFACTS:
        (output_dir / name).write_text("stale")
    result = oi.OmniInteractCaseResult("1q1a", str(case.video_path), str(output_dir), error="boom")

    oi.write_failure_artifacts(tmp_path / "output", case, result)

    assert not any((output_dir / name).exists() for name in oi.SUCCESS_ARTIFACTS)
    assert json.loads((output_dir / ".failed.json").read_text())["error"] == "boom"


def test_success_artifact_write_failure_revokes_eligibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    case = _case(tmp_path)
    collector = _collector(
        (_created("response"), 1.0),
        (_audio("response"), 1.1),
        (_done("response"), 1.2),
    )
    result = oi.OmniInteractCaseResult("1q1a", str(case.video_path), "unused")

    def fail_wav(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(oi, "_atomic_write_wav", fail_wav)

    with pytest.raises(OSError, match="disk full"):
        oi.write_success_artifacts(
            tmp_path / "output",
            case,
            collector,
            stream_start=1.0,
            video_duration_s=1.0,
            require_response=False,
            result=result,
        )

    assert result.success is False
    assert result.eligible_for_official_eval is False
    assert result.official_eval_ineligible_reasons == ["artifact_write_failed"]
    output_dir = oi._output_dir(tmp_path / "output", case)
    assert not any((output_dir / name).exists() for name in oi.SUCCESS_ARTIFACTS)


def test_batch_artifacts_match_official_evaluator_handoff(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    case = _case(tmp_path)
    clipped_case = _case(tmp_path, name="clipped.mp4")
    cancelled_case = _case(tmp_path, name="cancelled.mp4")
    output_root = tmp_path / "output"
    result_specs = [
        (
            case,
            _collector((_created("normal"), 1.0), (_audio("normal"), 1.1), (_done("normal"), 1.2)),
        ),
        (
            clipped_case,
            _collector((_created("clipped"), 1.0), (_audio("clipped"), 2.1), (_done("clipped"), 2.2)),
        ),
        (
            cancelled_case,
            _collector(
                (_created("cancelled"), 1.0),
                (_audio("cancelled"), 1.1),
                (_done("cancelled", "cancelled"), 1.2),
            ),
        ),
    ]
    results = []
    for result_case, collector in result_specs:
        case_result = oi.OmniInteractCaseResult("1q1a", str(result_case.video_path), "unused")
        oi.write_success_artifacts(
            output_root,
            result_case,
            collector,
            stream_start=1.0,
            video_duration_s=1.0,
            require_response=False,
            result=case_result,
            persist=False,
        )
        results.append(case_result)

    oi.write_batch_artifacts(
        output_root,
        [case, clipped_case, cancelled_case],
        results,
    )

    summary = json.loads((output_root / "batch_summary.json").read_text())
    summary_rows = summary["results"]
    manifest_rows = [json.loads(row) for row in (output_root / "official_eval_manifest.jsonl").read_text().splitlines()]
    assert [row["success"] for row in summary_rows] == [True, True, True]
    assert summary["eligible_for_official_eval"] == 1
    assert summary["successful_but_ineligible"] == 2
    assert results[1].official_eval_ineligible_reasons == ["audio_clipped"]
    assert results[2].official_eval_ineligible_reasons == ["cancelled_response"]
    assert len(manifest_rows) == 1
    assert manifest_rows[0] == data.case_manifest(case, oi._output_dir(output_root, case))
    assert "audio_clipped=1" in caplog.text
    assert "cancelled_response=1" in caplog.text


def test_response_metrics_use_response_created_as_latency_origin(tmp_path: Path):
    collector = _collector(
        (_created("response"), 10.0),
        (_transcript("response"), 10.1),
        (_audio("response", frames=4_800), 10.2),
        (_done("response"), 10.3),
    )
    result = oi.OmniInteractCaseResult("1q1a", str(tmp_path / "video.mp4"), "unused")

    oi._populate_response_metrics(result, collector, stream_start=1.0)

    metric = result.duplex_request_metrics[0]
    assert metric["ttft_ms"] == pytest.approx(100.0)
    assert metric["ttfp_ms"] == pytest.approx(200.0)
    assert metric["measurement_origin"] == {
        "ttft": "response.created client receive to first non-empty text delta",
        "ttfp": "response.created client receive to first audio packet",
        "rtf": "response.created client receive to last audio packet divided by emitted audio duration",
    }


def test_response_metrics_preserve_stage_tokens_without_audio_metrics(tmp_path: Path):
    collector = RealtimeEventCollector()
    collector.add(_created("response"), received_at_s=10.0)
    collector.add(
        {
            "type": "response.done",
            "response": {"id": "response"},
            "metadata": {
                "vllm_omni": {
                    "stage_metrics": {
                        "0": {
                            "num_tokens_out": 3,
                            "vllm_ttft_ms": 12.0,
                            "vllm_tpot_ms": 4.0,
                            "vllm_itls_ms": [3.0, 5.0],
                        }
                    }
                }
            },
        },
        received_at_s=10.1,
    )
    result = oi.OmniInteractCaseResult("1q1a", str(tmp_path / "video.mp4"), "unused")

    oi._populate_response_metrics(result, collector, stream_start=1.0)

    assert result.output_tokens == 3
    assert result.duplex_request_metrics[0]["stage0_tokens"] == {
        "source": "engine_stage_metrics",
        "output_token_count": 3,
        "ttft_ms": 12.0,
        "tpot_ms": 4.0,
        "itls_ms": [3.0, 5.0],
        "inter_token_interval_ms": {
            "count": 2,
            "mean": 4.0,
            "p50": 3.0,
            "p95": 5.0,
            "max": 5.0,
        },
    }


class _FakeClient:
    def __init__(self, collector: RealtimeEventCollector):
        self.events = collector
        self.sent: list[dict[str, object]] = []
        self.reader_error: Exception | None = None

    async def send(self, event: dict[str, object]) -> None:
        self.sent.append(event)

    async def send_playback_ack(self, response_id: str, played_ms: int) -> None:
        await self.send(
            {
                "type": "playback.ack",
                "response_id": response_id,
                "item_id": f"item_{response_id}",
                "played_ms": played_ms,
                "committed_ms": played_ms,
            }
        )

    def raise_if_reader_stopped(self) -> None:
        if self.reader_error:
            raise self.reader_error


@pytest.mark.asyncio
async def test_playback_commits_completed_response_only_after_audio_drains():
    collector = _collector(
        (_created("response"), 9.9),
        (_audio("response", frames=24_000), 10.0),
        (_done("response"), 10.01),
    )
    client = _FakeClient(collector)
    playback = oi._Playback()

    await playback.acknowledge(client, now=10.2)
    assert client.sent == []

    await playback.acknowledge(client, now=11.0)
    assert client.sent == [
        {
            "type": "playback.ack",
            "response_id": "response",
            "item_id": "item_response",
            "played_ms": 1_000,
            "committed_ms": 1_000,
        }
    ]


def test_playback_warns_when_audio_metadata_is_omitted():
    audio = _audio("response")
    audio.pop("format")
    audio.pop("sample_rate_hz")
    playback = oi._Playback()

    playback.ingest(
        _collector(
            (_created("response"), 1.0),
            (audio, 1.1),
            (_done("response"), 1.2),
        )
    )

    assert playback.warnings == [
        "response.audio.delta omitted format; assumed pcm16",
        "response.audio.delta omitted sample_rate_hz; assumed 24000",
    ]


@pytest.mark.asyncio
async def test_playback_aggregates_sub_millisecond_interleaved_deltas_once():
    events = [(_created("one"), 9.8), (_created("two"), 9.9)]
    for index, response_id in enumerate(("one", "two", "one", "two")):
        events.append((_audio(response_id, frames=15), 10.0 + index * 0.0001))
    events.extend([(_done("one"), 10.01), (_done("two"), 10.02)])
    client = _FakeClient(_collector(*events))
    playback = oi._Playback()

    await playback.acknowledge(client, now=11.0)
    await playback.acknowledge(client, now=11.1)

    assert sorted((event["response_id"], event["played_ms"]) for event in client.sent) == [
        ("one", 1),
        ("two", 1),
    ]


@pytest.mark.parametrize(
    ("base_url", "endpoint", "expected_scheme"),
    [
        ("http://host:8000", "/v1/realtime", "ws"),
        ("https://host:8000", "/v1/realtime", "wss"),
        ("http://ignored", "ws://host:8000/v1/realtime", "ws"),
        ("http://ignored", "wss://host:8000/v1/realtime", "wss"),
    ],
)
def test_websocket_url_overrides_reserved_query_per_case(base_url: str, endpoint: str, expected_scheme: str):
    separator = "&" if "?" in endpoint else "?"
    endpoint += f"{separator}trace=keep&session_id=shared&model=wrong&duplex=0&minicpmo45_native_duplex=0&autostart=1"
    config = oi.OmniInteractBenchmarkConfig(base_url=base_url, endpoint=endpoint, model="expected/model")

    result = urlsplit(oi._websocket_url(config, "unique-session"))
    query = parse_qs(result.query)

    assert result.scheme == expected_scheme
    assert query == {
        "trace": ["keep"],
        "session_id": ["unique-session"],
        "model": ["expected/model"],
        "duplex": ["1"],
        "minicpmo45_native_duplex": ["1"],
        "autostart": ["0"],
    }


@pytest.mark.asyncio
async def test_public_case_runner_supports_functional_e2e(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    instances = []

    class FakeRealtimeClient:
        def __init__(self, url: str):
            self.url = url
            self.events = RealtimeEventCollector()
            self.sent: list[dict[str, object]] = []
            self.configured: dict[str, object] = {}
            instances.append(self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def configure(self, model: str, **kwargs) -> None:
            self.configured = {"model": model, **kwargs}
            self.events.add({"type": "session.created"})

        async def send(self, event: dict[str, object]) -> None:
            self.sent.append(event)

        async def stream_pcm16(self, pcm16: bytes, **kwargs):
            return await stream_pcm16_chunks(self.send, pcm16, **kwargs)

        async def send_playback_ack(self, response_id: str, played_ms: int) -> None:
            await self.send(
                {
                    "type": "playback.ack",
                    "response_id": response_id,
                    "item_id": f"item_{response_id}",
                    "played_ms": played_ms,
                    "committed_ms": played_ms,
                }
            )

        async def commit(self) -> None:
            await self.send({"type": "input_audio_buffer.commit", "final": True})
            now = time.monotonic()
            self.events.add({"type": "input_audio_buffer.committed"}, received_at_s=now)
            self.events.add(_created("r1"), received_at_s=now)
            self.events.add(_audio("r1"), received_at_s=now)
            self.events.add(_transcript("r1", "functional response"), received_at_s=now)
            self.events.add(_done("r1"), received_at_s=now)

        async def close_session(self, *, timeout_s: float) -> None:
            await self.send({"type": "session.close"})
            self.events.add({"type": "session.closed"})

        def raise_if_reader_stopped(self) -> None:
            return None

    monkeypatch.setattr(
        oi,
        "prepare_media",
        lambda *args, **kwargs: (2.0, bytes(32_000), [base64.b64encode(b"frame").decode()]),
    )
    monkeypatch.setattr(oi, "RealtimeDuplexClient", FakeRealtimeClient)
    monkeypatch.setattr(oi, "_COMPLETION_SETTLE_S", 0)
    case = _case(tmp_path)
    ref_audio = tmp_path / "reference.wav"
    ref_audio.write_bytes(b"test reference audio")
    config = oi.OmniInteractBenchmarkConfig(
        output_root=tmp_path / "output",
        ref_audio=str(ref_audio),
        timeout_s=0.2,
        require_response=True,
    )

    result = await oi.run_omniinteract_case(case, config, request_index=0)

    assert result.success, result.error
    assert result.transcript == "functional response"
    assert result.input_video_frames == 1
    assert Path(result.output_dir, ".done").is_file()
    assert instances[0].configured["idle_timeout_s"] == config.timeout_s
    assert instances[0].configured["ref_audio"] == "data:audio/wav;base64,dGVzdCByZWZlcmVuY2UgYXVkaW8="
    appends = [event for event in instances[0].sent if event["type"] == "input_audio_buffer.append"]
    assert [event["audio_end_ms"] for event in appends] == [200, 400, 600, 800, 999]
    assert sum(len(base64.b64decode(event["audio"])) for event in appends) == 31_998
    assert ["video_frames" in event for event in appends] == [False, False, True, False, False]
    assert {event["type"] for event in instances[0].sent} >= {
        "input_audio_buffer.append",
        "input_audio_buffer.commit",
        "playback.ack",
        "session.close",
    }


@pytest.mark.asyncio
async def test_case_times_out_when_realtime_upload_stalls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class StalledRealtimeClient:
        def __init__(self, url: str):
            self.events = RealtimeEventCollector()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def configure(self, model: str, **kwargs) -> None:
            self.events.add({"type": "session.created"})

        async def send(self, event: dict[str, object]) -> None:
            await asyncio.Future()

        async def stream_pcm16(self, pcm16: bytes, **kwargs):
            return await stream_pcm16_chunks(self.send, pcm16, **kwargs)

        async def close_session(self, *, timeout_s: float) -> None:
            return None

    encoded_frame = base64.b64encode(b"frame").decode()
    monkeypatch.setattr(oi, "prepare_media", lambda *args, **kwargs: (0.2, bytes(6_400), [encoded_frame]))
    monkeypatch.setattr(oi, "RealtimeDuplexClient", StalledRealtimeClient)
    case = _case(tmp_path)
    ref_audio = tmp_path / "reference.wav"
    ref_audio.write_bytes(b"test reference audio")
    config = oi.OmniInteractBenchmarkConfig(
        output_root=tmp_path / "output",
        ref_audio=str(ref_audio),
        timeout_s=0.02,
    )

    result = await asyncio.wait_for(oi.run_omniinteract_case(case, config, request_index=0), timeout=1)

    assert result.error == "Realtime upload timed out after 0.22s"
    assert Path(result.output_dir, ".failed.json").is_file()


@pytest.mark.asyncio
async def test_completion_waits_for_final_decision_after_current_commit():
    collector = _collector(({"type": "input_audio_buffer.committed"}, 1.0))
    client = _FakeClient(collector)
    commit_from = len(collector.events)

    async def complete_current_commit():
        await asyncio.sleep(0.01)
        collector.add({"type": "input_audio_buffer.committed"})
        collector.add(_listen(model_listen=False, buffering=True, reason="buffering"))
        await asyncio.sleep(0.13)
        collector.add(_listen())

    task = asyncio.create_task(complete_current_commit())
    index = await oi.wait_for_session_completion(
        client,
        oi._Playback(),
        commit_from=commit_from,
        timeout_s=0.4,
        settle_s=0.01,
    )

    assert task.done()
    assert index == commit_from


@pytest.mark.asyncio
async def test_completion_ignores_response_that_releases_deferred_final_input():
    collector = _collector((_created("active"), 1.0))
    client = _FakeClient(collector)
    commit_from = len(collector.events)

    async def complete_deferred_commit():
        await asyncio.sleep(0.01)
        collector.add(
            {
                "type": "input_audio_buffer.committed",
                "event": {"overlap_deferred": True},
            }
        )
        await asyncio.sleep(0.02)
        collector.add(_done("active"))
        await asyncio.sleep(0.08)
        collector.add(_listen())

    task = asyncio.create_task(complete_deferred_commit())
    index = await oi.wait_for_session_completion(
        client,
        oi._Playback(),
        commit_from=commit_from,
        timeout_s=0.4,
        settle_s=0.01,
    )

    assert task.done()
    assert index == commit_from


@pytest.mark.asyncio
async def test_completion_does_not_accept_buffering_listen_as_final_decision():
    collector = RealtimeEventCollector()
    client = _FakeClient(collector)

    async def report_prefill_failure():
        await asyncio.sleep(0.01)
        collector.add({"type": "input_audio_buffer.committed"})
        collector.add(_listen(model_listen=False, buffering=True, reason="prefill_failed"))

    task = asyncio.create_task(report_prefill_failure())
    with pytest.raises(TimeoutError, match="Timed out waiting for committed input"):
        await oi.wait_for_session_completion(
            client,
            oi._Playback(),
            commit_from=0,
            timeout_s=0.08,
            settle_s=0.01,
        )
    assert task.done()


def test_exact_model_unit_reserves_one_sample_for_final_commit():
    session_created = {
        "type": "session.created",
        "session": {"capabilities": {"chunk_period_ms": 1000}},
    }

    assert len(oi._ensure_final_commit_tail(bytes(32_000), [session_created])) == 31_998
    assert len(oi._ensure_final_commit_tail(bytes(31_998), [session_created])) == 31_998


def test_explicit_close_rejects_racing_timeout_event():
    collector = _collector(
        ({"type": "session.created"}, 1.0),
        ({"type": "session.closed", "event": {"reason": "timeout"}}, 1.1),
    )

    with pytest.raises(RuntimeError, match="session.closed: timeout"):
        oi._validate_explicit_session_close(collector, session_from=0, close_from=1)


@pytest.mark.asyncio
async def test_completion_requires_each_created_response_to_finish():
    collector = _collector(
        (_created("r1"), 1.0),
        ({"type": "input_audio_buffer.committed"}, 1.1),
    )

    with pytest.raises(TimeoutError, match="unfinished response_ids=.*r1"):
        await oi.wait_for_session_completion(
            _FakeClient(collector),
            oi._Playback(),
            commit_from=1,
            timeout_s=0.06,
            settle_s=0.01,
        )


@pytest.mark.asyncio
async def test_completion_fails_fast_on_transport_close():
    collector = _collector(({"type": "input_audio_buffer.committed"}, 1.0))
    client = _FakeClient(collector)
    client.reader_error = ConnectionError("closed")

    with pytest.raises(ConnectionError, match="closed"):
        await oi.wait_for_session_completion(
            client,
            oi._Playback(),
            commit_from=0,
            timeout_s=10,
            settle_s=0,
        )


@pytest.mark.asyncio
async def test_completion_rejects_session_closed_before_final_commit():
    collector = _collector(
        ({"type": "session.closed", "reason": "idle_timeout"}, 1.0),
        ({"type": "input_audio_buffer.committed"}, 1.1),
    )

    with pytest.raises(RuntimeError, match="session.closed: idle_timeout"):
        await oi.wait_for_session_completion(
            _FakeClient(collector),
            oi._Playback(),
            commit_from=1,
            session_from=0,
            timeout_s=0.2,
            settle_s=0,
        )


@pytest.mark.asyncio
async def test_case_invalidates_stale_done_before_preprocessing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    case = _case(tmp_path)
    ref_audio = tmp_path / "reference.wav"
    ref_audio.write_bytes(b"reference audio")
    config = oi.OmniInteractBenchmarkConfig(
        output_root=tmp_path / "output",
        ref_audio=str(ref_audio),
    )
    output_dir = oi._output_dir(config.output_root, case)
    output_dir.mkdir(parents=True)
    (output_dir / ".done").write_text("stale")

    def fail_preprocessing(*args, **kwargs):
        raise RuntimeError("preprocessing failed")

    monkeypatch.setattr(oi, "prepare_media", fail_preprocessing)

    result = await oi.run_omniinteract_case(case, config, request_index=0)

    assert not (output_dir / ".done").exists()
    assert (output_dir / ".failed.json").is_file()
    assert result.error == "preprocessing failed"


@pytest.mark.asyncio
async def test_case_rejects_media_without_decoded_video_frames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    case = _case(tmp_path)
    ref_audio = tmp_path / "reference.wav"
    ref_audio.write_bytes(b"reference audio")
    config = oi.OmniInteractBenchmarkConfig(
        output_root=tmp_path / "output",
        ref_audio=str(ref_audio),
    )
    monkeypatch.setattr(oi, "prepare_media", lambda *args, **kwargs: (1.0, bytes(32_000), [None]))

    result = await oi.run_omniinteract_case(case, config, request_index=0)

    assert "No video frames were decoded" in result.error
    assert Path(result.output_dir, ".failed.json").is_file()


def test_serve_cli_loads_omniinteract_as_standard_samples(
    tmp_path: Path,
):
    parser = TrackingArgumentParser()
    OmniBenchmarkServingSubcommand.add_cli_args(parser)
    _write_dataset(tmp_path)
    ref_audio = tmp_path / "reference.wav"
    ref_audio.write_bytes(b"reference audio")

    args = parser.parse_args(
        [
            "--backend",
            "openai-realtime-duplex",
            "--dataset-name",
            "omniinteract",
            "--dataset-path",
            str(tmp_path),
            "--model",
            "openbmb/MiniCPM-o-4_5",
            "--base-url",
            "http://server:8000",
            "--endpoint",
            "/v1/realtime",
            "--omniinteract-subsets",
            "1q1a",
            "1qna",
            "--num-prompts",
            "8",
            "--max-concurrency",
            "2",
            "--omniinteract-output-dir",
            str(tmp_path / "artifacts"),
            "--omniinteract-ref-audio",
            str(ref_audio),
            "--omniinteract-require-response",
        ]
    )
    preprocess_serve_args(args)
    samples = benchmark_patch.get_samples(args, tokenizer=None)

    assert args.dataset_name == "omniinteract"
    assert args.num_prompts == 2
    assert len(samples) == 2
    assert all(isinstance(sample, data.OmniInteractSampleRequest) for sample in samples)
    assert {sample.omniinteract_case.subset for sample in samples} == {"1q1a", "1qna"}
    assert all(sample.omniinteract_options.output_root == tmp_path / "artifacts" for sample in samples)
    assert all(sample.omniinteract_options.ref_audio == str(ref_audio) for sample in samples)
    assert all(sample.omniinteract_options.require_response is True for sample in samples)


def test_serve_cli_uses_zero_to_select_all_omniinteract_samples(tmp_path: Path):
    parser = TrackingArgumentParser()
    OmniBenchmarkServingSubcommand.add_cli_args(parser)
    ref_audio = tmp_path / "reference.wav"
    ref_audio.touch()

    args = parser.parse_args(
        [
            "--backend",
            "openai-realtime-duplex",
            "--dataset-name",
            "omniinteract",
            "--endpoint",
            "/v1/realtime",
            "--omniinteract-ref-audio",
            str(ref_audio),
            "--dataset-path",
            str(tmp_path),
            "--num-prompts",
            "0",
        ]
    )

    _write_dataset(tmp_path)
    preprocess_serve_args(args)
    samples = benchmark_patch.get_samples(args, tokenizer=None)

    assert len(samples) == 3
    assert args.num_prompts == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(("request_id", "persist_artifacts"), [(None, False), ("measured-0", True)])
@pytest.mark.parametrize(
    "metric_mode",
    ["exact", "fallback", "mixed_missing", "exact_missing_tpot", "coverage_mismatch"],
)
async def test_realtime_backend_runs_one_dataset_session_and_only_persists_measured_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request_id: str | None,
    persist_artifacts: bool,
    metric_mode: str,
):
    case = _case(tmp_path)
    options = data.OmniInteractSessionOptions(
        output_root=tmp_path / "artifacts",
        timeout_s=10,
        media_timeout_s=5,
        ref_audio=str(tmp_path / "reference.wav"),
        require_response=True,
    )
    calls: list[dict[str, object]] = []

    async def fake_run(case_arg, config, *, request_index, persist_artifacts):
        calls.append(
            {
                "case": case_arg,
                "config": config,
                "request_index": request_index,
                "persist_artifacts": persist_artifacts,
            }
        )
        first_stage = {"output_token_count": 3, "tpot_ms": 15.0}
        second_stage = {"output_token_count": 2, "tpot_ms": 40.0}
        if metric_mode in {"exact", "exact_missing_tpot"}:
            first_stage["itls_ms"] = [10.0, 20.0]
            second_stage["itls_ms"] = [40.0]
        elif metric_mode == "mixed_missing":
            first_stage["itls_ms"] = [10.0, 20.0]
        if metric_mode in {"mixed_missing", "exact_missing_tpot"}:
            second_stage["tpot_ms"] = None
        return oi.OmniInteractCaseResult(
            subset=case_arg.subset,
            video=str(case_arg.video_path),
            output_dir=str(options.output_root),
            success=True,
            eligible_for_official_eval=True,
            transcript="answer",
            output_tokens=6 if metric_mode == "coverage_mismatch" else 5,
            duplex_request_metrics=[
                {"stage0_tokens": first_stage},
                {"stage0_tokens": second_stage},
            ],
            duplex_session_metrics={"mean_ttft_ms": 100.0, "mean_ttfp_ms": 200.0, "mean_rtf": 0.5},
        )

    monkeypatch.setattr(benchmark_patch, "run_omniinteract_case", fake_run)
    request = RequestFuncInput(
        model=oi.DEFAULT_MODEL,
        model_name=oi.DEFAULT_MODEL,
        prompt="",
        api_url="http://server:8000/v1/realtime",
        prompt_len=0,
        output_len=0,
        logprobs=None,
        multi_modal_content=None,
        ignore_eos=False,
        request_id=request_id,
    )
    request.omniinteract_case = case
    request.omniinteract_options = options

    output = await benchmark_patch.async_request_openai_realtime_duplex(request, session=None)

    assert output.success is True
    assert output.generated_text == "answer"
    assert output.output_tokens == (6 if metric_mode == "coverage_mismatch" else 5)
    assert output.ttft == pytest.approx(0.1)
    assert output.audio_ttfp == pytest.approx(0.2)
    assert output.audio_rtf == pytest.approx(0.5)
    if metric_mode in {"exact", "exact_missing_tpot"}:
        assert output.itl == pytest.approx([0.01, 0.02, 0.04])
        assert output.text_latency == pytest.approx(0.17)
    elif metric_mode == "fallback":
        assert output.itl == []
        assert output.text_latency == pytest.approx(0.1 + (0.015 * 2 + 0.04) / 3 * 4)
    else:
        assert output.itl == []
        assert output.text_latency == pytest.approx(0.1)
    assert calls[0]["case"] is case
    assert calls[0]["persist_artifacts"] is persist_artifacts
    assert (calls[0]["request_index"] == request_id) if request_id else bool(calls[0]["request_index"])


def test_standard_benchmark_finalizes_only_measured_omniinteract_outputs(tmp_path: Path):
    case = _case(tmp_path)
    options = data.OmniInteractSessionOptions(
        output_root=tmp_path / "artifacts",
        timeout_s=10,
        media_timeout_s=5,
        ref_audio="reference.wav",
    )
    sample = data.OmniInteractSampleRequest(
        prompt="",
        prompt_len=0,
        expected_output_len=0,
        multi_modal_data=None,
        request_id="measured-0",
        omniinteract_case=case,
        omniinteract_options=options,
    )
    output = benchmark_patch.MixRequestFuncOutput(success=True)
    output.omniinteract_case_result = oi.OmniInteractCaseResult(
        subset=case.subset,
        video=str(case.video_path),
        output_dir=str(options.output_root),
        success=True,
        eligible_for_official_eval=True,
    )

    summary = benchmark_patch._finalize_omniinteract_batch([sample], [output])

    assert summary == {
        "total": 1,
        "success": 1,
        "failed": 0,
        "eligible_for_official_eval": 1,
        "successful_but_ineligible": 0,
        "audio_clipped_bytes": 0,
    }
    assert len((options.output_root / "official_eval_manifest.jsonl").read_text().splitlines()) == 1


def test_serve_cli_requires_reference_audio_for_omniinteract():
    args = argparse.Namespace(
        dataset_name="omniinteract",
        backend="openai-realtime-duplex",
        endpoint="/v1/realtime",
        omniinteract_ref_audio=None,
    )

    with pytest.raises(ValueError, match="--omniinteract-ref-audio"):
        preprocess_serve_args(args)


def test_serve_cli_requires_realtime_endpoint_for_omniinteract():
    args = argparse.Namespace(
        dataset_name="omniinteract",
        backend="openai-realtime-duplex",
        endpoint="/other",
        omniinteract_ref_audio="reference.wav",
    )

    with pytest.raises(ValueError, match="--endpoint /v1/realtime"):
        preprocess_serve_args(args)


@pytest.mark.parametrize("max_concurrency", [0, -1])
def test_serve_cli_rejects_non_positive_concurrency(max_concurrency: int):
    args = argparse.Namespace(
        dataset_name="omniinteract",
        backend="openai-realtime-duplex",
        endpoint="/v1/realtime",
        omniinteract_ref_audio="reference.wav",
        max_concurrency=max_concurrency,
    )

    with pytest.raises(ValueError, match="--max-concurrency to be positive"):
        preprocess_serve_args(args)


def test_serve_cli_rejects_skip_tokenizer_init():
    args = argparse.Namespace(
        dataset_name="omniinteract",
        backend="openai-realtime-duplex",
        endpoint="/v1/realtime",
        omniinteract_ref_audio="reference.wav",
        skip_tokenizer_init=True,
    )

    with pytest.raises(ValueError, match="does not support --skip-tokenizer-init"):
        preprocess_serve_args(args)


def test_plain_namespace_cannot_reuse_upstream_default_endpoint(tmp_path: Path):
    ref_audio = tmp_path / "reference.wav"
    ref_audio.touch()
    args = argparse.Namespace(
        dataset_name="omniinteract",
        backend="openai-realtime-duplex",
        base_url="http://server:8000",
        host="127.0.0.1",
        port=8000,
        endpoint="/v1/completions",
        model=oi.DEFAULT_MODEL,
        dataset_path=None,
        max_concurrency=1,
        result_dir=None,
        omniinteract_output_dir=tmp_path / "artifacts",
        omniinteract_subsets=list(data.OMNIINTERACT_SUBSETS),
        num_prompts=1000,
        omniinteract_timeout_s=900.0,
        omniinteract_media_timeout_s=600.0,
        omniinteract_ref_audio=str(ref_audio),
        omniinteract_require_response=False,
        seed=0,
        disable_shuffle=False,
    )

    with pytest.raises(ValueError, match="--endpoint /v1/realtime"):
        preprocess_serve_args(args)


@pytest.mark.parametrize("value", ["0", "nan", "inf", "-inf"])
def test_serve_cli_rejects_non_positive_or_non_finite_timeout(value: str):
    parser = argparse.ArgumentParser()

    add_omniinteract_cli_args(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(["--omniinteract-timeout-s", value])


def test_serve_cli_rejects_missing_reference_audio_file(tmp_path: Path):
    parser = argparse.ArgumentParser()

    add_omniinteract_cli_args(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(["--omniinteract-ref-audio", str(tmp_path / "missing.wav")])


def test_serve_cli_does_not_require_omniinteract_options_for_other_datasets():
    parser = argparse.ArgumentParser()
    OmniBenchmarkServingSubcommand.add_cli_args(parser)

    args = parser.parse_args(["--dataset-name", "daily-omni"])

    assert args.dataset_name == "daily-omni"
    assert args.omniinteract_ref_audio is None


def test_serve_main_dispatches_omniinteract_through_standard_main_async(monkeypatch: pytest.MonkeyPatch):
    expected = {"total": 1, "success": 1, "failed": 0}

    async def fake_main_async(args):
        assert args.dataset_name == "omniinteract"
        return expected

    monkeypatch.setattr(benchmark_serve, "main_async", fake_main_async)

    result = benchmark_serve.main(argparse.Namespace(dataset_name="omniinteract"))

    assert result == expected
