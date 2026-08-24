# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import argparse
import asyncio
import base64
import errno
import fcntl
import io
import json
import subprocess
import tarfile
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from vllm_omni.benchmarks import omniinteract as oi
from vllm_omni.benchmarks import serve as benchmark_serve
from vllm_omni.benchmarks.data_modules import omniinteract_dataset as data
from vllm_omni.entrypoints.cli.benchmark.serve import OmniBenchmarkServingSubcommand
from vllm_omni.experimental.fullduplex.client import RealtimeEventCollector
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


def test_dataset_rejects_duplicate_subsets_and_missing_capacity(tmp_path: Path):
    root = _write_dataset(tmp_path)

    with pytest.raises(ValueError, match="must not contain duplicates"):
        data.discover_omniinteract_cases(root, ("1q1a", "1q1a"), num_prompts=1)
    with pytest.raises(ValueError, match="only 1 are available"):
        data.discover_omniinteract_cases(root, ("1q1a",), num_prompts=2)


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


def test_archive_extraction_is_serialized_across_shared_cache_users(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source_root = _write_dataset(tmp_path / "source")
    archive = tmp_path / "data.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(source_root, arcname="data")
    target = tmp_path / "cache" / "dataset"
    extraction_count = 0
    original_extract = data._safe_extract

    def slow_extract(handle: tarfile.TarFile, directory: Path) -> None:
        nonlocal extraction_count
        extraction_count += 1
        time.sleep(0.05)
        original_extract(handle, directory)

    monkeypatch.setattr(data, "_safe_extract", slow_extract)
    with ThreadPoolExecutor(max_workers=2) as executor:
        roots = list(executor.map(lambda _: data._extract_archive(archive, target), range(2)))

    assert roots == [target / "data", target / "data"]
    assert extraction_count == 1
    assert (roots[0] / "1q1a" / "video_json_map.json").is_file()


def test_downloads_dataset_archive_through_vllm_hf_filesystem(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source_root = _write_dataset(tmp_path / "source")
    archive = tmp_path / "data.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(source_root, arcname="data")
    downloads: list[str] = []

    class FakeHfFileSystem:
        def download(self, remote: str, local: str) -> None:
            downloads.append(remote)
            Path(local).write_bytes(archive.read_bytes())

    monkeypatch.setenv("HF_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(data, "hf_fs", FakeHfFileSystem)

    root = data.resolve_omniinteract_root(None, "owner/dataset")

    assert (root / "1q1a").is_dir()
    assert downloads == ["datasets/owner/dataset/data.tar.gz"]


def test_shared_cache_download_and_extract_use_enolck_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source_root = _write_dataset(tmp_path / "source")
    archive = tmp_path / "data.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(source_root, arcname="data")

    cache_root = tmp_path / "cache"
    target = cache_root / "vllm_omni" / "omniinteract" / "owner__dataset"
    fallback = target.parent / (data.hub_prefetch._safe_repo_filename(data._archive_lock_key(target)) + ".dir")
    downloads: list[str] = []
    extractions: list[Path] = []
    original_extract = data._safe_extract

    class FakeHfFileSystem:
        def download(self, remote: str, local: str) -> None:
            assert fallback.is_dir()
            downloads.append(remote)
            time.sleep(0.05)
            Path(local).write_bytes(archive.read_bytes())

    def locked_extract(handle: tarfile.TarFile, directory: Path) -> None:
        assert fallback.is_dir()
        extractions.append(directory)
        original_extract(handle, directory)

    def unsupported_flock(*args) -> None:
        raise OSError(errno.ENOLCK, "flock unavailable")

    monkeypatch.setenv("HF_HOME", str(cache_root))
    monkeypatch.setattr(data, "hf_fs", FakeHfFileSystem)
    monkeypatch.setattr(data, "_safe_extract", locked_extract)
    monkeypatch.setattr(fcntl, "flock", unsupported_flock)
    with ThreadPoolExecutor(max_workers=2) as executor:
        roots = list(executor.map(lambda _: data.resolve_omniinteract_root(None, "owner/dataset"), range(2)))

    assert roots == [target / "data", target / "data"]
    assert downloads == ["datasets/owner/dataset/data.tar.gz"]
    assert len(extractions) == 1
    assert not fallback.exists()


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


def test_config_requires_reference_audio():
    with pytest.raises(ValueError, match="ref_audio is required"):
        oi.validate_config(oi.OmniInteractBenchmarkConfig())


@pytest.mark.parametrize("field", ["timeout_s", "settle_s", "media_timeout_s"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_config_rejects_non_finite_timeouts(field: str, value: float):
    config = oi.OmniInteractBenchmarkConfig(ref_audio="unused.wav", **{field: value})

    with pytest.raises(ValueError, match="timeouts must be finite"):
        oi.validate_config(config)


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
    assert summary["audio_overwritten_bytes"] == 0
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
    assert summary["audio_overwritten_bytes"] == 0


def test_output_dir_flattens_backslash_traversal(tmp_path: Path):
    case = _case(tmp_path, name=r"..\..\escape/video.mp4")
    output_root = tmp_path / "output"

    output_dir = oi._output_dir(output_root, case)

    assert output_dir.resolve().is_relative_to(output_root.resolve())


def test_listen_only_is_valid_but_response_required_e2e_fails(tmp_path: Path):
    case = _case(tmp_path)
    collector = _collector(({"type": "response.listen"}, 1.0))
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
    assert json.loads((output_dir / "wav_transcript.json").read_text()) == {"text": "", "chunks": []}

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


def test_batch_artifacts_match_official_evaluator_handoff(tmp_path: Path):
    case = _case(tmp_path)
    clipped_case = _case(tmp_path, name="clipped.mp4")
    overwritten_case = _case(tmp_path, name="overwritten.mp4")
    output_root = tmp_path / "output"
    result = oi.OmniInteractCaseResult(
        "1q1a",
        str(case.video_path),
        str(oi._output_dir(output_root, case)),
        success=True,
    )
    clipped_result = oi.OmniInteractCaseResult(
        "1q1a",
        str(clipped_case.video_path),
        str(oi._output_dir(output_root, clipped_case)),
        success=True,
        audio_clipped_bytes=2,
    )
    overwritten_result = oi.OmniInteractCaseResult(
        "1q1a",
        str(overwritten_case.video_path),
        str(oi._output_dir(output_root, overwritten_case)),
        success=True,
        audio_overwritten_bytes=2,
    )

    oi.write_batch_artifacts(
        output_root,
        [case, clipped_case, overwritten_case],
        oi.OmniInteractBenchmarkResult([result, clipped_result, overwritten_result]),
    )

    summary_rows = json.loads((output_root / "batch_summary.json").read_text())["results"]
    manifest_rows = [json.loads(row) for row in (output_root / "official_eval_manifest.jsonl").read_text().splitlines()]
    assert [row["status"] for row in summary_rows] == ["ok", "ok", "ok"]
    assert len(manifest_rows) == 1
    assert manifest_rows[0] == data.case_manifest(case, oi._output_dir(output_root, case))


class _FakeClient:
    def __init__(self, collector: RealtimeEventCollector):
        self.events = collector
        self.sent: list[dict[str, object]] = []
        self.reader_error: Exception | None = None

    async def send(self, event: dict[str, object]) -> None:
        self.sent.append(event)

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
        lambda *args, **kwargs: (1.0, bytes(32_000), [base64.b64encode(b"frame").decode()]),
    )
    monkeypatch.setattr(oi, "RealtimeDuplexClient", FakeRealtimeClient)
    case = _case(tmp_path)
    ref_audio = tmp_path / "reference.wav"
    ref_audio.write_bytes(b"test reference audio")
    config = oi.OmniInteractBenchmarkConfig(
        data_root=str(tmp_path),
        output_root=tmp_path / "output",
        ref_audio=str(ref_audio),
        timeout_s=0.2,
        settle_s=0,
        pace=False,
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
    assert [event["audio_end_ms"] for event in appends] == [200, 400, 600, 800, 1000]
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

        async def close_session(self, *, timeout_s: float) -> None:
            return None

    encoded_frame = base64.b64encode(b"frame").decode()
    monkeypatch.setattr(oi, "prepare_media", lambda *args, **kwargs: (0.2, bytes(6_400), [encoded_frame]))
    monkeypatch.setattr(oi, "RealtimeDuplexClient", StalledRealtimeClient)
    case = _case(tmp_path)
    ref_audio = tmp_path / "reference.wav"
    ref_audio.write_bytes(b"test reference audio")
    config = oi.OmniInteractBenchmarkConfig(
        data_root=str(tmp_path),
        output_root=tmp_path / "output",
        ref_audio=str(ref_audio),
        timeout_s=0.02,
        settle_s=0,
        pace=False,
    )

    result = await asyncio.wait_for(oi.run_omniinteract_case(case, config, request_index=0), timeout=1)

    assert result.error == "Realtime upload timed out after 0.02s"
    assert Path(result.output_dir, ".failed.json").is_file()


@pytest.mark.asyncio
async def test_completion_ignores_old_commit_and_has_no_input_seq_dependency():
    collector = _collector(({"type": "input_audio_buffer.committed"}, 1.0))
    client = _FakeClient(collector)
    commit_from = len(collector.events)

    async def complete_current_commit():
        await asyncio.sleep(0.01)
        collector.add({"type": "input_audio_buffer.committed"})

    task = asyncio.create_task(complete_current_commit())
    index = await oi.wait_for_session_completion(
        client,
        oi._Playback(),
        pcm_bytes=32_000,
        commit_from=commit_from,
        timeout_s=0.2,
        settle_s=0.01,
    )
    await task

    assert index == commit_from
    assert "accepted_input_seq" not in collector.events[index]


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
            pcm_bytes=16_000,
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
            pcm_bytes=32_000,
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
            pcm_bytes=32_000,
            commit_from=1,
            session_from=0,
            timeout_s=0.2,
            settle_s=0,
        )


@pytest.mark.asyncio
async def test_case_invalidates_stale_done_before_preprocessing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    case = _case(tmp_path)
    config = oi.OmniInteractBenchmarkConfig(
        data_root=str(tmp_path),
        output_root=tmp_path / "output",
        ref_audio="unused.wav",
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
    config = oi.OmniInteractBenchmarkConfig(
        data_root=str(tmp_path),
        output_root=tmp_path / "output",
        ref_audio="unused.wav",
    )
    monkeypatch.setattr(oi, "prepare_media", lambda *args, **kwargs: (1.0, bytes(32_000), [None]))

    result = await oi.run_omniinteract_case(case, config, request_index=0)

    assert "No video frames were decoded" in result.error
    assert Path(result.output_dir, ".failed.json").is_file()


@pytest.mark.asyncio
async def test_benchmark_bounds_preprocessing_and_session_concurrency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cases = [_case(tmp_path, name=f"video-{index}.mp4") for index in range(4)]
    active = 0
    peak = 0

    monkeypatch.setattr(oi, "resolve_omniinteract_root", lambda *_: tmp_path)
    monkeypatch.setattr(oi, "discover_omniinteract_cases", lambda *args, **kwargs: cases)

    async def fake_run(case, config, *, request_index):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return oi.OmniInteractCaseResult(
            subset=case.subset,
            video=str(case.video_path),
            output_dir=str(oi._output_dir(config.output_root, case)),
            success=True,
        )

    monkeypatch.setattr(oi, "run_omniinteract_case", fake_run)
    config = oi.OmniInteractBenchmarkConfig(
        data_root=str(tmp_path),
        output_root=tmp_path / "output",
        ref_audio="unused.wav",
        num_prompts=4,
        max_concurrency=2,
    )

    benchmark = await oi.run_omniinteract_benchmark(config)

    assert peak == 2
    assert benchmark.succeeded == 4
    assert json.loads((config.output_root / "batch_summary.json").read_text())["success"] == 4
    assert len((config.output_root / "official_eval_manifest.jsonl").read_text().splitlines()) == 4


@pytest.mark.parametrize(
    ("endpoint_args", "expected_endpoint"),
    [([], "/v1/realtime"), (["--endpoint", "/custom/realtime"], "/custom/realtime")],
)
def test_serve_cli_maps_omniinteract_dataset_to_realtime_config(
    tmp_path: Path,
    endpoint_args: list[str],
    expected_endpoint: str,
):
    parser = TrackingArgumentParser()
    OmniBenchmarkServingSubcommand.add_cli_args(parser)

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
            "--omniinteract-subsets",
            "1q1a",
            "1qna",
            "--num-prompts",
            "8",
            "--max-concurrency",
            "2",
            "--result-dir",
            str(tmp_path / "results"),
            "--omniinteract-ref-audio",
            "/data/reference.wav",
            "--omniinteract-require-response",
            *endpoint_args,
        ]
    )
    config = benchmark_serve._omniinteract_config_from_args(args)

    assert args.dataset_name == "omniinteract"
    assert config.endpoint == expected_endpoint
    assert config.data_root == str(tmp_path)
    assert config.dataset_repo == data.DEFAULT_OMNIINTERACT_REPO
    assert config.subsets == ("1q1a", "1qna")
    assert config.output_root == tmp_path / "results"
    assert config.num_prompts == 8
    assert config.max_concurrency == 2
    assert config.ref_audio == "/data/reference.wav"
    assert config.require_response is True


def test_serve_cli_does_not_require_omniinteract_options_for_other_datasets():
    parser = argparse.ArgumentParser()
    OmniBenchmarkServingSubcommand.add_cli_args(parser)

    args = parser.parse_args(["--dataset-name", "daily-omni"])

    assert args.dataset_name == "daily-omni"
    assert args.omniinteract_ref_audio is None


def test_serve_main_dispatches_omniinteract_runner(monkeypatch: pytest.MonkeyPatch):
    expected = {"total": 1, "success": 1, "failed": 0}
    monkeypatch.setattr(benchmark_serve, "_run_omniinteract", lambda _args: expected)

    result = benchmark_serve.main(argparse.Namespace(dataset_name="omniinteract"))

    assert result == expected
