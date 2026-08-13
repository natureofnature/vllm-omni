from __future__ import annotations

import asyncio
import base64
import io
import json
import tarfile
import threading
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_omni.benchmarks.data_modules import omniinteract as oi
from vllm_omni.benchmarks.patch.patch import (
    _attach_omniinteract,
    _prepare_omniinteract_warmup,
    _project_omniinteract_result,
    get_samples,
)
from vllm_omni.experimental.fullduplex.client import RealtimeEventCollector

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _config(output_root: str | None = None) -> oi.OmniInteractConfig:
    return oi.OmniInteractConfig(output_root=output_root)


def _case(tmp_path: Path, config: oi.OmniInteractConfig | None = None) -> oi.OmniInteractCase:
    video, annotation = tmp_path / "clip.mp4", tmp_path / "clip.json"
    video.write_bytes(b"video")
    annotation.write_text("[]")
    return oi.OmniInteractCase(
        "1q1a", "videos/clip.mp4", str(video), str(annotation), "multi_turn", config or _config()
    )


def _collector(*events: dict[str, object]) -> RealtimeEventCollector:
    collector = RealtimeEventCollector()
    for index, event in enumerate(events):
        collector.add(event, received_at_s=1 + index / 10)
    return collector


def _audio(response_id: str = "r1", value: int = 1) -> dict[str, object]:
    return {
        "type": "response.audio.delta",
        "response_id": response_id,
        "format": "pcm16",
        "sample_rate_hz": 24_000,
        "delta": base64.b64encode(bytes((value, 0)) * 2400).decode(),
    }


def _identity(event_type: str, seq: int = 7, **extra: object) -> dict[str, object]:
    key = "accepted_input_seq" if event_type.endswith("committed") else "processed_input_seq"
    return {"type": event_type, "session_id": "s", "epoch": 2, key: seq, **extra}


def test_tar_extract_is_python310_compatible_and_safe(tmp_path: Path):
    good = tmp_path / "good.tar"
    with tarfile.open(good, "w") as handle:
        member = tarfile.TarInfo("data/1q1a/videos/clip.mp4")
        member.size = 5
        handle.addfile(member, io.BytesIO(b"video"))
    root = oi._extract(good, tmp_path / "good")
    assert (root / "1q1a/videos/clip.mp4").read_bytes() == b"video"

    bad = tmp_path / "bad.tar"
    with tarfile.open(bad, "w") as handle:
        member = tarfile.TarInfo("../escape")
        member.size = 1
        handle.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(ValueError):
        oi._extract(bad, tmp_path / "bad")


def test_bench_dataset_adapter_warmup_and_metric_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    subset = tmp_path / "1q1a"
    (subset / "videos").mkdir(parents=True)
    (subset / "annotations").mkdir()
    (subset / "videos" / "a.mp4").write_bytes(b"video")
    (subset / "annotations" / "a.json").write_text("[]")
    (subset / "video_json_map.json").write_text(
        json.dumps({"entries": [{"video": "videos/a.mp4", "annotation": "annotations/a.json"}]})
    )
    monkeypatch.setattr(oi, "get_cached_tokenizer", lambda tokenizer: tokenizer)
    args = SimpleNamespace(
        dataset_name="omniinteract",
        backend="minicpmo-realtime",
        dataset_path=str(tmp_path),
        hf_name=None,
        omniinteract_root=None,
        omniinteract_subsets="1q1a",
        omniinteract_official_output_dir=None,
        omniinteract_realtime_chunk_ms=200,
        omniinteract_realtime_video_fps=1.0,
        omniinteract_realtime_ref_audio=None,
        omniinteract_realtime_no_pace=False,
        omniinteract_realtime_timeout_s=30,
        no_oversample=True,
        num_prompts=1,
        request_id_prefix="req-",
        seed=0,
        disable_shuffle=True,
    )
    [sample] = get_samples(args, SimpleNamespace(encode=lambda text: list(text)))
    request = SimpleNamespace()
    _attach_omniinteract(sample, request)
    warmups = [_prepare_omniinteract_warmup(sample, index) for index in range(2)]
    assert request.omniinteract.video_rel == "videos/a.mp4"
    assert warmups[0].request_id != warmups[1].request_id
    assert all(item.omniinteract.config.output_root is None for item in warmups)

    result = {"mean_ttft_ms": 1, "p99_tpot_ms": 2, "total_output_tokens": 3}
    _project_omniinteract_result(result, [SimpleNamespace(omniinteract={"success": True})], [sample])
    assert "mean_ttft_ms" not in result and "total_output_tokens" not in result
    assert result["omniinteract_realtime_metric_scope"].startswith("continuous_session")

    args.omniinteract_official_output_dir, args.no_oversample = str(tmp_path / "out"), False
    with pytest.raises(ValueError, match="no-oversample"):
        get_samples(args, SimpleNamespace(encode=lambda text: list(text)))


@pytest.mark.asyncio
async def test_final_watermark_accepts_exact_precommit_speak():
    await oi._wait_final(
        _collector(
            _identity("input_audio_buffer.processed", outcome="speak", response_id="r1"),
            {"type": "response.created", "response": {"id": "r1"}},
            _audio(),
            {"type": "response.done", "response": {"id": "r1", "status": "completed"}},
            _identity("input_audio_buffer.committed"),
        ),
        0,
        0.1,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("events", "error"),
    [
        ([_identity("input_audio_buffer.processed", outcome="listen")], TimeoutError),
        (
            [_identity("input_audio_buffer.committed"), _identity("input_audio_buffer.processed", 8, outcome="listen")],
            TimeoutError,
        ),
        (
            [_identity("input_audio_buffer.committed"), _identity("input_audio_buffer.processed", outcome="unknown")],
            RuntimeError,
        ),
        ([_identity("input_audio_buffer.committed"), {"type": "session.closed", "reason": "timeout"}], RuntimeError),
    ],
)
async def test_final_watermark_fails_closed(events: list[dict[str, object]], error: type[Exception]):
    with pytest.raises(error):
        await oi._wait_final(_collector(*events), 0, 0.03)


def test_official_output_and_artifacts_are_strict(tmp_path: Path):
    case = _case(tmp_path)
    with pytest.raises(ValueError):
        oi.validate_output(_collector({**_audio(), "delta": "not base64"}))
    with pytest.raises(ValueError, match="failure"):
        oi.validate_output(_collector({"type": "response.done", "status": "failed"}))

    collector = _collector(
        {"type": "response.created", "response": {"id": "r1"}},
        {"type": "response.output_text.delta", "response_id": "r1", "delta": "one"},
        _audio("r1", 1),
        {"type": "response.created", "response": {"id": "r2"}},
        {"type": "response.output_text.delta", "response_id": "r2", "delta": "two"},
        _audio("r2", 2),
        {"type": "response.done", "response": {"id": "r1", "status": "completed"}},
        {"type": "response.done", "response": {"id": "r2", "status": "completed"}},
    )
    output = tmp_path / "output"
    success = oi.write_artifacts(output, case, collector, 1.0, 2.2, 2.3, {})
    failed = oi.failure_summary(output, oi.OmniInteractCase(**{**case.__dict__, "video_rel": "fail.mp4"}), "x")
    oi.write_batch(output, [success, failed])
    directory = Path(success["output_dir"])
    with wave.open(str(directory / "output.wav")) as wav_file:
        assert wav_file.getnframes() == 3 * 24_000
    transcript = json.loads((directory / "wav_transcript.json").read_text())
    assert [chunk["text"] for chunk in transcript["chunks"]] == ["one", "two"]
    assert len((output / "official_eval_manifest.jsonl").read_text().splitlines()) == 1


class _Client:
    def __init__(self) -> None:
        self.events = RealtimeEventCollector()
        self.sent: list[dict[str, object]] = []

    async def send(self, payload: dict[str, object]) -> None:
        self.sent.append(payload)


@pytest.mark.asyncio
async def test_stream_paces_audio_video_and_playback_ack():
    client = _Client()
    pcm = bytes(oi.PCM16_SAMPLE_RATE * oi.PCM16_BYTES_PER_SAMPLE)
    chunks, frames, _, _ = await oi._stream(client, pcm, ["frame"], _config(), oi._Playback())
    assert (chunks, frames) == (5, 1)
    assert [index for index, event in enumerate(client.sent) if event.get("video_frames")] == [2]


class _Realtime(_Client):
    fail = False
    instance: _Realtime | None = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        _Realtime.instance = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args) -> None:
        return None

    async def configure(self, *args, **kwargs) -> None:
        self.events.add({"type": "session.created"})

    async def commit(self) -> None:
        self.events.add(_identity("input_audio_buffer.committed", 1))
        self.events.add(
            {"type": "error", "message": "backend failed"}
            if self.fail
            else _identity("input_audio_buffer.processed", 1, outcome="listen")
        )

    async def close_session(self, **kwargs) -> None:
        self.events.add({"type": "session.closed"})


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_session_run_closes_and_writes_terminal_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: bool
):
    output = tmp_path / "output"
    _Realtime.fail = fail
    monkeypatch.setattr(oi, "RealtimeDuplexClient", _Realtime)
    monkeypatch.setattr(oi, "prepare_media", lambda *args: (1.0, bytes(32_000), [None]))

    async def no_sleep(*args) -> None:
        return None

    monkeypatch.setattr(oi.asyncio, "sleep", no_sleep)
    result = await oi.run_omniinteract(
        _case(tmp_path, _config(str(output))), "http://server/v1/realtime", "model", "request"
    )
    marker = ".failed.json" if fail else ".done"
    assert result.success is not fail
    assert (Path(result.official_summary["output_dir"]) / marker).is_file()
    assert _Realtime.instance
    assert _Realtime.instance.events.count("session.closed") == 1


@pytest.mark.asyncio
async def test_nonofficial_residual_waits_for_legacy_decision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class Legacy(_Realtime):
        async def commit(self) -> None:
            self.events.add({"type": "input_audio_buffer.committed"})
            asyncio.get_running_loop().call_soon(self.events.add, {"type": "response.listen"})

    monkeypatch.setattr(oi, "RealtimeDuplexClient", Legacy)
    monkeypatch.setattr(oi, "prepare_media", lambda *args: (1.0, bytes(32_002), []))

    async def streamed(*args):
        return 1, 0, 0.0, 0.0

    monkeypatch.setattr(oi, "_stream", streamed)
    result = await oi.run_omniinteract(_case(tmp_path), "http://server", "model", "legacy")
    assert result.success and Legacy.instance.events.count("response.listen") == 1


@pytest.mark.asyncio
async def test_slow_artifact_writer_does_not_block_peer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    started, release = threading.Event(), threading.Event()
    _Realtime.fail = False
    monkeypatch.setattr(oi, "RealtimeDuplexClient", _Realtime)
    monkeypatch.setattr(oi, "prepare_media", lambda *args: (1.0, bytes(32_000), [None]))

    async def no_sleep(*args):
        return None

    monkeypatch.setattr(oi.asyncio, "sleep", no_sleep)

    def slow_writer(*args, **kwargs):
        started.set()
        assert release.wait(2)
        return {"status": "ok", "output_dir": str(tmp_path / "out")}

    monkeypatch.setattr(oi, "write_artifacts", slow_writer)
    first = asyncio.create_task(
        oi.run_omniinteract(_case(tmp_path, _config(str(tmp_path / "out"))), "http://server", "model", "one")
    )
    assert await asyncio.to_thread(started.wait, 1)
    peer = await asyncio.wait_for(oi.run_omniinteract(_case(tmp_path), "http://server", "model", "two"), 0.5)
    release.set()
    assert peer.success and (await first).success
