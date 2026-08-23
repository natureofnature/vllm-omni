# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Local OmniInteract runner for MiniCPM-o native duplex serving."""

from __future__ import annotations

import asyncio
import binascii
import contextlib
import hashlib
import io
import json
import math
import subprocess
import time
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import pybase64 as base64

from vllm_omni.benchmarks.data_modules.omniinteract import (
    DEFAULT_OMNIINTERACT_REPO,
    OMNIINTERACT_SUBSETS,
    OmniInteractCase,
    case_manifest,
    discover_omniinteract_cases,
    resolve_omniinteract_root,
)
from vllm_omni.experimental.fullduplex.client import (
    PCM16_BYTES_PER_SAMPLE,
    PCM16_SAMPLE_RATE,
    RealtimeDuplexClient,
    RealtimeEventCollector,
)

OUTPUT_SAMPLE_RATE = 24_000
DEFAULT_MODEL = "openbmb/MiniCPM-o-4_5"
SUCCESS_ARTIFACTS = (".done", "output.wav", "wav_transcript.json", "events.json", "result.json")


@dataclass(frozen=True)
class OmniInteractBenchmarkConfig:
    base_url: str = "http://127.0.0.1:8000"
    endpoint: str = "/v1/realtime"
    model: str = DEFAULT_MODEL
    data_root: str | None = None
    dataset_repo: str = DEFAULT_OMNIINTERACT_REPO
    subsets: tuple[str, ...] = OMNIINTERACT_SUBSETS
    output_root: Path = Path("omniinteract-output")
    num_prompts: int = 1
    max_concurrency: int = 1
    chunk_ms: int = 200
    video_fps: float = 1.0
    timeout_s: float = 900.0
    settle_s: float = 2.0
    media_timeout_s: float = 600.0
    ref_audio: str | None = None
    pace: bool = True
    require_response: bool = False
    seed: int = 0
    disable_shuffle: bool = False


@dataclass
class OmniInteractCaseResult:
    subset: str
    video: str
    output_dir: str
    success: bool = False
    error: str = ""
    session_id: str = ""
    latency_s: float = 0.0
    input_audio_chunks: int = 0
    input_video_frames: int = 0
    pacing_mean_lag_s: float = 0.0
    pacing_max_lag_s: float = 0.0
    responses: int = 0
    audio_bytes: int = 0
    transcript: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "status": "ok" if self.success else "failed"}


@dataclass
class OmniInteractBenchmarkResult:
    results: list[OmniInteractCaseResult] = field(default_factory=list)

    @property
    def succeeded(self) -> int:
        return sum(result.success for result in self.results)

    @property
    def failed(self) -> int:
        return len(self.results) - self.succeeded

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": len(self.results),
            "success": self.succeeded,
            "failed": self.failed,
            "results": [result.as_dict() for result in self.results],
        }


def validate_config(config: OmniInteractBenchmarkConfig) -> None:
    if not config.base_url:
        raise ValueError("base_url is required")
    if not config.endpoint:
        raise ValueError("endpoint is required")
    if not config.model:
        raise ValueError("model is required")
    if not config.ref_audio:
        raise ValueError("ref_audio is required for MiniCPM-o native-duplex audio output")
    if config.num_prompts < 0:
        raise ValueError("num_prompts must be non-negative")
    if config.max_concurrency <= 0:
        raise ValueError("max_concurrency must be positive")
    if not 0 < config.chunk_ms <= 1000:
        raise ValueError("chunk_ms must be in [1, 1000]")
    if not math.isfinite(config.video_fps) or not 0 < config.video_fps <= 1:
        raise ValueError("video_fps must be in (0, 1]")
    timeouts = (config.timeout_s, config.settle_s, config.media_timeout_s)
    if not all(math.isfinite(value) for value in timeouts):
        raise ValueError("timeouts must be finite")
    if config.timeout_s <= 0 or config.settle_s < 0 or config.media_timeout_s <= 0:
        raise ValueError("timeouts must be positive and settle_s must be non-negative")
    if len(set(config.subsets)) != len(config.subsets):
        raise ValueError("subsets must not contain duplicates")
    invalid = set(config.subsets) - set(OMNIINTERACT_SUBSETS)
    if invalid:
        raise ValueError(f"Unsupported OmniInteract subsets: {sorted(invalid)}")


def _run_media_command(
    command: list[str],
    *,
    timeout_s: float,
    text: bool = False,
) -> subprocess.CompletedProcess[Any]:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=text,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"Media command timed out after {timeout_s:g}s: {command[0]}") from exc
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required media command is unavailable: {command[0]}") from exc
    if result.returncode:
        error = result.stderr if text else result.stderr.decode("utf-8", "ignore")
        raise RuntimeError(f"{command[0]} failed: {error.strip()}")
    return result


def prepare_media(video: Path, fps: float, *, timeout_s: float) -> tuple[float, bytes, list[str | None]]:
    """Decode one video to 16 kHz PCM and base64 JPEG frames."""

    duration = float(
        _run_media_command(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video),
            ],
            text=True,
            timeout_s=timeout_s,
        ).stdout.strip()
    )
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Invalid video duration for {video}: {duration!r}")
    pcm = _run_media_command(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vn",
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            str(PCM16_SAMPLE_RATE),
            "pipe:1",
        ],
        timeout_s=timeout_s,
    ).stdout
    target_bytes = math.ceil(duration) * PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE
    pcm = (pcm + bytes(max(0, target_bytes - len(pcm))))[:target_bytes]

    import imageio.v3 as iio
    from PIL import Image

    source_fps = float(iio.immeta(str(video)).get("fps") or 30)
    if not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError(f"Invalid source frame rate for {video}: {source_fps!r}")
    indices = [int((index + 0.5) * source_fps / fps) for index in range(math.ceil(duration * fps))]
    frames: list[str | None] = [None] * len(indices)
    cursor = 0
    for index, frame in enumerate(iio.imiter(str(video))):
        if cursor == len(indices):
            break
        if index < indices[cursor]:
            continue
        image = Image.fromarray(frame)
        image.thumbnail((640, 640))
        output = io.BytesIO()
        image.save(output, "JPEG", quality=85)
        encoded = base64.b64encode(output.getvalue()).decode("ascii")
        while cursor < len(indices) and index >= indices[cursor]:
            frames[cursor] = encoded
            cursor += 1
    return duration, pcm, frames


@dataclass(frozen=True)
class _AudioSegment:
    response_id: str
    start_s: float
    samples: int
    rate: int


class _Playback:
    def __init__(self) -> None:
        self.cursor = 0
        self.end_s = 0.0
        self.segments: list[_AudioSegment] = []
        self.acked: dict[str, int] = {}
        self.completed: set[str] = set()
        self.completion_acked: set[str] = set()

    async def acknowledge(self, client: RealtimeDuplexClient, now: float | None = None) -> None:
        events = client.events
        while self.cursor < len(events.events):
            index, self.cursor = self.cursor, self.cursor + 1
            event = events.events[index]
            response_id = events.response_id(event)
            if event.get("type") == "response.done":
                if response_id:
                    self.completed.add(response_id)
                continue
            if event.get("type") != "response.audio.delta":
                continue
            encoded = event.get("delta") or event.get("audio")
            if not response_id or not isinstance(encoded, str):
                continue
            try:
                samples = len(base64.b64decode(encoded, validate=True)) // PCM16_BYTES_PER_SAMPLE
            except (ValueError, binascii.Error):
                continue
            raw_rate = event.get("sample_rate_hz")
            rate = (
                raw_rate
                if isinstance(raw_rate, int) and not isinstance(raw_rate, bool) and raw_rate > 0
                else events.output_sample_rate_hz or OUTPUT_SAMPLE_RATE
            )
            start = max(events.event_received_at_s[index], self.end_s)
            self.segments.append(_AudioSegment(response_id, start, samples, rate))
            self.end_s = start + samples / rate

        now = time.monotonic() if now is None else now
        played: dict[str, int] = {}
        for segment in self.segments:
            samples = min(segment.samples, max(0, round((now - segment.start_s) * segment.rate)))
            played[segment.response_id] = played.get(segment.response_id, 0) + samples * 1000 // segment.rate
        for response_id, played_ms in played.items():
            completion_due = response_id in self.completed and response_id not in self.completion_acked
            if played_ms <= self.acked.get(response_id, -1) and not completion_due:
                continue
            await client.send(
                {
                    "type": "playback.ack",
                    "response_id": response_id,
                    "item_id": f"item_{response_id}",
                    "played_ms": played_ms,
                    "committed_ms": played_ms,
                }
            )
            self.acked[response_id] = played_ms
            if response_id in self.completed:
                self.completion_acked.add(response_id)


async def stream_inputs(
    client: RealtimeDuplexClient,
    pcm: bytes,
    frames: list[str | None],
    config: OmniInteractBenchmarkConfig,
    playback: _Playback,
) -> tuple[int, int, float, float]:
    """Pace interleaved PCM and frames over one Realtime session."""

    chunk_bytes = PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE * config.chunk_ms // 1000
    bytes_per_second = PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE
    started_at, frame_cursor, sent_frames = time.monotonic(), 0, 0
    lags: list[float] = []
    for offset in range(0, len(pcm), chunk_bytes):
        end = min(offset + chunk_bytes, len(pcm))
        end_ms = end * 1000 // bytes_per_second
        lags.append(max(0.0, time.monotonic() - (started_at + offset / bytes_per_second)))
        ready: list[str] = []
        while frame_cursor < len(frames) and end_ms >= (frame_cursor + 0.5) * 1000 / config.video_fps:
            if frames[frame_cursor]:
                ready.append(frames[frame_cursor] or "")
            frame_cursor += 1
        payload: dict[str, object] = {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm[offset:end]).decode("ascii"),
            "input_audio_format": "pcm16",
            "sample_rate_hz": PCM16_SAMPLE_RATE,
            "duration_ms": (end - offset) * 1000 // bytes_per_second,
            "audio_end_ms": end_ms,
        }
        if ready:
            payload["video_frames"] = ready
        await client.send(payload)
        sent_frames += len(ready)
        if config.pace:
            await playback.acknowledge(client)
            await asyncio.sleep(max(0.0, started_at + end_ms / 1000 - time.monotonic()))
    lags.append(max(0.0, time.monotonic() - (started_at + len(pcm) / bytes_per_second)))
    return math.ceil(len(pcm) / chunk_bytes), sent_frames, sum(lags) / len(lags), max(lags)


def _response_status(event: dict[str, object]) -> str | None:
    response = event.get("response")
    if isinstance(response, dict) and response.get("status"):
        return str(response["status"])
    status = event.get("status")
    return str(status) if status else None


def _response_states(event: dict[str, object]) -> set[object]:
    response = event.get("response")
    response = response if isinstance(response, dict) else {}
    details = event.get("status_details")
    details = details if isinstance(details, dict) else {}
    response_details = response.get("status_details")
    response_details = response_details if isinstance(response_details, dict) else {}
    return {event.get("status"), response.get("status"), details.get("type"), response_details.get("type")}


def response_ledger(collector: RealtimeEventCollector) -> tuple[set[str], set[str]]:
    """Validate exact response identities and return created/done sets."""

    created: set[str] = set()
    done: set[str] = set()
    for event in collector.events:
        event_type = event.get("type")
        if event_type not in {"response.created", "response.done"}:
            continue
        response_id = collector.response_id(event)
        if not response_id:
            raise ValueError(f"{event_type} has no response_id")
        if event_type == "response.created":
            if response_id in created:
                raise ValueError(f"duplicate response.created for {response_id}")
            created.add(response_id)
            continue
        if response_id not in created:
            raise ValueError(f"response.done without response.created for {response_id}")
        if response_id in done:
            raise ValueError(f"duplicate response.done for {response_id}")
        if "failed" in _response_states(event):
            raise ValueError(f"response.done reports failure for {response_id}")
        done.add(response_id)
    return created, done


def _raise_if_session_terminated(collector: RealtimeEventCollector, from_index: int) -> None:
    errors = collector.errors()
    if errors:
        raise RuntimeError(str(errors[-1]))
    for event in collector.events[from_index:]:
        event_type = event.get("type")
        if event_type not in {"session.expired", "session.closed"}:
            continue
        reason = event.get("reason")
        nested = event.get("event")
        if reason is None and isinstance(nested, dict):
            reason = nested.get("reason")
        detail = f": {reason}" if reason else ""
        raise RuntimeError(f"{event_type}{detail}")


def _chunk_period_ms(events: list[dict[str, object]]) -> int:
    for event in reversed(events):
        session = event.get("session")
        capabilities = session.get("capabilities") if isinstance(session, dict) else None
        value = capabilities.get("chunk_period_ms") if isinstance(capabilities, dict) else None
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return 1000


def _needs_post_commit_decision(pcm_bytes: int, events: list[dict[str, object]]) -> bool:
    period_ms = _chunk_period_ms(events)
    unit_bytes = PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE * period_ms // 1000
    created, done = _response_ledger_from_events(events)
    return bool(created - done) or bool(unit_bytes and pcm_bytes % unit_bytes)


def _response_ledger_from_events(events: list[dict[str, object]]) -> tuple[set[str], set[str]]:
    collector = RealtimeEventCollector()
    collector.events = events
    collector.event_received_at_s = [0.0] * len(events)
    return response_ledger(collector)


async def wait_for_session_completion(
    client: RealtimeDuplexClient,
    playback: _Playback,
    *,
    pcm_bytes: int,
    commit_from: int,
    session_from: int | None = None,
    timeout_s: float,
    settle_s: float,
) -> int:
    """Wait for this commit and a stable, identity-complete response ledger."""

    deadline = time.monotonic() + timeout_s
    committed_index: int | None = None
    require_decision = False
    last_event_count = len(client.events.events)
    stable_since = time.monotonic()
    while time.monotonic() < deadline:
        client.raise_if_reader_stopped()
        _raise_if_session_terminated(client.events, commit_from if session_from is None else session_from)
        await playback.acknowledge(client)
        if len(client.events.events) != last_event_count:
            last_event_count = len(client.events.events)
            stable_since = time.monotonic()
        if committed_index is None:
            committed_index = next(
                (
                    index
                    for index in range(commit_from, len(client.events.events))
                    if client.events.events[index].get("type") == "input_audio_buffer.committed"
                ),
                None,
            )
            if committed_index is not None:
                require_decision = _needs_post_commit_decision(pcm_bytes, client.events.events[: committed_index + 1])
                stable_since = time.monotonic()
        if committed_index is not None:
            created, done = response_ledger(client.events)
            post_commit = client.events.events[committed_index + 1 :]
            decision = any(
                event.get("type") == "response.listen"
                or (event.get("type") == "response.done" and _response_status(event) != "cancelled")
                for event in post_commit
            )
            if created == done and (decision or not require_decision) and time.monotonic() - stable_since >= settle_s:
                return committed_index
        await asyncio.sleep(0.05)
    missing: set[str] = set()
    with contextlib.suppress(ValueError):
        created, done = response_ledger(client.events)
        missing = created - done
    raise TimeoutError(
        "Timed out waiting for committed input and stable responses"
        + (f"; unfinished response_ids={sorted(missing)}" if missing else "")
    )


def _response_text(collector: RealtimeEventCollector, response_id: str) -> str:
    return "".join(
        str(event.get("delta") or "")
        for event in collector.events
        if collector.response_id(event) == response_id
        and event.get("type")
        in {"response.audio_transcript.delta", "response.output_text.delta", "response.text.delta"}
    )


def _output_dir(root: Path, case: OmniInteractCase) -> Path:
    relative = case.video_rel.replace("\\", "/")
    stem = Path(relative).with_suffix("").as_posix().replace("/", "__")
    digest = hashlib.sha256(f"{case.subset}/{relative}".encode()).hexdigest()[:8]
    return root / case.subset / f"{stem}--{digest}"


def _atomic_write_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value)
    temporary.replace(path)


def _atomic_write_json(path: Path, value: object) -> None:
    _atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _atomic_write_wav(path: Path, pcm: bytes, rate: int) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with wave.open(str(temporary), "wb") as output:
        output.setparams((1, PCM16_BYTES_PER_SAMPLE, rate, 0, "NONE", "not compressed"))
        output.writeframes(pcm)
    temporary.replace(path)


def _sanitized_events(collector: RealtimeEventCollector) -> list[dict[str, object]]:
    sanitized: list[dict[str, object]] = []
    for event in collector.events:
        if event.get("type") == "response.audio.delta":
            item = {key: value for key, value in event.items() if key not in {"delta", "audio"}}
            encoded = event.get("delta") or event.get("audio")
            item["audio_bytes"] = len(base64.b64decode(encoded, validate=True)) if isinstance(encoded, str) else 0
        else:
            item = dict(event)
        sanitized.append(item)
    return sanitized


def _build_output(
    collector: RealtimeEventCollector,
    *,
    stream_start: float,
    video_duration_s: float,
    require_response: bool,
) -> tuple[bytes, int, str, list[dict[str, object]]]:
    created, done = response_ledger(collector)
    if created != done:
        raise ValueError(f"unfinished response_ids: {sorted(created - done)}")
    terminal_responses = {
        response_id
        for event in collector.events
        if event.get("type") == "response.done"
        and (response_id := collector.response_id(event)) is not None
        and _response_status(event) != "cancelled"
    }
    horizon_s = math.ceil(video_duration_s)
    output = bytearray(horizon_s * OUTPUT_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE)
    cursor_s = 0.0
    response_times: dict[str, list[float]] = {}
    responses_with_audio: set[str] = set()
    for event, received_at in zip(collector.events, collector.event_received_at_s, strict=True):
        if event.get("type") != "response.audio.delta":
            continue
        response_id = collector.response_id(event)
        encoded = event.get("delta") or event.get("audio")
        if not response_id or response_id not in created:
            raise ValueError("response audio has no matching response.created")
        if event.get("format") != "pcm16":
            raise ValueError("OmniInteract output must be pcm16")
        rate = event.get("sample_rate_hz")
        if not isinstance(rate, int) or isinstance(rate, bool) or rate != OUTPUT_SAMPLE_RATE:
            raise ValueError(f"OmniInteract output must use {OUTPUT_SAMPLE_RATE} Hz audio")
        if not isinstance(encoded, str):
            raise ValueError("response audio payload is missing")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("response audio is not valid base64") from exc
        if not raw or len(raw) % PCM16_BYTES_PER_SAMPLE:
            raise ValueError("response audio is empty or not PCM16 aligned")
        start_s = max(0.0, received_at - stream_start, cursor_s)
        end_s = start_s + len(raw) / (rate * PCM16_BYTES_PER_SAMPLE)
        offset = round(start_s * rate) * PCM16_BYTES_PER_SAMPLE
        writable = max(0, min(len(raw), len(output) - offset))
        if writable:
            clipped = raw[:writable]
            output[offset : offset + writable] = clipped
            if any(clipped):
                responses_with_audio.add(response_id)
        cursor_s = end_s
        timing = response_times.setdefault(response_id, [start_s, end_s])
        timing[0], timing[1] = min(timing[0], start_s), max(timing[1], end_s)

    chunks: list[dict[str, object]] = []
    texts: list[str] = []
    text_event_types = {"response.audio_transcript.delta", "response.output_text.delta", "response.text.delta"}
    for event in collector.events:
        if event.get("type") not in text_event_types:
            continue
        response_id = collector.response_id(event)
        if not response_id or response_id not in created:
            raise ValueError("response transcript has no matching response.created")
    responses_with_text: set[str] = set()
    for response_id in collector.response_ids:
        text = _response_text(collector, response_id).strip()
        if not text:
            continue
        response_timing = response_times.get(response_id)
        if response_timing is None or response_timing[0] >= horizon_s:
            continue
        texts.append(text)
        responses_with_text.add(response_id)
        chunks.append(
            {
                "response_id": response_id,
                "text": text,
                "timestamp": [round(response_timing[0], 6), round(min(response_timing[1], horizon_s), 6)],
            }
        )
    transcript = " ".join(texts).strip()
    complete_outputs = terminal_responses & responses_with_audio & responses_with_text
    if require_response and not complete_outputs:
        raise ValueError("OmniInteract E2E requires a response with audio and transcript")
    return bytes(output), OUTPUT_SAMPLE_RATE, transcript, chunks


def write_success_artifacts(
    root: Path,
    case: OmniInteractCase,
    collector: RealtimeEventCollector,
    *,
    stream_start: float,
    video_duration_s: float,
    require_response: bool,
    result: OmniInteractCaseResult,
) -> dict[str, Any]:
    directory = _output_dir(root, case)
    directory.mkdir(parents=True, exist_ok=True)
    for name in (*SUCCESS_ARTIFACTS, ".failed.json"):
        (directory / name).unlink(missing_ok=True)
    try:
        pcm, rate, transcript, chunks = _build_output(
            collector,
            stream_start=stream_start,
            video_duration_s=video_duration_s,
            require_response=require_response,
        )
        result.audio_bytes = sum(len(collector.audio_bytes(response_id)) for response_id in collector.response_ids)
        result.transcript = transcript
        result.responses = len(collector.response_ids)
        result.output_dir = str(directory.resolve())
        result.success = True
        _atomic_write_wav(directory / "output.wav", pcm, rate)
        _atomic_write_json(directory / "wav_transcript.json", {"text": transcript, "chunks": chunks})
        _atomic_write_json(directory / "events.json", _sanitized_events(collector))
        _atomic_write_json(directory / "result.json", result.as_dict())
        summary = {
            **result.as_dict(),
            "annotation": str(case.annotation_path),
            "scene_type": "1QnA" if case.scene_type == "1qna" else case.scene_type,
        }
        _atomic_write_json(directory / ".done", summary)
        return summary
    except Exception:
        for name in SUCCESS_ARTIFACTS:
            (directory / name).unlink(missing_ok=True)
        raise


def write_failure_artifacts(root: Path, case: OmniInteractCase, result: OmniInteractCaseResult) -> None:
    directory = _output_dir(root, case)
    directory.mkdir(parents=True, exist_ok=True)
    for name in SUCCESS_ARTIFACTS:
        (directory / name).unlink(missing_ok=True)
    result.output_dir = str(directory.resolve())
    result.success = False
    summary = {
        **result.as_dict(),
        "annotation": str(case.annotation_path),
        "scene_type": "1QnA" if case.scene_type == "1qna" else case.scene_type,
    }
    _atomic_write_json(directory / ".failed.json", summary)


def clear_case_artifacts(root: Path, case: OmniInteractCase) -> None:
    """Invalidate a previous run before starting expensive preprocessing."""

    directory = _output_dir(root, case)
    directory.mkdir(parents=True, exist_ok=True)
    for name in (*SUCCESS_ARTIFACTS, ".failed.json"):
        (directory / name).unlink(missing_ok=True)


def write_batch_artifacts(
    root: Path,
    cases: list[OmniInteractCase],
    benchmark: OmniInteractBenchmarkResult,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(root / "batch_summary.json", benchmark.as_dict())
    rows = [
        case_manifest(case, _output_dir(root, case))
        for case, result in zip(cases, benchmark.results, strict=True)
        if result.success
    ]
    _atomic_write_text(
        root / "official_eval_manifest.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )


def _websocket_url(config: OmniInteractBenchmarkConfig, session_id: str) -> str:
    endpoint = (
        config.endpoint
        if urlsplit(config.endpoint).scheme
        else urljoin(config.base_url.rstrip("/") + "/", config.endpoint.lstrip("/"))
    )
    parts = urlsplit(endpoint)
    if parts.scheme not in {"http", "https", "ws", "wss"} or not parts.netloc:
        raise ValueError(f"Unsupported endpoint scheme: {parts.scheme!r}")
    if parts.scheme in {"http", "https"}:
        parts = parts._replace(scheme="ws" if parts.scheme == "http" else "wss")
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.update(
        {
            "duplex": "1",
            "model": config.model,
            "minicpmo45_native_duplex": "1",
            "autostart": "0",
            "session_id": session_id,
        }
    )
    return urlunsplit(parts._replace(query=urlencode(query)))


def _reference_audio(path: str | None) -> str | None:
    if not path:
        return None
    audio = Path(path).expanduser().resolve()
    if not audio.is_file():
        raise FileNotFoundError(f"Reference audio does not exist: {audio}")
    return "data:audio/wav;base64," + base64.b64encode(audio.read_bytes()).decode("ascii")


async def run_omniinteract_case(
    case: OmniInteractCase,
    config: OmniInteractBenchmarkConfig,
    *,
    request_index: int,
) -> OmniInteractCaseResult:
    """Run one case. This public coroutine is the hook used by E2E tests."""

    validate_config(config)
    output_dir = _output_dir(config.output_root, case)
    session_id = f"omniinteract:{case.subset}:{request_index}:{time.monotonic_ns()}"
    result = OmniInteractCaseResult(
        subset=case.subset,
        video=str(case.video_path),
        output_dir=str(output_dir.resolve()),
        session_id=session_id,
    )
    started_at = time.monotonic()
    try:
        clear_case_artifacts(config.output_root, case)
        duration, pcm, frames = await asyncio.to_thread(
            prepare_media,
            case.video_path,
            config.video_fps,
            timeout_s=config.media_timeout_s,
        )
        if not any(frames):
            raise ValueError(f"No video frames were decoded from {case.video_path}")
        async with RealtimeDuplexClient(_websocket_url(config, session_id)) as client:
            session_from = len(client.events.events)
            await client.configure(
                config.model,
                ref_audio=_reference_audio(config.ref_audio),
                session_id=session_id,
                idle_timeout_s=config.timeout_s,
                timeout_s=min(config.timeout_s, 20.0),
            )
            playback = _Playback()
            stream_start = time.monotonic()
            try:
                chunks, frame_count, mean_lag, max_lag = await stream_inputs(
                    client,
                    pcm,
                    frames,
                    config,
                    playback,
                )
                commit_from = len(client.events.events)
                await client.commit()
                await wait_for_session_completion(
                    client,
                    playback,
                    pcm_bytes=len(pcm),
                    commit_from=commit_from,
                    session_from=session_from,
                    timeout_s=config.timeout_s,
                    settle_s=config.settle_s,
                )
                await playback.acknowledge(client, time.monotonic())
                _raise_if_session_terminated(client.events, session_from)
                close_from = len(client.events.events)
                await client.close_session(timeout_s=min(config.timeout_s, 20.0))
            except Exception:
                with contextlib.suppress(Exception):
                    await client.close_session(timeout_s=min(config.timeout_s, 20.0))
                raise
            errors = client.events.errors()
            if errors:
                raise RuntimeError(str(errors[-1]))
            for index, event in enumerate(client.events.events[session_from:], start=session_from):
                if event.get("type") == "session.expired" or (
                    event.get("type") == "session.closed" and index < close_from
                ):
                    raise RuntimeError(f"Unexpected {event.get('type')} before explicit session close")
            result.latency_s = time.monotonic() - started_at
            result.input_audio_chunks = chunks
            result.input_video_frames = frame_count
            result.pacing_mean_lag_s = mean_lag
            result.pacing_max_lag_s = max_lag
            write_success_artifacts(
                config.output_root,
                case,
                client.events,
                stream_start=stream_start,
                video_duration_s=duration,
                require_response=config.require_response,
                result=result,
            )
    except Exception as exc:
        result.error = str(exc)
        result.latency_s = max(0.0, time.monotonic() - started_at)
        write_failure_artifacts(config.output_root, case, result)
    return result


async def run_omniinteract_benchmark(config: OmniInteractBenchmarkConfig) -> OmniInteractBenchmarkResult:
    """Run selected cases with bounded preprocessing and WebSocket concurrency."""

    validate_config(config)
    root = await asyncio.to_thread(resolve_omniinteract_root, config.data_root, config.dataset_repo)
    cases = await asyncio.to_thread(
        discover_omniinteract_cases,
        root,
        config.subsets,
        num_prompts=config.num_prompts,
        seed=config.seed,
        disable_shuffle=config.disable_shuffle,
    )
    config.output_root.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(config.max_concurrency)

    async def run(index: int, case: OmniInteractCase) -> OmniInteractCaseResult:
        async with semaphore:
            return await run_omniinteract_case(case, config, request_index=index)

    results = await asyncio.gather(*(run(index, case) for index, case in enumerate(cases)))
    benchmark = OmniInteractBenchmarkResult(results=list(results))
    await asyncio.to_thread(write_batch_artifacts, config.output_root, cases, benchmark)
    return benchmark
