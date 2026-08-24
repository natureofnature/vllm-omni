# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""OmniInteract session execution and artifacts for serving benchmarks."""

from __future__ import annotations

import asyncio
import binascii
import contextlib
import copy
import fcntl
import hashlib
import json
import logging
import math
import os
import subprocess
import tempfile
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, cast
from urllib.parse import urljoin, urlsplit

import pybase64 as base64

from vllm_omni.benchmarks.data_modules.omniinteract_dataset import (
    OmniInteractCase,
    OmniInteractPreparedInput,
    case_manifest,
)
from vllm_omni.experimental.fullduplex.client import (
    PCM16_BYTES_PER_SAMPLE,
    PCM16_SAMPLE_RATE,
    RealtimeDuplexClient,
    RealtimeEventCollector,
    build_realtime_url,
    chunk_period_ms,
    has_residual_model_unit,
    reference_audio_data_url,
    summarize_session_request_metrics,
    write_pcm16_wav,
)

OUTPUT_SAMPLE_RATE = 24_000
DEFAULT_MODEL = "openbmb/MiniCPM-o-4_5"
SUCCESS_ARTIFACTS = (".done", "output.wav", "wav_transcript.json", "events.json", "result.json")
BATCH_ARTIFACTS = ("batch_summary.json", "official_eval_manifest.jsonl")
ARTIFACT_LOCK_FILE = ".omniinteract.lock"
_INPUT_CHUNK_MS = 200
VIDEO_FPS = 1.0
_COMPLETION_SETTLE_S = 2.0
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OmniInteractBenchmarkConfig:
    base_url: str = "http://127.0.0.1:8000"
    endpoint: str = "/v1/realtime"
    model: str = DEFAULT_MODEL
    output_root: Path = Path("omniinteract-output")
    timeout_s: float = 900.0
    media_timeout_s: float = 600.0
    ref_audio: str | None = None
    require_response: bool = False


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
    audio_clipped_bytes: int = 0
    transcript: str = ""
    eligible_for_official_eval: bool = False
    official_eval_ineligible_reasons: list[str] = field(default_factory=list)
    artifact_warnings: list[str] = field(default_factory=list)
    output_tokens: int = 0
    duplex_request_metrics: list[dict[str, object]] = field(default_factory=list)
    duplex_session_metrics: dict[str, object] = field(default_factory=dict)
    _artifact_context: _DeferredArtifactContext | None = field(default=None, repr=False, compare=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            item.name: copy.deepcopy(getattr(self, item.name))
            for item in fields(self)
            if item.name != "_artifact_context"
        }


@dataclass(frozen=True)
class _ArtifactAudioSpan:
    offset: int
    pcm16: bytes


@dataclass(frozen=True)
class _DeferredArtifactContext:
    horizon_bytes: int
    spans: tuple[_ArtifactAudioSpan, ...]
    rate: int
    chunks: list[dict[str, object]]
    events: list[dict[str, object]]


def benchmark_summary(results: list[OmniInteractCaseResult]) -> dict[str, Any]:
    succeeded = sum(result.success for result in results)
    eligible = sum(result.success and result.eligible_for_official_eval for result in results)
    return {
        "total": len(results),
        "success": succeeded,
        "failed": len(results) - succeeded,
        "eligible_for_official_eval": eligible,
        "successful_but_ineligible": succeeded - eligible,
        "audio_clipped_bytes": sum(result.audio_clipped_bytes for result in results),
        "results": [result.as_dict() for result in results],
    }


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

    frame_count = math.ceil(duration * fps)
    frames: list[str | None] = [None] * frame_count
    with tempfile.TemporaryDirectory(prefix="vllm-omni-frames-") as temp_dir:
        output_pattern = Path(temp_dir) / "frame-%06d.jpg"
        frame_filter = f"select=gte(t\\,(selected_n+0.5)/{fps}),scale=640:640:force_original_aspect_ratio=decrease"
        _run_media_command(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-i",
                str(video),
                "-an",
                "-vf",
                frame_filter,
                "-frames:v",
                str(frame_count),
                "-vsync",
                "vfr",
                "-q:v",
                "5",
                str(output_pattern),
            ],
            timeout_s=timeout_s,
        )
        for index, frame_path in enumerate(sorted(Path(temp_dir).glob("frame-*.jpg"))[:frame_count]):
            frames[index] = base64.b64encode(frame_path.read_bytes()).decode("ascii")
    return duration, pcm, frames


@dataclass(frozen=True)
class _AudioSegment:
    event_index: int
    response_id: str
    start_s: float
    pcm16: bytes
    rate: int

    @property
    def samples(self) -> int:
        return len(self.pcm16) // PCM16_BYTES_PER_SAMPLE

    @property
    def end_s(self) -> float:
        return self.start_s + self.samples / self.rate


class _Playback:
    def __init__(self) -> None:
        self.cursor = 0
        self.end_s = 0.0
        self.segments: list[_AudioSegment] = []
        self.acked: dict[str, int] = {}
        self.completed: set[str] = set()
        self.completion_acked: set[str] = set()
        self.warnings: list[str] = []
        self._total_samples: dict[str, int] = {}
        self._fully_played_samples: dict[str, int] = {}
        self._drain_cursor = 0

    def _warn_once(self, warning: str) -> None:
        if warning not in self.warnings:
            self.warnings.append(warning)

    def ingest(self, events: RealtimeEventCollector) -> None:
        """Decode each new audio delta exactly once into the playback queue."""
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
            if not response_id:
                raise ValueError("response audio has no response_id")
            if not isinstance(encoded, str):
                raise ValueError("response audio payload is missing")
            output_format = event.get("format")
            if output_format is None:
                self._warn_once("response.audio.delta omitted format; assumed pcm16")
            elif output_format != "pcm16":
                raise ValueError("OmniInteract output must be pcm16")
            raw_rate = event.get("sample_rate_hz")
            if raw_rate is None:
                rate = events.output_sample_rate_hz or OUTPUT_SAMPLE_RATE
                self._warn_once(f"response.audio.delta omitted sample_rate_hz; assumed {rate}")
            elif isinstance(raw_rate, int) and not isinstance(raw_rate, bool) and raw_rate > 0:
                rate = raw_rate
            else:
                raise ValueError("response audio sample_rate_hz must be a positive integer")
            if rate != OUTPUT_SAMPLE_RATE:
                raise ValueError(f"OmniInteract output must use {OUTPUT_SAMPLE_RATE} Hz audio")
            try:
                raw = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ValueError("response audio is not valid base64") from exc
            if not raw or len(raw) % PCM16_BYTES_PER_SAMPLE:
                raise ValueError("response audio is empty or not PCM16 aligned")
            start = max(events.event_received_at_s[index], self.end_s)
            segment = _AudioSegment(index, response_id, start, raw, rate)
            self.segments.append(segment)
            self.end_s = segment.end_s
            self._total_samples[response_id] = self._total_samples.get(response_id, 0) + segment.samples

    async def acknowledge(self, client: RealtimeDuplexClient, now: float | None = None) -> None:
        events = client.events
        self.ingest(events)
        now = time.monotonic() if now is None else now
        while self._drain_cursor < len(self.segments) and self.segments[self._drain_cursor].end_s <= now:
            segment = self.segments[self._drain_cursor]
            self._fully_played_samples[segment.response_id] = (
                self._fully_played_samples.get(segment.response_id, 0) + segment.samples
            )
            self._drain_cursor += 1
        partial_response_id: str | None = None
        partial_samples = 0
        if self._drain_cursor < len(self.segments):
            segment = self.segments[self._drain_cursor]
            if now > segment.start_s:
                partial_response_id = segment.response_id
                partial_samples = min(segment.samples, round((now - segment.start_s) * segment.rate))
        for response_id in self.completed - self.completion_acked:
            played_samples = self._fully_played_samples.get(response_id, 0)
            if response_id == partial_response_id:
                played_samples += partial_samples
            total_samples = self._total_samples.get(response_id, 0)
            if not total_samples or played_samples < total_samples:
                continue
            played_ms = played_samples * 1000 // OUTPUT_SAMPLE_RATE
            await client.send_playback_ack(response_id, played_ms)
            self.acked[response_id] = played_ms
            self.completion_acked.add(response_id)


async def stream_inputs(
    client: RealtimeDuplexClient,
    pcm: bytes,
    frames: Sequence[str | None],
    playback: _Playback,
) -> tuple[int, int, float, float]:
    """Pace interleaved PCM and frames over one Realtime session."""

    frame_cursor, sent_frames = 0, 0

    def chunk_hints(_offset: int, end_ms: int) -> dict[str, object]:
        nonlocal frame_cursor, sent_frames
        ready: list[str] = []
        while frame_cursor < len(frames) and end_ms >= (frame_cursor + 0.5) * 1000 / VIDEO_FPS:
            if frames[frame_cursor]:
                ready.append(frames[frame_cursor] or "")
            frame_cursor += 1
        sent_frames += len(ready)
        return {"video_frames": ready} if ready else {}

    async def on_chunk_sent(_end: int, _end_ms: int) -> None:
        await playback.acknowledge(client)

    stats = await client.stream_pcm16(
        pcm,
        chunk_ms=_INPUT_CHUNK_MS,
        realtime=True,
        chunk_hints=chunk_hints,
        on_chunk_sent=on_chunk_sent,
    )
    return stats.chunks, sent_frames, stats.mean_lag_s, stats.max_lag_s


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


def _active_response_ids_before(collector: RealtimeEventCollector, end: int) -> set[str]:
    created: set[str] = set()
    done: set[str] = set()
    for event in collector.events[:end]:
        response_id = collector.response_id(event)
        if not response_id:
            continue
        if event.get("type") == "response.created":
            created.add(response_id)
        elif event.get("type") == "response.done":
            done.add(response_id)
    return created - done


def _commit_defers_response(event: dict[str, object]) -> bool:
    committed = event.get("event")
    return isinstance(committed, dict) and committed.get("overlap_deferred") is True


def _is_model_listen_decision(event: dict[str, object]) -> bool:
    """Distinguish a model LISTEN decision from a buffering notification."""

    metadata: dict[str, object] = event
    response = event.get("response")
    if isinstance(response, dict) and isinstance(response.get("metadata"), dict):
        metadata = response["metadata"]
    elif isinstance(event.get("metadata"), dict):
        metadata = event["metadata"]

    model_listen = metadata.get("model_listen")
    return metadata.get("buffering") is not True and model_listen is True


def _has_post_commit_decision(
    collector: RealtimeEventCollector,
    events: list[dict[str, object]],
    *,
    prior_response_ids: set[str],
) -> bool:
    """Ignore terminals that only release a deferred final input."""

    pending_prior = set(prior_response_ids)
    for event in events:
        event_type = event.get("type")
        if event_type == "response.done":
            response_id = collector.response_id(event)
            if response_id in pending_prior:
                pending_prior.remove(response_id)
                continue
            if not pending_prior and _response_status(event) != "cancelled":
                return True
        elif event_type == "response.listen" and not pending_prior and _is_model_listen_decision(event):
            return True
    return False


def _raise_if_session_terminated(collector: RealtimeEventCollector, from_index: int) -> None:
    errors = collector.errors()
    if errors:
        raise RuntimeError(str(errors[-1]))
    for event in collector.events[from_index:]:
        event_type = event.get("type")
        if event_type not in {"session.expired", "session.closed"}:
            continue
        reason = _session_close_reason(event)
        detail = f": {reason}" if reason else ""
        raise RuntimeError(f"{event_type}{detail}")


def _session_close_reason(event: dict[str, object]) -> object | None:
    reason = event.get("reason")
    nested = event.get("event")
    if reason is None and isinstance(nested, dict):
        reason = nested.get("reason")
    return reason


def _validate_explicit_session_close(
    collector: RealtimeEventCollector,
    *,
    session_from: int,
    close_from: int,
) -> None:
    for index, event in enumerate(collector.events[session_from:], start=session_from):
        event_type = event.get("type")
        reason = _session_close_reason(event)
        if event_type == "session.expired" or (
            event_type == "session.closed" and (index < close_from or reason is not None)
        ):
            detail = f": {reason}" if reason else ""
            raise RuntimeError(f"Unexpected {event_type}{detail} before explicit session close completed")


def _ensure_final_commit_tail(pcm: bytes, events: list[dict[str, object]]) -> bytes:
    """Keep one almost-full model unit for the final commit decision.

    A complete unit is emitted before commit and its asynchronous decision has
    no commit correlation. Removing one sample keeps that unit buffered until
    commit without adding silence or materially changing the input.
    """

    period_ms = chunk_period_ms(events)
    if len(pcm) >= PCM16_BYTES_PER_SAMPLE and not has_residual_model_unit(pcm, chunk_period_ms=period_ms):
        return pcm[:-PCM16_BYTES_PER_SAMPLE]
    return pcm


async def wait_for_session_completion(
    client: RealtimeDuplexClient,
    playback: _Playback,
    *,
    commit_from: int,
    session_from: int | None = None,
    timeout_s: float,
    settle_s: float,
) -> int:
    """Wait for this commit and a stable, identity-complete response ledger."""

    deadline = time.monotonic() + timeout_s
    committed_index: int | None = None
    prior_response_ids: set[str] = set()
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
                committed_event = client.events.events[committed_index]
                if _commit_defers_response(committed_event):
                    prior_response_ids = _active_response_ids_before(client.events, committed_index)
                stable_since = time.monotonic()
        if committed_index is not None:
            created, done = response_ledger(client.events)
            post_commit = client.events.events[committed_index + 1 :]
            decision = _has_post_commit_decision(
                client.events,
                post_commit,
                prior_response_ids=prior_response_ids,
            )
            if created == done and decision and time.monotonic() - stable_since >= settle_s:
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


def _output_dir(root: Path, case: OmniInteractCase) -> Path:
    relative = case.video_rel.replace("\\", "/")
    stem = Path(relative).with_suffix("").as_posix().replace("/", "__")
    digest = hashlib.sha256(f"{case.subset}/{relative}".encode()).hexdigest()[:8]
    return root / case.subset / f"{stem}--{digest}"


@contextlib.contextmanager
def _artifact_output_lock(root: Path) -> Iterator[None]:
    """Serialize artifact mutations by benchmark processes sharing a root."""

    root.mkdir(parents=True, exist_ok=True)
    with (root / ARTIFACT_LOCK_FILE).open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _atomic_write_text(path: Path, value: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(value, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: object) -> None:
    _atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _atomic_write_wav(path: Path, pcm: bytes | bytearray, rate: int) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        # wave.writeframes accepts bytearray; retain the single materialized
        # horizon buffer instead of copying it solely for the helper's annotation.
        write_pcm16_wav(temporary, cast(bytes, pcm), sample_rate_hz=rate)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _sanitized_events(
    collector: RealtimeEventCollector,
    *,
    audio_bytes_by_event: dict[int, int] | None = None,
) -> list[dict[str, object]]:
    sanitized: list[dict[str, object]] = []
    for index, event in enumerate(collector.events):
        if event.get("type") == "response.audio.delta":
            item = {key: value for key, value in event.items() if key not in {"delta", "audio"}}
            if audio_bytes_by_event is not None:
                item["audio_bytes"] = audio_bytes_by_event[index]
            else:
                encoded = event.get("delta") or event.get("audio")
                item["audio_bytes"] = len(base64.b64decode(encoded, validate=True)) if isinstance(encoded, str) else 0
        else:
            item = dict(event)
        sanitized.append(item)
    return sanitized


def _artifact_summary(case: OmniInteractCase, result: OmniInteractCaseResult) -> dict[str, Any]:
    return {
        **result.as_dict(),
        "annotation": str(case.annotation_path),
        "scene_type": "1QnA" if case.scene_type == "1qna" else case.scene_type,
    }


def _publish_success_artifacts(
    directory: Path,
    result: OmniInteractCaseResult,
    *,
    pcm: bytes | bytearray,
    rate: int,
    chunks: list[dict[str, object]],
    events: list[dict[str, object]],
    summary: dict[str, Any],
) -> None:
    _atomic_write_wav(directory / "output.wav", pcm, rate)
    _atomic_write_json(
        directory / "wav_transcript.json",
        {
            "text": result.transcript,
            "chunks": chunks,
            "timestamp_semantics": "serialized playback queue time relative to input streaming start",
        },
    )
    _atomic_write_json(directory / "events.json", events)
    _atomic_write_json(directory / "result.json", result.as_dict())
    _atomic_write_json(directory / ".done", summary)


def _replace_success_artifacts(
    root: Path,
    directory: Path,
    result: OmniInteractCaseResult,
    *,
    pcm: bytes | bytearray,
    rate: int,
    chunks: list[dict[str, object]],
    events: list[dict[str, object]],
    summary: dict[str, Any],
) -> None:
    with _artifact_output_lock(root):
        directory.mkdir(parents=True, exist_ok=True)
        try:
            for name in (*SUCCESS_ARTIFACTS, ".failed.json"):
                (directory / name).unlink(missing_ok=True)
            _publish_success_artifacts(
                directory,
                result,
                pcm=pcm,
                rate=rate,
                chunks=chunks,
                events=events,
                summary=summary,
            )
        except Exception:
            for name in SUCCESS_ARTIFACTS:
                (directory / name).unlink(missing_ok=True)
            raise


def _build_output(
    collector: RealtimeEventCollector,
    playback: _Playback,
    *,
    stream_start: float,
    video_duration_s: float,
    require_response: bool,
    materialize_pcm: bool,
) -> tuple[
    bytes | None,
    int,
    str,
    list[dict[str, object]],
    int,
    int,
    tuple[_ArtifactAudioSpan, ...],
]:
    playback.ingest(collector)
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
    horizon_bytes = horizon_s * OUTPUT_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE
    output = bytearray(horizon_bytes) if materialize_pcm else None
    spans: list[_ArtifactAudioSpan] = []
    clipped_bytes = 0
    response_times: dict[str, list[float]] = {}
    responses_with_audio: set[str] = set()
    for segment in playback.segments:
        response_id = segment.response_id
        if response_id not in created:
            raise ValueError("response audio has no matching response.created")
        start_s = max(0.0, segment.start_s - stream_start)
        end_s = max(start_s, segment.end_s - stream_start)
        offset = round(start_s * segment.rate) * PCM16_BYTES_PER_SAMPLE
        writable = max(0, min(len(segment.pcm16), horizon_bytes - offset))
        clipped_bytes += len(segment.pcm16) - writable
        if writable:
            clipped = segment.pcm16[:writable]
            spans.append(_ArtifactAudioSpan(offset=offset, pcm16=clipped))
            if output is not None:
                output[offset : offset + writable] = clipped
            if any(clipped):
                responses_with_audio.add(response_id)
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
        text = collector.response_text(response_id).strip()
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
    pcm = bytes(output) if output is not None else None
    return pcm, OUTPUT_SAMPLE_RATE, transcript, chunks, clipped_bytes, horizon_bytes, tuple(spans)


def write_success_artifacts(
    root: Path,
    case: OmniInteractCase,
    collector: RealtimeEventCollector,
    *,
    playback: _Playback | None = None,
    stream_start: float,
    video_duration_s: float,
    require_response: bool,
    result: OmniInteractCaseResult,
    persist: bool = True,
    defer: bool = False,
) -> dict[str, Any]:
    if persist and defer:
        raise ValueError("persist and defer are mutually exclusive")
    directory = _output_dir(root, case)
    if persist:
        clear_case_artifacts(root, case)
    publishing = False
    try:
        playback = playback or _Playback()
        pcm, rate, transcript, chunks, clipped_bytes, horizon_bytes, spans = _build_output(
            collector,
            playback,
            stream_start=stream_start,
            video_duration_s=video_duration_s,
            require_response=require_response,
            materialize_pcm=persist,
        )
        result.audio_bytes = sum(len(segment.pcm16) for segment in playback.segments)
        result.audio_clipped_bytes = clipped_bytes
        result.transcript = transcript
        result.responses = len(collector.response_ids)
        result.output_dir = str(directory.resolve())
        result.success = True
        result.artifact_warnings = list(playback.warnings)
        reasons: list[str] = []
        if clipped_bytes:
            reasons.append("audio_clipped")
        if any(
            event.get("type") == "response.done" and _response_status(event) == "cancelled"
            for event in collector.events
        ):
            reasons.append("cancelled_response")
        result.official_eval_ineligible_reasons = reasons
        result.eligible_for_official_eval = not reasons
        summary = _artifact_summary(case, result)
        events: list[dict[str, object]] | None = None
        if persist or defer:
            events = _sanitized_events(
                collector,
                audio_bytes_by_event={segment.event_index: len(segment.pcm16) for segment in playback.segments},
            )
        if defer:
            assert events is not None
            result._artifact_context = _DeferredArtifactContext(
                horizon_bytes=horizon_bytes,
                spans=spans,
                rate=rate,
                chunks=chunks,
                events=events,
            )
        elif persist:
            assert pcm is not None and events is not None
            publishing = True
            _replace_success_artifacts(
                root,
                directory,
                result,
                pcm=pcm,
                rate=rate,
                chunks=chunks,
                events=events,
                summary=summary,
            )
        return summary
    except Exception:
        result.success = False
        result.eligible_for_official_eval = False
        if publishing and "artifact_write_failed" not in result.official_eval_ineligible_reasons:
            result.official_eval_ineligible_reasons.append("artifact_write_failed")
        raise


def write_failure_artifacts(root: Path, case: OmniInteractCase, result: OmniInteractCaseResult) -> None:
    directory = _output_dir(root, case)
    result.output_dir = str(directory.resolve())
    result.success = False
    result.eligible_for_official_eval = False
    with _artifact_output_lock(root):
        directory.mkdir(parents=True, exist_ok=True)
        for name in SUCCESS_ARTIFACTS:
            (directory / name).unlink(missing_ok=True)
        _atomic_write_json(directory / ".failed.json", _artifact_summary(case, result))


def publish_deferred_case_artifacts(
    root: Path,
    case: OmniInteractCase,
    result: OmniInteractCaseResult,
) -> None:
    """Publish one measured case after the benchmark clock is frozen."""

    context = result._artifact_context
    result._artifact_context = None
    if not result.success:
        write_failure_artifacts(root, case, result)
        return
    if context is None:
        raise RuntimeError("OmniInteract benchmark output lost its deferred artifact context")
    directory = _output_dir(root, case)
    try:
        pcm = bytearray(context.horizon_bytes)
        for span in context.spans:
            pcm[span.offset : span.offset + len(span.pcm16)] = span.pcm16
        _replace_success_artifacts(
            root,
            directory,
            result,
            pcm=pcm,
            rate=context.rate,
            chunks=context.chunks,
            events=context.events,
            summary=_artifact_summary(case, result),
        )
    except Exception:
        result.success = False
        result.eligible_for_official_eval = False
        if "artifact_write_failed" not in result.official_eval_ineligible_reasons:
            result.official_eval_ineligible_reasons.append("artifact_write_failed")
        raise


def clear_case_artifacts(root: Path, case: OmniInteractCase) -> None:
    """Invalidate a previous run before starting expensive preprocessing."""

    directory = _output_dir(root, case)
    with _artifact_output_lock(root):
        directory.mkdir(parents=True, exist_ok=True)
        for name in (*SUCCESS_ARTIFACTS, ".failed.json"):
            (directory / name).unlink(missing_ok=True)


def clear_batch_artifacts(root: Path) -> None:
    """Invalidate aggregate handoff files before a measured batch starts."""

    with _artifact_output_lock(root):
        for name in BATCH_ARTIFACTS:
            (root / name).unlink(missing_ok=True)


def write_batch_artifacts(
    root: Path,
    cases: list[OmniInteractCase],
    results: list[OmniInteractCaseResult],
) -> None:
    rows = [
        case_manifest(case, _output_dir(root, case))
        for case, result in zip(cases, results, strict=True)
        if result.success and result.eligible_for_official_eval
    ]
    ineligible = [result for result in results if result.success and not result.eligible_for_official_eval]
    if ineligible:
        reason_counts: dict[str, int] = {}
        for result in ineligible:
            for reason in result.official_eval_ineligible_reasons or ["unspecified"]:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        logger.warning(
            "%d successful OmniInteract cases were excluded from official evaluation: %s",
            len(ineligible),
            ", ".join(f"{reason}={count}" for reason, count in sorted(reason_counts.items())),
        )
    with _artifact_output_lock(root):
        _atomic_write_json(root / "batch_summary.json", benchmark_summary(results))
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
    return build_realtime_url(
        endpoint,
        config.model,
        autostart=False,
        native_duplex=True,
        session_id=session_id,
    )


def _populate_response_metrics(
    result: OmniInteractCaseResult,
    collector: RealtimeEventCollector,
    *,
    stream_start: float,
) -> None:
    measurement_origin = {
        "ttft": "response.created client receive to first non-empty text delta",
        "ttfp": "response.created client receive to first audio packet",
        "rtf": "response.created client receive to last audio packet divided by emitted audio duration",
    }
    request_metrics: list[dict[str, object]] = []
    output_tokens = 0
    for request_index, response_id in enumerate(collector.response_ids):
        timing = collector.timing_summary(
            after_s=stream_start,
            input_committed_at_s=None,
            response_id=response_id,
            measurement_origin=measurement_origin,
        )
        metric = timing.get("request_metrics")
        stage0 = timing.get("stage0_tokens")
        if isinstance(metric, dict) or isinstance(stage0, dict):
            request_metric = {
                "session_id": result.session_id,
                "request_index": request_index,
                "response_id": response_id,
            }
            if isinstance(metric, dict):
                request_metric.update(metric)
            if isinstance(stage0, dict):
                request_metric["stage0_tokens"] = dict(stage0)
            request_metrics.append(request_metric)
        if isinstance(stage0, dict):
            output_tokens += int(stage0.get("output_token_count") or 0)
    result.output_tokens = output_tokens
    result.duplex_request_metrics = request_metrics
    result.duplex_session_metrics = summarize_session_request_metrics(
        request_metrics,
        session_id=result.session_id,
    )


async def run_omniinteract_case(
    case: OmniInteractCase,
    config: OmniInteractBenchmarkConfig,
    *,
    request_index: int | str,
    persist_artifacts: bool = True,
    defer_artifacts: bool = False,
    prepared_input: OmniInteractPreparedInput | None = None,
) -> OmniInteractCaseResult:
    """Run one case. This public coroutine is the hook used by E2E tests."""

    if not config.ref_audio:
        raise ValueError("ref_audio is required for MiniCPM-o native-duplex audio output")
    if not math.isfinite(config.timeout_s) or config.timeout_s <= 0:
        raise ValueError("timeout_s must be finite and positive")
    if not math.isfinite(config.media_timeout_s) or config.media_timeout_s <= 0:
        raise ValueError("media_timeout_s must be finite and positive")
    if persist_artifacts and defer_artifacts:
        raise ValueError("persist_artifacts and defer_artifacts are mutually exclusive")
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
        if persist_artifacts:
            await asyncio.to_thread(clear_case_artifacts, config.output_root, case)
        if prepared_input is None:
            reference_audio = reference_audio_data_url(config.ref_audio)
            duration, pcm, frames = await asyncio.to_thread(
                prepare_media,
                case.video_path,
                VIDEO_FPS,
                timeout_s=config.media_timeout_s,
            )
        else:
            reference_audio = prepared_input.ref_audio_data_url
            duration = prepared_input.duration_s
            pcm = prepared_input.pcm16
            frames = prepared_input.video_frames
        if not any(frames):
            raise ValueError(f"No video frames were decoded from {case.video_path}")
        async with RealtimeDuplexClient(_websocket_url(config, session_id)) as client:
            session_from = len(client.events.events)
            await client.configure(
                config.model,
                ref_audio=reference_audio,
                session_id=session_id,
                idle_timeout_s=config.timeout_s,
                timeout_s=min(config.timeout_s, 20.0),
            )
            pcm = _ensure_final_commit_tail(pcm, client.events.events)
            playback = _Playback()
            stream_start = time.monotonic()
            try:
                input_duration_s = len(pcm) / (PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE)
                upload_timeout_s = config.timeout_s + input_duration_s
                try:
                    chunks, frame_count, mean_lag, max_lag = await asyncio.wait_for(
                        stream_inputs(
                            client,
                            pcm,
                            frames,
                            playback,
                        ),
                        timeout=upload_timeout_s,
                    )
                except asyncio.TimeoutError as exc:
                    raise TimeoutError(f"Realtime upload timed out after {upload_timeout_s:g}s") from exc
                commit_from = len(client.events.events)
                await client.commit()
                await wait_for_session_completion(
                    client,
                    playback,
                    commit_from=commit_from,
                    session_from=session_from,
                    timeout_s=config.timeout_s,
                    settle_s=_COMPLETION_SETTLE_S,
                )
                await playback.acknowledge(client, playback.end_s)
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
            _validate_explicit_session_close(
                client.events,
                session_from=session_from,
                close_from=close_from,
            )
            result.latency_s = time.monotonic() - started_at
            result.input_audio_chunks = chunks
            result.input_video_frames = frame_count
            result.pacing_mean_lag_s = mean_lag
            result.pacing_max_lag_s = max_lag
            _populate_response_metrics(result, client.events, stream_start=stream_start)
            await asyncio.to_thread(
                write_success_artifacts,
                config.output_root,
                case,
                client.events,
                playback=playback,
                stream_start=stream_start,
                video_duration_s=duration,
                require_response=config.require_response,
                result=result,
                persist=persist_artifacts,
                defer=defer_artifacts,
            )
    except Exception as exc:
        result.success = False
        result.eligible_for_official_eval = False
        if "case_failed" not in result.official_eval_ineligible_reasons:
            result.official_eval_ineligible_reasons.append("case_failed")
        result.error = str(exc)
        result.latency_s = max(0.0, time.monotonic() - started_at)
        if persist_artifacts:
            await asyncio.to_thread(write_failure_artifacts, config.output_root, case, result)
    return result
