"""OmniInteract full-duplex realtime helpers for continuous video sessions."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import math
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from vllm_omni.benchmarks.data_modules.omniinteract_dataset import OmniInteractQASlot
from vllm_omni.experimental.fullduplex.client import (
    PCM16_BYTES_PER_SAMPLE,
    PCM16_SAMPLE_RATE,
    RealtimeDuplexClient,
    RealtimeEventCollector,
    build_realtime_url,
    wait_for,
)

logger = logging.getLogger(__name__)


@dataclass
class OmniInteractRealtimeTurnMetrics:
    turn_index: int
    response_id: str | None
    video_time_s: float = 0.0
    ttft_s: float = 0.0
    tpot_s: float = 0.0
    rtf: float = 0.0
    audio_duration_s: float = 0.0
    response_generation_s: float = 0.0
    generated_text: str = ""
    success: bool = False
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "response_id": self.response_id,
            "video_time_s": self.video_time_s,
            "ttft_s": self.ttft_s,
            "tpot_s": self.tpot_s,
            "rtf": self.rtf,
            "audio_duration_s": self.audio_duration_s,
            "response_generation_s": self.response_generation_s,
            "generated_text": self.generated_text,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class OmniInteractRealtimeSessionResult:
    session_key: str
    turn_metrics: list[OmniInteractRealtimeTurnMetrics] = field(default_factory=list)
    turn_outputs: list[dict[str, Any]] = field(default_factory=list)
    success: bool = False
    error: str = ""
    preprocess_s: float = 0.0
    latency_s: float = 0.0
    ttft_s: float = 0.0
    tpot_s: float = 0.0
    audio_rtf: float = 0.0
    pacing_mean_lag_s: float = 0.0
    pacing_max_lag_s: float = 0.0
    official_summary: dict[str, Any] | None = None


@dataclass(frozen=True)
class _PlaybackSegment:
    response_id: str
    start_s: float
    sample_count: int
    sample_rate: int

    @property
    def duration_s(self) -> float:
        return self.sample_count / self.sample_rate

    @property
    def duration_ms(self) -> int:
        return self.sample_count * 1000 // self.sample_rate


class RealtimePlaybackAcknowledger:
    """Acknowledge only audio that a realtime client could have played.

    The generic demo acknowledges all generated audio after a one-shot input.
    OmniInteract keeps sending input for several minutes, so doing that only at
    session close leaves every earlier response marked as unplayed.  This
    tracker models one serial audio device from client receive timestamps and
    advances each response's playback cursor while the video is streaming.
    """

    def __init__(self) -> None:
        self._event_cursor = 0
        self._playback_cursor_s = 0.0
        self._segments: list[_PlaybackSegment] = []
        self._acked_ms: dict[str, int] = {}
        self._completed_responses: set[str] = set()
        self._completion_acked: set[str] = set()

    def _ingest(self, collector: RealtimeEventCollector) -> None:
        events = collector.events
        received_times = collector.event_received_at_s
        while self._event_cursor < len(events):
            index = self._event_cursor
            self._event_cursor += 1
            event = events[index]
            if event.get("type") == "response.done":
                response_id = collector.response_id(event)
                if response_id:
                    self._completed_responses.add(response_id)
                continue
            if event.get("type") != "response.audio.delta":
                continue
            response_id = collector.response_id(event)
            encoded = event.get("delta") or event.get("audio")
            if not response_id or not isinstance(encoded, str):
                continue
            try:
                pcm16 = base64.b64decode(encoded, validate=True)
            except ValueError:
                continue
            sample_rate = event.get("sample_rate_hz")
            if not isinstance(sample_rate, int) or sample_rate <= 0:
                sample_rate = collector.output_sample_rate_hz or 24_000
            samples = len(pcm16) // PCM16_BYTES_PER_SAMPLE
            if samples <= 0:
                continue
            received_s = received_times[index]
            start_s = max(received_s, self._playback_cursor_s)
            segment = _PlaybackSegment(response_id, start_s, samples, sample_rate)
            self._segments.append(segment)
            self._playback_cursor_s = start_s + segment.duration_s

    def played_ms(self, collector: RealtimeEventCollector, *, now_s: float) -> dict[str, int]:
        self._ingest(collector)
        played: dict[str, int] = {}
        for segment in self._segments:
            elapsed_samples = max(0, int(round((now_s - segment.start_s) * segment.sample_rate)))
            if elapsed_samples >= segment.sample_count:
                elapsed_ms = segment.duration_ms
            else:
                elapsed_ms = elapsed_samples * 1000 // segment.sample_rate
            played[segment.response_id] = played.get(segment.response_id, 0) + elapsed_ms
        return played

    async def acknowledge(
        self,
        client: RealtimeDuplexClient,
        collector: RealtimeEventCollector,
        *,
        now_s: float | None = None,
    ) -> None:
        cursors = self.played_ms(collector, now_s=time.monotonic() if now_s is None else now_s)
        for response_id, played_ms in cursors.items():
            completion_ack_due = response_id in self._completed_responses and response_id not in self._completion_acked
            if played_ms <= self._acked_ms.get(response_id, 0) and not completion_ack_due:
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
            self._acked_ms[response_id] = played_ms
            if response_id in self._completed_responses:
                self._completion_acked.add(response_id)


def http_url_to_ws_url(url: str) -> str:
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme in {"ws", "wss"}:
        return url
    if scheme == "http":
        ws_scheme = "ws"
    elif scheme == "https":
        ws_scheme = "wss"
    else:
        raise ValueError(f"Unsupported realtime URL scheme: {parts.scheme!r}")
    return urlunsplit((ws_scheme, parts.netloc, parts.path, parts.query, parts.fragment))


def _ref_audio_data_url(path: str | None) -> str | None:
    if not path:
        return None
    ref_path = Path(path).expanduser()
    if not ref_path.is_file():
        raise FileNotFoundError(f"OmniInteract realtime ref audio does not exist: {path}")
    return "data:audio/wav;base64," + base64.b64encode(ref_path.read_bytes()).decode("ascii")


def probe_video_duration_s(video_path: Path) -> float:
    """Return source-video duration without decoding the full file."""
    if shutil.which("ffprobe") is not None:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            check=False,
            text=True,
        )
        if proc.returncode == 0:
            try:
                duration = float(proc.stdout.strip())
            except ValueError:
                duration = 0.0
            if math.isfinite(duration) and duration > 0:
                return duration

    try:
        import imageio.v3 as iio

        metadata = iio.immeta(str(video_path))
        duration = float(metadata.get("duration") or 0.0)
        if duration > 0 and math.isfinite(duration):
            return duration
        fps = float(metadata.get("fps") or 0.0)
        frame_count = float(metadata.get("nframes") or metadata.get("n_frames") or 0.0)
        if fps > 0 and frame_count > 0:
            return frame_count / fps
    except Exception:
        pass
    raise RuntimeError(f"Could not determine video duration: {video_path}")


def validate_realtime_video_fps(fps: float) -> float:
    """Validate the cadence supported by the MiniCPM duplex adapter.

    Stage0 consumes at most one queued image for each roughly one-second model
    unit.  Accepting a higher rate would silently delay frames into later units
    and make accuracy numbers meaningless.
    """
    value = float(fps)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("OmniInteract realtime video fps must be finite and positive")
    if value > 1.0:
        raise ValueError(
            "OmniInteract MiniCPM duplex supports at most 1 video frame per second; "
            f"got {value:g}. Higher rates queue stale frames in later model units."
        )
    return value


def validate_realtime_chunk_ms(chunk_ms: int) -> int:
    """Validate one append cannot span multiple one-second model units."""
    value = int(chunk_ms)
    if value <= 0 or value > 1000:
        raise ValueError("OmniInteract realtime chunk size must be in the range [1, 1000] ms")
    return value


def extract_pcm16_from_video(video_path: Path, *, duration_s: float | None = None) -> bytes:
    """Extract mono 16 kHz PCM16 from a video, matching OmniInteract MiniCPM-o."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to extract OmniInteract duplex audio from video")
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vn",
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(PCM16_SAMPLE_RATE),
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to extract audio from {video_path}: {proc.stderr.decode('utf-8', 'ignore')}")
    pcm16 = proc.stdout
    if duration_s is None:
        return pcm16
    # The official runner steps the model for ceil(video_duration) one-second
    # units.  Keep video-only tails by padding a shorter audio track with silence.
    target_bytes = int(math.ceil(duration_s) * PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE)
    if len(pcm16) < target_bytes:
        return pcm16 + bytes(target_bytes - len(pcm16))
    return pcm16[:target_bytes]


def sample_video_jpeg_frames(
    video_path: Path,
    fps: float,
    *,
    duration_s: float | None = None,
) -> list[str | None]:
    """Sample midpoint frames, matching the official pseudo-online runner."""
    fps = validate_realtime_video_fps(fps)
    try:
        import imageio.v3 as iio
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "OmniInteract realtime video sampling requires imageio and Pillow. Install with: pip install imageio pillow"
        ) from exc

    video_fps = 30.0
    try:
        meta = iio.immeta(str(video_path))
        video_fps = float(meta.get("fps") or 30.0)
    except Exception:
        pass
    if duration_s is None:
        duration_s = probe_video_duration_s(video_path)
    sample_count = max(1, int(math.ceil(duration_s * fps)))
    target_indices = [int((sample_index + 0.5) * video_fps / fps) for sample_index in range(sample_count)]
    frames_b64: list[str | None] = [None] * sample_count
    target_cursor = 0
    for idx, frame in enumerate(iio.imiter(str(video_path))):
        if target_cursor >= len(target_indices):
            break
        if idx < target_indices[target_cursor]:
            continue
        while target_cursor < len(target_indices) and idx >= target_indices[target_cursor]:
            image = Image.fromarray(frame)
            image.thumbnail((640, 640))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            frames_b64[target_cursor] = base64.b64encode(buffer.getvalue()).decode("ascii")
            target_cursor += 1
    return frames_b64


def _transcript_for_response(collector: RealtimeEventCollector, response_id: str | None) -> str:
    if not response_id:
        return ""
    parts: list[str] = []
    for event in collector.events:
        if collector.response_id(event) != response_id:
            continue
        if event.get("type") not in {"response.audio_transcript.delta", "response.output_text.delta"}:
            continue
        delta = event.get("delta")
        if isinstance(delta, str) and delta:
            parts.append(delta)
    return "".join(parts)


def _event_stage0_metrics(event: dict[str, object]) -> dict[str, object] | None:
    candidates: list[object] = [event.get("vllm_omni")]
    metadata = event.get("metadata")
    if isinstance(metadata, dict):
        candidates.extend((metadata, metadata.get("vllm_omni")))
    response = event.get("response")
    if isinstance(response, dict):
        response_metadata = response.get("metadata")
        if isinstance(response_metadata, dict):
            candidates.extend((response_metadata, response_metadata.get("vllm_omni")))
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        stage_metrics = candidate.get("stage_metrics")
        if isinstance(stage_metrics, dict):
            stage0 = stage_metrics.get("0")
            if isinstance(stage0, dict):
                return stage0
    return None


def compute_turn_metrics(
    collector: RealtimeEventCollector,
    *,
    response_id: str | None,
    turn_start_s: float,
    stream_start_s: float,
) -> OmniInteractRealtimeTurnMetrics:
    metrics = OmniInteractRealtimeTurnMetrics(turn_index=0, response_id=response_id)
    if response_id is None:
        metrics.error = "missing response_id"
        return metrics

    response_created_at_s: float | None = None
    first_text_at_s: float | None = None
    first_audio_at_s: float | None = None
    response_done_at_s: float | None = None
    stage0_metrics: dict[str, object] | None = None
    audio_bytes = collector.audio_bytes(response_id)
    sample_rate = collector.output_sample_rate_hz or 24_000

    for event, received_at_s in zip(collector.events, collector.event_received_at_s, strict=True):
        if collector.response_id(event) != response_id:
            continue
        event_type = event.get("type")
        if event_type == "response.created" and response_created_at_s is None:
            response_created_at_s = received_at_s
        if event_type in {"response.audio_transcript.delta", "response.output_text.delta"} and first_text_at_s is None:
            first_text_at_s = received_at_s
        if event_type == "response.audio.delta" and first_audio_at_s is None:
            first_audio_at_s = received_at_s
        if event_type == "response.done":
            response_done_at_s = received_at_s
        stage0 = _event_stage0_metrics(event)
        if isinstance(stage0, dict):
            stage0_metrics = stage0

    origin = response_created_at_s if response_created_at_s is not None else turn_start_s
    metrics.video_time_s = max(0.0, origin - stream_start_s)
    metrics.generated_text = _transcript_for_response(collector, response_id)
    metrics.audio_duration_s = len(audio_bytes) / (sample_rate * PCM16_BYTES_PER_SAMPLE)
    metrics.response_generation_s = (
        (response_done_at_s - response_created_at_s)
        if response_done_at_s is not None and response_created_at_s is not None
        else 0.0
    )
    if stage0_metrics is not None:
        ttft_ms = float(stage0_metrics.get("vllm_ttft_ms") or 0.0)
        tpot_ms = float(stage0_metrics.get("vllm_tpot_ms") or 0.0)
        if ttft_ms > 0:
            metrics.ttft_s = ttft_ms / 1000.0
        if tpot_ms > 0:
            metrics.tpot_s = tpot_ms / 1000.0
    if metrics.ttft_s <= 0 and first_text_at_s is not None:
        metrics.ttft_s = max(0.0, first_text_at_s - origin)
    elif metrics.ttft_s <= 0 and first_audio_at_s is not None:
        metrics.ttft_s = max(0.0, first_audio_at_s - origin)
    if metrics.audio_duration_s > 0 and metrics.response_generation_s > 0:
        metrics.rtf = metrics.response_generation_s / metrics.audio_duration_s
    metrics.success = bool(metrics.generated_text.strip() or metrics.audio_duration_s > 0)
    return metrics


def _slot_windows(
    slots: list[OmniInteractQASlot],
) -> list[tuple[OmniInteractQASlot, float, float, float]]:
    """Build OmniInteract interaction windows ``[t_start, t_a, t_end)``.

    Matches paper Sec. 3.3.1: ``t_start`` is the query/observation onset,
    ``t_a`` is the earliest valid core-answer time, and ``t_end`` is the next
    slot's ``t_start`` (or +inf for the last slot). Nested overlap prefers the
    latest ``t_start``.
    """
    ordered = sorted(
        (slot for slot in slots if slot.question_time_s is not None),
        key=lambda slot: (float(slot.question_time_s), slot.slot_index),
    )
    windows: list[tuple[OmniInteractQASlot, float, float, float]] = []
    for index, slot in enumerate(ordered):
        t_start = float(slot.question_time_s)
        t_a = float(slot.answer_time_s) if slot.answer_time_s is not None else t_start
        t_end = float(ordered[index + 1].question_time_s) if index + 1 < len(ordered) else float("inf")
        windows.append((slot, t_start, t_a, t_end))
    return windows


def _match_slot(
    slots: list[OmniInteractQASlot],
    *,
    video_time_s: float,
    used: set[int] | None = None,
) -> OmniInteractQASlot | None:
    """Assign a response chunk to the open slot whose window contains its time.

    Official rule: map a chunk to the slot with the latest ``t_start`` among
    windows where ``t_start <= t < t_end``. Multiple chunks may share one slot;
    ``used`` is retained only for backward-compatible call sites and is ignored.
    """
    del used  # multi-chunk-per-slot aggregation owns exclusivity downstream
    candidates: list[tuple[float, OmniInteractQASlot]] = []
    for slot, t_start, _t_a, t_end in _slot_windows(slots):
        if t_start <= video_time_s < t_end:
            candidates.append((t_start, slot))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1].slot_index))
    return candidates[-1][1]


async def _stream_pcm16_with_video(
    client: RealtimeDuplexClient,
    pcm16: bytes,
    *,
    chunk_ms: int,
    realtime: bool,
    video_frames: list[str | None],
    video_fps: float,
    playback_acknowledger: RealtimePlaybackAcknowledger | None = None,
) -> tuple[int, int, float, float]:
    chunk_bytes = PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE * chunk_ms // 1000
    bytes_per_second = PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE
    audio_end_ms = 0
    frame_cursor = 0
    audio_chunks_sent = 0
    video_frames_sent = 0
    pacing_start_s = time.monotonic()
    pacing_lags_s: list[float] = []
    for offset in range(0, len(pcm16), chunk_bytes):
        if realtime:
            expected_send_s = pacing_start_s + offset / bytes_per_second
            pacing_lags_s.append(max(0.0, time.monotonic() - expected_send_s))
        chunk = pcm16[offset : offset + chunk_bytes]
        next_audio_end_ms = (offset + len(chunk)) * 1000 // bytes_per_second
        duration_ms = next_audio_end_ms - audio_end_ms
        audio_end_ms = next_audio_end_ms
        payload: dict[str, object] = {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(chunk).decode("ascii"),
            "input_audio_format": "pcm16",
            "sample_rate_hz": PCM16_SAMPLE_RATE,
            "duration_ms": duration_ms,
            "audio_end_ms": audio_end_ms,
        }
        # Each sampled frame represents the midpoint of its source-video
        # interval. Attach it before the corresponding one-second model unit is
        # formed, rather than front-loading the first frame at t=0.
        ready_frames: list[str] = []
        while frame_cursor < len(video_frames):
            frame_time_ms = (frame_cursor + 0.5) * 1000.0 / video_fps
            if audio_end_ms < frame_time_ms:
                break
            frame = video_frames[frame_cursor]
            if frame:
                ready_frames.append(frame)
            frame_cursor += 1
        if ready_frames:
            payload["video_frames"] = ready_frames
        await client.send(payload)
        audio_chunks_sent += 1
        video_frames_sent += len(ready_frames)
        if realtime:
            if playback_acknowledger is not None:
                await playback_acknowledger.acknowledge(client, client.events)
            deadline_s = pacing_start_s + audio_end_ms / 1000
            await asyncio.sleep(max(0.0, deadline_s - time.monotonic()))
    if realtime:
        final_deadline_s = pacing_start_s + audio_end_ms / 1000
        pacing_lags_s.append(max(0.0, time.monotonic() - final_deadline_s))
    pacing_mean_lag_s = sum(pacing_lags_s) / len(pacing_lags_s) if pacing_lags_s else 0.0
    pacing_max_lag_s = max(pacing_lags_s, default=0.0)
    return audio_chunks_sent, video_frames_sent, pacing_mean_lag_s, pacing_max_lag_s


async def _drain_active_responses(
    collector: RealtimeEventCollector,
    *,
    timeout_s: float,
) -> None:
    await wait_for(
        lambda: collector.count("response.created") <= collector.count("response.done"),
        timeout_s=timeout_s,
        label="active duplex responses to finish",
    )


def _event_index(events: list[dict[str, object]], event_type: str, after: int) -> int | None:
    return next(
        (index for index, event in enumerate(events[after:], start=after) if event.get("type") == event_type),
        None,
    )


def _committed_input_watermark(event: dict[str, object]) -> tuple[str, int, int]:
    session_id = event.get("session_id")
    epoch = event.get("epoch")
    accepted_input_seq = event.get("accepted_input_seq")
    if (
        not isinstance(session_id, str)
        or not session_id
        or not isinstance(epoch, int)
        or isinstance(epoch, bool)
        or not isinstance(accepted_input_seq, int)
        or isinstance(accepted_input_seq, bool)
    ):
        raise RuntimeError(
            "Official OmniInteract accuracy requires input_audio_buffer.committed "
            "to include session_id, epoch, and accepted_input_seq"
        )
    return session_id, epoch, accepted_input_seq


def _processed_input_event(
    events: list[dict[str, object]],
    *,
    after: int,
    session_id: str,
    epoch: int,
    accepted_input_seq: int,
) -> dict[str, object] | None:
    for event in events[after + 1 :]:
        if event.get("type") != "input_audio_buffer.processed":
            continue
        processed_input_seq = event.get("processed_input_seq")
        if (
            event.get("session_id") == session_id
            and event.get("epoch") == epoch
            and isinstance(processed_input_seq, int)
            and not isinstance(processed_input_seq, bool)
            and processed_input_seq == accepted_input_seq
        ):
            return event
    return None


def _response_done_event_for_id(
    events: list[dict[str, object]], *, after: int, response_id: str
) -> dict[str, object] | None:
    for event in events[after + 1 :]:
        if event.get("type") != "response.done":
            continue
        response = event.get("response")
        event_response_id = response.get("id") if isinstance(response, dict) else event.get("response_id")
        if event_response_id == response_id:
            return event
    return None


def _validate_final_response_done(event: dict[str, object]) -> None:
    response = event.get("response")
    status = response.get("status") if isinstance(response, dict) else event.get("status")
    if status != "completed":
        raise RuntimeError(f"The final response ended with status {status!r}")


def _final_processing_outcome(event: dict[str, object]) -> tuple[str, str | None]:
    outcome = event.get("outcome")
    if outcome == "failed":
        raise RuntimeError("The runtime reported that the final accepted input failed")
    if outcome not in {"listen", "speak"}:
        raise RuntimeError(f"Invalid final input processing outcome: {outcome!r}")
    response_id = event.get("response_id")
    if outcome == "speak" and (not isinstance(response_id, str) or not response_id):
        raise RuntimeError("A processed speak outcome has no response_id")
    return outcome, response_id if isinstance(response_id, str) else None


def _post_commit_decision(events: list[dict[str, object]], committed_index: int) -> bool:
    for event in events[committed_index + 1 :]:
        if event.get("type") == "response.listen":
            return True
        if event.get("type") == "response.done":
            response = event.get("response")
            if not isinstance(response, dict) or response.get("status") != "cancelled":
                return True
    return False


def _response_in_progress(events: list[dict[str, object]]) -> bool:
    return sum(event.get("type") == "response.created" for event in events) > sum(
        event.get("type") == "response.done" for event in events
    )


def _has_residual_model_unit(pcm16: bytes, events: list[dict[str, object]]) -> bool:
    chunk_period_ms = 1000
    for event in reversed(events):
        session = event.get("session")
        capabilities = session.get("capabilities") if isinstance(session, dict) else None
        period = capabilities.get("chunk_period_ms") if isinstance(capabilities, dict) else None
        if isinstance(period, int) and period > 0:
            chunk_period_ms = period
            break
    unit_bytes = PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE * chunk_period_ms // 1000
    return bool(unit_bytes and len(pcm16) % unit_bytes)


async def run_omniinteract_realtime_session(
    *,
    api_url: str,
    model: str,
    video_path: Path,
    session_key: str,
    slots: list[OmniInteractQASlot],
    subset: str = "",
    video_rel: str = "",
    annotation_path: Path | None = None,
    scene_type: str = "multi_turn",
    official_output_root: Path | None = None,
    ref_audio: str | None = None,
    chunk_ms: int = 200,
    video_fps: float = 1.0,
    realtime_pacing: bool = True,
    timeout_s: float = 120.0,
) -> OmniInteractRealtimeSessionResult:
    result = OmniInteractRealtimeSessionResult(session_key=session_key)
    commit_completed = False
    wait_error = ""
    close_error = ""
    pacing_error = ""
    output_validation_error = ""
    preprocess_start_s = time.monotonic()
    try:
        validate_realtime_video_fps(video_fps)
        validate_realtime_chunk_ms(chunk_ms)
        if official_output_root is not None and not realtime_pacing:
            raise ValueError("Official OmniInteract accuracy output requires realtime pacing")
        if official_output_root is not None and (chunk_ms != 200 or video_fps != 1.0):
            raise ValueError("Official OmniInteract accuracy output requires 200 ms PCM chunks and 1 FPS video")
        video_duration_s = await asyncio.to_thread(probe_video_duration_s, video_path)
        pcm16, video_frames = await asyncio.gather(
            asyncio.to_thread(extract_pcm16_from_video, video_path, duration_s=video_duration_s),
            asyncio.to_thread(sample_video_jpeg_frames, video_path, video_fps, duration_s=video_duration_s),
        )
    except Exception as exc:
        result.preprocess_s = time.monotonic() - preprocess_start_s
        result.error = str(exc)
        return result
    result.preprocess_s = time.monotonic() - preprocess_start_s
    if not pcm16:
        result.error = f"No audio extracted from video: {video_path}"
        return result
    # When ref_audio is supplied via session.update, disable server-side
    # autostart so the duplex session does not reject the socket for missing
    # ref_audio before configure() can send it.
    ws_url = build_realtime_url(
        http_url_to_ws_url(api_url),
        model,
        autostart=False if ref_audio else None,
        session_id=session_key,
    )

    session_start_s = time.monotonic()
    request_done_s: float | None = None
    try:
        async with RealtimeDuplexClient(ws_url) as client:
            await client.configure(
                model,
                ref_audio=_ref_audio_data_url(ref_audio),
                session_id=session_key,
                timeout_s=timeout_s,
            )
            stream_start_s = time.monotonic()
            before_created = client.events.count("response.created")
            playback_acknowledger = RealtimePlaybackAcknowledger() if realtime_pacing else None
            (
                input_audio_chunks,
                input_video_frames,
                result.pacing_mean_lag_s,
                result.pacing_max_lag_s,
            ) = await _stream_pcm16_with_video(
                client,
                pcm16,
                chunk_ms=chunk_ms,
                realtime=realtime_pacing,
                video_frames=video_frames,
                video_fps=video_fps,
                playback_acknowledger=playback_acknowledger,
            )
            commit_cursor = len(client.events.events)
            await client.commit()
            try:
                await wait_for(
                    lambda: _event_index(
                        client.events.events,
                        "input_audio_buffer.committed",
                        commit_cursor,
                    )
                    is not None,
                    timeout_s=timeout_s,
                    label="input_audio_buffer.committed",
                )
                committed_index = _event_index(
                    client.events.events,
                    "input_audio_buffer.committed",
                    commit_cursor,
                )
                assert committed_index is not None
                if official_output_root is not None:
                    session_id, epoch, accepted_input_seq = _committed_input_watermark(
                        client.events.events[committed_index]
                    )
                    await wait_for(
                        lambda: _processed_input_event(
                            client.events.events,
                            after=-1,
                            session_id=session_id,
                            epoch=epoch,
                            accepted_input_seq=accepted_input_seq,
                        )
                        is not None,
                        timeout_s=timeout_s,
                        label="final accepted input to be processed",
                    )
                    processed_event = _processed_input_event(
                        client.events.events,
                        after=-1,
                        session_id=session_id,
                        epoch=epoch,
                        accepted_input_seq=accepted_input_seq,
                    )
                    assert processed_event is not None
                    outcome, response_id = _final_processing_outcome(processed_event)
                    if outcome == "speak":
                        assert response_id is not None
                        await wait_for(
                            lambda: _response_done_event_for_id(
                                client.events.events,
                                after=-1,
                                response_id=response_id,
                            )
                            is not None,
                            timeout_s=timeout_s,
                            label=f"final response {response_id} to finish",
                        )
                        response_done = _response_done_event_for_id(
                            client.events.events,
                            after=-1,
                            response_id=response_id,
                        )
                        assert response_done is not None
                        _validate_final_response_done(response_done)
                else:
                    events_at_commit = client.events.events[: committed_index + 1]
                    if _has_residual_model_unit(pcm16, events_at_commit) or _response_in_progress(events_at_commit):
                        await wait_for(
                            lambda: _post_commit_decision(client.events.events, committed_index),
                            timeout_s=timeout_s,
                            label="post-commit model decision or response drain",
                        )
                await _drain_active_responses(client.events, timeout_s=timeout_s)
                commit_completed = True
            except (RuntimeError, TimeoutError) as exc:
                wait_error = str(exc)

            used_slots: set[int] = set()
            slot_text_parts: dict[int, list[str]] = {}
            slot_primary_turn: dict[int, int] = {}
            response_ids = client.events.response_ids[before_created:]
            for turn_index, response_id in enumerate(response_ids):
                turn_metrics = compute_turn_metrics(
                    client.events,
                    response_id=response_id,
                    turn_start_s=stream_start_s,
                    stream_start_s=stream_start_s,
                )
                turn_metrics.turn_index = turn_index
                result.turn_metrics.append(turn_metrics)
                matched = _match_slot(slots, video_time_s=turn_metrics.video_time_s)
                matched_index = matched.slot_index if matched is not None else None
                if matched is not None:
                    used_slots.add(matched.slot_index)
                    text = (turn_metrics.generated_text or "").strip()
                    if text:
                        slot_text_parts.setdefault(matched.slot_index, []).append(text)
                        # Prefer the chunk nearest answer_time for primary metrics.
                        answer_s = matched.answer_time_s
                        if answer_s is None:
                            slot_primary_turn.setdefault(matched.slot_index, turn_index)
                        else:
                            prev = slot_primary_turn.get(matched.slot_index)
                            if prev is None:
                                slot_primary_turn[matched.slot_index] = turn_index
                            else:
                                prev_t = result.turn_metrics[prev].video_time_s
                                cur_t = turn_metrics.video_time_s
                                if abs(cur_t - answer_s) < abs(prev_t - answer_s):
                                    slot_primary_turn[matched.slot_index] = turn_index
                result.turn_outputs.append(
                    {
                        "turn_index": turn_index,
                        "question_text": matched.question_text if matched else "",
                        "gold_answer": matched.answer_text if matched else "",
                        "generated_text": turn_metrics.generated_text,
                        "success": turn_metrics.success,
                        "subset": matched.subset if matched else (slots[0].subset if slots else ""),
                        "question_type": matched.question_type if matched else "",
                        "video": matched.video_rel if matched else (slots[0].video_rel if slots else ""),
                        "scene_type": matched.scene_type if matched else (slots[0].scene_type if slots else ""),
                        "nested_group_id": matched.nested_group_id if matched else None,
                        "nested_role": matched.nested_role if matched else "",
                        "question_time": matched.question_time if matched else "",
                        "answer_time": matched.answer_time if matched else "",
                        "is_interrupted": matched.is_interrupted if matched else None,
                        "matched_slot_index": matched_index,
                        **turn_metrics.as_dict(),
                    }
                )

            # Collapse multi-chunk slot hits into one eval row per slot (paper:
            # chunks share a slot; proxy QA uses the concatenated transcript).
            collapsed: list[dict[str, Any]] = []
            emitted_slots: set[int] = set()
            for row in result.turn_outputs:
                slot_idx = row.get("matched_slot_index")
                if not isinstance(slot_idx, int):
                    # Unmatched model chunks become FP rows (empty gold skipped
                    # by exact/soft; marked failed so FN/FP accounting can use
                    # question_type when present). Keep for turn metrics only.
                    continue
                if slot_idx in emitted_slots:
                    continue
                emitted_slots.add(slot_idx)
                primary_idx = slot_primary_turn.get(slot_idx, int(row["turn_index"]))
                primary = result.turn_outputs[primary_idx]
                merged_text = " ".join(slot_text_parts.get(slot_idx, [])).strip()
                collapsed.append(
                    {
                        **primary,
                        "turn_index": len(collapsed),
                        "generated_text": merged_text or str(primary.get("generated_text") or ""),
                        "success": bool(merged_text) or bool(primary.get("success")),
                    }
                )
            for slot in slots:
                if slot.slot_index in emitted_slots:
                    continue
                collapsed.append(
                    {
                        "turn_index": len(collapsed),
                        "question_text": slot.question_text,
                        "gold_answer": slot.answer_text,
                        "generated_text": "",
                        "success": False,
                        "error": "no_response_matched_slot",
                        "subset": slot.subset,
                        "question_type": slot.question_type,
                        "video": slot.video_rel,
                        "scene_type": slot.scene_type,
                        "nested_group_id": slot.nested_group_id,
                        "nested_role": slot.nested_role,
                        "question_time": slot.question_time,
                        "answer_time": slot.answer_time,
                        "is_interrupted": slot.is_interrupted,
                        "matched_slot_index": slot.slot_index,
                        "ttft_s": 0.0,
                        "tpot_s": 0.0,
                        "rtf": 0.0,
                    }
                )
            # Keep per-response rows for performance; replace eval rows.
            result.turn_outputs = collapsed

            request_done_s = time.monotonic()
            result.latency_s = request_done_s - session_start_s
            if playback_acknowledger is not None:
                await playback_acknowledger.acknowledge(client, client.events, now_s=request_done_s)
            else:
                # Non-realtime mode is for load debugging only. It has no
                # meaningful playback clock, so acknowledge the completed audio.
                await client.acknowledge_playback()
            try:
                await client.close_session(timeout_s=min(timeout_s, 20.0))
            except TimeoutError as exc:
                close_error = f"Session close acknowledgement timed out: {exc}"
                logger.warning("OmniInteract %s", close_error)

            if official_output_root is not None:
                from vllm_omni.benchmarks.data_modules.omniinteract_official import (
                    build_official_failure_summary,
                    write_official_session_artifacts,
                )

                protocol_errors = client.events.errors()
                if result.pacing_max_lag_s > chunk_ms / 1000:
                    pacing_error = (
                        f"Realtime pacing lag {result.pacing_max_lag_s:.3f}s exceeded one {chunk_ms}ms input chunk"
                    )
                status = (
                    "ok"
                    if commit_completed and not protocol_errors and not close_error and not pacing_error
                    else "error"
                )
                try:
                    result.official_summary = await asyncio.to_thread(
                        write_official_session_artifacts,
                        output_root=official_output_root,
                        subset=subset,
                        video_rel=video_rel or video_path.name,
                        video_path=video_path,
                        annotation_path=annotation_path,
                        scene_type=scene_type,
                        duration_s=video_duration_s,
                        inference_s=request_done_s - stream_start_s,
                        collector=client.events,
                        stream_start_s=stream_start_s,
                        status=status,
                        preprocess_s=result.preprocess_s,
                        error=wait_error
                        or close_error
                        or pacing_error
                        or (str(protocol_errors[-1]) if protocol_errors else ""),
                        input_audio_chunks=input_audio_chunks,
                        input_video_frames=input_video_frames,
                        pacing_mean_lag_s=result.pacing_mean_lag_s,
                        pacing_max_lag_s=result.pacing_max_lag_s,
                    )
                except ValueError as exc:
                    output_validation_error = str(exc)
                    result.official_summary = await asyncio.to_thread(
                        build_official_failure_summary,
                        output_root=official_output_root,
                        subset=subset,
                        video_rel=video_rel or video_path.name,
                        video_path=video_path,
                        annotation_path=annotation_path,
                        scene_type=scene_type,
                        error=output_validation_error,
                    )
    except Exception as exc:
        result.error = str(exc)
        result.success = False
        result.latency_s = (request_done_s or time.monotonic()) - session_start_s
        return result

    if request_done_s is None:
        request_done_s = time.monotonic()
        result.latency_s = request_done_s - session_start_s
    protocol_errors = client.events.errors()
    result.success = (
        commit_completed
        and not protocol_errors
        and not close_error
        and not pacing_error
        and not output_validation_error
    )
    if not result.success and wait_error:
        result.error = wait_error
    elif not result.success and close_error:
        result.error = close_error
    elif not result.success and pacing_error:
        result.error = pacing_error
    elif not result.success and output_validation_error:
        result.error = output_validation_error
    elif not result.success and protocol_errors:
        last_error = protocol_errors[-1]
        result.error = str(last_error.get("error") or last_error.get("message") or last_error)
    if result.turn_metrics:
        result.ttft_s = result.turn_metrics[0].ttft_s
        tpots = [metric.tpot_s for metric in result.turn_metrics if metric.tpot_s > 0]
        result.tpot_s = sum(tpots) / len(tpots) if tpots else 0.0
        rtfs = [metric.rtf for metric in result.turn_metrics if metric.rtf > 0]
        result.audio_rtf = sum(rtfs) / len(rtfs) if rtfs else 0.0
    return result


def summarize_turn_metrics(turn_metrics: list[OmniInteractRealtimeTurnMetrics]) -> dict[str, Any]:
    if not turn_metrics:
        return {
            "omniinteract_realtime_turn_count": 0,
            "omniinteract_realtime_turn_ttft_mean_s": None,
            "omniinteract_realtime_turn_tpot_mean_s": None,
            "omniinteract_realtime_turn_rtf_mean": None,
            "omniinteract_realtime_turn_metrics": [],
        }

    def _mean(values: list[float]) -> float | None:
        clean = [value for value in values if math.isfinite(value) and value > 0]
        return (sum(clean) / len(clean)) if clean else None

    return {
        "omniinteract_realtime_turn_count": len(turn_metrics),
        "omniinteract_realtime_turn_ttft_mean_s": _mean([metric.ttft_s for metric in turn_metrics]),
        "omniinteract_realtime_turn_tpot_mean_s": _mean([metric.tpot_s for metric in turn_metrics]),
        "omniinteract_realtime_turn_rtf_mean": _mean([metric.rtf for metric in turn_metrics]),
        "omniinteract_realtime_turn_metrics": [metric.as_dict() for metric in turn_metrics],
    }
