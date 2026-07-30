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
    latency_s: float = 0.0
    ttft_s: float = 0.0
    tpot_s: float = 0.0
    audio_rtf: float = 0.0


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


def extract_pcm16_from_video(video_path: Path) -> bytes:
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
    return proc.stdout


def sample_video_jpeg_frames(video_path: Path, fps: float) -> list[str]:
    """Sample one JPEG frame per ``1/fps`` seconds in presentation order."""
    if fps <= 0:
        raise ValueError("video fps must be positive")
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
    step = max(1, int(round(video_fps / fps)))
    frames_b64: list[str] = []
    for idx, frame in enumerate(iio.imiter(str(video_path))):
        if idx % step != 0:
            continue
        image = Image.fromarray(frame)
        # Keep frames small enough for realtime append validation.
        image.thumbnail((640, 640))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        frames_b64.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
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


def _match_slot(
    slots: list[OmniInteractQASlot],
    *,
    video_time_s: float,
    used: set[int],
) -> OmniInteractQASlot | None:
    candidates: list[tuple[float, OmniInteractQASlot]] = []
    for slot in slots:
        if slot.slot_index in used:
            continue
        q_s = slot.question_time_s
        a_s = slot.answer_time_s
        if q_s is None:
            continue
        if a_s is not None and q_s <= video_time_s <= a_s:
            candidates.append((0.0, slot))
        elif video_time_s >= q_s:
            candidates.append((video_time_s - q_s, slot))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1].slot_index))
    return candidates[0][1]


async def _stream_pcm16_with_video(
    client: RealtimeDuplexClient,
    pcm16: bytes,
    *,
    chunk_ms: int,
    realtime: bool,
    video_frames: list[str],
) -> None:
    chunk_bytes = max(
        PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE * chunk_ms // 1000,
        PCM16_BYTES_PER_SAMPLE,
    )
    audio_end_ms = 0
    frames_sent = 0
    for offset in range(0, len(pcm16), chunk_bytes):
        chunk = pcm16[offset : offset + chunk_bytes]
        duration_ms = len(chunk) * 1000 // (PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE)
        audio_end_ms += duration_ms
        payload: dict[str, object] = {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(chunk).decode("ascii"),
            "input_audio_format": "pcm16",
            "sample_rate_hz": PCM16_SAMPLE_RATE,
            "duration_ms": duration_ms,
            "audio_end_ms": audio_end_ms,
        }
        # Official MiniCPM-o cadence: one camera frame per ~1 s of audio.
        if video_frames and audio_end_ms > frames_sent * 1000:
            frame_index = min(frames_sent, len(video_frames) - 1)
            payload["video_frames"] = [video_frames[frame_index]]
            frames_sent += 1
        await client.send(payload)
        if realtime:
            await asyncio.sleep(duration_ms / 1000)


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
    ref_audio: str | None = None,
    chunk_ms: int = 200,
    video_fps: float = 1.0,
    realtime_pacing: bool = True,
    timeout_s: float = 120.0,
) -> OmniInteractRealtimeSessionResult:
    result = OmniInteractRealtimeSessionResult(session_key=session_key)
    session_start = time.monotonic()
    commit_completed = False
    wait_error = ""
    pcm16, video_frames = await asyncio.gather(
        asyncio.to_thread(extract_pcm16_from_video, video_path),
        asyncio.to_thread(sample_video_jpeg_frames, video_path, video_fps),
    )
    if not pcm16:
        result.error = f"No audio extracted from video: {video_path}"
        result.latency_s = time.monotonic() - session_start
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
            await _stream_pcm16_with_video(
                client,
                pcm16,
                chunk_ms=chunk_ms,
                realtime=realtime_pacing,
                video_frames=video_frames,
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
                events_at_commit = client.events.events[: committed_index + 1]
                if _has_residual_model_unit(pcm16, events_at_commit) or _response_in_progress(events_at_commit):
                    await wait_for(
                        lambda: _post_commit_decision(client.events.events, committed_index),
                        timeout_s=timeout_s,
                        label="post-commit model decision or response drain",
                    )
                await _drain_active_responses(client.events, timeout_s=timeout_s)
                commit_completed = True
            except TimeoutError as exc:
                wait_error = str(exc)

            used_slots: set[int] = set()
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
                matched = _match_slot(slots, video_time_s=turn_metrics.video_time_s, used=used_slots)
                if matched is not None:
                    used_slots.add(matched.slot_index)
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
                        "matched_slot_index": matched.slot_index if matched else None,
                        **turn_metrics.as_dict(),
                    }
                )

            # Unmatched GT slots become FN rows for soft QA metrics.
            for slot in slots:
                if slot.slot_index in used_slots:
                    continue
                result.turn_outputs.append(
                    {
                        "turn_index": len(result.turn_outputs),
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

            await client.acknowledge_playback()
            try:
                await client.close_session(timeout_s=min(timeout_s, 20.0))
            except TimeoutError as exc:
                logger.warning("OmniInteract session close acknowledgement timed out: %s", exc)
    except Exception as exc:
        result.error = str(exc)
        result.success = False
        result.latency_s = time.monotonic() - session_start
        return result

    result.latency_s = time.monotonic() - session_start
    protocol_errors = client.events.errors()
    result.success = commit_completed and not protocol_errors
    if not result.success and wait_error:
        result.error = wait_error
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
