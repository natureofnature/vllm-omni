"""OmniInteract full-duplex realtime helpers for MiniCPM-o 4.5."""

from __future__ import annotations

import asyncio
import base64
import io
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from vllm_omni.experimental.fullduplex.client import (
    PCM16_BYTES_PER_SAMPLE,
    PCM16_SAMPLE_RATE,
    RealtimeDuplexClient,
    RealtimeEventCollector,
    build_realtime_url,
    read_pcm16_wav,
)


@dataclass(frozen=True)
class OmniInteractRealtimeTurn:
    """One user turn inside a duplex OmniInteract session."""

    turn_index: int
    audio_path: Path
    gold_answer: str
    question_text: str = ""
    question_type: str = ""
    question_time: str = ""
    answer_time: str = ""
    is_interrupted: bool | None = None
    nested_group_id: int | None = None
    nested_role: str = ""
    subset: str = ""
    video_rel: str = ""
    scene_type: str = ""


@dataclass
class OmniInteractRealtimeTurnMetrics:
    turn_index: int
    response_id: str | None
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


def sample_video_jpeg_frames(video_path: Path, fps: float) -> list[str]:
    """Sample a subvideo into base64 JPEG frames at approximately ``fps``."""
    if fps <= 0:
        raise ValueError("video fps must be positive")
    try:
        import imageio.v3 as iio
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional benchmark dependency
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
        event_type = event.get("type")
        if event_type not in {
            "response.audio_transcript.delta",
            "response.output_text.delta",
        }:
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
        if received_at_s < turn_start_s:
            continue
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
        metrics.ttft_s = max(0.0, first_text_at_s - turn_start_s)
    elif metrics.ttft_s <= 0 and first_audio_at_s is not None:
        metrics.ttft_s = max(0.0, first_audio_at_s - turn_start_s)

    if metrics.audio_duration_s > 0 and metrics.response_generation_s > 0:
        metrics.rtf = metrics.response_generation_s / metrics.audio_duration_s
    metrics.success = bool(metrics.generated_text.strip() or metrics.audio_duration_s > 0)
    return metrics


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
        if video_frames and audio_end_ms > frames_sent * 1000:
            frame_index = min(frames_sent, len(video_frames) - 1)
            payload["video_frames"] = [video_frames[frame_index]]
            frames_sent += 1
        await client.send(payload)
        if realtime:
            await asyncio.sleep(duration_ms / 1000)


async def _wait_for_response_done(
    collector: RealtimeEventCollector,
    *,
    before_created: int,
    timeout_s: float,
) -> str | None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if collector.count("response.created") > before_created:
            response_id = collector.response_ids[before_created]
            done_for_response = sum(
                1
                for event in collector.events
                if event.get("type") == "response.done" and collector.response_id(event) == response_id
            )
            if done_for_response > 0:
                return response_id
        await asyncio.sleep(0.02)
    raise TimeoutError("Timed out waiting for response.done")


async def run_omniinteract_realtime_session(
    *,
    api_url: str,
    model: str,
    video_path: Path,
    turns: list[OmniInteractRealtimeTurn],
    session_key: str,
    ref_audio: str | None = None,
    chunk_ms: int = 200,
    video_fps: float = 1.0,
    realtime_pacing: bool = True,
    timeout_s: float = 120.0,
) -> OmniInteractRealtimeSessionResult:
    if not turns:
        raise ValueError("OmniInteract realtime session requires at least one turn")

    result = OmniInteractRealtimeSessionResult(session_key=session_key)
    session_start = time.monotonic()
    video_frames = sample_video_jpeg_frames(video_path, video_fps)
    ws_url = build_realtime_url(http_url_to_ws_url(api_url), model, session_id=session_key)

    try:
        async with RealtimeDuplexClient(ws_url) as client:
            await client.configure(
                model,
                ref_audio=_ref_audio_data_url(ref_audio),
                session_id=session_key,
                timeout_s=timeout_s,
            )
            for turn in turns:
                turn_start_s = time.monotonic()
                before_created = client.events.count("response.created")
                pcm16 = read_pcm16_wav(turn.audio_path)
                await _stream_pcm16_with_video(
                    client,
                    pcm16,
                    chunk_ms=chunk_ms,
                    realtime=realtime_pacing,
                    video_frames=video_frames,
                )
                await client.commit()
                try:
                    response_id = await _wait_for_response_done(
                        client.events,
                        before_created=before_created,
                        timeout_s=timeout_s,
                    )
                except TimeoutError as exc:
                    turn_metrics = OmniInteractRealtimeTurnMetrics(
                        turn_index=turn.turn_index,
                        response_id=None,
                        error=str(exc),
                    )
                    result.turn_metrics.append(turn_metrics)
                    result.turn_outputs.append(
                        {
                            "turn_index": turn.turn_index,
                            "gold_answer": turn.gold_answer,
                            "generated_text": "",
                            "success": False,
                            "error": str(exc),
                            **turn_metrics.as_dict(),
                        }
                    )
                    continue

                turn_metrics = compute_turn_metrics(
                    client.events,
                    response_id=response_id,
                    turn_start_s=turn_start_s,
                )
                turn_metrics.turn_index = turn.turn_index
                result.turn_metrics.append(turn_metrics)
                result.turn_outputs.append(
                    {
                        "turn_index": turn.turn_index,
                        "gold_answer": turn.gold_answer,
                        "generated_text": turn_metrics.generated_text,
                        "success": turn_metrics.success,
                        "subset": turn.subset,
                        "question_type": turn.question_type,
                        "video": turn.video_rel,
                        "scene_type": turn.scene_type,
                        "nested_group_id": turn.nested_group_id,
                        "nested_role": turn.nested_role,
                        "question_time": turn.question_time,
                        "answer_time": turn.answer_time,
                        "is_interrupted": turn.is_interrupted,
                        **turn_metrics.as_dict(),
                    }
                )
                await client.acknowledge_playback()

            await client.close_session(timeout_s=timeout_s)
    except Exception as exc:
        result.error = str(exc)
        result.success = False
        result.latency_s = time.monotonic() - session_start
        return result

    result.latency_s = time.monotonic() - session_start
    result.success = all(metric.success for metric in result.turn_metrics) and not result.error
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


def flatten_realtime_eval_pairs(
    session_results: list[OmniInteractRealtimeSessionResult],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return flattened per-turn request/output dicts for OmniInteract QA eval."""
    requests: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    for session in session_results:
        for turn_output in session.turn_outputs:
            requests.append(turn_output)
            outputs.append(
                {
                    "success": bool(turn_output.get("success")),
                    "generated_text": str(turn_output.get("generated_text") or ""),
                    "error": str(turn_output.get("error") or ""),
                }
            )
    return requests, outputs
