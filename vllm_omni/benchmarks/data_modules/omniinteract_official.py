"""Official-compatible OmniInteract output artifacts.

The official evaluator consumes a time-aligned output WAV, a native model
transcript, and one batch record per source video.  This module converts the
Realtime API event stream into that interchange format without copying the
official ASR, forced-alignment, or LLM-judge implementation into vLLM-Omni.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
import os
import shutil
import wave
from pathlib import Path
from typing import Any

from vllm_omni.experimental.fullduplex.client import RealtimeEventCollector

OUTPUT_SAMPLE_RATE = 24_000
PCM16_BYTES_PER_SAMPLE = 2


def validate_official_event_stream(collector: RealtimeEventCollector) -> None:
    """Reject malformed output that would make an official score meaningless."""
    output_sample_rate_hz: int | None = None
    for index, event in enumerate(collector.events):
        event_type = event.get("type")
        if event_type == "response.done":
            response = event.get("response")
            nested_status = response.get("status") if isinstance(response, dict) else None
            top_status = event.get("status")
            nested_details = response.get("status_details") if isinstance(response, dict) else None
            top_details = event.get("status_details")
            nested_details_type = nested_details.get("type") if isinstance(nested_details, dict) else None
            top_details_type = top_details.get("type") if isinstance(top_details, dict) else None
            if "failed" in {nested_status, top_status, nested_details_type, top_details_type}:
                raise ValueError(f"response.done event {index} reports a failed response")
            continue
        if event_type != "response.audio.delta":
            continue
        response_id = collector.response_id(event)
        if not response_id:
            raise ValueError(f"response.audio.delta event {index} has no response identity")
        if event.get("format") != "pcm16":
            raise ValueError(f"response.audio.delta event {index} must use format=pcm16")
        sample_rate_hz = event.get("sample_rate_hz")
        if isinstance(sample_rate_hz, bool) or not isinstance(sample_rate_hz, int) or sample_rate_hz <= 0:
            raise ValueError(f"response.audio.delta event {index} has no positive integer sample_rate_hz")
        if output_sample_rate_hz is None:
            output_sample_rate_hz = sample_rate_hz
        elif sample_rate_hz != output_sample_rate_hz:
            raise ValueError(
                "response.audio.delta events use inconsistent sample_rate_hz values: "
                f"{output_sample_rate_hz} and {sample_rate_hz}"
            )
        delta = event.get("delta") or event.get("audio")
        if not isinstance(delta, str) or not delta:
            raise ValueError(f"response.audio.delta event {index} has no audio payload")
        try:
            pcm16 = base64.b64decode(delta, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError(f"response.audio.delta event {index} contains invalid base64") from exc
        if len(pcm16) % PCM16_BYTES_PER_SAMPLE:
            raise ValueError(f"response.audio.delta event {index} contains an odd number of PCM16 bytes")


def official_output_dir(output_root: Path, *, subset: str, video_rel: str) -> Path:
    """Return the directory layout used by the official OmniInteract runners."""
    normalized = video_rel.replace("\\", "/").lstrip("./")
    if subset.lower() == "1qna" and normalized.startswith("videos_bench/"):
        normalized = normalized.removeprefix("videos_bench/")
    name = normalized.replace("/", "__")
    if name.lower().endswith(".mp4"):
        name = name[:-4]
    return output_root / subset / name


def _response_text(collector: RealtimeEventCollector, response_id: str) -> str:
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


def _json_event(event: dict[str, object], received_at_s: float, stream_start_s: float) -> dict[str, object]:
    """Keep protocol evidence while avoiding a second copy of base64 audio."""
    out = {key: value for key, value in event.items() if key != "_client_received_at_s"}
    out["received_at_second"] = round(max(0.0, received_at_s - stream_start_s), 6)
    if out.get("type") == "response.audio.delta":
        delta = out.pop("delta", None) or out.pop("audio", None)
        if isinstance(delta, str):
            try:
                out["audio_bytes"] = len(base64.b64decode(delta, validate=True))
            except (ValueError, binascii.Error) as exc:
                raise ValueError("response.audio.delta contains invalid base64") from exc
    return out


def _build_playback(
    collector: RealtimeEventCollector,
    *,
    stream_start_s: float,
    duration_s: float,
) -> tuple[bytes, list[dict[str, Any]], dict[str, Any]]:
    """Build a deterministic client-playback timeline from received audio deltas."""
    sample_rate = int(collector.output_sample_rate_hz or OUTPUT_SAMPLE_RATE)
    # The official runner processes and writes ceil(source_duration) complete
    # one-second units. Preserve the whole final unit instead of clipping its
    # answer at the source video's fractional-second boundary.
    output_duration_s = float(math.ceil(duration_s))
    total_samples = max(0, int(output_duration_s * sample_rate))
    output = bytearray(total_samples * PCM16_BYTES_PER_SAMPLE)
    playback_cursor_s = 0.0
    response_times: dict[str, dict[str, float]] = {}

    for event, received_at_s in zip(collector.events, collector.event_received_at_s, strict=True):
        if event.get("type") != "response.audio.delta":
            continue
        response_id = collector.response_id(event)
        delta = event.get("delta") or event.get("audio")
        if not response_id:
            raise ValueError("response.audio.delta has no response identity")
        if not isinstance(delta, str) or not delta:
            raise ValueError("response.audio.delta has no audio payload")
        try:
            pcm16 = base64.b64decode(delta, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("response.audio.delta contains invalid base64") from exc
        if len(pcm16) % PCM16_BYTES_PER_SAMPLE:
            raise ValueError("response.audio.delta contains an odd number of PCM16 bytes")
        if not pcm16:
            continue
        received_s = max(0.0, received_at_s - stream_start_s)
        start_s = max(received_s, playback_cursor_s)
        audio_duration_s = len(pcm16) / (sample_rate * PCM16_BYTES_PER_SAMPLE)
        end_s = start_s + audio_duration_s
        playback_cursor_s = end_s

        start_byte = int(round(start_s * sample_rate)) * PCM16_BYTES_PER_SAMPLE
        if start_byte < len(output):
            copy_len = min(len(pcm16), len(output) - start_byte)
            output[start_byte : start_byte + copy_len] = pcm16[:copy_len]
        timing = response_times.setdefault(response_id, {"start": start_s, "end": end_s, "audio_duration": 0.0})
        timing["start"] = min(timing["start"], start_s)
        timing["end"] = max(timing["end"], end_s)
        timing["audio_duration"] += audio_duration_s

    responses: list[dict[str, Any]] = []
    transcript_chunks: list[dict[str, Any]] = []
    for response_id in collector.response_ids:
        text = _response_text(collector, response_id)
        timing = response_times.get(response_id)
        created_s = next(
            (
                max(0.0, received - stream_start_s)
                for event, received in zip(collector.events, collector.event_received_at_s, strict=True)
                if event.get("type") == "response.created" and collector.response_id(event) == response_id
            ),
            0.0,
        )
        done_s = next(
            (
                max(0.0, received - stream_start_s)
                for event, received in zip(collector.events, collector.event_received_at_s, strict=True)
                if event.get("type") == "response.done" and collector.response_id(event) == response_id
            ),
            created_s,
        )
        start_s = timing["start"] if timing else created_s
        end_s = timing["end"] if timing else max(start_s, done_s)
        chunks = [{"text": text, "timestamp": [round(start_s, 6), round(end_s, 6)]}] if text else []
        responses.append(
            {
                "response_id": response_id,
                "triggered_at_second": round(created_s, 6),
                "done_at_second": round(max(done_s, end_s), 6),
                "text": text,
                "chunks": chunks,
                "audio_duration_sec": round(timing["audio_duration"], 6) if timing else 0.0,
            }
        )
        clipped_start_s = min(max(start_s, 0.0), output_duration_s)
        clipped_end_s = min(max(end_s, clipped_start_s), output_duration_s)
        if text and timing and clipped_start_s < clipped_end_s:
            transcript_chunks.append(
                {
                    "text": text,
                    "timestamp": [round(clipped_start_s, 6), round(clipped_end_s, 6)],
                }
            )

    transcript = {
        "text": " ".join(chunk["text"] for chunk in transcript_chunks),
        "chunks": transcript_chunks,
    }
    return bytes(output), responses, transcript


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_wav(path: Path, pcm16: bytes, *, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(PCM16_BYTES_PER_SAMPLE)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16)


def write_official_session_artifacts(
    *,
    output_root: Path,
    subset: str,
    video_rel: str,
    video_path: Path,
    annotation_path: Path | None,
    scene_type: str,
    duration_s: float,
    inference_s: float,
    collector: RealtimeEventCollector,
    stream_start_s: float,
    status: str,
    preprocess_s: float = 0.0,
    error: str = "",
    input_audio_chunks: int | None = None,
    input_video_frames: int | None = None,
    pacing_mean_lag_s: float | None = None,
    pacing_max_lag_s: float | None = None,
) -> dict[str, Any]:
    """Write one session in the format consumed by the official evaluator."""
    output_dir = official_output_dir(output_root, subset=subset, video_rel=video_rel)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Every inference run invalidates evaluator-derived files. Reusing an old
    # ASR/alignment/judge result would silently score a previous model output.
    for stale_name in (
        ".done",
        ".failed.json",
        "output.json",
        "wav_transcript_aligned.json",
        "precise_truncation.json",
    ):
        (output_dir / stale_name).unlink(missing_ok=True)
    validate_official_event_stream(collector)
    tmp_dir = output_dir / ".tmp"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    audio_dir = tmp_dir / "audio_per_second"
    audio_dir.mkdir(parents=True)

    pcm16, responses, transcript = _build_playback(
        collector,
        stream_start_s=stream_start_s,
        duration_s=duration_s,
    )
    sample_rate = int(collector.output_sample_rate_hz or OUTPUT_SAMPLE_RATE)
    _write_wav(tmp_dir / "output.wav", pcm16, sample_rate=sample_rate)
    _write_json(tmp_dir / "wav_transcript.json", transcript)
    _write_jsonl(tmp_dir / "responses.jsonl", responses)
    compact_events = [
        _json_event(event, received, stream_start_s)
        for event, received in zip(collector.events, collector.event_received_at_s, strict=True)
    ]
    _write_jsonl(tmp_dir / "events.jsonl", compact_events)

    seconds = max(1, int(math.ceil(duration_s)))
    per_second_rows: list[dict[str, Any]] = []
    text_lines: list[str] = []
    bytes_per_second = sample_rate * PCM16_BYTES_PER_SAMPLE
    for second in range(seconds):
        start = second * bytes_per_second
        chunk = pcm16[start : min(len(pcm16), start + bytes_per_second)]
        has_audio = any(chunk)
        audio_path: str | None = None
        if has_audio:
            pcm_path = audio_dir / f"{second:04d}.pcm"
            pcm_path.write_bytes(chunk)
            audio_path = f"audio_per_second/{pcm_path.name}"
        active = [
            response
            for response in responses
            if response["triggered_at_second"] < second + 1 and response["done_at_second"] >= second
        ]
        started = [response for response in responses if second <= response["triggered_at_second"] < second + 1]
        text = "".join(str(response["text"]) for response in started)
        row = {
            "second": second,
            "text": text,
            "is_listen": not active,
            "end_of_turn": any(second <= response["done_at_second"] < second + 1 for response in responses),
            "audio_pcm_path": audio_path,
            "audio_sample_rate": sample_rate,
            "audio_channels": 1,
            "audio_sample_width": PCM16_BYTES_PER_SAMPLE,
            "audio_bytes": len(chunk) if has_audio else 0,
        }
        per_second_rows.append(row)
        text_lines.append(f"[{second:04d}s] TEXT: {text}" if text else f"[{second:04d}s] listen...")
    _write_jsonl(tmp_dir / "model_output.jsonl", per_second_rows)
    (tmp_dir / "model_output.txt").write_text("\n".join(text_lines) + "\n", encoding="utf-8")

    for name in (
        "output.wav",
        "wav_transcript.json",
        "responses.jsonl",
        "events.jsonl",
        "model_output.jsonl",
        "model_output.txt",
    ):
        os.replace(tmp_dir / name, output_dir / name)
    final_audio_dir = output_dir / "audio_per_second"
    shutil.rmtree(final_audio_dir, ignore_errors=True)
    os.replace(audio_dir, final_audio_dir)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    summary: dict[str, Any] = {
        "video": str(video_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "duration_sec": round(duration_s, 6),
        "num_chunks": int(math.ceil(duration_s)),
        "preprocess_sec": round(preprocess_s, 6),
        "inference_sec": round(inference_s, 6),
        "paced_e2e_ratio": round(inference_s / max(duration_s, 0.001), 6),
        "num_responses": len(responses),
        "status": status,
        "subset": subset,
        "scene_type": "1QnA" if scene_type.lower() == "1qna" else scene_type,
        "annotation": str(annotation_path.resolve()) if annotation_path else "",
    }
    if input_audio_chunks is not None:
        summary["input_audio_chunks"] = input_audio_chunks
    if input_video_frames is not None:
        summary["input_video_frames"] = input_video_frames
    if pacing_mean_lag_s is not None:
        summary["pacing_mean_lag_sec"] = round(pacing_mean_lag_s, 6)
    if pacing_max_lag_s is not None:
        summary["pacing_max_lag_sec"] = round(pacing_max_lag_s, 6)
    if error:
        summary["error"] = error
    marker = ".done" if status == "ok" else ".failed.json"
    _write_json(output_dir / marker, summary)
    return summary


def build_official_failure_summary(
    *,
    output_root: Path,
    subset: str,
    video_rel: str,
    video_path: Path | None,
    annotation_path: Path | None,
    scene_type: str,
    error: str,
) -> dict[str, Any]:
    """Represent a failed requested sample without making it evaluable."""
    output_dir = official_output_dir(output_root, subset=subset, video_rel=video_rel)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in (
        ".done",
        "output.json",
        "wav_transcript_aligned.json",
        "precise_truncation.json",
    ):
        (output_dir / stale_name).unlink(missing_ok=True)
    failure = {
        "video": str(video_path.resolve()) if video_path is not None else video_rel,
        "output_dir": str(output_dir.resolve()),
        "status": "error",
        "subset": subset,
        "scene_type": "1QnA" if scene_type.lower() == "1qna" else scene_type,
        "annotation": str(annotation_path.resolve()) if annotation_path else "",
        "error": error,
    }
    _write_json(output_dir / ".failed.json", failure)
    return failure


def write_official_batch_files(output_root: Path, summaries: list[dict[str, Any]]) -> tuple[Path, Path]:
    """Write the official batch summary and evaluator manifest."""
    output_root.mkdir(parents=True, exist_ok=True)
    successes = [row for row in summaries if row.get("status") == "ok"]
    batch = {
        "total": len(summaries),
        "success": len(successes),
        "failed": len(summaries) - len(successes),
        "results": summaries,
    }
    batch_path = output_root / "batch_summary.json"
    _write_json(batch_path, batch)

    manifest_rows: list[dict[str, Any]] = []
    for index, row in enumerate(successes):
        annotation = str(row.get("annotation") or "")
        output_dir = Path(str(row.get("output_dir") or ""))
        if not annotation or not output_dir:
            continue
        manifest_rows.append(
            {
                "sample_id": f"{row.get('subset', 'sample')}__{output_dir.name or index}",
                "gt_json": annotation,
                "model_json": str((output_dir / "wav_transcript.json").resolve()),
                "scene_type": row.get("scene_type") or "multi_turn",
            }
        )
    manifest_path = output_root / "official_eval_manifest.jsonl"
    _write_jsonl(manifest_path, manifest_rows)
    return batch_path, manifest_path


__all__ = [
    "build_official_failure_summary",
    "official_output_dir",
    "validate_official_event_stream",
    "write_official_batch_files",
    "write_official_session_artifacts",
]
