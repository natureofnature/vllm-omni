"""Unit tests for OmniInteract realtime benchmark helpers."""

from __future__ import annotations

import base64
import io
import json
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_omni.benchmarks.data_modules.omniinteract_realtime import (
    RealtimePlaybackAcknowledger,
    _committed_input_watermark,
    _event_index,
    _final_processing_outcome,
    _has_residual_model_unit,
    _post_commit_decision,
    _processed_input_event,
    _response_done_event_for_id,
    _stream_pcm16_with_video,
    _validate_final_response_done,
    compute_turn_metrics,
    extract_pcm16_from_video,
    http_url_to_ws_url,
    run_omniinteract_realtime_session,
    sample_video_jpeg_frames,
    summarize_turn_metrics,
    validate_realtime_chunk_ms,
    validate_realtime_video_fps,
)
from vllm_omni.benchmarks.patch.patch import (
    _project_continuous_session_result,
    benchmark,
)
from vllm_omni.experimental.fullduplex.client import (
    PCM16_BYTES_PER_SAMPLE,
    PCM16_SAMPLE_RATE,
)
from vllm_omni.experimental.fullduplex.client import (
    RealtimeEventCollector as ClientCollector,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
@pytest.mark.parametrize("metric", ["ttft", "tpot", "audio_ttft"])
async def test_continuous_session_rejects_undefined_goodput(metric: str):
    with pytest.raises(ValueError, match="undefined for a continuous Duplex Session"):
        await benchmark(
            task_type=None,
            endpoint_type="minicpmo-realtime",
            api_url="",
            base_url="",
            model_id="",
            model_name="",
            tokenizer=None,
            input_requests=[],
            logprobs=None,
            request_rate=float("inf"),
            burstiness=1.0,
            disable_tqdm=True,
            num_warmups=0,
            profile=False,
            selected_percentile_metrics=[],
            selected_percentiles=[],
            ignore_eos=False,
            goodput_config_dict={metric: 1.0},
            max_concurrency=None,
            lora_modules=None,
            extra_headers=None,
            extra_body=None,
        )


def test_http_url_to_ws_url_converts_http_scheme():
    assert http_url_to_ws_url("http://127.0.0.1:8099/v1/realtime") == "ws://127.0.0.1:8099/v1/realtime"


def test_post_commit_wait_ignores_precommit_decisions():
    events = [
        {"type": "response.listen"},
        {"type": "input_audio_buffer.committed"},
    ]
    committed_index = _event_index(events, "input_audio_buffer.committed", 0)
    assert committed_index == 1
    assert not _post_commit_decision(events, committed_index)

    events.append({"type": "response.listen"})
    assert _post_commit_decision(events, committed_index)


def test_residual_model_unit_requires_post_commit_decision():
    unit_bytes = PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE
    session_events = [{"session": {"capabilities": {"chunk_period_ms": 1000}}}]
    assert not _has_residual_model_unit(b"\0" * unit_bytes, session_events)
    assert _has_residual_model_unit(b"\0" * (unit_bytes + 2), session_events)


def test_final_input_watermark_rejects_uncorrelated_commit():
    with pytest.raises(RuntimeError, match="accepted_input_seq"):
        _committed_input_watermark({"type": "input_audio_buffer.committed"})


def test_final_input_watermark_ignores_prior_and_stale_events():
    events = [
        {"type": "response.done", "response": {"id": "old"}},
        {
            "type": "input_audio_buffer.committed",
            "session_id": "session-a",
            "epoch": 2,
            "accepted_input_seq": 8,
        },
        {
            "type": "input_audio_buffer.processed",
            "session_id": "session-a",
            "epoch": 1,
            "processed_input_seq": 8,
            "outcome": "listen",
        },
        {
            "type": "input_audio_buffer.processed",
            "session_id": "session-b",
            "epoch": 2,
            "processed_input_seq": 8,
            "outcome": "listen",
        },
        {
            "type": "input_audio_buffer.processed",
            "session_id": "session-a",
            "epoch": 2,
            "processed_input_seq": 7,
            "outcome": "listen",
        },
        {
            "type": "input_audio_buffer.processed",
            "session_id": "session-a",
            "epoch": 2,
            "processed_input_seq": 9,
            "outcome": "listen",
        },
    ]
    assert (
        _processed_input_event(
            events,
            after=1,
            session_id="session-a",
            epoch=2,
            accepted_input_seq=8,
        )
        is None
    )

    expected = {
        "type": "input_audio_buffer.processed",
        "session_id": "session-a",
        "epoch": 2,
        "processed_input_seq": 8,
        "outcome": "speak",
        "response_id": "response-final",
    }
    events.append(expected)
    assert (
        _processed_input_event(
            events,
            after=-1,
            session_id="session-a",
            epoch=2,
            accepted_input_seq=8,
        )
        is expected
    )
    assert _response_done_event_for_id(events, after=-1, response_id="response-final") is None
    events.append({"type": "response.done", "response": {"id": "response-final"}})
    assert _response_done_event_for_id(events, after=-1, response_id="response-final") is events[-1]


def test_final_processed_watermark_may_precede_commit_ack():
    processed = {
        "type": "input_audio_buffer.processed",
        "session_id": "session-a",
        "epoch": 2,
        "processed_input_seq": 8,
        "outcome": "speak",
        "response_id": "response-final",
    }
    response_done = {
        "type": "response.done",
        "response": {"id": "response-final", "status": "completed"},
    }
    events = [
        response_done,
        processed,
        {
            "type": "input_audio_buffer.committed",
            "session_id": "session-a",
            "epoch": 2,
            "accepted_input_seq": 8,
        },
    ]

    matched = _processed_input_event(
        events,
        after=-1,
        session_id="session-a",
        epoch=2,
        accepted_input_seq=8,
    )
    assert matched is processed
    matched_done = _response_done_event_for_id(events, after=-1, response_id="response-final")
    assert matched_done is response_done
    _validate_final_response_done(matched_done)


@pytest.mark.parametrize("status", [None, "unknown", "cancelled", "failed", "incomplete"])
def test_final_response_rejects_unsuccessful_terminal_status(status: str | None):
    with pytest.raises(RuntimeError, match="status"):
        _validate_final_response_done({"type": "response.done", "response": {"status": status}})


def test_final_response_accepts_completed_terminal_status():
    _validate_final_response_done({"type": "response.done", "response": {"status": "completed"}})


def test_final_input_outcome_validates_failure_and_speak_identity():
    assert _final_processing_outcome({"outcome": "listen"}) == ("listen", None)
    assert _final_processing_outcome({"outcome": "speak", "response_id": "resp-1"}) == (
        "speak",
        "resp-1",
    )
    with pytest.raises(RuntimeError, match="final accepted input failed"):
        _final_processing_outcome({"outcome": "failed"})
    with pytest.raises(RuntimeError, match="no response_id"):
        _final_processing_outcome({"outcome": "speak"})


def test_audio_extract_pads_video_only_tail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from vllm_omni.benchmarks.data_modules import omniinteract_realtime

    one_second = b"\x01\x00" * PCM16_SAMPLE_RATE
    monkeypatch.setattr(omniinteract_realtime.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        omniinteract_realtime.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=one_second, stderr=b""),
    )

    result = extract_pcm16_from_video(tmp_path / "sample.mp4", duration_s=2.2)

    assert len(result) == 3 * PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE
    assert result[: len(one_second)] == one_second
    assert not any(result[len(one_second) :])


def test_realtime_video_fps_rejects_frames_stage0_cannot_consume():
    assert validate_realtime_video_fps(1.0) == 1.0
    with pytest.raises(ValueError, match="at most 1 video frame per second"):
        validate_realtime_video_fps(2.0)


def test_realtime_chunk_size_rejects_invalid_or_multi_unit_appends():
    assert validate_realtime_chunk_ms(200) == 200
    with pytest.raises(ValueError, match=r"\[1, 1000\]"):
        validate_realtime_chunk_ms(0)
    with pytest.raises(ValueError, match=r"\[1, 1000\]"):
        validate_realtime_chunk_ms(1001)


def test_continuous_session_json_omits_undefined_generic_metrics():
    result = {
        "duration": 10.0,
        "completed": 2,
        "request_throughput": 0.2,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "output_throughput": 0.0,
        "total_token_throughput": 0.0,
        "audio_throughput": 0.0,
        "ttfts": [0.7],
        "itls": [[]],
        "mean_ttft_ms": 700.0,
        "median_tpot_ms": 20.0,
        "p99_itl_ms": 30.0,
        "mean_audio_ttft_ms": 0.0,
        "p99_audio_rtf": 0.5,
        "mean_e2el_ms": 1_000.0,
        "omniinteract_realtime_turn_ttft_mean_s": 0.2,
    }

    projected = _project_continuous_session_result(result)

    assert projected["completed"] == 2
    assert projected["request_throughput"] == pytest.approx(0.2)
    assert "total_input_tokens" not in projected
    assert "total_output_tokens" not in projected
    assert "audio_throughput" not in projected
    assert "ttfts" not in projected
    assert "itls" not in projected
    assert "mean_ttft_ms" not in projected
    assert "median_tpot_ms" not in projected
    assert "p99_itl_ms" not in projected
    assert "mean_audio_ttft_ms" not in projected
    assert "p99_audio_rtf" not in projected
    assert projected["mean_e2el_ms"] == 1_000.0
    assert projected["omniinteract_realtime_turn_ttft_mean_s"] == 0.2
    assert "generic token" in projected["omniinteract_realtime_metric_scope"]


def test_video_sampler_uses_each_interval_midpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    np = pytest.importorskip("numpy")
    iio = pytest.importorskip("imageio.v3")
    image_module = pytest.importorskip("PIL.Image")
    red = np.zeros((8, 8, 3), dtype=np.uint8)
    red[:, :, 0] = 255
    blue = np.zeros((8, 8, 3), dtype=np.uint8)
    blue[:, :, 2] = 255
    source_frames = [red, red, blue, blue]
    monkeypatch.setattr(iio, "immeta", lambda _: {"fps": 2.0})
    monkeypatch.setattr(iio, "imiter", lambda _: iter(source_frames))

    encoded = sample_video_jpeg_frames(tmp_path / "synthetic.mp4", 1.0, duration_s=2.0)
    decoded = [np.asarray(image_module.open(io.BytesIO(base64.b64decode(frame)))) for frame in encoded]

    assert len(decoded) == 2
    assert decoded[0][:, :, 0].mean() > decoded[0][:, :, 2].mean()
    assert decoded[1][:, :, 2].mean() > decoded[1][:, :, 0].mean()


@pytest.mark.asyncio
async def test_official_accuracy_requires_realtime_pacing(tmp_path: Path):
    result = await run_omniinteract_realtime_session(
        api_url="http://127.0.0.1:1/v1/realtime",
        model="model",
        video_path=tmp_path / "unused.mp4",
        session_key="sample",
        slots=[],
        official_output_root=tmp_path / "output",
        realtime_pacing=False,
    )
    assert not result.success
    assert "requires realtime pacing" in result.error


@pytest.mark.asyncio
async def test_playback_ack_advances_only_as_audio_can_play():
    collector = ClientCollector()
    pcm16 = b"\x01\x00" * 24_000
    collector.add(
        {
            "type": "response.audio.delta",
            "response_id": "resp-1",
            "delta": base64.b64encode(pcm16).decode("ascii"),
            "sample_rate_hz": 24_000,
        },
        received_at_s=10.0,
    )

    class FakeClient:
        def __init__(self):
            self.sent: list[dict[str, object]] = []

        async def send(self, event: dict[str, object]) -> None:
            self.sent.append(event)

    client = FakeClient()
    acknowledger = RealtimePlaybackAcknowledger()
    await acknowledger.acknowledge(client, collector, now_s=10.25)  # type: ignore[arg-type]
    await acknowledger.acknowledge(client, collector, now_s=10.75)  # type: ignore[arg-type]
    await acknowledger.acknowledge(client, collector, now_s=10.75)  # type: ignore[arg-type]

    assert [event["played_ms"] for event in client.sent] == [250, 750]
    assert all(event["committed_ms"] == event["played_ms"] for event in client.sent)

    # The server can emit response.done after the audio has already played.
    # Re-acknowledge the unchanged cursor once so it can commit that response
    # into playback-qualified history.
    collector.add({"type": "response.done", "response": {"id": "resp-1"}}, received_at_s=10.75)
    await acknowledger.acknowledge(client, collector, now_s=10.75)  # type: ignore[arg-type]
    await acknowledger.acknowledge(client, collector, now_s=10.75)  # type: ignore[arg-type]
    assert [event["played_ms"] for event in client.sent] == [250, 750, 750]


@pytest.mark.asyncio
async def test_video_frame_is_attached_at_interval_midpoint():
    class FakeClient:
        def __init__(self):
            self.events: list[dict[str, object]] = []

        async def send(self, event: dict[str, object]) -> None:
            self.events.append(event)

    client = FakeClient()
    pcm16 = bytes(PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE)
    audio_chunks_sent, video_frames_sent, pacing_mean_lag_s, pacing_max_lag_s = await _stream_pcm16_with_video(
        client,  # type: ignore[arg-type]
        pcm16,
        chunk_ms=200,
        realtime=False,
        video_frames=["frame-at-0.5s"],
        video_fps=1.0,
    )

    frame_events = [event for event in client.events if event.get("video_frames")]
    assert len(frame_events) == 1
    assert frame_events[0]["audio_end_ms"] == 600
    assert frame_events[0]["video_frames"] == ["frame-at-0.5s"]
    assert audio_chunks_sent == 5
    assert video_frames_sent == 1
    assert pacing_mean_lag_s == 0
    assert pacing_max_lag_s == 0


@pytest.mark.asyncio
async def test_realtime_stream_uses_absolute_media_deadlines(monkeypatch: pytest.MonkeyPatch):
    from vllm_omni.benchmarks.data_modules import omniinteract_realtime

    clock = {"now": 0.0}

    class FakeClient:
        def __init__(self):
            self.sent_at: list[float] = []

        async def send(self, event: dict[str, object]) -> None:
            self.sent_at.append(clock["now"])
            clock["now"] += 0.03

    async def fake_sleep(delay_s: float) -> None:
        clock["now"] += delay_s

    monkeypatch.setattr(omniinteract_realtime.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(omniinteract_realtime.asyncio, "sleep", fake_sleep)
    client = FakeClient()
    pcm16 = bytes(PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE)

    _chunks, _frames, mean_lag_s, max_lag_s = await _stream_pcm16_with_video(
        client,  # type: ignore[arg-type]
        pcm16,
        chunk_ms=200,
        realtime=True,
        video_frames=[],
        video_fps=1.0,
    )

    assert client.sent_at == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8])
    assert mean_lag_s == pytest.approx(0.0)
    assert max_lag_s == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_playback_ack_uses_exact_completed_segment_samples():
    collector = ClientCollector()
    for samples, received_at_s in ((20_160, 10.0), (11_520, 10.84)):
        collector.add(
            {
                "type": "response.audio.delta",
                "response_id": "resp-exact",
                "delta": base64.b64encode(b"\x01\x00" * samples).decode("ascii"),
                "sample_rate_hz": 24_000,
            },
            received_at_s=received_at_s,
        )

    acknowledger = RealtimePlaybackAcknowledger()
    assert acknowledger.played_ms(collector, now_s=11.32) == {"resp-exact": 1320}


def test_compute_turn_metrics_uses_stage0_engine_metrics():
    collector = ClientCollector()
    collector.add(
        {"type": "response.created", "response": {"id": "resp-a"}},
        received_at_s=10.0,
    )
    collector.add(
        {
            "type": "response.audio.delta",
            "response_id": "resp-a",
            "delta": base64.b64encode(b"audio").decode("ascii"),
            "sample_rate_hz": 24_000,
            "metadata": {
                "audio_duration_ms": 1000,
                "vllm_omni": {
                    "stage_metrics": {
                        "0": {
                            "vllm_ttft_ms": 150.0,
                            "vllm_tpot_ms": 20.0,
                        }
                    }
                },
            },
        },
        received_at_s=10.5,
    )
    collector.add(
        {"type": "response.done", "response_id": "resp-a"},
        received_at_s=11.0,
    )

    metrics = compute_turn_metrics(
        collector,
        response_id="resp-a",
        turn_start_s=9.5,
        stream_start_s=9.0,
    )

    assert metrics.ttft_s == pytest.approx(0.15)
    assert metrics.tpot_s == pytest.approx(0.02)
    assert metrics.audio_duration_s > 0
    assert metrics.success is True


def test_soft_match_rejects_tiny_ack_but_accepts_key_fragments():
    from vllm_omni.benchmarks.data_modules.omniinteract_eval import _is_soft_match, _normalize_text

    assert not _is_soft_match(_normalize_text("好的"), _normalize_text("好的，参谋团的参谋长是刘伯承。"))
    assert _is_soft_match(
        _normalize_text("参谋长是邓演达。秘书长是吴玉章。"),
        _normalize_text("好的，秘书厅的秘书长是吴玉章。"),
    )
    assert _is_soft_match(
        _normalize_text("你看这里有一台缝纫机哦。"),
        _normalize_text("好的，现在出现了缝纫机。"),
    )


def test_match_slot_uses_official_start_end_windows():
    from vllm_omni.benchmarks.data_modules.omniinteract_dataset import OmniInteractQASlot
    from vllm_omni.benchmarks.data_modules.omniinteract_realtime import _match_slot

    slots = [
        OmniInteractQASlot(
            0,
            "q0",
            "sewing",
            "00:57",
            "01:08",
            "proactive",
            question_time_s=57.0,
            answer_time_s=68.0,
            subset="1q1a",
            video_rel="v",
        ),
        OmniInteractQASlot(
            1,
            "q1",
            "liubocheng",
            "01:31",
            "01:35",
            "realtime",
            question_time_s=91.0,
            answer_time_s=95.0,
            subset="1q1a",
            video_rel="v",
        ),
        OmniInteractQASlot(
            2,
            "q2",
            "wuyuzhang",
            "01:36",
            "01:41",
            "realtime",
            question_time_s=96.0,
            answer_time_s=101.0,
            subset="1q1a",
            video_rel="v",
        ),
    ]

    # Early ack and later core answer share the proactive window.
    assert _match_slot(slots, video_time_s=60.8).slot_index == 0
    assert _match_slot(slots, video_time_s=70.0).slot_index == 0
    # Adjacent realtime windows do not steal each other's chunks.
    assert _match_slot(slots, video_time_s=95.3).slot_index == 1
    assert _match_slot(slots, video_time_s=98.4).slot_index == 2
    assert _match_slot(slots, video_time_s=101.2).slot_index == 2
    assert _match_slot(slots, video_time_s=10.0) is None


def test_summarize_turn_metrics_reports_session_level_means():
    from vllm_omni.benchmarks.data_modules.omniinteract_realtime import OmniInteractRealtimeTurnMetrics

    summary = summarize_turn_metrics(
        [
            OmniInteractRealtimeTurnMetrics(
                turn_index=0,
                response_id="a",
                ttft_s=0.1,
                tpot_s=0.02,
                rtf=0.8,
                audio_duration_s=1.0,
                response_generation_s=0.8,
                success=True,
            ),
            OmniInteractRealtimeTurnMetrics(
                turn_index=1,
                response_id="b",
                ttft_s=0.2,
                tpot_s=0.04,
                rtf=1.0,
                audio_duration_s=1.2,
                response_generation_s=1.2,
                success=True,
            ),
        ]
    )

    assert summary["omniinteract_realtime_turn_count"] == 2
    assert summary["omniinteract_realtime_turn_ttft_mean_s"] == pytest.approx(0.15)
    assert summary["omniinteract_realtime_turn_tpot_mean_s"] == pytest.approx(0.03)
    assert summary["omniinteract_realtime_turn_rtf_mean"] == pytest.approx(0.9)


def test_official_recorder_writes_time_aligned_multiturn_artifacts(tmp_path: Path):
    from vllm_omni.benchmarks.data_modules.omniinteract_official import (
        build_official_failure_summary,
        write_official_batch_files,
        write_official_session_artifacts,
    )

    collector = ClientCollector()
    pcm_chunk = b"\x01\x00" * 2_400  # 100 ms at 24 kHz.
    for response_id, text, created_at, audio_at, done_at in (
        ("resp-1", "first answer", 10.2, 10.4, 10.6),
        ("resp-2", "second answer", 11.1, 11.2, 11.4),
        ("resp-after-video", "too late", 12.1, 12.2, 12.4),
    ):
        collector.add(
            {"type": "response.created", "response": {"id": response_id}},
            received_at_s=created_at,
        )
        collector.add(
            {"type": "response.audio_transcript.delta", "response_id": response_id, "delta": text},
            received_at_s=audio_at,
        )
        collector.add(
            {
                "type": "response.audio.delta",
                "response_id": response_id,
                "format": "pcm16",
                "delta": base64.b64encode(pcm_chunk).decode("ascii"),
                "sample_rate_hz": 24_000,
            },
            received_at_s=audio_at,
        )
        collector.add({"type": "response.done", "response_id": response_id}, received_at_s=done_at)

    annotation = tmp_path / "annotations" / "sample.json"
    annotation.parent.mkdir()
    annotation.write_text("[]", encoding="utf-8")
    video = tmp_path / "1q1a" / "videos" / "sample.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    output_root = tmp_path / "outputs"
    summary = write_official_session_artifacts(
        output_root=output_root,
        subset="1q1a",
        video_rel="videos/sample.mp4",
        video_path=video,
        annotation_path=annotation,
        scene_type="multi_turn",
        duration_s=2.0,
        inference_s=2.1,
        collector=collector,
        stream_start_s=10.0,
        status="ok",
        preprocess_s=0.25,
        input_audio_chunks=10,
        input_video_frames=2,
    )

    output_dir = Path(summary["output_dir"])
    with wave.open(str(output_dir / "output.wav"), "rb") as wav_file:
        assert wav_file.getframerate() == 24_000
        assert wav_file.getnframes() == 48_000
    responses = [json.loads(line) for line in (output_dir / "responses.jsonl").read_text().splitlines()]
    transcript = json.loads((output_dir / "wav_transcript.json").read_text())
    compact_events = [json.loads(line) for line in (output_dir / "events.jsonl").read_text().splitlines()]
    assert [response["text"] for response in responses] == ["first answer", "second answer", "too late"]
    assert len(transcript["chunks"]) == 2
    assert transcript["text"] == "first answer second answer"
    assert summary["preprocess_sec"] == pytest.approx(0.25)
    assert summary["inference_sec"] == pytest.approx(2.1)
    assert summary["paced_e2e_ratio"] == pytest.approx(1.05)
    assert "rtf" not in summary
    assert summary["input_audio_chunks"] == 10
    assert summary["input_video_frames"] == 2
    assert all("delta" not in event for event in compact_events if event["type"] == "response.audio.delta")

    batch_path, manifest_path = write_official_batch_files(output_root, [summary])
    batch = json.loads(batch_path.read_text())
    manifest = json.loads(manifest_path.read_text().strip())
    assert batch["success"] == 1
    assert manifest["gt_json"] == str(annotation.resolve())
    assert manifest["scene_type"] == "multi_turn"

    (output_dir / "output.json").write_text("{}", encoding="utf-8")
    (output_dir / "precise_truncation.json").write_text("{}", encoding="utf-8")
    errored = write_official_session_artifacts(
        output_root=output_root,
        subset="1q1a",
        video_rel="videos/sample.mp4",
        video_path=video,
        annotation_path=annotation,
        scene_type="multi_turn",
        duration_s=2.0,
        inference_s=2.1,
        collector=collector,
        stream_start_s=10.0,
        status="error",
        error="protocol failed",
    )
    assert errored["status"] == "error"
    assert not (output_dir / ".done").exists()
    assert not (output_dir / "output.json").exists()
    assert not (output_dir / "precise_truncation.json").exists()
    assert json.loads((output_dir / ".failed.json").read_text())["error"] == "protocol failed"

    failure = build_official_failure_summary(
        output_root=output_root,
        subset="1q1a",
        video_rel="videos/sample.mp4",
        video_path=video,
        annotation_path=annotation,
        scene_type="multi_turn",
        error="request failed",
    )
    assert not (output_dir / ".done").exists()
    assert json.loads((output_dir / ".failed.json").read_text())["error"] == "request failed"
    batch_path, manifest_path = write_official_batch_files(output_root, [summary, failure])
    batch = json.loads(batch_path.read_text())
    assert (batch["total"], batch["success"], batch["failed"]) == (2, 1, 1)
    assert len(manifest_path.read_text().splitlines()) == 1


def test_official_recorder_preserves_fractional_video_final_unit(tmp_path: Path):
    from vllm_omni.benchmarks.data_modules.omniinteract_official import (
        write_official_session_artifacts,
    )

    collector = ClientCollector()
    response_id = "response-final-unit"
    pcm_chunk = b"\x01\x00" * 4_800  # 200 ms at 24 kHz.
    collector.add({"type": "response.created", "response": {"id": response_id}}, received_at_s=2.3)
    collector.add(
        {"type": "response.audio_transcript.delta", "response_id": response_id, "delta": "tail answer"},
        received_at_s=2.4,
    )
    collector.add(
        {
            "type": "response.audio.delta",
            "response_id": response_id,
            "format": "pcm16",
            "delta": base64.b64encode(pcm_chunk).decode("ascii"),
            "sample_rate_hz": 24_000,
        },
        received_at_s=2.4,
    )
    collector.add(
        {"type": "response.done", "response": {"id": response_id, "status": "completed"}},
        received_at_s=2.6,
    )
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"video")

    summary = write_official_session_artifacts(
        output_root=tmp_path / "outputs",
        subset="1q1a",
        video_rel="sample.mp4",
        video_path=video,
        annotation_path=None,
        scene_type="multi_turn",
        duration_s=2.2,
        inference_s=2.7,
        collector=collector,
        stream_start_s=0.0,
        status="ok",
    )

    output_dir = Path(summary["output_dir"])
    with wave.open(str(output_dir / "output.wav"), "rb") as wav_file:
        assert wav_file.getnframes() == 3 * 24_000
        assert any(wav_file.readframes(wav_file.getnframes()))
    transcript = json.loads((output_dir / "wav_transcript.json").read_text())
    assert transcript["text"] == "tail answer"
    assert transcript["chunks"][0]["timestamp"] == [2.4, 2.6]
    assert summary["duration_sec"] == pytest.approx(2.2)


@pytest.mark.parametrize(
    "failed_event",
    [
        {"type": "response.done", "response_id": "early", "status": "failed"},
        {
            "type": "response.done",
            "response": {"id": "early", "status": "failed"},
        },
        {
            "type": "response.done",
            "response_id": "early",
            "status_details": {"type": "failed"},
        },
        {
            "type": "response.done",
            "response": {
                "id": "early",
                "status_details": {"type": "failed"},
            },
        },
        {
            "type": "response.done",
            "status": "failed",
            "response": {"id": "early", "status": "completed"},
        },
        {
            "type": "response.done",
            "status_details": {"type": "failed"},
            "response": {
                "id": "early",
                "status_details": {"type": "completed"},
            },
        },
    ],
)
def test_official_recorder_rejects_early_failed_response(
    tmp_path: Path,
    failed_event: dict[str, object],
):
    from vllm_omni.benchmarks.data_modules.omniinteract_official import (
        build_official_failure_summary,
        write_official_batch_files,
        write_official_session_artifacts,
    )

    collector = ClientCollector()
    collector.add(failed_event, received_at_s=0.5)
    # A later successful response must not hide the earlier failed unit.
    collector.add(
        {
            "type": "response.done",
            "response": {"id": "final", "status": "completed"},
        },
        received_at_s=1.0,
    )
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"video")
    output_root = tmp_path / "outputs"

    with pytest.raises(ValueError, match="reports a failed response"):
        write_official_session_artifacts(
            output_root=output_root,
            subset="1q1a",
            video_rel="sample.mp4",
            video_path=video,
            annotation_path=None,
            scene_type="multi_turn",
            duration_s=1.0,
            inference_s=1.0,
            collector=collector,
            stream_start_s=0.0,
            status="ok",
        )

    failure = build_official_failure_summary(
        output_root=output_root,
        subset="1q1a",
        video_rel="sample.mp4",
        video_path=video,
        annotation_path=None,
        scene_type="multi_turn",
        error="response.done reports a failed response",
    )
    _, manifest_path = write_official_batch_files(output_root, [failure])
    output_dir = Path(failure["output_dir"])
    assert (output_dir / ".failed.json").is_file()
    assert not (output_dir / ".done").exists()
    assert manifest_path.read_text() == ""


@pytest.mark.parametrize(
    "cancelled_event",
    [
        {"type": "response.done", "response_id": "interrupted", "status": "cancelled"},
        {
            "type": "response.done",
            "response": {"id": "interrupted", "status": "cancelled"},
        },
    ],
)
def test_official_recorder_allows_cancelled_response(cancelled_event: dict[str, object]):
    from vllm_omni.benchmarks.data_modules.omniinteract_official import (
        validate_official_event_stream,
    )

    collector = ClientCollector()
    collector.add(cancelled_event, received_at_s=0.5)
    validate_official_event_stream(collector)


@pytest.mark.parametrize(
    "audio_event, error",
    [
        (
            {"type": "response.audio.delta", "delta": base64.b64encode(b"\x00\x00").decode("ascii")},
            "response identity",
        ),
        (
            {
                "type": "response.audio.delta",
                "response_id": "resp-1",
                "format": "pcm16",
                "sample_rate_hz": 24_000,
                "delta": "not base64!",
            },
            "invalid base64",
        ),
        (
            {
                "type": "response.audio.delta",
                "response_id": "resp-1",
                "format": "pcm16",
                "sample_rate_hz": 24_000,
                "delta": base64.b64encode(b"\x00").decode("ascii"),
            },
            "odd number of PCM16 bytes",
        ),
        (
            {
                "type": "response.audio.delta",
                "response_id": "resp-1",
                "format": "wav",
                "sample_rate_hz": 24_000,
                "delta": base64.b64encode(b"RIFF").decode("ascii"),
            },
            "format=pcm16",
        ),
        (
            {
                "type": "response.audio.delta",
                "response_id": "resp-1",
                "format": "pcm16",
                "delta": base64.b64encode(b"\x00\x00").decode("ascii"),
            },
            "positive integer sample_rate_hz",
        ),
    ],
)
def test_official_recorder_rejects_malformed_audio(
    tmp_path: Path,
    audio_event: dict[str, object],
    error: str,
):
    from vllm_omni.benchmarks.data_modules.omniinteract_official import (
        write_official_session_artifacts,
    )

    collector = ClientCollector()
    collector.add(audio_event, received_at_s=1.0)
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"video")
    output_root = tmp_path / "outputs"
    output_dir = output_root / "1q1a" / "sample"
    output_dir.mkdir(parents=True)
    (output_dir / ".done").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        write_official_session_artifacts(
            output_root=output_root,
            subset="1q1a",
            video_rel="sample.mp4",
            video_path=video,
            annotation_path=None,
            scene_type="multi_turn",
            duration_s=1.0,
            inference_s=1.0,
            collector=collector,
            stream_start_s=0.0,
            status="ok",
        )

    assert not (output_dir / ".done").exists()


def test_official_recorder_rejects_mixed_output_sample_rates(tmp_path: Path):
    from vllm_omni.benchmarks.data_modules.omniinteract_official import (
        write_official_session_artifacts,
    )

    collector = ClientCollector()
    for sample_rate_hz in (24_000, 16_000):
        collector.add(
            {
                "type": "response.audio.delta",
                "response_id": "resp-1",
                "format": "pcm16",
                "sample_rate_hz": sample_rate_hz,
                "delta": base64.b64encode(b"\x00\x00").decode("ascii"),
            },
            received_at_s=1.0,
        )
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"video")

    with pytest.raises(ValueError, match="inconsistent sample_rate_hz"):
        write_official_session_artifacts(
            output_root=tmp_path / "outputs",
            subset="1q1a",
            video_rel="sample.mp4",
            video_path=video,
            annotation_path=None,
            scene_type="multi_turn",
            duration_s=1.0,
            inference_s=1.0,
            collector=collector,
            stream_start_s=0.0,
            status="ok",
        )


def test_official_1qna_output_layout_strips_dataset_transport_prefix(tmp_path: Path):
    from vllm_omni.benchmarks.data_modules.omniinteract_official import official_output_dir

    assert official_output_dir(
        tmp_path,
        subset="1qna",
        video_rel="videos_bench/activitynet/clip.mp4",
    ) == (tmp_path / "1qna" / "activitynet__clip")
