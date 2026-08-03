"""Unit tests for OmniInteract realtime benchmark helpers."""

from __future__ import annotations

import base64

import pytest

from vllm_omni.benchmarks.data_modules.omniinteract_realtime import (
    _event_index,
    _has_residual_model_unit,
    _post_commit_decision,
    compute_turn_metrics,
    http_url_to_ws_url,
    summarize_turn_metrics,
)
from vllm_omni.experimental.fullduplex.client import (
    PCM16_BYTES_PER_SAMPLE,
    PCM16_SAMPLE_RATE,
)
from vllm_omni.experimental.fullduplex.client import (
    RealtimeEventCollector as ClientCollector,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


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
