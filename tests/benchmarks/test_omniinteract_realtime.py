"""Unit tests for OmniInteract realtime benchmark helpers."""

from __future__ import annotations

import base64

import pytest

from vllm_omni.benchmarks.data_modules.omniinteract_realtime import (
    compute_turn_metrics,
    http_url_to_ws_url,
    summarize_turn_metrics,
)
from vllm_omni.experimental.fullduplex.client import RealtimeEventCollector as ClientCollector

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_http_url_to_ws_url_converts_http_scheme():
    assert http_url_to_ws_url("http://127.0.0.1:8099/v1/realtime") == "ws://127.0.0.1:8099/v1/realtime"


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
