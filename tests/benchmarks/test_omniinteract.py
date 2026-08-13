from __future__ import annotations

import base64
import io
import tarfile
from pathlib import Path

import pytest

from vllm_omni.benchmarks.data_modules import omniinteract as oi
from vllm_omni.experimental.fullduplex.client import RealtimeEventCollector

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


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
