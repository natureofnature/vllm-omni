"""code2wav async finish-sentinel terminal handling.

The runner marks terminal real codec payloads with ``is_finished=True`` so the
normal code2wav branch can flush any trailing partial chunk.  The later sentinel
is a control-only marker and must not rebuild audio from cached history.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.data_entry_keys import ASYNC_FINISH_SENTINEL_KEY
from vllm_omni.model_executor.stage_input_processors.qwen3_omni import talker2code2wav_async_chunk


def _tm(accumulated, chunk_frames=4, left_frames=25, initial_frames=None):
    cfg = {"codec_chunk_frames": chunk_frames, "codec_left_context_frames": left_frames}
    if initial_frames is not None:
        cfg["initial_codec_chunk_frames"] = initial_frames
    return SimpleNamespace(
        code_prompt_token_ids=dict(accumulated),
        connector=SimpleNamespace(config={"extra": cfg}),
    )


def _sentinel_payload():
    return {ASYNC_FINISH_SENTINEL_KEY: True}


def test_codec_config_can_come_from_runner_model_config():
    cfg = {
        "codec_chunk_frames": 25,
        "codec_left_context_frames": 25,
        "initial_codec_chunk_frames": 4,
    }
    model_config = SimpleNamespace(stage_connector_config={"extra": cfg})
    tm = SimpleNamespace(
        code_prompt_token_ids={"r": []},
        put_req_chunk={"r": 0},
    )
    tm._get_model_config = lambda: model_config
    req = SimpleNamespace(external_req_id="r", sampling_params=None)
    frame = torch.ones((1, 16), dtype=torch.long)

    for _ in range(3):
        assert talker2code2wav_async_chunk(tm, {"codes": {"audio": frame}}, req) is None
    out = talker2code2wav_async_chunk(tm, {"codes": {"audio": frame}}, req)

    assert out is not None
    assert out.codes.audio.numel() == 64
    assert out.meta.left_context_size == 0


def test_talker2code2wav_async_chunk_accepts_flat_payload():
    tm = _tm({"r": []}, chunk_frames=1, left_frames=0)
    tm.put_req_chunk = {"r": 0}
    req = SimpleNamespace(external_req_id="r", sampling_params=None)
    frame = torch.arange(1, 17, dtype=torch.long).reshape(1, 16)

    out = talker2code2wav_async_chunk(tm, {"codes.audio": frame}, req)

    assert out is not None
    assert torch.equal(out.codes.audio, frame.transpose(0, 1).reshape(-1))


@pytest.mark.parametrize(
    ("chunk_frames", "initial_frames"),
    [(4, None), (25, 4)],
    ids=["regular-chunk", "initial-chunk"],
)
def test_finished_real_payload_flushes_partial_tail(chunk_frames: int, initial_frames: int | None):
    tm = _tm(
        {"r": [torch.tensor([[i]]) for i in range(1, 6)]},
        chunk_frames=chunk_frames,
        initial_frames=initial_frames,
    )
    tm.put_req_chunk = {"r": 1}
    req = SimpleNamespace(external_req_id="r")
    frame = torch.tensor([[6]])

    out = talker2code2wav_async_chunk(tm, {"codes": {"audio": frame}}, req, is_finished=True)

    assert out is not None
    assert bool(out.meta.finished) is True
    assert out.meta.left_context_size == 4
    assert isinstance(out.codes.audio, torch.Tensor)
    assert out.codes.audio.numel() == 6


def test_finish_sentinel_flushes_partial_tail_without_finishing_request():
    tm = _tm({"r": [torch.tensor([[i]]) for i in range(1, 4)]}, chunk_frames=4, left_frames=25)
    tm.put_req_chunk = {"r": 0}
    req = SimpleNamespace(external_req_id="r")

    out = talker2code2wav_async_chunk(tm, _sentinel_payload(), req, is_finished=False)

    assert out is not None
    assert bool(out.meta.finished) is False
    assert out.meta.left_context_size == 0
    assert isinstance(out.codes.audio, torch.Tensor)
    assert out.codes.audio.numel() == 3


@pytest.mark.parametrize(
    "chunk_config",
    [{"chunk_frames": 4}, {"chunk_frames": 25, "initial_frames": 4}],
    ids=["regular-chunk", "initial-chunk"],
)
def test_finish_sentinel_on_chunk_boundary_emits_flag_only(chunk_config: dict[str, int]):
    tm = _tm({"r": [torch.tensor([[i]]) for i in range(1, 5)]}, **chunk_config)
    tm.put_req_chunk = {"r": 1}
    req = SimpleNamespace(external_req_id="r")

    out = talker2code2wav_async_chunk(tm, _sentinel_payload(), req, is_finished=True)

    assert out is not None
    assert bool(out.meta.finished) is True
    assert out.codes is None, "boundary finish must not re-send the last full chunk"


def test_finish_sentinel_with_no_sent_chunks_emits_flag_only():
    tm = _tm({}, chunk_frames=4)
    req = SimpleNamespace(external_req_id="missing")

    out = talker2code2wav_async_chunk(tm, _sentinel_payload(), req, is_finished=True)

    assert out is not None
    assert bool(out.meta.finished) is True
    assert out.codes is None


def test_non_sentinel_empty_call_is_unchanged():
    # Without the marker, an empty/codeless call returns None as before -> the
    # adapter path (which never sets the marker) is byte-identical.
    tm = _tm({"r": [torch.tensor([[1]]), torch.tensor([[2]])]}, chunk_frames=4)
    req = SimpleNamespace(external_req_id="r")

    assert talker2code2wav_async_chunk(tm, {"codes": {}}, req, is_finished=True) is None
    assert talker2code2wav_async_chunk(tm, {}, req, is_finished=True) is None


def test_finish_sentinel_after_initial_partial_flushes_tail():
    tm = _tm({"r": [torch.tensor([[i]]) for i in range(1, 7)]}, chunk_frames=25, initial_frames=4)
    tm.put_req_chunk = {"r": 1}
    req = SimpleNamespace(external_req_id="r")
    out = talker2code2wav_async_chunk(tm, _sentinel_payload(), req, is_finished=True)

    assert out is not None
    assert bool(out.meta.finished) is True
    assert out.meta.left_context_size == 4
    assert isinstance(out.codes.audio, torch.Tensor)
    assert out.codes.audio.numel() == 6


def test_finished_real_payload_flushes_regular_chunk_after_initial_boundary():
    tm = _tm({"r": [torch.tensor([[i]]) for i in range(1, 29)]}, chunk_frames=25, initial_frames=4)
    tm.put_req_chunk = {"r": 1}
    req = SimpleNamespace(external_req_id="r")
    frame = torch.tensor([[29]])

    out = talker2code2wav_async_chunk(tm, {"codes": {"audio": frame}}, req, is_finished=True)

    assert out is not None
    assert bool(out.meta.finished) is True
    assert out.meta.left_context_size == 4
    assert isinstance(out.codes.audio, torch.Tensor)
    assert out.codes.audio.numel() == 29
