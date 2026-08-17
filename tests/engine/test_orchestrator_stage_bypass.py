# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""AURA async_chunk orchestrator hooks (prewarm)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from vllm_omni.engine.messages import ErrorMessage
from vllm_omni.engine.orchestrator import (
    Orchestrator,
    OrchestratorRequestState,
    _stage0_hard_bypass_enabled,
)

from .test_orchestrator import (
    FakeOutputProcessor,
    FakeStageClient,
    _build_harness,
    _build_stage_pools,
    _sampling_params,
    _shutdown_orchestrator,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _aura_stage_clients() -> list[FakeStageClient]:
    return [
        FakeStageClient(stage_type="llm", final_output=False),
        FakeStageClient(stage_type="llm", final_output=False, model_stage="aura"),
        FakeStageClient(
            stage_type="llm",
            final_output=False,
            model_stage="qwen3_tts",
            final_output_type="latent",
        ),
        FakeStageClient(
            stage_type="llm",
            final_output=True,
            final_output_type="audio",
            model_stage="code2wav",
        ),
    ]


def _aura_stage_vllm_configs(clients: list[FakeStageClient]) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=64, model_stage=client.model_stage, worker_type="ar")
        )
        if client.model_stage != "code2wav"
        else SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=64, model_stage="code2wav", worker_type="generation")
        )
        for client in clients
    ]


def _inline_aura_orchestrator() -> tuple[Orchestrator, asyncio.Queue, list[FakeStageClient]]:
    """Orchestrator on the current loop so output_async_queue is awaitable in-test."""
    clients = _aura_stage_clients()
    output_queue: asyncio.Queue = asyncio.Queue()
    orchestrator = Orchestrator(
        request_async_queue=asyncio.Queue(),
        output_async_queue=output_queue,
        rpc_async_queue=asyncio.Queue(),
        stage_pools=_build_stage_pools(
            [[client] for client in clients],
            output_processors=[FakeOutputProcessor() for _ in clients],
            stage_vllm_configs=_aura_stage_vllm_configs(clients),
        ),
        async_chunk=True,
    )
    return orchestrator, output_queue, clients


def _bypass_request_state(request_id: str) -> OrchestratorRequestState:
    return OrchestratorRequestState(
        request_id=request_id,
        prompt={"prompt": "video-only"},
        sampling_params_list=[_sampling_params() for _ in range(4)],
        final_stage_id=3,
        final_output_stage_ids={3},
    )


@pytest.mark.asyncio
async def test_stage0_bypass_keeps_final_stage_for_vision_spoken_tts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage0 bypass must not clamp final_stage_id→1 (vision-only may still speak)."""
    stage0 = FakeStageClient(stage_type="llm", final_output=False)
    stage1 = FakeStageClient(stage_type="llm", final_output=False, model_stage="aura")
    stage2 = FakeStageClient(
        stage_type="llm",
        final_output=False,
        model_stage="qwen3_tts",
        final_output_type="latent",
    )
    stage3 = FakeStageClient(
        stage_type="llm",
        final_output=True,
        final_output_type="audio",
        model_stage="code2wav",
    )
    processors = [FakeOutputProcessor() for _ in range(4)]
    stage_vllm_configs = [
        SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=64, model_stage=client.model_stage, worker_type="ar")
        )
        if client.model_stage != "code2wav"
        else SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=64, model_stage="code2wav", worker_type="generation")
        )
        for client in (stage0, stage1, stage2, stage3)
    ]
    fixture = _build_harness(
        [stage0, stage1, stage2, stage3],
        output_processors=processors,
        stage_vllm_configs=stage_vllm_configs,
        async_chunk=True,
    )
    request = SimpleNamespace(
        request_id="req-bypass-keep-tts",
        prompt_token_ids=[1, 2, 3],
        additional_information={"omni_skip_stages": [0]},
    )
    req_state = OrchestratorRequestState(
        request_id="req-bypass-keep-tts",
        prompt={"prompt": "video-only"},
        sampling_params_list=[_sampling_params() for _ in range(4)],
        final_stage_id=3,
        final_output_stage_ids={3},
    )
    fixture.orchestrator.request_states["req-bypass-keep-tts"] = req_state
    monkeypatch.setenv("VLLM_AURA_STAGE0_BYPASS", "1")
    # Even if a stale shell exports the removed flag, bypass must not clamp.
    monkeypatch.setenv("VLLM_AURA_SILENT_STOP_AT_STAGE1", "1")

    async def _fake_inject(request_id: str, additional_info: dict) -> bool:
        del request_id, additional_info
        return True

    monkeypatch.setattr(fixture.orchestrator, "_inject_bypassed_stage0_chunk", _fake_inject)

    try:
        await fixture.orchestrator._bypass_stage0("req-bypass-keep-tts", request, req_state)
        assert req_state.final_stage_id == 3
        assert req_state.final_output_stage_ids == {3}
        assert req_state.stage0_bypassed is True
        assert len(stage2.add_request_calls) == 1
        assert len(stage3.add_request_calls) == 1
    finally:
        await _shutdown_orchestrator(fixture)


@pytest.mark.asyncio
async def test_async_chunk_prewarm_uses_empty_prompt_for_qwen3_tts() -> None:
    stage0 = FakeStageClient(stage_type="llm", final_output=False)
    stage1 = FakeStageClient(stage_type="llm", final_output=False, model_stage="aura")
    stage2 = FakeStageClient(
        stage_type="llm",
        final_output=False,
        model_stage="qwen3_tts",
        final_output_type="latent",
    )
    stage3 = FakeStageClient(
        stage_type="llm",
        final_output=True,
        final_output_type="audio",
        model_stage="code2wav",
    )
    processors = [FakeOutputProcessor() for _ in range(4)]
    stage_vllm_configs = [
        SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=64, model_stage=client.model_stage, worker_type="ar")
        )
        if client.model_stage != "code2wav"
        else SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=64, model_stage="code2wav", worker_type="generation")
        )
        for client in (stage0, stage1, stage2, stage3)
    ]
    fixture = _build_harness(
        [stage0, stage1, stage2, stage3],
        output_processors=processors,
        stage_vllm_configs=stage_vllm_configs,
        async_chunk=True,
    )
    request = SimpleNamespace(request_id="req-prewarm-tts", prompt_token_ids=[1, 2, 3, 4, 5])
    req_state = OrchestratorRequestState(
        request_id="req-prewarm-tts",
        prompt={"prompt": "video-only"},
        sampling_params_list=[_sampling_params() for _ in range(4)],
        final_stage_id=3,
    )
    fixture.orchestrator.request_states["req-prewarm-tts"] = req_state

    try:
        await fixture.orchestrator._prewarm_async_chunk_stages("req-prewarm-tts", request, req_state)

        assert len(stage2.add_request_calls) == 1
        talker_request = stage2.add_request_calls[0][0]
        assert talker_request.prompt_token_ids == []
        assert len(stage3.add_request_calls) == 1
        codec_request = stage3.add_request_calls[0][0]
        assert codec_request.prompt_token_ids == []
    finally:
        await _shutdown_orchestrator(fixture)


@pytest.mark.asyncio
async def test_stage0_bypass_inject_failure_fails_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SHM inject failure must abort the request instead of stranding prewarmed stages."""
    orchestrator, output_queue, clients = _inline_aura_orchestrator()
    request_id = "req-bypass-inject-fail"
    request = SimpleNamespace(
        request_id=request_id,
        prompt_token_ids=[1, 2, 3],
        additional_information={"omni_skip_stages": [0]},
    )
    req_state = _bypass_request_state(request_id)
    orchestrator.request_states[request_id] = req_state
    monkeypatch.setenv("VLLM_AURA_STAGE0_BYPASS", "1")

    async def _failing_inject(rid: str, additional_info: dict) -> bool:
        del rid, additional_info
        return False

    monkeypatch.setattr(orchestrator, "_inject_bypassed_stage0_chunk", _failing_inject)

    await orchestrator._bypass_stage0(request_id, request, req_state)

    # Downstream stages were prewarmed before the inject attempt...
    assert len(clients[2].add_request_calls) == 1
    assert len(clients[3].add_request_calls) == 1
    # ...so the failure must surface an error and abort/cleanup the request.
    error = output_queue.get_nowait()
    assert isinstance(error, ErrorMessage)
    assert error.request_id == request_id
    assert "Stage0" in error.error
    assert request_id not in orchestrator.request_states
    assert any(request_id in aborted for aborted in clients[2].abort_calls)
    assert any(request_id in aborted for aborted in clients[3].abort_calls)


@pytest.mark.asyncio
async def test_stage0_bypass_disabled_env_skips_inject_and_prewarm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VLLM_AURA_STAGE0_BYPASS=0 must leave the request untouched (Stage0 GPU runs)."""
    orchestrator, output_queue, clients = _inline_aura_orchestrator()
    request_id = "req-bypass-disabled"
    request = SimpleNamespace(
        request_id=request_id,
        prompt_token_ids=[1, 2, 3],
        additional_information={"omni_skip_stages": [0]},
    )
    req_state = _bypass_request_state(request_id)
    orchestrator.request_states[request_id] = req_state
    monkeypatch.setenv("VLLM_AURA_STAGE0_BYPASS", "0")

    inject_calls: list[str] = []

    async def _tracking_inject(rid: str, additional_info: dict) -> bool:
        del additional_info
        inject_calls.append(rid)
        return True

    monkeypatch.setattr(orchestrator, "_inject_bypassed_stage0_chunk", _tracking_inject)

    await orchestrator._bypass_stage0(request_id, request, req_state)

    assert inject_calls == []
    assert req_state.stage0_bypassed is False
    assert len(clients[2].add_request_calls) == 0
    assert len(clients[3].add_request_calls) == 0
    assert output_queue.empty()
    assert request_id in orchestrator.request_states


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, True),
        ("1", True),
        ("true", True),
        ("0", False),
        ("false", False),
        ("off", False),
        ("no", False),
        ("NO", False),
    ],
)
def test_stage0_hard_bypass_env_parsing(
    monkeypatch: pytest.MonkeyPatch,
    raw: str | None,
    expected: bool,
) -> None:
    if raw is None:
        monkeypatch.delenv("VLLM_AURA_STAGE0_BYPASS", raising=False)
    else:
        monkeypatch.setenv("VLLM_AURA_STAGE0_BYPASS", raw)
    assert _stage0_hard_bypass_enabled() is expected
