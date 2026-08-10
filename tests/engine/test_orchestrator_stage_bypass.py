# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""AURA async_chunk orchestrator hooks (prewarm)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.engine.orchestrator import OrchestratorRequestState

from .test_orchestrator import (
    FakeOutputProcessor,
    FakeStageClient,
    _build_harness,
    _sampling_params,
    _shutdown_orchestrator,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


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

    async def _fake_inject(request_id: str, additional_info: dict) -> None:
        del request_id, additional_info

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
