# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from starlette.requests import Request
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)

from vllm_omni.config.pipeline_registry import resolve_pipeline_config
from vllm_omni.entrypoints.openai.api_server import reset_session as reset_session_route
from vllm_omni.entrypoints.openai.session_adapter import SessionServingAdapter
from vllm_omni.experimental.fullduplex.joyvl.memory.memory import MidTermSummary
from vllm_omni.experimental.fullduplex.joyvl.serving.config import InteractionConfig
from vllm_omni.experimental.fullduplex.joyvl.serving.omni_adapter import (
    JoyVLSessionServingAdapter,
    OmniChatBackend,
)
from vllm_omni.experimental.fullduplex.joyvl.serving.server import (
    SessionManager,
    SessionOperationConflictError,
    StaleSessionEpochError,
    _operation_id,
    create_app,
)
from vllm_omni.experimental.fullduplex.joyvl.serving.session import InteractionSession

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Backend:
    def __init__(self, response: str = "</response> ok") -> None:
        self.response = response
        self.calls: list[list[dict[str, Any]]] = []
        self.closed = 0
        self.fail_next = False
        self.on_generate = None

    async def generate(self, messages, **_kwargs):
        self.calls.append(messages)
        if self.on_generate is not None:
            self.on_generate()
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("inference failed")
        return self.response, None

    async def aclose(self):
        self.closed += 1


class _FirstCallGateBackend(_Backend):
    def __init__(self) -> None:
        super().__init__()
        self.first_started = asyncio.Event()
        self.release_first = asyncio.Event()

    async def generate(self, messages, **kwargs):
        call_index = len(self.calls)
        self.calls.append(messages)
        if call_index == 0:
            self.first_started.set()
            await self.release_first.wait()
        return self.response, None


def _config() -> InteractionConfig:
    return InteractionConfig(enable_memory=False, enable_delegation=False)


@pytest.mark.asyncio
async def test_interaction_session_rolls_back_when_inference_fails():
    backend = _Backend()
    backend.fail_next = True
    session = InteractionSession("s", _config(), backend)
    brain = session._policy.brain

    def commit_concurrent_summary():
        brain.memory.mid_term_summaries.append(MidTermSummary(1, "0-1", "committed summary"))

    backend.on_generate = commit_concurrent_summary

    with pytest.raises(RuntimeError, match="inference failed"):
        await session.step(["data:image/jpeg;base64,AAA"], "first query")

    backend.on_generate = None
    assert [item.summary_text for item in brain.memory.mid_term_summaries] == ["committed summary"]
    assert brain.frame_index == 0
    assert brain.current_query is None
    assert brain.working_frames == []
    assert brain.response_records == []
    assert session.chunk.messages == []

    result = await session.step(["data:image/jpeg;base64,AAA"], "first query")
    assert result.frame_index == 1
    assert len(backend.calls) == 2


@pytest.mark.asyncio
async def test_manager_deduplicates_in_flight_and_completed_operation():
    backend = _FirstCallGateBackend()
    manager = SessionManager(_config(), backend=backend)

    first = asyncio.create_task(manager.step("s", ["frame-a"], "query", operation_id="op-1", epoch=0))
    await backend.first_started.wait()
    duplicate = asyncio.create_task(manager.step("s", ["frame-a"], "query", operation_id="op-1", epoch=0))
    await asyncio.sleep(0)
    backend.release_first.set()

    first_result, duplicate_result = await asyncio.gather(first, duplicate)
    cached_result = await manager.step(
        "s",
        ["frame-a"],
        "query",
        operation_id="op-1",
        epoch=0,
    )

    assert first_result == duplicate_result == cached_result
    assert len(backend.calls) == 1
    await manager.aclose()


@pytest.mark.asyncio
async def test_numeric_zero_input_seq_is_idempotent_and_conflict_checked():
    raw_request = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    operation_id = _operation_id(raw_request, {"input_seq": 0})
    assert operation_id == "0"

    backend = _FirstCallGateBackend()
    manager = SessionManager(_config(), backend=backend)

    first = asyncio.create_task(manager.step("s", ["frame-a"], "query", operation_id=operation_id, epoch=0))
    await backend.first_started.wait()
    duplicate = asyncio.create_task(manager.step("s", ["frame-a"], "query", operation_id=operation_id, epoch=0))
    await asyncio.sleep(0)
    backend.release_first.set()

    first_result, duplicate_result = await asyncio.gather(first, duplicate)
    cached_result = await manager.step("s", ["frame-a"], "query", operation_id=operation_id, epoch=0)

    assert first_result == duplicate_result == cached_result
    with pytest.raises(SessionOperationConflictError):
        await manager.step("s", ["frame-b"], "query", operation_id=operation_id, epoch=0)
    assert len(backend.calls) == 1
    await manager.aclose()


def test_explicit_empty_operation_id_is_rejected():
    raw_request = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    assert _operation_id(raw_request, {}) is None
    for payload in ({"operation_id": ""}, {"operation_id": "  "}, {"input_seq": ""}):
        with pytest.raises(ValueError, match="non-empty"):
            _operation_id(raw_request, payload)

    empty_header = Request({"type": "http", "method": "POST", "path": "/", "headers": [(b"x-operation-id", b"")]})
    with pytest.raises(ValueError, match="non-empty"):
        _operation_id(empty_header, {})


@pytest.mark.asyncio
async def test_manager_rejects_operation_id_reuse_with_different_input():
    backend = _Backend()
    manager = SessionManager(_config(), backend=backend)
    await manager.step("s", ["frame-a"], "query", operation_id="op-1")

    with pytest.raises(SessionOperationConflictError):
        await manager.step("s", ["frame-b"], "query", operation_id="op-1")

    assert len(backend.calls) == 1
    await manager.aclose()


@pytest.mark.asyncio
async def test_reset_filters_late_result_and_isolates_new_epoch():
    backend = _FirstCallGateBackend()
    manager = SessionManager(_config(), backend=backend)

    old_step = asyncio.create_task(manager.step("s", ["old-frame"], "old query", operation_id="old", epoch=0))
    await backend.first_started.wait()

    new_epoch = await manager.reset("s", expected_epoch=0)
    assert new_epoch == 1

    new_result = await manager.step(
        "s",
        ["new-frame"],
        "new query",
        operation_id="new",
        epoch=new_epoch,
    )
    backend.release_first.set()

    with pytest.raises(StaleSessionEpochError):
        await old_step

    assert new_result.frame_index == 1
    with pytest.raises(StaleSessionEpochError):
        await manager.step("s", ["late-frame"], "late", epoch=0)
    await manager.aclose()


@pytest.mark.asyncio
async def test_reset_wins_before_delegation_commit_has_no_external_side_effect():
    backend = _FirstCallGateBackend()
    backend.response = "</response> note </delegation> question"
    config = InteractionConfig(
        enable_memory=False,
        enable_delegation=True,
        delegation_kind="stub",
    )
    manager = SessionManager(config, backend=backend)
    bridge = manager._delegation
    assert bridge is not None

    old_step = asyncio.create_task(manager.step("s", ["old-frame"], "old query", operation_id="old", epoch=0))
    await backend.first_started.wait()

    assert await manager.reset("s", expected_epoch=0) == 1
    backend.release_first.set()
    with pytest.raises(StaleSessionEpochError):
        await old_step

    assert bridge._counter == 0
    await manager.aclose()


@pytest.mark.asyncio
async def test_reset_filters_committed_operation_before_creator_returns(monkeypatch):
    backend = _Backend()
    manager = SessionManager(_config(), backend=backend)
    real_shield = asyncio.shield
    operation_completed = asyncio.Event()
    release_output = asyncio.Event()

    async def hold_completed_operation(task):
        result = await real_shield(task)
        operation_completed.set()
        await release_output.wait()
        return result

    monkeypatch.setattr(asyncio, "shield", hold_completed_operation)
    old_step = asyncio.create_task(manager.step("s", ["old-frame"], "old query", operation_id="old", epoch=0))
    await operation_completed.wait()

    assert await manager.reset("s", expected_epoch=0) == 1
    release_output.set()
    with pytest.raises(StaleSessionEpochError):
        await old_step

    assert len(backend.calls) == 1
    await manager.aclose()


@pytest.mark.asyncio
async def test_reset_filters_creator_and_duplicate_after_shared_operation_commits(monkeypatch):
    backend = _FirstCallGateBackend()
    manager = SessionManager(_config(), backend=backend)
    real_shield = asyncio.shield
    completed_waiters = 0
    both_operations_completed = asyncio.Event()
    release_outputs = asyncio.Event()

    async def hold_completed_operation(task):
        nonlocal completed_waiters
        result = await real_shield(task)
        completed_waiters += 1
        if completed_waiters == 2:
            both_operations_completed.set()
        await release_outputs.wait()
        return result

    monkeypatch.setattr(asyncio, "shield", hold_completed_operation)
    creator = asyncio.create_task(manager.step("s", ["old-frame"], "old query", operation_id="old", epoch=0))
    await backend.first_started.wait()
    duplicate = asyncio.create_task(manager.step("s", ["old-frame"], "old query", operation_id="old", epoch=0))
    await asyncio.sleep(0)
    backend.release_first.set()
    await both_operations_completed.wait()

    assert await manager.reset("s", expected_epoch=0) == 1
    release_outputs.set()
    creator_result, duplicate_result = await asyncio.gather(
        creator,
        duplicate,
        return_exceptions=True,
    )

    assert isinstance(creator_result, StaleSessionEpochError)
    assert isinstance(duplicate_result, StaleSessionEpochError)
    assert len(backend.calls) == 1
    await manager.aclose()


@pytest.mark.asyncio
async def test_reset_rejects_queued_old_epoch_before_backend_inference():
    backend = _FirstCallGateBackend()
    manager = SessionManager(_config(), backend=backend)

    active = asyncio.create_task(manager.step("s", ["active-frame"], "active query", operation_id="active", epoch=0))
    await backend.first_started.wait()
    queued = asyncio.create_task(manager.step("s", ["queued-frame"], "queued query", operation_id="queued", epoch=0))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert len(backend.calls) == 1

    assert await manager.reset("s", expected_epoch=0) == 1
    backend.release_first.set()
    active_result, queued_result = await asyncio.gather(active, queued, return_exceptions=True)

    assert isinstance(active_result, StaleSessionEpochError)
    assert isinstance(queued_result, StaleSessionEpochError)
    assert len(backend.calls) == 1

    fresh = await manager.step("s", ["fresh-frame"], "fresh query", operation_id="fresh", epoch=1)
    assert fresh.frame_index == 1
    assert len(backend.calls) == 2
    await manager.aclose()


@pytest.mark.asyncio
async def test_reset_retry_does_not_invalidate_new_epoch_work():
    backend = _FirstCallGateBackend()
    manager = SessionManager(_config(), backend=backend)

    new_epoch = await manager.reset("s", expected_epoch=0)
    assert new_epoch == 1
    new_step = asyncio.create_task(manager.step("s", ["new-frame"], "new query", operation_id="new", epoch=new_epoch))
    await backend.first_started.wait()

    retried_epoch = await manager.reset("s", expected_epoch=0)
    assert retried_epoch == new_epoch
    assert manager.current_epoch("s") == new_epoch
    backend.release_first.set()
    result = await new_step
    assert result.frame_index == 1

    next_epoch = await manager.reset("s", expected_epoch=new_epoch)
    assert next_epoch == 2
    with pytest.raises(StaleSessionEpochError):
        await manager.reset("s", expected_epoch=0)
    with pytest.raises(StaleSessionEpochError):
        await manager.reset("s", expected_epoch=3)
    await manager.aclose()


@pytest.mark.asyncio
async def test_sessions_keep_independent_history():
    backend = _Backend()
    manager = SessionManager(_config(), backend=backend)

    first = await manager.step("s1", ["frame-a"], "alpha")
    second = await manager.step("s2", ["frame-b"], "beta")

    assert first.frame_index == second.frame_index == 1
    assert "alpha" not in json.dumps(backend.calls[1])
    assert "beta" in json.dumps(backend.calls[1])
    await manager.aclose()


class _FakeChatService:
    def __init__(self) -> None:
        self.requests: list[ChatCompletionRequest] = []

    async def create_chat_completion(self, request, raw_request=None):
        assert raw_request is None
        self.requests.append(request)
        return ChatCompletionResponse.model_validate(
            {
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "</response> local result"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
            }
        )


@pytest.mark.asyncio
async def test_omni_backend_uses_local_chat_service():
    chat_service = _FakeChatService()
    backend = OmniChatBackend(chat_service, "model")
    text, usage = await backend.generate(
        [{"role": "user", "content": "hello"}],
        max_tokens=32,
        temperature=0.1,
        top_p=0.9,
        extra_body={"top_k": 5},
    )

    assert text == "</response> local result"
    assert usage["total_tokens"] == 12
    assert chat_service.requests[0].top_k == 5


@pytest.mark.asyncio
async def test_session_adapter_runs_policy_above_ordinary_chat_service():
    chat_service = _FakeChatService()
    adapter = JoyVLSessionServingAdapter(chat_service=chat_service, model_name="model")
    raw_request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [
                (b"x-session-id", b"session-a"),
                (b"x-operation-id", b"operation-1"),
            ],
        }
    )
    request = ChatCompletionRequest.model_validate(
        {
            "model": "model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAA"}},
                        {"type": "text", "text": "what happened?"},
                    ],
                }
            ],
        }
    )

    missing_session_request = Request({"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": []})
    missing_response = await adapter.create_chat_completion(request, missing_session_request)
    assert missing_response.status_code == 400
    assert json.loads(missing_response.body)["type"] == "missing_session_id"
    assert len(chat_service.requests) == 0

    response = await adapter.create_chat_completion(request, raw_request)
    body = json.loads(response.body)

    assert body["interaction"]["session_id"] == "session-a"
    assert body["interaction"]["operation_id"] == "operation-1"
    assert body["interaction"]["epoch"] == 0
    assert body["interaction"]["text"] == "local result"
    assert len(chat_service.requests) == 1
    missing_epoch_reset = await adapter.reset_session(raw_request, {})
    assert missing_epoch_reset.status_code == 400
    assert json.loads(missing_epoch_reset.body)["type"] == "missing_session_epoch"

    first_reset = await adapter.reset_session(raw_request, {"session_epoch": 0})
    first_reset_body = json.loads(first_reset.body)
    assert first_reset_body["epoch"] == 1
    retry_reset = await adapter.reset_session(raw_request, {"session_epoch": 0})
    assert json.loads(retry_reset.body)["epoch"] == 1

    missing_epoch_response = await adapter.create_chat_completion(request, raw_request)
    assert missing_epoch_response.status_code == 409
    missing_epoch_body = json.loads(missing_epoch_response.body)
    assert missing_epoch_body["type"] == "missing_session_epoch"
    assert missing_epoch_body["current_epoch"] == 1
    assert len(chat_service.requests) == 1

    await adapter.aclose()


@pytest.mark.asyncio
async def test_public_reset_route_rejects_invalid_json_with_4xx():
    adapter = JoyVLSessionServingAdapter(chat_service=_FakeChatService(), model_name="model")
    app = FastAPI()
    app.post("/v1/session/reset")(reset_session_route)
    app.state.openai_session_adapter = adapter
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    headers = {"content-type": "application/json", "x-session-id": "session-a"}

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        malformed = await client.post(
            "/v1/session/reset",
            content="{",
            headers=headers,
        )
        assert malformed.status_code == 400
        assert malformed.json()["detail"] == "Invalid JSON request body"

        for payload in ([], "bad"):
            response = await client.post("/v1/session/reset", json=payload, headers=headers)
            assert response.status_code == 400
            assert response.json()["detail"] == "Session reset body must be a JSON object"

        wrong_epoch = await client.post("/v1/session/reset", json={"session_epoch": []}, headers=headers)
        assert wrong_epoch.status_code == 400
        assert wrong_epoch.json()["type"] == "invalid_request"

        valid = await client.post("/v1/session/reset", json={"session_epoch": 0}, headers=headers)
        assert valid.status_code == 200
        assert valid.json()["epoch"] == 1

    await adapter.aclose()


def test_joyvl_pipeline_selects_session_adapter_without_custom_runtime():
    pipeline = resolve_pipeline_config("joyvl_interaction")

    assert pipeline is not None
    assert pipeline.session_serving_adapter.endswith("JoyVLSessionServingAdapter")
    assert len(pipeline.stages) == 1
    assert pipeline.stages[0].model_stage == "vlm"
    assert pipeline.stages[0].custom_process_input_func is None
    assert not hasattr(SessionServingAdapter, "set_persona")


@pytest.mark.asyncio
async def test_external_sidecar_routes_validate_json_and_report_reset_replays():
    app = create_app(_config())
    async with app.router.lifespan_context(app):
        paths = {route.path for route in app.routes}
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        headers = {"content-type": "application/json", "x-session-id": "session-a"}
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            for path in (
                "/v1/chat/completions",
                "/reset",
                "/v1/streaming/persona",
            ):
                malformed = await client.post(path, content="{", headers=headers)
                assert malformed.status_code == 400
                assert malformed.json()["type"] == "invalid_request"

                non_object = await client.post(path, json=[], headers=headers)
                assert non_object.status_code == 400
                assert non_object.json()["type"] == "invalid_request"

            first = await client.post("/reset", json={"session_id": "restart", "session_epoch": 0})
            assert first.status_code == 200
            assert first.json()["epoch"] == 1
            assert first.json()["advanced"] is True

            replay = await client.post("/reset", json={"session_id": "restart", "session_epoch": 0})
            assert replay.json()["epoch"] == 1
            assert replay.json()["advanced"] is False

            next_reset = await client.post("/reset", json={"session_id": "restart", "session_epoch": 1})
            assert next_reset.json()["epoch"] == 2
            assert next_reset.json()["advanced"] is True

    assert "/health" in paths
    assert "/v1/chat/completions" in paths
    assert "/v1/streaming/persona" in paths
