# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse, Response
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.engine.protocol import ErrorResponse

from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat
from vllm_omni.experimental.fullduplex.joyvl.bridges.backend import ModelBackend
from vllm_omni.experimental.fullduplex.joyvl.serving.config import InteractionConfig
from vllm_omni.experimental.fullduplex.joyvl.serving.server import (
    SessionManager,
    SessionOperationConflictError,
    StaleSessionEpochError,
    _completion_response,
    _explicit_session_id,
    _extract_frames_and_query,
    _operation_id,
    _session_epoch,
)


class OmniChatBackend(ModelBackend):
    """Run JoyAI policy requests through the local AsyncOmni chat service."""

    def __init__(self, chat_service: OmniOpenAIServingChat, model_name: str) -> None:
        self._chat_service = chat_service
        self._model_name = model_name

    async def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        extra_body: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any] | None]:
        body: dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }
        body.update(extra_body or {})
        request = ChatCompletionRequest(**body)
        response = await self._chat_service.create_chat_completion(request, raw_request=None)
        if isinstance(response, ErrorResponse):
            message = response.error.message if response.error is not None else "AsyncOmni request failed"
            raise RuntimeError(message)
        if not isinstance(response, ChatCompletionResponse):
            raise RuntimeError("JoyAI Session Adapter requires a non-streaming ChatCompletionResponse")
        if not response.choices:
            return "", None
        text = response.choices[0].message.content or ""
        if not isinstance(text, str):
            text = str(text)
        usage = response.usage.model_dump() if response.usage is not None else None
        return text, usage

    async def aclose(self) -> None:
        # The API server owns the AsyncOmni engine and closes it separately.
        return None


class JoyVLSessionServingAdapter:
    """JoyAI Session state in the API Server, with ordinary AsyncOmni execution."""

    def __init__(self, chat_service: OmniOpenAIServingChat, model_name: str) -> None:
        self._model_name = model_name
        backend = OmniChatBackend(chat_service, model_name)
        config = InteractionConfig(main_model=model_name)
        self._manager = SessionManager(
            config,
            backend=backend,
            summarizer_backend=backend,
        )

    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Request,
    ) -> Response:
        if request.stream:
            return self._error("JoyAI Session Adapter does not support stream=true yet", "unsupported_stream", 400)

        payload = request.model_dump(mode="json", exclude_none=True)
        frames, query = _extract_frames_and_query(payload)
        if not frames:
            return self._error(
                "JoyAI interaction requires at least one image_url frame",
                "missing_frame",
                400,
            )

        try:
            session_id = _explicit_session_id(raw_request, payload)
            if session_id is None:
                return self._error("session_id is required", "missing_session_id", 400)
            operation_id = _operation_id(raw_request, payload)
            epoch = _session_epoch(raw_request, payload)
            current_epoch = self._manager.current_epoch(session_id)
            if epoch is None and current_epoch != 0:
                return self._error(
                    "session_epoch is required after a session reset",
                    "missing_session_epoch",
                    409,
                    current_epoch=current_epoch,
                )
            epoch = current_epoch if epoch is None else epoch
            result = await self._manager.step(
                session_id,
                frames,
                query,
                operation_id=operation_id,
                epoch=epoch,
            )
        except SessionOperationConflictError as exc:
            return self._error(str(exc), "operation_conflict", 409)
        except StaleSessionEpochError as exc:
            return self._error(
                str(exc),
                "stale_session_epoch",
                409,
                current_epoch=self._manager.current_epoch(session_id),
            )
        except ValueError as exc:
            return self._error(str(exc), "invalid_request", 400)

        return JSONResponse(
            _completion_response(
                self._model_name,
                result,
                session_id=session_id,
                epoch=self._manager.current_epoch(session_id),
                operation_id=operation_id,
            )
        )

    async def reset_session(self, raw_request: Request, payload: dict[str, Any]) -> Response:
        try:
            session_id = _explicit_session_id(raw_request, payload)
            if session_id is None:
                return self._error("session_id is required", "missing_session_id", 400)
            expected_epoch = _session_epoch(raw_request, payload)
            if expected_epoch is None:
                return self._error("session_epoch is required to reset a session", "missing_session_epoch", 400)
            epoch = await self._manager.reset(session_id, expected_epoch=expected_epoch)
        except ValueError as exc:
            return self._error(str(exc), "invalid_request", 400)
        except StaleSessionEpochError as exc:
            return self._error(
                str(exc), "stale_session_epoch", 409, current_epoch=self._manager.current_epoch(session_id)
            )
        return JSONResponse({"status": "reset", "session_id": session_id, "epoch": epoch})

    async def aclose(self) -> None:
        await self._manager.aclose()

    @staticmethod
    def _error(
        message: str,
        error_type: str,
        status_code: int,
        *,
        current_epoch: int | None = None,
    ) -> JSONResponse:
        content: dict[str, Any] = {"error": message, "type": error_type}
        if current_epoch is not None:
            content["current_epoch"] = current_epoch
        return JSONResponse(content, status_code=status_code)


__all__ = ["JoyVLSessionServingAdapter", "OmniChatBackend"]
