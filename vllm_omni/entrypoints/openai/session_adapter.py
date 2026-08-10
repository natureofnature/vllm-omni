# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from fastapi import Request
from fastapi.responses import Response
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat


@runtime_checkable
class SessionServingAdapter(Protocol):
    """Optional model-owned Session behavior above the ordinary request path."""

    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Request,
    ) -> Response: ...

    async def reset_session(self, raw_request: Request, payload: dict[str, Any]) -> Response: ...

    async def aclose(self) -> None: ...


def load_session_serving_adapter(
    path: str,
    chat_service: OmniOpenAIServingChat,
    model_name: str,
) -> SessionServingAdapter:
    module_name, separator, attribute_name = path.rpartition(".")
    if not separator:
        raise ValueError(f"Invalid Session serving adapter path: {path!r}")
    adapter_type = getattr(import_module(module_name), attribute_name)
    adapter = adapter_type(chat_service=chat_service, model_name=model_name)
    if not isinstance(adapter, SessionServingAdapter):
        raise TypeError(f"Session serving adapter {path!r} does not implement SessionServingAdapter")
    return adapter


__all__ = ["SessionServingAdapter", "load_session_serving_adapter"]
