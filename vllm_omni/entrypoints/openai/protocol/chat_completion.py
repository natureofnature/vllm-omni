from typing import Any, ClassVar

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
)


class OmniChatCompletionRequest(ChatCompletionRequest):
    field_names: ClassVar[set[str] | None] = None
    sampling_params_list: list[dict[str, Any]] | None = None
    modalities: list[str] | None = None


class OmniChatCompletionStreamResponse(ChatCompletionStreamResponse):
    modality: str | None = "text"
    metrics: dict[str, Any] | None = None


class OmniChatCompletionResponse(ChatCompletionResponse):
    metrics: dict[str, Any] | None = None
