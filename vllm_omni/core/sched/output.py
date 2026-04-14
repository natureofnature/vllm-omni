from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.request import Request

from vllm_omni.engine import PromptEmbedsPayload


def _normalize_initial_model_buffer(payload: Any | None) -> dict[str, Any] | None:
    """Convert a request seed payload into plain runner buffer data."""
    if payload is None:
        return None
    if isinstance(payload, dict):
        info_dict = payload
    else:
        info_dict = {}
        entries = getattr(payload, "entries", None)
        if isinstance(entries, Mapping):
            for key, entry in entries.items():
                tensor_data = getattr(entry, "tensor_data", None)
                if tensor_data is not None:
                    dtype = np.dtype(getattr(entry, "tensor_dtype", "float32"))
                    array = np.frombuffer(tensor_data, dtype=dtype)
                    array = array.reshape(getattr(entry, "tensor_shape", ()))
                    info_dict[key] = torch.from_numpy(array.copy())
                else:
                    info_dict[key] = getattr(entry, "list_data", None)
    return info_dict or None


def _get_request_initial_model_buffer(request: Request) -> dict[str, Any] | None:
    """Return the runner-facing seed payload for a scheduled request.

    Runtime stage handoff should prefer the coordinator-owned internal seed
    field over the public ``additional_information`` request payload.
    """
    runtime_seed = getattr(request, "_omni_initial_model_buffer", None)
    if runtime_seed is not None:
        return _normalize_initial_model_buffer(runtime_seed)
    return _normalize_initial_model_buffer(getattr(request, "additional_information", None))


@dataclass
class OmniNewRequestData(NewRequestData):
    """New request data for omni models with embeddings support.

    Extends NewRequestData to include prompt embeddings and additional
    information for direct transfer between pipeline stages.

    Args:
        prompt_embeds: Optional serialized prompt embeddings payload
            (overrides parent's torch.Tensor type with PromptEmbedsPayload
            for cross-process serialization)
        external_req_id: Optional external request ID for tracking
        initial_model_buffer: Optional plain per-request runtime payload
            dictionary used to seed ``model_intermediate_buffer``
    """

    # Optional serialized prompt embeddings (override parent type for serialization)
    prompt_embeds: PromptEmbedsPayload | None = None  # type: ignore[assignment]
    # Optional external request ID for tracking
    external_req_id: str | None = None
    # Optional decoded bootstrap payload for model_intermediate_buffer.
    initial_model_buffer: dict[str, Any] | None = None

    @classmethod
    def from_request(
        cls,
        request: Request,
        block_ids: tuple[list[int], ...],
        prefill_token_ids: list[int] | None = None,
    ) -> "OmniNewRequestData":
        """Create OmniNewRequestData from a Request object.

        Args:
            request: Request object to convert
            block_ids: Tuple of block ID lists for KV cache allocation
            prefill_token_ids: Optional prefill token IDs for v2 model runner

        Returns:
            OmniNewRequestData instance with data from the request
        """
        return cls(
            req_id=request.request_id,
            external_req_id=getattr(request, "external_req_id", None),
            prompt_token_ids=request.prompt_token_ids,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            block_ids=block_ids,
            num_computed_tokens=request.num_computed_tokens,
            lora_request=request.lora_request,
            prompt_embeds=getattr(request, "prompt_embeds", None),
            prefill_token_ids=prefill_token_ids,
            initial_model_buffer=_get_request_initial_model_buffer(request),
        )

    @classmethod
    def from_scheduled_request_data(
        cls,
        new_req_data: NewRequestData,
        request: Request | None,
    ) -> "OmniNewRequestData":
        """Attach omni-specific request payloads to scheduler-produced request data."""
        return cls(
            req_id=new_req_data.req_id,
            external_req_id=(
                getattr(request, "external_req_id", None)
                if request is not None
                else getattr(new_req_data, "external_req_id", None)
            ),
            prompt_token_ids=new_req_data.prompt_token_ids,
            mm_features=new_req_data.mm_features,
            sampling_params=new_req_data.sampling_params,
            pooling_params=new_req_data.pooling_params,
            block_ids=new_req_data.block_ids,
            num_computed_tokens=new_req_data.num_computed_tokens,
            lora_request=new_req_data.lora_request,
            prompt_embeds=(
                getattr(request, "prompt_embeds", None)
                if request is not None
                else getattr(new_req_data, "prompt_embeds", None)
            ),
            prefill_token_ids=getattr(new_req_data, "prefill_token_ids", None),
            initial_model_buffer=(
                _get_request_initial_model_buffer(request)
                if request is not None
                else getattr(new_req_data, "initial_model_buffer", None)
            ),
        )


@dataclass
class OmniCachedRequestData(CachedRequestData):
    """Cached request data for omni models with embeddings support.

    Args:
        prompt_token_ids: Mapping from request ID to list of prompt token IDs
    """

    prompt_token_ids: dict[str, list[int]]


@dataclass
class OmniSchedulerOutput(SchedulerOutput):
    """Scheduler output with omni-specific transfer metadata."""

    finished_requests_needing_kv_transfer: dict[str, dict] = field(default_factory=dict)
    # Requests that need to be registered for chunk recv by the Model Runner's
    # background thread. Populated by ChunkSchedulingCoordinator.
    pending_chunk_registrations: list = field(default_factory=list)
    # Requests that need to be registered for batch input recv by the
    # Model Runner's background thread. Populated by OmniSchedulingCoordinator.
    pending_input_registrations: list = field(default_factory=list)
