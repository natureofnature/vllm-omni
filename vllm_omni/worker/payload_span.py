# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers for explicit thinker decode span metadata."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

THINKER_DECODE_EMBEDDINGS_KEY = "thinker_decode_embeddings"
THINKER_OUTPUT_TOKEN_IDS_KEY = "thinker_output_token_ids"
THINKER_DECODE_TOKEN_START_KEY = "thinker_decode_embeddings_token_start"
THINKER_DECODE_TOKEN_END_KEY = "thinker_decode_embeddings_token_end"

CACHED_THINKER_DECODE_EMBEDDINGS_KEY = "cached_thinker_decode_embeddings"
CACHED_THINKER_DECODE_TOKEN_START_KEY = "cached_thinker_decode_embeddings_token_start"
CACHED_THINKER_DECODE_TOKEN_END_KEY = "cached_thinker_decode_embeddings_token_end"

TensorSpan = tuple[torch.Tensor, int, int]


@dataclass(frozen=True)
class ThinkerDecodeStepState:
    thinker_embed: torch.Tensor | None
    start_index: int
    available_end: int
    legacy_decode_end: int


def get_tensor_span(payload: Mapping[str, Any], *, tensor_key: str, start_key: str, end_key: str) -> TensorSpan | None:
    tensor = payload.get(tensor_key)
    start = payload.get(start_key)
    end = payload.get(end_key)
    if not isinstance(tensor, torch.Tensor):
        return None
    if not isinstance(start, int) or not isinstance(end, int):
        return None
    if start < 0 or end < start or (end - start) != int(tensor.shape[0]):
        return None
    return tensor, start, end


def merge_tensor_spans(existing_span: TensorSpan | None, incoming_span: TensorSpan | None) -> TensorSpan | None:
    if existing_span is None or incoming_span is None:
        return None

    existing_tensor, existing_start, existing_end = existing_span
    incoming_tensor, incoming_start, incoming_end = incoming_span
    if incoming_tensor.device != existing_tensor.device or incoming_tensor.dtype != existing_tensor.dtype:
        incoming_tensor = incoming_tensor.to(device=existing_tensor.device, dtype=existing_tensor.dtype)
    if incoming_start == existing_end:
        return torch.cat([existing_tensor, incoming_tensor], dim=0), existing_start, incoming_end
    if incoming_start < existing_end:
        overlap = existing_end - incoming_start
        if overlap >= int(incoming_tensor.shape[0]):
            return existing_tensor, existing_start, existing_end
        trimmed_tensor = incoming_tensor[overlap:]
        return (
            torch.cat([existing_tensor, trimmed_tensor], dim=0),
            existing_start,
            existing_end + int(trimmed_tensor.shape[0]),
        )
    return None


def get_tensor_span_row(span: TensorSpan | None, index: int) -> torch.Tensor | None:
    if span is None:
        return None
    tensor, start, end = span
    if index < start or index >= end:
        return None
    return tensor[index - start]


def _cast_tensor_span(
    span: TensorSpan | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> TensorSpan | None:
    if span is None:
        return None
    tensor, start, end = span
    return tensor.to(device=device, dtype=dtype), start, end


def _store_tensor_span(
    payload: dict[str, Any],
    *,
    span: TensorSpan,
    tensor_key: str,
    start_key: str,
    end_key: str,
) -> None:
    payload[tensor_key], payload[start_key], payload[end_key] = span


def _merge_or_replace_tensor_span(
    cached_span: TensorSpan,
    incoming_span: TensorSpan,
    *,
    start_index: int,
    logger: Any = None,
    stale_warning: str,
) -> tuple[TensorSpan, bool]:
    merged_span = merge_tensor_spans(cached_span, incoming_span)
    if merged_span is not None:
        return merged_span, True
    if incoming_span[1] >= cached_span[2] and start_index >= cached_span[2]:
        if logger is not None:
            logger.warning(
                stale_warning,
                cached_span[1],
                cached_span[2],
                incoming_span[1],
                incoming_span[2],
            )
        return incoming_span, True
    return cached_span, False


def cache_thinker_decode_span(
    payload: Mapping[str, Any],
    update_dict: dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
    logger: Any = None,
) -> None:
    thinker_decode_embeds = payload.get(THINKER_DECODE_EMBEDDINGS_KEY)
    if thinker_decode_embeds is None:
        return

    start_index = int(payload.get("num_processed_tokens", 0))
    clear_incoming_decode_embed = True
    incoming_span = _cast_tensor_span(
        get_tensor_span(
            payload,
            tensor_key=THINKER_DECODE_EMBEDDINGS_KEY,
            start_key=THINKER_DECODE_TOKEN_START_KEY,
            end_key=THINKER_DECODE_TOKEN_END_KEY,
        ),
        device=device,
        dtype=dtype,
    )
    cached_span = _cast_tensor_span(
        get_tensor_span(
            payload,
            tensor_key=CACHED_THINKER_DECODE_EMBEDDINGS_KEY,
            start_key=CACHED_THINKER_DECODE_TOKEN_START_KEY,
            end_key=CACHED_THINKER_DECODE_TOKEN_END_KEY,
        ),
        device=device,
        dtype=dtype,
    )

    if cached_span is None:
        if incoming_span is not None:
            _store_tensor_span(
                update_dict,
                span=incoming_span,
                tensor_key=CACHED_THINKER_DECODE_EMBEDDINGS_KEY,
                start_key=CACHED_THINKER_DECODE_TOKEN_START_KEY,
                end_key=CACHED_THINKER_DECODE_TOKEN_END_KEY,
            )
        else:
            update_dict[CACHED_THINKER_DECODE_EMBEDDINGS_KEY] = thinker_decode_embeds.to(device=device, dtype=dtype)
    else:
        if incoming_span is not None:
            next_span, clear_incoming_decode_embed = _merge_or_replace_tensor_span(
                cached_span,
                incoming_span,
                start_index=start_index,
                logger=logger,
                stale_warning=(
                    "Talker decode cache replaced stale non-contiguous thinker span: existing=(%s,%s) incoming=(%s,%s)"
                ),
            )
            if clear_incoming_decode_embed:
                _store_tensor_span(
                    update_dict,
                    span=next_span,
                    tensor_key=CACHED_THINKER_DECODE_EMBEDDINGS_KEY,
                    start_key=CACHED_THINKER_DECODE_TOKEN_START_KEY,
                    end_key=CACHED_THINKER_DECODE_TOKEN_END_KEY,
                )
        else:
            thinker_decode_embeds = thinker_decode_embeds.to(device=device, dtype=dtype)
            merged_tensor = torch.cat([cached_span[0], thinker_decode_embeds], dim=0)
            _store_tensor_span(
                update_dict,
                span=(merged_tensor, cached_span[1], cached_span[1] + int(merged_tensor.shape[0])),
                tensor_key=CACHED_THINKER_DECODE_EMBEDDINGS_KEY,
                start_key=CACHED_THINKER_DECODE_TOKEN_START_KEY,
                end_key=CACHED_THINKER_DECODE_TOKEN_END_KEY,
            )

    if clear_incoming_decode_embed:
        update_dict[THINKER_DECODE_EMBEDDINGS_KEY] = None


def resolve_thinker_decode_step(
    payload: Mapping[str, Any],
    update_dict: dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
    logger: Any = None,
) -> ThinkerDecodeStepState:
    start_index = int(payload.get("num_processed_tokens", 0))
    thinker_decode_embed = payload.get(THINKER_DECODE_EMBEDDINGS_KEY)
    clear_incoming_decode_embed = thinker_decode_embed is not None
    cached_span = _cast_tensor_span(
        get_tensor_span(
            payload,
            tensor_key=CACHED_THINKER_DECODE_EMBEDDINGS_KEY,
            start_key=CACHED_THINKER_DECODE_TOKEN_START_KEY,
            end_key=CACHED_THINKER_DECODE_TOKEN_END_KEY,
        ),
        device=device,
        dtype=dtype,
    )
    incoming_span = _cast_tensor_span(
        get_tensor_span(
            payload,
            tensor_key=THINKER_DECODE_EMBEDDINGS_KEY,
            start_key=THINKER_DECODE_TOKEN_START_KEY,
            end_key=THINKER_DECODE_TOKEN_END_KEY,
        ),
        device=device,
        dtype=dtype,
    )

    if incoming_span is not None:
        if cached_span is None:
            cached_span = incoming_span
        else:
            cached_span, clear_incoming_decode_embed = _merge_or_replace_tensor_span(
                cached_span,
                incoming_span,
                start_index=start_index,
                logger=logger,
                stale_warning=(
                    "Talker decode replaced stale cached span with newer incoming span: "
                    "existing=(%s,%s) incoming=(%s,%s)"
                ),
            )
        if clear_incoming_decode_embed:
            _store_tensor_span(
                update_dict,
                span=cached_span,
                tensor_key=CACHED_THINKER_DECODE_EMBEDDINGS_KEY,
                start_key=CACHED_THINKER_DECODE_TOKEN_START_KEY,
                end_key=CACHED_THINKER_DECODE_TOKEN_END_KEY,
            )
            thinker_decode_embed = None
            incoming_span = None
    elif (
        isinstance(thinker_decode_embed, torch.Tensor)
        and thinker_decode_embed.ndim >= 2
        and thinker_decode_embed.shape[0] > 1
    ):
        thinker_decode_embed = thinker_decode_embed.to(device=device, dtype=dtype)
        if cached_span is None:
            cached_span = (thinker_decode_embed, 0, int(thinker_decode_embed.shape[0]))
        else:
            merged_tensor = torch.cat([cached_span[0], thinker_decode_embed], dim=0)
            cached_span = (merged_tensor, cached_span[1], cached_span[2] + int(thinker_decode_embed.shape[0]))
        _store_tensor_span(
            update_dict,
            span=cached_span,
            tensor_key=CACHED_THINKER_DECODE_EMBEDDINGS_KEY,
            start_key=CACHED_THINKER_DECODE_TOKEN_START_KEY,
            end_key=CACHED_THINKER_DECODE_TOKEN_END_KEY,
        )
        thinker_decode_embed = None

    thinker_output_token_ids = payload.get(THINKER_OUTPUT_TOKEN_IDS_KEY, [])
    thinker_embed = get_tensor_span_row(cached_span, start_index)
    if thinker_embed is None:
        thinker_embed = get_tensor_span_row(incoming_span, start_index)
    if thinker_embed is None and isinstance(thinker_decode_embed, torch.Tensor):
        thinker_decode_embed = thinker_decode_embed.to(device=device, dtype=dtype)
        if thinker_decode_embed.ndim == 1:
            thinker_embed = thinker_decode_embed
        elif thinker_decode_embed.shape[0] == 1:
            thinker_embed = thinker_decode_embed[0]

    available_end = -1
    if cached_span is not None:
        available_end = max(available_end, cached_span[2])
    if incoming_span is not None:
        available_end = max(available_end, incoming_span[2])
    if available_end < 0 and isinstance(thinker_decode_embed, torch.Tensor):
        available_end = start_index + 1
    legacy_decode_end = len(thinker_output_token_ids) - 1

    if clear_incoming_decode_embed:
        update_dict[THINKER_DECODE_EMBEDDINGS_KEY] = None

    return ThinkerDecodeStepState(
        thinker_embed=thinker_embed,
        start_index=start_index,
        available_end=available_end,
        legacy_decode_end=legacy_decode_end,
    )
