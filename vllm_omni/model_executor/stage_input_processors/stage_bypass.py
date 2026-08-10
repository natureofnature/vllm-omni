# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generic optional stage-skip metadata helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

OMNI_SKIP_STAGES_KEY = "omni_skip_stages"


def _parse_skip_stage_ids(value: Any) -> frozenset[int]:
    if value is None:
        return frozenset()
    if isinstance(value, int):
        return frozenset([value])
    if isinstance(value, list):
        stage_ids: list[int] = []
        for item in value:
            if isinstance(item, int):
                stage_ids.append(item)
            elif isinstance(item, str) and item.strip().isdigit():
                stage_ids.append(int(item.strip()))
        return frozenset(stage_ids)
    return frozenset()


def should_skip_stage(prompt: Any, stage_id: int) -> bool:
    """Return True when ``additional_information.omni_skip_stages`` includes ``stage_id``."""
    if not isinstance(prompt, dict):
        return False
    additional_info = prompt.get("additional_information")
    return should_skip_stage_from_info(additional_info, stage_id)


def should_skip_stage_from_info(additional_info: Any, stage_id: int) -> bool:
    """Return True when a per-request ``additional_information`` dict requests stage skip."""
    if not isinstance(additional_info, dict):
        return False
    return stage_id in _parse_skip_stage_ids(additional_info.get(OMNI_SKIP_STAGES_KEY))


def make_mock_text_stage_output(request_id: str, text: str = "", *, finished: bool = True) -> Any:
    """Synthetic text-stage output used when an upstream stage is bypassed (sync path)."""
    output = SimpleNamespace(
        text=text,
        cumulative_text=text,
        cumulative_token_ids=[],
        multimodal_output={},
        finished=finished,
    )
    return SimpleNamespace(
        request_id=request_id,
        outputs=[output],
        finished=finished,
    )


def build_empty_asr_aura_chunk_payload(
    additional_info: dict[str, Any] | None = None,
    *,
    mm_processor_kwargs: Any = None,
) -> dict[str, Any]:
    """Build a finished empty ASR→AURA async_chunk payload for orchestrator SHM inject.

    Shape matches ``asr2aura_async_chunk`` finish emit plus the meta stamps that
    ``OmniChunkTransferAdapter._send_single_request`` would attach.
    """
    info = dict(additional_info) if isinstance(additional_info, dict) else {}
    payload: dict[str, Any] = {
        "aura_asr_transcript": "",
        "additional_information": info,
        "mm_processor_kwargs": mm_processor_kwargs,
        "meta": {
            "finished": torch.tensor(True, dtype=torch.bool),
            "is_segment_finished": torch.tensor(True, dtype=torch.bool),
        },
    }
    # Best-effort: attach packed turn video like the real Stage0 processor.
    try:
        from vllm_omni.model_executor.stage_input_processors.aura_omni import (
            _attach_aura_turn_video_payload,
        )

        multi_modal_data: dict[str, Any] = {}
        deferred = info.get("deferred_multi_modal_data")
        if isinstance(deferred, dict):
            multi_modal_data.update(deferred)
        _attach_aura_turn_video_payload(payload, info, multi_modal_data)
    except Exception:
        pass
    return payload
