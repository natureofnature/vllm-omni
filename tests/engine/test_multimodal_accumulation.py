# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.outputs.mm_outputs import MultimodalPayload
from vllm_omni.outputs.multimodal_accumulation import (
    drain_delta_payload,
    is_non_final_delta_audio_chunk,
    replace_nemotron_snapshot_keys,
    replace_snapshot_keys,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_chunk_accumulation_policy_replaces_snapshots_and_drains_delta_state():
    accumulated = MultimodalPayload.from_dict(
        {
            "audio": torch.tensor([1.0]),
            "meta.segment_end": torch.tensor([0]),
            "meta.tts_is_last_chunk": torch.tensor([0]),
            "meta.turn_end": torch.tensor([0]),
            "meta.stable_request_value": "keep",
        }
    )
    incoming = MultimodalPayload.from_dict(
        {
            "audio": torch.tensor([2.0]),
            "meta.segment_end": torch.tensor([1]),
            "meta.tts_is_last_chunk": torch.tensor([1]),
            "meta.turn_end": torch.tensor([1]),
        }
    )
    assert accumulated is not None
    assert incoming is not None

    replace_snapshot_keys(accumulated, incoming)
    merged = accumulated.merged_with(incoming)

    assert not is_non_final_delta_audio_chunk(merged, "audio")

    drain_delta_payload(merged)

    assert "audio" not in merged
    assert "meta.segment_end" not in merged
    assert "meta.tts_is_last_chunk" not in merged
    assert "meta.turn_end" not in merged
    assert merged.metadata["meta.stable_request_value"] == "keep"


def test_replace_nemotron_snapshot_keys_is_scoped_to_nemotron_marker():
    marker = "meta.nvc_logical_prompt_len"
    accumulated = MultimodalPayload.from_dict(
        {
            "codes.audio": torch.ones(2, 4, dtype=torch.long),
            "meta.nvc_tts_step": torch.tensor([1]),
            marker: 3,
        }
    )
    assert accumulated is not None

    # Marked wake re-ships the whole prefix -> prior tensors dropped before merge.
    marked = MultimodalPayload.from_dict(
        {
            "codes.audio": torch.ones(5, 4, dtype=torch.long),
            "meta.nvc_tts_step": torch.tensor([2]),
            marker: 3,
        }
    )
    assert marked is not None
    replace_nemotron_snapshot_keys(accumulated, marked)
    merged = accumulated.merged_with(marked)
    assert isinstance(merged["codes.audio"], torch.Tensor)
    assert merged["codes.audio"].shape[0] == 5
    assert isinstance(merged["meta.nvc_tts_step"], torch.Tensor)
    assert merged["meta.nvc_tts_step"].tolist() == [2]

    # Unmarked codec deltas keep accumulating under the same key.
    unmarked = MultimodalPayload.from_dict({"codes.audio": torch.ones(3, 4, dtype=torch.long)})
    assert unmarked is not None
    replace_nemotron_snapshot_keys(merged, unmarked)
    grown = merged.merged_with(unmarked)
    assert isinstance(grown["codes.audio"], list)
    assert sum(t.shape[0] for t in grown["codes.audio"]) == 8


def test_replace_nemotron_snapshot_keys_detects_nested_meta_marker():
    # Marker nested under a "meta" dict (unflattened producer form) must still
    # scope the REPLACE, with no global key registration required.
    incoming = MultimodalPayload.from_dict(
        {"codes.audio": torch.ones(1, 4, dtype=torch.long), "meta": {"nvc_logical_prompt_len": 3}}
    )
    assert incoming is not None
    accumulated = MultimodalPayload.from_dict({"codes.audio": torch.ones(4, 4, dtype=torch.long)})
    assert accumulated is not None

    replace_nemotron_snapshot_keys(accumulated, incoming)
    assert "codes.audio" not in accumulated
