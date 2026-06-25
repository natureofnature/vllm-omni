# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_omni.core.sched.output import OmniChunkRecvHandle, OmniSchedulerOutput


def _empty_scheduler_output(**kwargs) -> OmniSchedulerOutput:
    return OmniSchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=None,
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        **kwargs,
    )


def test_pending_input_registrations_aliases_connector_registrations() -> None:
    output = _empty_scheduler_output()
    handle = OmniChunkRecvHandle(request_id="r1", external_req_id="external-r1")

    output.pending_input_registrations = [handle]

    assert output.pending_connector_registrations == [handle]
    assert output.pending_input_registrations == [handle]


def test_pending_input_registrations_constructor_alias() -> None:
    handle = OmniChunkRecvHandle(request_id="r1", external_req_id="external-r1")

    output = _empty_scheduler_output(pending_input_registrations=[handle])

    assert output.pending_connector_registrations == [handle]
    assert output.pending_input_registrations == [handle]
