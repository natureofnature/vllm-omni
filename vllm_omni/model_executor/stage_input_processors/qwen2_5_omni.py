import torch
from vllm.inputs import TextPrompt

from vllm_omni.data_entry_keys import (
    EmbeddingsStruct,
    HiddenStatesStruct,
    IdsStruct,
    OmniPayload,
    OmniPayloadStruct,
    to_dict,
)
from vllm_omni.inputs.data import OmniTokensPrompt

TALKER_CODEC_PAD_TOKEN_ID = 8292
TALKER_CODEC_START_TOKEN_ID = 8293
TALKER_CODEC_END_TOKEN_ID = 8294


def thinker2talker(
    source_outputs,
    prompt: OmniTokensPrompt | TextPrompt = None,
    requires_multimodal_data: bool = False,
):
    thinker_outputs = source_outputs
    talker_inputs = []
    if not isinstance(prompt, list):
        prompt = [prompt]
    multi_modal_data = {
        thinker_output.request_id: p.get("multi_modal_data", None) for thinker_output, p in zip(thinker_outputs, prompt)
    }

    for i, thinker_output in enumerate(thinker_outputs):
        output = thinker_output.outputs[0]
        prompt_token_ids = thinker_output.prompt_token_ids
        thinker_output_ids = output.cumulative_token_ids
        prompt_token_ids_len = len(prompt_token_ids)
        mm: OmniPayload = output.multimodal_output
        latent = mm["latent"]
        thinker_hidden_states = latent.clone().detach().to(latent.device)
        decode_hidden = thinker_hidden_states[prompt_token_ids_len:].to(torch.float32)
        prefill_hidden = thinker_hidden_states[:prompt_token_ids_len].to(torch.float32)
        additional_information = to_dict(
            OmniPayloadStruct(
                hidden_states=HiddenStatesStruct(output=decode_hidden, output_shape=list(decode_hidden.shape)),
                embed=EmbeddingsStruct(prefill=prefill_hidden, prefill_shape=list(prefill_hidden.shape)),
                ids=IdsStruct(prompt=list(prompt_token_ids), output=list(thinker_output_ids)),
            )
        )
        talker_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[TALKER_CODEC_START_TOKEN_ID]
                + [TALKER_CODEC_PAD_TOKEN_ID] * (len(prompt_token_ids))
                + [TALKER_CODEC_END_TOKEN_ID],
                additional_information=additional_information,
                multi_modal_data=(
                    multi_modal_data[thinker_output.request_id]
                    if requires_multimodal_data and multi_modal_data is not None
                    else None
                ),
                mm_processor_kwargs=None,
            )
        )
    return talker_inputs


def talker2code2wav(
    source_outputs,
    _prompt: OmniTokensPrompt | TextPrompt = None,
    _requires_multimodal_data: bool = False,
):
    code2wav_inputs = []
    for talker_output in source_outputs:
        output = talker_output.outputs[0]
        token_ids = list(output.cumulative_token_ids)
        if token_ids and token_ids[0] == TALKER_CODEC_START_TOKEN_ID:
            token_ids = token_ids[1:]
        if token_ids and token_ids[-1] == TALKER_CODEC_END_TOKEN_ID:
            token_ids = token_ids[:-1]
        if not token_ids:
            continue
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=token_ids,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return code2wav_inputs


# ============================================================================
# PR3 worker-connector data plane (non-async-chunk path) — Group B half.
# Only talker→code2wav is migrated in this commit; thinker→talker (Group A)
# requires model-side pooler_output emit and is deferred.
# ============================================================================

# Per-model REPLACE-keys for the full-payload accumulator.  qwen2_5_omni's
# producer side does not emit model_outputs through pooler_output (it ships
# token_ids on the request directly), so the empty set preserves correctness.
_FULL_PAYLOAD_REPLACE_KEYS: frozenset[str] = frozenset()


def _strip_codec_boundaries(token_ids: list[int]) -> list[int]:
    """Drop TALKER_CODEC_START/END boundary tokens (mirror talker2code2wav)."""
    tids = list(token_ids)
    if tids and tids[0] == TALKER_CODEC_START_TOKEN_ID:
        tids = tids[1:]
    if tids and tids[-1] == TALKER_CODEC_END_TOKEN_ID:
        tids = tids[:-1]
    return tids


def talker2code2wav_token_only(
    source_outputs,
    _prompt: OmniTokensPrompt | TextPrompt = None,
    _requires_multimodal_data: bool = False,
):
    """Sync-side placeholder for Stage-2 input (code2wav).

    Returns OmniTokensPrompt sized to the stripped codec token count.
    Actual codec ids are delivered via the worker connector payload built
    by ``talker2code2wav_full_payload``.
    """
    code2wav_inputs = []
    for talker_output in source_outputs:
        output = talker_output.outputs[0]
        token_ids = _strip_codec_boundaries(list(output.cumulative_token_ids))
        if not token_ids:
            continue
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0] * len(token_ids),
                additional_information=None,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return code2wav_inputs


talker2code2wav_token_only._is_sync_input = True


def talker2code2wav_full_payload(
    transfer_manager,
    pooling_output: dict,
    request,
) -> dict | None:
    """Producer-side packer: ship the stripped codec ids via connector.

    Group B shape — token_ids only.  The talker stage's output already
    carries the codec ids on ``request.output_token_ids``; we strip the
    boundary tokens and pack a minimal payload.
    """
    del transfer_manager
    token_ids = list(getattr(request, "output_token_ids", None) or [])
    if not token_ids:
        return None
    token_ids = _strip_codec_boundaries(token_ids)
    if not token_ids:
        return None
    return {
        "codes": {"audio": token_ids},
        "meta": {"finished": torch.tensor(True, dtype=torch.bool)},
    }


# ============================================================================
# PR3 worker-connector data plane (non-async-chunk path) — Group A reduced
# to D-minimal shape.
#
# Three subagent investigations (2026-05-16, audits/) confirmed:
# - qwen2_5_omni talker consumes ONE tensor (last-layer hidden state) via
#   Linear(3584, 896); no early-layer-0 consumer, no `accept_hidden_layer`
#   HF config field.
# - `text_hidden_states` is NOT plumbed into the AR runner pooler_output
#   chain, so the existing accumulator cannot ship it.
# So the PR3 migration is structural-only: thinker2talker_token_only mirrors
# the legacy body so additional_information continues to carry the latent
# tensor (same as cosyvoice3's post-fix state).  full_payload returns None.
# ============================================================================

_FULL_PAYLOAD_REPLACE_KEYS: frozenset[str] = frozenset()


def thinker2talker_token_only(
    source_outputs,
    prompt: OmniTokensPrompt | TextPrompt = None,
    requires_multimodal_data: bool = False,
):
    """Sync-side builder for the non-async-chunk thinker->talker path.

    Body is identical to legacy ``thinker2talker`` above — preserves the
    orchestrator-shaped data path (latent in additional_information) so
    the talker stage receives thinker hidden states without requiring the
    worker connector to deliver them.  Filed as a Phase 4 follow-up to
    route the latent via connector once the AR runner's text_hidden_states
    plumbing is wired into pooler_output / model_intermediate_buffer.

    The ``_is_sync_input = True`` marker below activates the Phase 2a
    structural gate so the rest of the PR3 infrastructure (gen scheduler
    bridge, runner lifecycle, full-payload accumulator) participates
    consistently with the other 8 migrated transitions.
    """
    thinker_outputs = source_outputs
    talker_inputs = []
    if not isinstance(prompt, list):
        prompt = [prompt]
    multi_modal_data = {
        thinker_output.request_id: p.get("multi_modal_data", None) for thinker_output, p in zip(thinker_outputs, prompt)
    }

    for i, thinker_output in enumerate(thinker_outputs):
        output = thinker_output.outputs[0]
        prompt_token_ids = thinker_output.prompt_token_ids
        thinker_output_ids = output.cumulative_token_ids
        prompt_token_ids_len = len(prompt_token_ids)
        mm: OmniPayload = output.multimodal_output
        latent = mm["latent"]
        thinker_hidden_states = latent.clone().detach().to(latent.device)
        decode_hidden = thinker_hidden_states[prompt_token_ids_len:].to(torch.float32)
        prefill_hidden = thinker_hidden_states[:prompt_token_ids_len].to(torch.float32)
        additional_information = to_dict(
            OmniPayloadStruct(
                hidden_states=HiddenStatesStruct(output=decode_hidden, output_shape=list(decode_hidden.shape)),
                embed=EmbeddingsStruct(prefill=prefill_hidden, prefill_shape=list(prefill_hidden.shape)),
                ids=IdsStruct(prompt=list(prompt_token_ids), output=list(thinker_output_ids)),
            )
        )
        talker_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[TALKER_CODEC_START_TOKEN_ID]
                + [TALKER_CODEC_PAD_TOKEN_ID] * (len(prompt_token_ids))
                + [TALKER_CODEC_END_TOKEN_ID],
                additional_information=additional_information,
                multi_modal_data=(
                    multi_modal_data[thinker_output.request_id]
                    if requires_multimodal_data and multi_modal_data is not None
                    else None
                ),
                mm_processor_kwargs=None,
            )
        )

    return talker_inputs


thinker2talker_token_only._is_sync_input = True


def thinker2talker_full_payload(
    transfer_manager,
    pooling_output,
    request,
):
    """Producer-side packer — no-op.

    qwen2_5_omni's thinker emits its last-layer hidden state via
    ``text_hidden_states`` (the OmniOutput field), which is materialized
    into ``multimodal_output["latent"]`` at the engine boundary.  That
    field is NOT plumbed into the AR runner's pooler_output chain
    (data_entry_keys.flatten_payload + gpu_ar_model_runner emit), so the
    accumulator cannot ship it via the worker connector today.

    Returning None tells the connector to skip the send for this
    transition; the consumer reads the latent via additional_information
    (preserved by thinker2talker_token_only).  Filed as Phase 4 follow-up
    alongside #90 (speaker/language) and cosyvoice3 (embed.*).
    """
    del transfer_manager, pooling_output, request
    return None
