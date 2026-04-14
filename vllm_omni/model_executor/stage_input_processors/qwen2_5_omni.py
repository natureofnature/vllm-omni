from typing import Any

import torch
from vllm.inputs import TextPrompt
from vllm.logger import init_logger

from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.stage_input_processors._common import ensure_list

TALKER_CODEC_PAD_TOKEN_ID = 8292
TALKER_CODEC_START_TOKEN_ID = 8293
TALKER_CODEC_END_TOKEN_ID = 8294

logger = init_logger(__name__)


# =========================
# full_payload_mode processors (new mixin path)
# =========================


def thinker2talker_full_payload(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: Any,
) -> dict[str, Any] | None:
    """full_payload_mode thinker->talker processor for Qwen2.5-Omni.

    Called once when the thinker finishes.  Packs the full hidden states
    (prompt + generated) into a dict that becomes the talker's
    ``additional_information``.
    """
    hidden = pooling_output.get("hidden")
    if hidden is None:
        return None

    if hasattr(request, "all_token_ids"):
        prompt_token_ids = ensure_list(getattr(request, "prompt_token_ids", []) or [])
        output_token_ids = ensure_list(getattr(request, "output_token_ids", []) or [])
    else:
        prompt_token_ids = ensure_list(getattr(request, "prompt_token_ids", []) or [])
        output_token_ids = ensure_list(getattr(request, "output_token_ids", []) or [])

    prompt_len = len(prompt_token_ids)
    h = hidden.detach().cpu().to(torch.float32)

    # Qwen2.5: talker prompt is [START] + [PAD]*prompt_len + [END] → length = prompt_len + 2
    next_stage_prompt_len = prompt_len + 2

    payload = {
        "thinker_result": h[prompt_len:],
        "prompt_embeds": h[:prompt_len],
        "prompt_token_ids": prompt_token_ids,
        "thinker_output_token_ids": output_token_ids,
        "thinker_result_shape": list(h[prompt_len:].shape),
        "prompt_embeds_shape": list(h[:prompt_len].shape),
        "next_stage_prompt_len": next_stage_prompt_len,
        "finished": torch.tensor(True, dtype=torch.bool),
    }

    logger.debug(
        (
            "thinker2talker_full_payload: prompt_len=%s thinker_result.shape=%s "
            "prompt_embeds.shape=%s output_token_ids_len=%s"
        ),
        prompt_len,
        tuple(h[prompt_len:].shape),
        tuple(h[:prompt_len].shape),
        len(output_token_ids),
    )

    return payload


def talker2code2wav_full_payload(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: Any,
) -> dict[str, Any] | None:
    """full_payload_mode talker->code2wav processor for Qwen2.5-Omni.

    The talker's sampled token IDs are the codec codes.  We pass them
    through so that ``update_request_metadata`` can set them as the code2wav
    stage's ``prompt_token_ids``.
    """
    output_ids = ensure_list(getattr(request, "output_token_ids", []) or [])
    if not output_ids:
        return None

    # Remove end token if present (8294)
    if output_ids and output_ids[-1] == TALKER_CODEC_END_TOKEN_ID:
        output_ids = output_ids[:-1]

    # Filter out special tokens: start (8293), pad (8292), mask (8296), and negative values
    # Valid codec tokens are in range [0, 8291] (vocab_size=8448, but codec range is smaller)
    filtered_ids = []
    for tid in output_ids:
        if tid >= 0 and tid < TALKER_CODEC_PAD_TOKEN_ID:  # < 8292
            filtered_ids.append(tid)

    logger.debug(
        "talker2code2wav_full_payload: original_len=%s filtered_len=%s first_10=%s last_10=%s",
        len(output_ids),
        len(filtered_ids),
        filtered_ids[:10],
        filtered_ids[-10:],
    )

    return {
        "code_predictor_codes": filtered_ids,
        "finished": torch.tensor(True, dtype=torch.bool),
    }


# Backward-compatible aliases for configs that still reference the old suffix.
thinker2talker_batch = thinker2talker_full_payload
talker2code2wav_batch = talker2code2wav_full_payload


# =========================
# Old orchestrator-style processors
# =========================


def thinker2talker(
    stage_list,
    engine_input_source,
    prompt: OmniTokensPrompt | TextPrompt = None,
    requires_multimodal_data: bool = False,
):
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")
    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {source_stage_id}")
    if stage_list[source_stage_id].engine_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")
    thinker_outputs = stage_list[source_stage_id].engine_outputs
    talker_inputs = []
    if not isinstance(prompt, list):
        prompt = [prompt]
    multi_modal_data = {
        thinker_output.request_id: p.get("multi_modal_data", None) for thinker_output, p in zip(thinker_outputs, prompt)
    }

    for i, thinker_output in enumerate(thinker_outputs):
        output = thinker_output.outputs[0]
        prompt_token_ids = thinker_output.prompt_token_ids
        thinker_output_ids = output.token_ids
        prompt_token_ids_len = len(prompt_token_ids)
        latent = output.multimodal_output["latent"]
        thinker_hidden_states = latent.clone().detach().to(latent.device)
        additional_information = {
            "thinker_result": thinker_hidden_states[prompt_token_ids_len:].to(torch.float32),
            "prompt_embeds": thinker_hidden_states[:prompt_token_ids_len].to(torch.float32),
            "prompt_token_ids": prompt_token_ids,
            "thinker_output_token_ids": thinker_output_ids,
            "thinker_result_shape": list(thinker_hidden_states[prompt_token_ids_len:].shape),
            "prompt_embeds_shape": list(thinker_hidden_states[:prompt_token_ids_len].shape),
        }
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
