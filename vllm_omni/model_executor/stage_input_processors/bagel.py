from vllm.inputs import TextPrompt
from vllm.logger import init_logger

from vllm_omni.inputs.data import OmniTokensPrompt

logger = init_logger(__name__)


def ar2dit(
    stage_list,
    engine_input_source,
    prompt: OmniTokensPrompt | TextPrompt = None,
    requires_multimodal_data: bool = False,
):
    """
    Convert AR Stage outputs to DiT Stage inputs, explicitly handling KV Cache transfer.
    """
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        return []

    source_stage = stage_list[source_stage_id]
    if not source_stage.engine_outputs:
        return []

    source_outputs = source_stage.engine_outputs
    engine_inputs = []

    for req_output in source_outputs:
        generated_text = ""
        if hasattr(req_output, "outputs") and len(req_output.outputs) > 0:
            generated_text = req_output.outputs[0].text

        # Not finished! Now I can only get length and block ids
        # TODO: Extract past_key_values from block ids
        # kv_info = getattr(req_output, "kv_transfer_params", {}) or {}
        # past_key_values = kv_info.get("past_key_values")
        past_key_values = None

        input_dict = {
            "prompt": generated_text,
            "request_id": req_output.request_id,
        }

        if past_key_values is not None:
            input_dict["past_key_values"] = past_key_values
        else:
            pass

        engine_inputs.append(input_dict)

    return engine_inputs
