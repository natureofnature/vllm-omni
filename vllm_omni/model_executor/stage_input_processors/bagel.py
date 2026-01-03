from vllm.inputs import TextPrompt

from vllm_omni.inputs.data import OmniTokensPrompt


def ar2dit(
    stage_list,
    engine_input_source,
    prompt: OmniTokensPrompt | TextPrompt = None,
    requires_multimodal_data: bool = False,
):
    """
    Placeholder for passing Thinker output (Text + KV Cache) to DiT stage.
    Current implementation assumes a direct passthrough or minimal processing.
    """
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    source_stage_id = engine_input_source[0]
    # Check if stage_list has valid output
    if source_stage_id >= len(stage_list):
        return []

    source_stage = stage_list[source_stage_id]
    if not source_stage.engine_outputs:
        return []

    source_outputs = source_stage.engine_outputs

    # Process outputs
    engine_inputs = []

    # If prompt is provided (original request prompts), we can try to align them
    # But usually engine_outputs has request_ids

    for req_output in source_outputs:
        # req_output is RequestOutput
        if hasattr(req_output, "outputs") and len(req_output.outputs) > 0:
            generated_text = req_output.outputs[0].text

            # Create input for DiT
            input_dict = {
                "prompt": generated_text,
                "request_id": req_output.request_id,
            }

            # Check for KV cache in additional_information
            # This relies on the AR stage (Runner/Scheduler) attaching the extracted KV to the output
            # or the OmniLLM flow preserving it.
            add_info = getattr(req_output, "additional_information", None) or {}

            if "past_key_values" in add_info:
                input_dict["past_key_values"] = add_info["past_key_values"]

            if "kv_metadata" in add_info:
                input_dict["kv_metadata"] = add_info["kv_metadata"]

            engine_inputs.append(input_dict)

    return engine_inputs
