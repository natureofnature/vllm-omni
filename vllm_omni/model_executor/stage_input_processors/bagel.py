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
        # For Bagel AR, we expect generated text.
        if hasattr(req_output, "outputs") and len(req_output.outputs) > 0:
            generated_text = req_output.outputs[0].text
            # Create input for DiT
            # For now, we just pass the generated text as the prompt for DiT
            # Logic from test.py: output = image_generator.generate(...)
            # The prompt for DiT is implicit or same as AR?
            # In test.py: kv_cache = encode_text(prompt) -> kv_cache passed.
            # The DiT prompts are usually the same or conditioned.
            # But here we treat the AR output as the "context" potentially.
            # If ar2dit is strict, we need to know what Stage 1 expects.
            # Qwen2ImageGenerator expects prompts?

            # We assume Stage 1 needs a prompt.
            engine_inputs.append({"prompt": generated_text, "request_id": req_output.request_id})

    return engine_inputs
