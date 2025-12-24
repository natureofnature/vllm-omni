from typing import Union

from vllm.inputs import TextPrompt

from vllm_omni.inputs.data import OmniTokensPrompt


def ar2dit(
    stage_list,
    engine_input_source,
    prompt: Union[OmniTokensPrompt, TextPrompt] = None,
    requires_multimodal_data: bool = False,
):
    """
    Placeholder for passing Thinker output (Text + KV Cache) to DiT stage.
    Current implementation assumes a direct passthrough or minimal processing.
    """
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    # Placeholder: Just return empty list or dummy prompt for now as requested
    # Real implementation will need to package text_hidden_states into additional_information
    return []
