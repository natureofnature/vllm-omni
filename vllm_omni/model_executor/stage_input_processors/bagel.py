# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The vLLM team.
"""Stage input processor for Bagel: Thinker -> Gen -> VAE transition."""

from typing import Any, Union, List

import torch
from vllm.inputs import TextPrompt, TokensPrompt

from vllm_omni.inputs.data import OmniTokensPrompt


def thinker2gen(
    stage_list: List[Any],
    engine_input_source: List[int],
    prompt: Union[OmniTokensPrompt, TextPrompt, TokensPrompt, None] = None,
    requires_multimodal_data: bool = False,
) -> List[OmniTokensPrompt]:
    """
    Process thinker outputs to create gen (diffusion) inputs.
    
    Assumption: Thinker generates text description which serves as the prompt for generation.
    """
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {source_stage_id}")

    if stage_list[source_stage_id].engine_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")

    thinker_outputs = stage_list[source_stage_id].engine_outputs
    gen_inputs = []

    for thinker_output in thinker_outputs:
        output = thinker_output.outputs[0]
        # Use the generated text (token ids) as the prompt for the diffusion model
        # We assume the thinker outputs sequence of token IDs that represent the prompt
        
        # Note: generated_token_ids might include special tokens, we might need to filter or process them
        prompt_ids = list(output.token_ids)
        
        gen_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=prompt_ids,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return gen_inputs


def gen2vae(
    stage_list: List[Any],
    engine_input_source: List[int],
    prompt: Union[OmniTokensPrompt, TextPrompt, TokensPrompt, None] = None,
    requires_multimodal_data: bool = False,
) -> List[OmniTokensPrompt]:
    """
    Process gen outputs (latents) to create VAE inputs.
    """
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {source_stage_id}")

    if stage_list[source_stage_id].engine_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")

    gen_outputs = stage_list[source_stage_id].engine_outputs
    vae_inputs = []

    for gen_output in gen_outputs:
        output = gen_output.outputs[0]
        
        # We expect the latent tensor to be available in multimodal_output
        # The key should match what the worker/model returns.
        # Assuming "latents" or similar.
        
        # In a real implementation, the Gen worker must return 'latents' in multimodal_output.
        # For this processor, we'll assume it's there.
        
        if output.multimodal_output and "latents" in output.multimodal_output:
            latents = output.multimodal_output["latents"]
            
            # Pass latents as "additional_information" or repackage them
            # The VAE stage typically doesn't take "tokens" but we use OmniTokensPrompt as a carrier
            info = {
                "latents": latents
            }
            
            vae_inputs.append(
                OmniTokensPrompt(
                    prompt_token_ids=[0], # Dummy tokens
                    additional_information=info,
                    multi_modal_data=None, # Or pass as multi_modal_data?
                )
            )
        else:
            # Fallback or error handling
            # If no latents, maybe it failed or we are in a different mode
            pass

    return vae_inputs
