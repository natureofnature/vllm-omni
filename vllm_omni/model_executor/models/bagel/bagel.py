# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 Bytedance Ltd. and/or its affiliates.

from typing import Optional, Union, List, Any, Dict, Iterable

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.v1.sample.metadata import SamplingMetadata

# Import the Thinker Processor from standard vllm (Bagel)
from vllm.model_executor.models.bagel import (
    BagelMultiModalProcessor,
    BagelProcessingInfo,
    BagelDummyInputsBuilder,
)

# Import local Gen components
from .bagel_core import Bagel, BagelConfig
from .autoencoder import AutoEncoder, load_ae

logger = init_logger(__name__)

# We reuse the BagelProcessingInfo from vllm but might need to extend it for Gen if needed.
# For now, we register the same processor for both stages as the inputs are similar (multimodal).
@MULTIMODAL_REGISTRY.register_processor(
    BagelMultiModalProcessor,
    info=BagelProcessingInfo,
    dummy_inputs=BagelDummyInputsBuilder,
)
class BagelForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):
    """
    Unified Bagel model for vllm-omni.
    
    Supports three stages via `model_stage` config:
    - "thinker": The AR model (Qwen2-VL based) for image understanding.
    - "gen": The Diffusion model (Bagel MoT) for image generation.
    - "vae_decoder": The VAE Decoder for latent-to-image conversion.
    """
    
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config
        self.model_stage = vllm_config.model_config.model_stage

        if self.model_stage == "thinker":
            # Initialize standard vLLM Bagel model
            # We use the standard Bagel implementation from vllm.model_executor.models.bagel
            # But we need to initialize it with the correct config
            from vllm.model_executor.models.bagel import BagelForConditionalGeneration as BagelAR
            
            self.thinker = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "thinker"),
                hf_config=self.config,
                architectures=["BagelForConditionalGeneration"],
            )
            self.model = self.thinker
            self.gen = None
            self.vae = None

        elif self.model_stage == "gen":
            self.thinker = None
            self.vae = None
            
            # Initialize local Bagel Diffusion model
            # This uses the ported bagel_core.Bagel class
            # We assume config is a BagelConfig
            
            # First, we need to construct the Qwen2 Language Model part using vLLM's optimized kernels if possible
            # OR use the ported Qwen2Navit from bagel_core which supports the specific MoT attention.
            # The ported bagel_core.Bagel expects a 'language_model' argument which is the Qwen2ForCausalLM.
            # In bagel_core.py, Qwen2ForCausalLM is imported from .qwen2_navit
            
            from .qwen2_navit import Qwen2ForCausalLM, Qwen2Config
            
            # We might need to adapt the config to Qwen2Config for the inner LM
            llm_config = self.config.llm_config
            # Ensure layer_module is set correctly for MoT
            if not hasattr(llm_config, "layer_module"):
                llm_config.layer_module = "Qwen2MoTDecoderLayer"
                
            # Initialize the custom Qwen2 LM for Bagel Gen
            language_model = Qwen2ForCausalLM(llm_config)
            
            self.gen = Bagel(
                language_model=language_model,
                config=self.config,
            )
            self.model = self.gen

        elif self.model_stage == "vae_decoder":
            self.thinker = None
            self.gen = None
            
            # Initialize VAE Decoder
            # We use load_ae from autoencoder.py
            # But we want to initialize it from config/weights, not from file path hardcoded
            # AutoEncoder params are dataclass.
            
            from .autoencoder import AutoEncoderParams
            
            vae_config = self.config.vae_config
            ae_params = AutoEncoderParams(
                resolution=self.config.vit_config.image_size, # Assuming resolution matches
                in_channels=3, 
                downsample=vae_config.downsample,
                ch=128, # Default from load_ae? Need to check config
                out_ch=3,
                ch_mult=[1, 2, 4, 4], # Default
                num_res_blocks=2, # Default
                z_channels=vae_config.z_channels,
                scale_factor=0.3611, # Default
                shift_factor=0.1159, # Default
            )
            
            # TODO: Populate params from self.config if available
            
            self.vae = AutoEncoder(ae_params)
            self.model = self.vae

        else:
            raise ValueError(f"Invalid model_stage: {self.model_stage}. Must be 'thinker', 'gen', or 'vae_decoder'.")

        self.make_empty_intermediate_tensors = (
            self.thinker.make_empty_intermediate_tensors if self.thinker else lambda: None
        )

    def get_multimodal_embeddings(self, **kwargs):
        if self.thinker:
            return self.thinker.get_multimodal_embeddings(**kwargs)
        return None # Gen stage might handle embeddings differently or internally

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        
        if self.model_stage == "thinker":
            return self.thinker(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **kwargs
            )
        elif self.model_stage == "gen":
            # For Gen stage, we primarily support generation via custom methods,
            # but vLLM might call forward for prefill.
            # The ported Bagel Gen uses custom 'prepare_prompts' and 'forward_cache_update_text' etc.
            # standard vLLM forward might not be directly compatible without adaptation.
            # However, for now, we can expose the underlying model's forward if it exists.
            
            # TODO: Adapt standard vLLM forward to Bagel Gen's specific needs if necessary.
            # The Bagel Gen model in bagel_core.py is a PreTrainedModel but its usage in pipeline
            # is typically via specific prepare methods + forward_inference.
            
            raise NotImplementedError("Standard forward pass not yet implemented for Bagel Gen stage. Use custom generation methods.")
            
        return None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if self.model_stage == "thinker":
            # Delegate to standard Bagel loader
            return self.thinker.load_weights(weights)
        elif self.model_stage == "gen":
            # Use AutoWeightsLoader for the local Bagel model
            # We might need to use the logic from pipeline_bagel.py's load_weights
            # to handle prefix mapping and shape mismatches.
            
            from vllm.model_executor.models.utils import AutoWeightsLoader
            
            # Define a custom filter similar to pipeline_bagel.py if needed
            # For now, attempt standard load and refine if strict loading fails
            
            loader = AutoWeightsLoader(self.gen)
            return loader.load_weights(weights)
            
        return set()

    # Expose custom generation methods for Gen stage
    def generate_image(self, *args, **kwargs):
        if self.gen:
            return self.gen.generate_image(*args, **kwargs)
        raise ValueError("generate_image only available in 'gen' stage")
