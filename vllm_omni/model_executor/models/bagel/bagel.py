# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
"""Inference-only BAGEL model compatible with HuggingFace weights.

BAGEL is a unified multimodal model for image understanding and generation.
For vLLM, we focus on the image understanding (vision-to-text) capabilities.
"""

from collections.abc import Iterable, Mapping, Sequence

# from pydantic.dataclasses import dataclass
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import torch
import torch.nn as nn
from pydantic import Field
from transformers import Qwen2Config
from vllm.config import CacheConfig, VllmConfig

# from vllm.config.multimodal import BaseDummyOptions
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.qwen2 import (
    Qwen2Attention as VllmQwen2Attention,
)
from vllm.model_executor.models.qwen2 import (
    Qwen2DecoderLayer as VllmQwen2DecoderLayer,
)
from vllm.model_executor.models.qwen2 import (
    Qwen2ForCausalLM as VllmQwen2ForCausalLM,
)
from vllm.model_executor.models.qwen2 import (
    Qwen2Model as VllmQwen2Model,
)
from vllm.model_executor.models.siglip import SiglipVisionModel
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema

from .bagel_processor import BagelProcessor

logger = init_logger(__name__)


@dataclass
class BaseDummyOptions:
    """Base options for generating dummy data during profiling."""

    count: int = Field(999, ge=0)


class BagelImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size * number of images
        - c: Number of channels (3)
        - h: Height of each image
        - w: Width of each image
    """

    type: Literal["pixel_values"]
    pixel_values: torch.Tensor  # Shape: (bn, 3, h, w)


BagelImageInputs: TypeAlias = BagelImagePixelInputs


class BagelVisionMLP(nn.Module):
    """MLP connector for vision features."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        act_layer: str = "gelu_pytorch_tanh",
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.fc1 = ColumnParallelLinear(
            in_features,
            hidden_features,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )
        self.act = get_act_fn(act_layer)
        self.fc2 = RowParallelLinear(
            hidden_features,
            out_features,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc1(x)
        x = self.act(x)
        x, _ = self.fc2(x)
        return x


class PositionEmbedding(nn.Module):
    """2D position embedding for vision tokens using sin-cos embeddings."""

    def __init__(self, max_num_patch_per_side: int, hidden_size: int):
        super().__init__()
        self.max_num_patch_per_side = max_num_patch_per_side
        self.hidden_size = hidden_size

        # Create learnable 2D position embeddings (frozen sin-cos)
        pos_embed = self._get_2d_sincos_pos_embed(hidden_size, max_num_patch_per_side)
        self.register_buffer(
            "pos_embed",
            torch.from_numpy(pos_embed).float(),
            persistent=False,
        )

    @staticmethod
    def _get_2d_sincos_pos_embed(embed_dim: int, grid_size: int):
        """Generate 2D sin-cos position embeddings."""
        import numpy as np

        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)  # w goes first
        grid = np.stack(grid, axis=0)
        grid = grid.reshape([2, 1, grid_size, grid_size])
        pos_embed = PositionEmbedding._get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
        return pos_embed

    @staticmethod
    def _get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid):
        """Generate 2D sin-cos position embeddings from grid."""
        import numpy as np

        assert embed_dim % 2 == 0
        # use half of dimensions to encode grid_h
        emb_h = PositionEmbedding._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
        emb_w = PositionEmbedding._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
        emb = np.concatenate([emb_h, emb_w], axis=1)
        return emb

    @staticmethod
    def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos):
        """Generate 1D sin-cos position embeddings."""
        import numpy as np

        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float64)
        omega /= embed_dim / 2.0
        omega = 1.0 / 10000**omega

        pos = pos.reshape(-1)
        out = np.einsum("m,d->md", pos, omega)

        emb_sin = np.sin(out)
        emb_cos = np.cos(out)
        emb = np.concatenate([emb_sin, emb_cos], axis=1)
        return emb

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            position_ids: Flattened position IDs, shape (N,) where each ID
                         corresponds to a position in the flattened grid
        Returns:
            Position embeddings of shape (N, hidden_size)
        """
        # Ensure position_ids are on the same device as pos_embed
        position_ids = position_ids.to(self.pos_embed.device)
        return self.pos_embed[position_ids]


class BagelProcessingInfo(BaseProcessingInfo):
    """Processing information for BAGEL model."""

    def get_hf_processor(self, **kwargs: object) -> BagelProcessor:
        from vllm.transformers_utils.processor import cached_get_image_processor

        image_processor = cached_get_image_processor(
            self.ctx.model_config.model,
            revision=self.ctx.model_config.revision,
            trust_remote_code=self.ctx.model_config.trust_remote_code,
        )

        tokenizer = self.get_tokenizer()

        return BagelProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            **kwargs,
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        hf_config = self.get_hf_config()
        # Calculate max tokens per image
        # For BAGEL: (vit_max_num_patch_per_side) ** 2
        max_num_patches = hf_config.vit_max_num_patch_per_side**2
        return {"image": max_num_patches}

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        hf_config = self.get_hf_config()
        vit_config = hf_config.vit_config
        patch_size = vit_config.patch_size

        # Calculate number of patches
        num_patches_h = image_height // patch_size
        num_patches_w = image_width // patch_size
        return num_patches_h * num_patches_w


class BagelDummyInputsBuilder(BaseDummyInputsBuilder[BagelProcessingInfo]):
    """Build dummy inputs for BAGEL model profiling."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        # Use a simple placeholder for each image
        return "<|image_pad|>" * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        hf_config = self.info.get_hf_config()
        vit_config = hf_config.vit_config

        # Use the configured image size
        image_size = vit_config.image_size

        return {
            "image": self._get_dummy_images(
                width=image_size,
                height=image_size,
                num_images=num_images,
                # overrides=image_overrides,
            ),
        }


class BagelMultiModalProcessor(BaseMultiModalProcessor[BagelProcessingInfo]):
    """Multimodal processor for BAGEL model."""

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return False

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptReplacement]:
        """Replace image placeholders with the correct number of tokens."""
        hf_config = self.info.get_hf_config()

        # Get the tokenizer to look up the image token ID
        tokenizer = self.info.get_tokenizer()
        image_token_id = tokenizer.get_vocab().get("<|image_pad|>")
        if image_token_id is None:
            raise ValueError("Image token '<|image_pad|>' not found in tokenizer vocabulary")

        def get_replacement_bagel(item_idx: int):
            # For BAGEL, calculate number of tokens based on max patch size
            num_tokens = hf_config.vit_max_num_patch_per_side**2
            # Use the image token ID from tokenizer
            return [image_token_id] * num_tokens

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement_bagel,
            )
        ]

    def _get_mm_fields_config(
        self,
        hf_inputs: Any,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return {
            "pixel_values": MultiModalFieldConfig.batched("image"),
        }


class Qwen2BagelAttention(VllmQwen2Attention):
    """Qwen2 Attention with QK normalization support for BAGEL."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict[str, Any],
        max_position: int = 4096 * 32,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_type: str = "DECODER",
        dual_chunk_attention_config: dict[str, Any] | None = None,
        qk_norm: bool = False,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        # Initialize parent class
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            rope_parameters=rope_parameters,
            max_position=max_position,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            attn_type=attn_type,
        )

        # Add QK normalization support
        from vllm.model_executor.layers.layernorm import RMSNorm

        self.qk_norm = qk_norm
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

        # Override rotary embedding to use custom rope
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Apply QK normalization if enabled (before RoPE)
        if self.qk_norm:
            # Reshape to apply per-head normalization
            total_tokens = q.shape[0]
            q = q.view(total_tokens, self.num_heads, self.head_dim)
            k = k.view(total_tokens, self.num_kv_heads, self.head_dim)

            # Apply normalization
            q = self.q_norm(q)
            k = self.k_norm(k)

            # Reshape back
            q = q.view(total_tokens, self.q_size)
            k = k.view(total_tokens, self.kv_size)

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen2BagelMLP(nn.Module):
    """MLP module for Qwen2Bagel."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        from vllm.model_executor.layers.activation import SiluAndMul
        from vllm.model_executor.layers.linear import (
            MergedColumnParallelLinear,
            RowParallelLinear,
        )

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Qwen2BagelDecoderLayer(VllmQwen2DecoderLayer):
    """Qwen2 Decoder Layer with QK normalization support for BAGEL."""

    def __init__(
        self,
        config: Qwen2Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        # We need to override __init__ to use Qwen2BagelAttention
        nn.Module.__init__(self)  # Skip parent's __init__

        self.hidden_size = config.hidden_size

        # Import dependencies needed for attention initialization
        from vllm.attention.backends.abstract import AttentionType
        from vllm.model_executor.layers.layernorm import RMSNorm
        from vllm.transformers_utils.config import set_default_rope_theta

        set_default_rope_theta(config, default_theta=1000000)
        dual_chunk_attention_config = getattr(config, "dual_chunk_attention_config", None)

        # Determine attention type
        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        # Check if QK normalization is enabled
        qk_norm = getattr(config, "qk_norm", False)

        # Use Qwen2BagelAttention instead of standard Qwen2Attention
        self.self_attn = Qwen2BagelAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            rope_parameters=config.rope_parameters,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
            dual_chunk_attention_config=dual_chunk_attention_config,
            qk_norm=qk_norm,
            rms_norm_eps=config.rms_norm_eps,
        )

        # MLP - use custom MLP class
        self.mlp = Qwen2BagelMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

        # Layer norms
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Qwen2BagelModel(VllmQwen2Model):
    """Qwen2 Model with QK normalization support for BAGEL."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        hf_config_override: Qwen2Config | None = None,
    ):
        # Temporarily set hf_config_override if provided (for nested models like BAGEL)
        original_hf_config = None
        if hf_config_override is not None:
            original_hf_config = vllm_config.model_config.hf_config
            vllm_config.model_config.hf_config = hf_config_override

        try:
            # Initialize with custom decoder layer
            super().__init__(
                vllm_config=vllm_config,
                prefix=prefix,
                decoder_layer_type=Qwen2BagelDecoderLayer,
            )
        finally:
            # Restore original hf_config
            if original_hf_config is not None:
                vllm_config.model_config.hf_config = original_hf_config


class Qwen2BagelForCausalLM(VllmQwen2ForCausalLM):
    """Qwen2 for Causal LM with QK normalization support for BAGEL."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        hf_config_override: Qwen2Config | None = None,
    ):
        # We need to override to use Qwen2BagelModel
        nn.Module.__init__(self)  # Skip parent's __init__

        from vllm.distributed import get_pp_group
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
        from vllm.model_executor.models.utils import PPMissingLayer

        # Allow override of hf_config for nested models like BAGEL
        if hf_config_override is not None:
            config = hf_config_override
        else:
            config = vllm_config.model_config.hf_config.get_text_config()
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config

        # Use Qwen2BagelModel instead of standard Qwen2Model
        self.model = Qwen2BagelModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            hf_config_override=hf_config_override,
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

        # Copy attributes for SupportsLoRA, SupportsPP, SupportsEagle3
        self.packed_modules_mapping = VllmQwen2ForCausalLM.packed_modules_mapping


@MULTIMODAL_REGISTRY.register_processor(
    BagelMultiModalProcessor,
    info=BagelProcessingInfo,
    dummy_inputs=BagelDummyInputsBuilder,
)
class BagelForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsLoRA, SupportsPP):
    """
    BAGEL: A unified multimodal model for image understanding and generation.

    For vLLM, we focus on the image understanding (vision-to-text) capabilities.
    The image generation part is not supported in vLLM.
    """

    # Weight mapping from HF to vLLM
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "language_model.": "language_model.",
            "vit_model.": "vit_model.",
            "connector.": "connector.",
            "vit_pos_embed.": "vit_pos_embed.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        # Ensure we have a BagelConfig (check by name to handle trust_remote_code)
        # When trust_remote_code=True, the config comes from transformers_modules
        if type(config).__name__ != "BagelConfig":
            raise ValueError(
                f"Expected BagelConfig, got {type(config).__name__}. Make sure the model config is properly loaded."
            )

        self.config = config
        self.multimodal_config = multimodal_config

        self.language_model = Qwen2BagelForCausalLM(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
            hf_config_override=config.llm_config,
        )

        # Initialize vision model (SigLIP) if visual understanding is enabled
        if config.visual_und:
            # Fix vit_config: checkpoint has 26 layers (0-25) but config says 27
            # Also disable head as it's not in checkpoint
            vit_config = config.vit_config

            if vit_config.num_hidden_layers == 27:
                logger.warning(
                    "Overriding vit_config.num_hidden_layers from 27 to 26 to match the Bagel model checkpoint."
                )
                vit_config.num_hidden_layers = 26
            if not hasattr(vit_config, "vision_use_head"):
                logger.warning(
                    "Setting vit_config.vision_use_head to False as it is not present in the Bagel model checkpoint."
                )
                vit_config.vision_use_head = False

            self.vit_model = SiglipVisionModel(
                config=vit_config,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "vit_model"),
            )

            # Initialize connector (MLP)
            # vit_hidden_size = config.vit_config.hidden_size
            vit_hidden_size = vit_config.hidden_size
            llm_hidden_size = config.llm_config.hidden_size

            self.connector = BagelVisionMLP(
                in_features=vit_hidden_size,
                hidden_features=llm_hidden_size,
                out_features=llm_hidden_size,
                act_layer=config.connector_act,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "connector"),
            )

            # Position embedding for vision tokens
            self.vit_pos_embed = PositionEmbedding(
                max_num_patch_per_side=config.vit_max_num_patch_per_side,
                hidden_size=llm_hidden_size,
            )
        else:
            self.vit_model = None
            self.connector = None
            self.vit_pos_embed = None

        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

    def _parse_and_validate_image_input(self, **kwargs: object) -> BagelImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)

        if pixel_values is None:
            return None

        return BagelImagePixelInputs(
            type="pixel_values",
            pixel_values=pixel_values,
        )

    def _process_image_input(self, image_input: BagelImageInputs) -> tuple[torch.Tensor, ...]:
        """Process image inputs through vision encoder and connector."""
        pixel_values = image_input["pixel_values"]

        # Handle potential extra batch dimension
        # Expected shape: (batch_size * num_images, 3, H, W)
        # But might receive: (batch_size, num_images, 3, H, W)
        if pixel_values.ndim == 5:
            # Flatten batch and num_images dimensions
            batch_size, num_images, channels, height, width = pixel_values.shape
            pixel_values = pixel_values.reshape(batch_size * num_images, channels, height, width)

        # Get vision features from SigLIP
        # pixel_values shape: (batch_size * num_images, 3, H, W)
        vision_features = self.vit_model(pixel_values)

        # Pass through connector
        vision_embeds = self.connector(vision_features)

        # Add position embeddings
        batch_size, num_patches, hidden_size = vision_embeds.shape
        patch_size = self.config.vit_config.patch_size
        image_size = self.config.vit_config.image_size

        # Calculate grid dimensions
        num_patches_per_side = image_size // patch_size

        # Create flattened position IDs (0 to num_patches-1)
        # For BAGEL, we use extrapolate mode by default
        h_coords = torch.arange(num_patches_per_side, device=vision_embeds.device)
        w_coords = torch.arange(num_patches_per_side, device=vision_embeds.device)
        position_ids = (h_coords[:, None] * self.config.vit_max_num_patch_per_side + w_coords).flatten()
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1).flatten()

        # Add position embeddings
        pos_embeds = self.vit_pos_embed(position_ids)
        pos_embeds = pos_embeds.reshape(batch_size, num_patches, hidden_size)
        # Ensure pos_embeds are on the same device as vision_embeds
        pos_embeds = pos_embeds.to(vision_embeds.device)
        vision_embeds = vision_embeds + pos_embeds

        # Split by image
        return tuple(vision_embeds)

    def get_multimodal_embeddings(self, **kwargs: object) -> MultiModalEmbeddings:
        """Get multimodal embeddings from input."""
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []

        return self._process_image_input(image_input)

    def get_language_model(self) -> nn.Module:
        return self.language_model

    def get_input_embeddings(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        if hasattr(self.language_model, "get_input_embeddings"):
            return self.language_model.get_input_embeddings(input_ids)
        return self.language_model.model.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        """Run forward pass for BAGEL.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a batch.
            positions: Flattened (concatenated) position ids corresponding to a batch.
            intermediate_tensors: Intermediate tensors from prior forward pass.
            inputs_embeds: Optional tensor of input embeddings.
        """
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from checkpoint."""
        skip_prefixes = []
        # Skip vit_pos_embed.pos_embed as it's handled by PositionEmbedding module
        skip_prefixes.append("vit_pos_embed.pos_embed")

        # If visual understanding is disabled, skip vision-related weights
        if self.vit_model is None:
            skip_prefixes.extend(["vit_model.", "connector.", "vit_pos_embed"])

        generation_keywords = [
            "moe_gen",
            "latent_pos_embed",
            "llm2vae",
            "vae2llm",
            "time_embedder",
        ]
        vae_prefixes = [
            "decoder.",
            "encoder.",
        ]  # VAE encoder/decoder, not vision encoder
        filtered_weights = []
        for name, tensor in weights:
            # Skip generation-related keywords
            if any(skip in name for skip in generation_keywords):
                continue
            if any(name.startswith(prefix) for prefix in vae_prefixes):
                continue

            if "patch_embedding.weight" in name and tensor.ndim == 2:
                out_channels = tensor.shape[0]
                in_features = tensor.shape[1]
                patch_size = self.config.vit_config.patch_size
                in_channels = self.config.vit_config.num_channels
                if in_features == in_channels * patch_size * patch_size:
                    tensor = tensor.reshape(out_channels, patch_size, patch_size, in_channels)
                    tensor = tensor.permute(0, 3, 1, 2).contiguous()

            filtered_weights.append((name, tensor))

        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)
        return loader.load_weights(filtered_weights, mapper=self.hf_to_vllm_mapper)
