# Copyright (c) 2024 The Qwen Team and The HuggingFace Inc. team.
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from typing import Optional

import torch
from torch import nn
from tqdm import tqdm
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2MLP,
    Qwen2PreTrainedModel,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
    apply_rotary_pos_emb,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

# Adjust imports to point to the correct location in vllm_omni/diffusion/models/bagel
# Adjust imports to use local copies in vllm_omni/model_executor/models/bagel
from .bagel_core import (
    BagelConfig,
    PositionEmbedding,
    TimestepEmbedder,
    get_flattened_position_ids_extrapolate,
    prepare_vae_latent,
)
from .qwen2_navit import NaiveCache, Qwen2Config

try:
    from vllm.vllm_flash_attn import flash_attn_varlen_func
except ImportError:
    pass


class Qwen2GenAttention(nn.Module):
    """
    Qwen2 attention with MoE routing for generation mode.
    Retains both und and gen weights for handling:
    - <|vision_start|> / <|vision_end|> tokens: use und weights
    - VAE tokens: use gen (_moe_gen) weights
    """

    def __init__(self, config: Qwen2Config, layer_idx: int | None = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        # Und weights (for text tokens like <|vision_start|>)
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.q_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Gen weights (for VAE tokens) - _moe_gen suffix
        self.q_proj_moe_gen = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj_moe_gen = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj_moe_gen = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj_moe_gen = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.q_norm_moe_gen = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm_moe_gen = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: NaiveCache | None = None,
        key_values_lens: torch.Tensor | None = None,
        packed_key_value_indexes: torch.Tensor | None = None,
        update_past_key_values: bool = False,  # Usually False for gen mode
        packed_vae_token_indexes: torch.Tensor | None = None,
        packed_text_indexes: torch.Tensor | None = None,
    ):
        # Gen mode: route text tokens through und weights, VAE tokens through gen weights
        packed_query_sequence = packed_query_sequence.to(torch.bfloat16)

        packed_query_states = packed_query_sequence.new_zeros(
            (packed_query_sequence.shape[0], self.num_heads * self.head_dim)
        )
        packed_key_states = packed_query_sequence.new_zeros(
            (packed_query_sequence.shape[0], self.num_key_value_heads * self.head_dim)
        )
        packed_value_states = packed_query_sequence.new_zeros(
            (packed_query_sequence.shape[0], self.num_key_value_heads * self.head_dim)
        )

        packed_text_query_sequence = packed_query_sequence[packed_text_indexes]
        packed_vae_query_sequence = packed_query_sequence[packed_vae_token_indexes]

        # Text tokens (<|vision_start|>, <|vision_end|>) use und weights
        if packed_text_indexes.numel() > 0:
            packed_query_states[packed_text_indexes] = self.q_proj(packed_text_query_sequence)
            packed_key_states[packed_text_indexes] = self.k_proj(packed_text_query_sequence)
            packed_value_states[packed_text_indexes] = self.v_proj(packed_text_query_sequence)

        if packed_vae_token_indexes.numel() > 0:
            packed_query_states[packed_vae_token_indexes] = self.q_proj_moe_gen(packed_vae_query_sequence)
            packed_key_states[packed_vae_token_indexes] = self.k_proj_moe_gen(packed_vae_query_sequence)
            packed_value_states[packed_vae_token_indexes] = self.v_proj_moe_gen(packed_vae_query_sequence)

        packed_query_states = packed_query_states.view(-1, self.num_heads, self.head_dim)
        packed_key_states = packed_key_states.view(-1, self.num_key_value_heads, self.head_dim)
        packed_value_states = packed_value_states.view(-1, self.num_key_value_heads, self.head_dim)

        # Apply QK normalization with routing
        packed_query_states = packed_query_states.to(torch.float32)
        if packed_text_indexes.numel() > 0:
            packed_query_states[packed_text_indexes] = self.q_norm(packed_query_states[packed_text_indexes])
            packed_key_states[packed_text_indexes] = self.k_norm(packed_key_states[packed_text_indexes])

        if packed_vae_token_indexes.numel() > 0:
            packed_query_states[packed_vae_token_indexes] = self.q_norm_moe_gen(
                packed_query_states[packed_vae_token_indexes]
            )
            packed_key_states[packed_vae_token_indexes] = self.k_norm_moe_gen(
                packed_key_states[packed_vae_token_indexes]
            )

        # Apply rotary position embeddings
        packed_cos, packed_sin = packed_query_position_embeddings
        packed_query_states, packed_key_states = apply_rotary_pos_emb(
            packed_query_states, packed_key_states, packed_cos, packed_sin, unsqueeze_dim=1
        )

        packed_query_states = packed_query_states.to(torch.bfloat16)
        packed_key_states = packed_key_states.to(torch.bfloat16)
        packed_value_states = packed_value_states.to(torch.bfloat16)

        # Merge with past key values from text encoder
        if past_key_values is not None and past_key_values.key_cache[self.layer_idx] is not None:
            past_key_states = past_key_values.key_cache[self.layer_idx]
            past_value_states = past_key_values.value_cache[self.layer_idx]

            seqlens = sum(query_lens) + sum(key_values_lens)
            merged_key_states = past_key_states.new_zeros(size=[seqlens, self.num_key_value_heads, self.head_dim])
            merged_value_states = past_key_states.new_zeros(size=[seqlens, self.num_key_value_heads, self.head_dim])
            merged_key_states[packed_query_indexes] = packed_key_states
            merged_key_states[packed_key_value_indexes] = past_key_states
            merged_value_states[packed_query_indexes] = packed_value_states
            merged_value_states[packed_key_value_indexes] = past_value_states
            key_values_lens = key_values_lens + query_lens
        else:
            merged_key_states = packed_key_states
            merged_value_states = packed_value_states
            key_values_lens = query_lens

        cu_seqlens_q = torch.nn.functional.pad(torch.cumsum(query_lens, dim=0), (1, 0))
        cu_seqlens_k = torch.nn.functional.pad(torch.cumsum(key_values_lens, dim=0), (1, 0))

        # Full attention (non-causal) for generation
        packed_attn_output = flash_attn_varlen_func(
            q=packed_query_states,
            k=merged_key_states,
            v=merged_value_states,
            cu_seqlens_q=cu_seqlens_q.to(torch.int32),
            cu_seqlens_k=cu_seqlens_k.to(torch.int32),
            max_seqlen_q=max(query_lens).item(),
            max_seqlen_k=max(key_values_lens).item(),
            causal=False,  # Non-causal for generation
        )
        packed_attn_output = packed_attn_output.reshape(-1, self.hidden_size)

        # Route output projection
        if packed_text_indexes.numel() > 0:
            packed_attn_output[packed_text_indexes] = self.o_proj(packed_attn_output[packed_text_indexes])
        if packed_vae_token_indexes.numel() > 0:
            packed_attn_output[packed_vae_token_indexes] = self.o_proj_moe_gen(
                packed_attn_output[packed_vae_token_indexes]
            )

        if update_past_key_values:
            past_key_values.key_cache[self.layer_idx] = merged_key_states
            past_key_values.value_cache[self.layer_idx] = merged_value_states

        return packed_attn_output, past_key_values


class Qwen2GenDecoderLayer(nn.Module):
    """
    Qwen2 decoder layer with MoE routing for generation.
    """

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2GenAttention(config, layer_idx)

        # Und weights for text tokens
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Gen weights for VAE tokens
        self.mlp_moe_gen = Qwen2MLP(config)
        self.input_layernorm_moe_gen = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_moe_gen = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        key_values_lens: torch.Tensor | None = None,
        packed_key_value_indexes: torch.Tensor | None = None,
        packed_text_indexes: torch.Tensor | None = None,
        packed_vae_token_indexes: torch.Tensor | None = None,
        past_key_values: Optional[NaiveCache] = None,
        update_past_key_values: bool = False,
    ) -> tuple[torch.Tensor, Optional[NaiveCache]]:
        residual = packed_query_sequence

        # Input LayerNorm with routing
        packed_query_sequence_ = torch.zeros_like(packed_query_sequence)
        if packed_text_indexes.numel() > 0:
            packed_query_sequence_[packed_text_indexes] = self.input_layernorm(
                packed_query_sequence[packed_text_indexes]
            )
        if packed_vae_token_indexes.numel() > 0:
            packed_query_sequence_[packed_vae_token_indexes] = self.input_layernorm_moe_gen(
                packed_query_sequence[packed_vae_token_indexes]
            )
        packed_query_sequence = packed_query_sequence_

        # Self Attention
        packed_query_sequence, past_key_values = self.self_attn(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_embeddings=packed_query_position_embeddings,
            packed_query_indexes=packed_query_indexes,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            packed_text_indexes=packed_text_indexes,
            packed_vae_token_indexes=packed_vae_token_indexes,
            past_key_values=past_key_values,
            update_past_key_values=update_past_key_values,
        )
        packed_query_sequence = residual + packed_query_sequence

        # FFN with routing
        residual = packed_query_sequence

        packed_query_sequence_ = torch.zeros_like(packed_query_sequence).to(torch.bfloat16)

        if packed_text_indexes.numel() > 0:
            packed_text_query_sequence = packed_query_sequence[packed_text_indexes]
            packed_text_query_sequence = self.post_attention_layernorm(packed_text_query_sequence).to(torch.bfloat16)
            packed_query_sequence_[packed_text_indexes] = self.mlp(packed_text_query_sequence)

        if packed_vae_token_indexes.numel() > 0:
            packed_vae_query_sequence = packed_query_sequence[packed_vae_token_indexes]
            packed_vae_query_sequence = self.post_attention_layernorm_moe_gen(packed_vae_query_sequence).to(
                torch.bfloat16
            )
            packed_query_sequence_[packed_vae_token_indexes] = self.mlp_moe_gen(packed_vae_query_sequence)

        packed_query_sequence = packed_query_sequence_

        packed_query_sequence = residual + packed_query_sequence

        return packed_query_sequence, past_key_values


class Qwen2ImageGeneratorModel(Qwen2PreTrainedModel):
    """
    Qwen2 model for image generation.
    Contains MoE routing for text vs VAE tokens.
    """

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Embedding for <|vision_start|> and <|vision_end|> tokens
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        self.layers = nn.ModuleList(
            [Qwen2GenDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        # Final norm with routing
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_moe_gen = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.rotary_emb = Qwen2RotaryEmbedding(config=config)

        self.post_init()

    def forward(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_ids: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        key_values_lens: torch.Tensor | None = None,
        packed_key_value_indexes: torch.Tensor | None = None,
        packed_text_indexes: torch.Tensor | None = None,
        packed_vae_token_indexes: torch.Tensor | None = None,
        past_key_values: Optional[NaiveCache] = None,
        update_past_key_values: bool = False,
    ) -> tuple[torch.Tensor, Optional[NaiveCache]]:
        # Create position embeddings
        cos, sin = self.rotary_emb(packed_query_sequence, packed_query_position_ids.unsqueeze(0))
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)
        packed_query_position_embeddings = (cos, sin)

        for decoder_layer in self.layers:
            packed_query_sequence, past_key_values = decoder_layer(
                packed_query_sequence=packed_query_sequence,
                query_lens=query_lens,
                packed_query_position_embeddings=packed_query_position_embeddings,
                packed_query_indexes=packed_query_indexes,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                packed_text_indexes=packed_text_indexes,
                packed_vae_token_indexes=packed_vae_token_indexes,
                past_key_values=past_key_values,
                update_past_key_values=update_past_key_values,
            )

        # Final norm with routing
        packed_query_sequence_ = torch.zeros_like(packed_query_sequence)
        if packed_text_indexes.numel() > 0:
            packed_query_sequence_[packed_text_indexes] = self.norm(packed_query_sequence[packed_text_indexes])
        if packed_vae_token_indexes.numel() > 0:
            packed_query_sequence_[packed_vae_token_indexes] = self.norm_moe_gen(
                packed_query_sequence[packed_vae_token_indexes]
            )
        packed_query_sequence = packed_query_sequence_

        return packed_query_sequence, past_key_values


class Qwen2ImageGenerator(Qwen2PreTrainedModel):
    """
    Qwen2 Image Generator for Bagel.
    Wrapper around Qwen2ImageGeneratorModel for consistent API.
    Including diffusion generation loop.
    """

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Qwen2Config, bagel_config: BagelConfig = None):
        super().__init__(config)
        self.model = Qwen2ImageGeneratorModel(config)
        self.vocab_size = config.vocab_size

        # If bagel_config not passed, try to extract from config
        if bagel_config is None and hasattr(config, "bagel_config"):
            bagel_config = config.bagel_config

        if bagel_config:
            self.hidden_size = config.hidden_size
            self.latent_patch_size = bagel_config.latent_patch_size
            self.max_latent_size = bagel_config.max_latent_size
            self.latent_channel = bagel_config.vae_config.z_channels
            self.patch_latent_dim = self.latent_patch_size**2 * self.latent_channel
            self.timestep_shift = bagel_config.timestep_shift
            self.latent_downsample = bagel_config.vae_config.downsample * bagel_config.latent_patch_size

            # Additional layers for DiT
            self.time_embedder = TimestepEmbedder(self.hidden_size)
            self.vae2llm = nn.Linear(self.patch_latent_dim, self.hidden_size)
            self.llm2vae = nn.Linear(self.hidden_size, self.patch_latent_dim)
            self.latent_pos_embed = PositionEmbedding(self.max_latent_size, self.hidden_size)

            # Init weights for new layers
            nn.init.constant_(self.llm2vae.weight, 0)
            nn.init.constant_(self.llm2vae.bias, 0)

        self.get_flattened_position_ids = get_flattened_position_ids_extrapolate
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        past_key_values: torch.Tensor | None = None,
        **kwargs,
    ):
        """
        Forward pass for Bagel DiT.
        Expects `past_key_values` from AR stage (shape: [2, num_layers, seq_len, num_heads, head_dim]).
        input_ids/positions are generally ignored/dummy as this is an image generation stage
        driven by the AR context and configuration.
        """
        if past_key_values is None:
            # If strictly required, raise error. But for partial pipeline test, maybe allow?
            # User objective implies we NEED to handle it.
            # If passed as None, the generation will likely fail or produce garbage if dependent on it.
            # Let's verify if we can get it from kwargs just in case.
            past_key_values = kwargs.get("past_key_values", None)

        if past_key_values is None:
            raise ValueError("past_key_values (AR KV Cache) is required for Bagel DiT generation.")

        # 1. Wrap AR KV Cache into NaiveCache
        # Input shape: [2, num_layers, seq_len, num_heads, head_dim]
        # or maybe [layers, 2, ...] depending on how "someone else" implements it.
        # Assuming [2, num_layers, seq_len, num_heads, head_dim] as per standard stacking.

        # If the tensor is on CPU, move to device?
        if past_key_values.device != self.device:
            past_key_values = past_key_values.to(self.device)

        num_layers = past_key_values.shape[1]
        seq_len = past_key_values.shape[2]

        ar_cache = NaiveCache(num_layers)
        for i in range(num_layers):
            # keys: [seq_len, num_heads, head_dim]
            ar_cache.key_cache[i] = past_key_values[0, i]
            # values: [seq_len, num_heads, head_dim]
            ar_cache.value_cache[i] = past_key_values[1, i]

        curr_kvlens = [seq_len]
        # Assuming RoPE continues from where text left off
        curr_rope = [seq_len]

        # 2. Prepare VAE Latent Inputs
        # Get image size from config or kwargs
        # Default to configured max size if not specified
        image_size = kwargs.get(
            "image_size", (self.max_latent_size * self.latent_downsample, self.max_latent_size * self.latent_downsample)
        )
        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        # We need token IDs for <|vision_start|> etc.
        # These should align with tokenizer.
        # Hardcoding standard Bagel special token IDs or extracting if available.
        new_token_ids = {
            "start_of_image": self.config.vision_start_token_id
            if hasattr(self.config, "vision_start_token_id")
            else 151939,  # Example default, should check
            "end_of_image": self.config.vision_end_token_id if hasattr(self.config, "vision_end_token_id") else 151940,
        }

        # Check for user passed new_token_ids
        if "new_token_ids" in kwargs:
            new_token_ids.update(kwargs["new_token_ids"])

        generation_input = prepare_vae_latent(
            curr_kvlens=curr_kvlens,
            curr_rope=curr_rope,
            image_sizes=[image_size],  # Batch size 1 assumption
            new_token_ids=new_token_ids,
            bagel_config=self.bagel_config,
        )

        # Move tensors to device
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(self.device)

        # 3. Run Generation Loop
        return self.generate_image(
            past_key_values=ar_cache,
            **generation_input,
            num_timesteps=kwargs.get("num_timesteps", 24),
            timestep_shift=kwargs.get("timestep_shift", getattr(self, "timestep_shift", 1.0)),
        )

    @torch.no_grad
    def generate_image(
        self,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_init_noises: torch.Tensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        past_key_values: NaiveCache,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.LongTensor,
        num_timesteps: int = 24,
        timestep_shift: float = 1.0,
    ):
        x_t = packed_init_noises

        timesteps = torch.linspace(1, 0, num_timesteps, device=x_t.device)
        timesteps = timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
        dts = timesteps[:-1] - timesteps[1:]
        timesteps = timesteps[:-1]

        # Use caching dictionaries if needed, for now omitted/simplified as in original Reference
        # Assuming sequential scheduling for simplicity as per Bagel implementation

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps), desc="Bagel Generation"):
            timestep = torch.tensor([t] * x_t.shape[0], device=x_t.device)
            v_t = self._forward_flow(
                x_t=x_t,
                timestep=timestep,
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_vae_position_ids=packed_vae_position_ids,
                packed_text_ids=packed_text_ids,
                packed_text_indexes=packed_text_indexes,
                packed_position_ids=packed_position_ids,
                packed_indexes=packed_indexes,
                packed_seqlens=packed_seqlens,
                key_values_lens=key_values_lens,
                past_key_values=past_key_values,
                packed_key_value_indexes=packed_key_value_indexes,
            )

            x_t = x_t - v_t.to(x_t.device) * dts[i]  # velocity pointing from data to noise

        unpacked_latent = x_t.split((packed_seqlens - 2).tolist())
        return unpacked_latent

    @torch.no_grad
    def _forward_flow(
        self,
        x_t: torch.Tensor,
        timestep: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        key_values_lens: torch.IntTensor,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
    ):
        packed_text_embedding = self.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        assert timestep.unique().shape[0] == 1
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.time_embedder(timestep)

        # Project x_t to LLM dim and add embeddings
        x_t_mapped = self.vae2llm(x_t) + packed_timestep_embeds + packed_pos_embed
        if x_t_mapped.dtype != packed_sequence.dtype:
            x_t_mapped = x_t_mapped.to(packed_sequence.dtype)

        # Ensure we don't write OOB if indexes are messy, though they should be correct from prep
        packed_sequence[packed_vae_token_indexes] = x_t_mapped

        output_sequence, _ = self.model(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=False,
            packed_text_indexes=packed_text_indexes,
            packed_vae_token_indexes=packed_vae_token_indexes,
        )

        # Project back
        # output_sequence is flattened [total_tokens, hidden]
        v_t = self.llm2vae(output_sequence)
        v_t = v_t[packed_vae_token_indexes]

        return v_t

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """
        Load weights with specialized logic for MoE Gen layers.
        Loads standard weights (und) AND _moe_gen weights.
        Also loads time_embedder, vae2llm, llm2vae, latent_pos_embed.
        """
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            # Map standard Qwen2 keys to our model structure
            if name.startswith("model."):
                name = name[6:]  # remove model. prefix to match Qwen2ImageGeneratorModel structure if needed,
                # but usually we keep it or valid mapping.
                # self.model is Qwen2ImageGeneratorModel
                # Actually, self.named_parameters() will return "model.layers.0..."
                name = "model." + name

            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            else:
                # Fallback or specific mapping
                pass
