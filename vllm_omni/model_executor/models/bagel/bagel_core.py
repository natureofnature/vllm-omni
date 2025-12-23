# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn.attention.flex_attention import and_masks, or_masks
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig

# -------------------------------------------------------------------------
# From utils.py
# -------------------------------------------------------------------------


@dataclass
class BagelGenParams:
    num_timesteps: int = 50
    timestep_shift: float = 1.0


def add_special_tokens(tokenizer):
    all_special_tokens = []
    for k, v in tokenizer.special_tokens_map.items():
        if isinstance(v, str):
            all_special_tokens.append(v)
        elif isinstance(v, list):
            all_special_tokens += v

    new_tokens = []

    if "<|im_start|>" not in all_special_tokens:
        new_tokens.append("<|im_start|>")

    if "<|im_end|>" not in all_special_tokens:
        new_tokens.append("<|im_end|>")

    if "<|vision_start|>" not in all_special_tokens:
        new_tokens.append("<|vision_start|>")

    if "<|vision_end|>" not in all_special_tokens:
        new_tokens.append("<|vision_end|>")

    num_new_tokens = tokenizer.add_tokens(new_tokens)
    bos_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    start_of_image = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    end_of_image = tokenizer.convert_tokens_to_ids("<|vision_end|>")

    new_token_ids = dict(
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        start_of_image=start_of_image,
        end_of_image=end_of_image,
    )

    return tokenizer, new_token_ids, num_new_tokens


def create_sparse_mask(document_lens, split_lens, attn_modes, device):
    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    def full_and_noise_mask(b, h, q_idx, kv_idx):
        return (full_and_noise_seq_id[q_idx] == full_and_noise_seq_id[kv_idx]) & (full_and_noise_seq_id[q_idx] >= 0)

    def remove_noise_mask(b, h, q_idx, kv_idx):
        return ~((noise_seq_id[kv_idx] >= 0) & (noise_seq_id[q_idx] != noise_seq_id[kv_idx]))

    def sample_mask(b, h, q_idx, kv_idx):
        return document_id[q_idx] == document_id[kv_idx]

    full_and_noise_tmp = []
    noise_tmp = []

    for i, (length, model) in enumerate(zip(split_lens, attn_modes)):
        value = i if model in ["full", "noise"] else -1
        full_and_noise_tmp.extend([value] * length)
        value_noise = i if model == "noise" else -1
        noise_tmp.extend([value_noise] * length)

    full_and_noise_seq_id = torch.Tensor(full_and_noise_tmp).to(device)
    noise_seq_id = torch.Tensor(noise_tmp).to(device)

    document_id = torch.cat([torch.full((length,), i) for i, length in enumerate(document_lens, start=1)]).to(device)

    return and_masks(or_masks(causal_mask, full_and_noise_mask), remove_noise_mask, sample_mask)


def patchify(image, patch_size):
    p = patch_size
    c, h, w = image.shape
    assert h % p == 0 and w % p == 0
    image = image.reshape(c, h // p, p, w // p, p)
    image = torch.einsum("chpwq->hwpqc", image)
    image = image.reshape(-1, p**2 * c)
    return image


def get_flattened_position_ids_extrapolate(img_h, img_w, patch_size, max_num_patches_per_side):
    num_patches_h, num_patches_w = img_h // patch_size, img_w // patch_size
    coords_h = torch.arange(0, num_patches_h)
    coords_w = torch.arange(0, num_patches_w)
    pos_ids = (coords_h[:, None] * max_num_patches_per_side + coords_w).flatten()
    return pos_ids


def get_flattened_position_ids_interpolate(img_h, img_w, patch_size, max_num_patches_per_side):
    num_patches_h, num_patches_w = img_h // patch_size, img_w // patch_size
    boundaries = torch.arange(1 / max_num_patches_per_side, 1.0, 1 / max_num_patches_per_side)
    fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / num_patches_h)
    fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / num_patches_w)
    bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
    bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)
    pos_ids = (bucket_coords_h[:, None] * max_num_patches_per_side + bucket_coords_w).flatten()
    return pos_ids


# -------------------------------------------------------------------------
# From modeling_utils.py
# -------------------------------------------------------------------------


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class MLPconnector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_act: str):
        super().__init__()
        self.activation_fn = ACT2FN[hidden_act]
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class PositionEmbedding(nn.Module):
    def __init__(self, max_num_patch_per_side, hidden_size):
        super().__init__()
        self.max_num_patch_per_side = max_num_patch_per_side
        self.hidden_size = hidden_size
        self.pos_embed = nn.Parameter(torch.zeros(max_num_patch_per_side**2, hidden_size), requires_grad=False)
        self._init_weights()

    def _init_weights(self):
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, self.max_num_patch_per_side)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

    def forward(self, position_ids):
        return self.pos_embed[position_ids]


# -------------------------------------------------------------------------
# Original bagel_core.py content
# -------------------------------------------------------------------------


class BagelConfig(PretrainedConfig):
    def __init__(
        self,
        llm_config=None,
        vae_config=None,
        latent_patch_size=2,
        max_latent_size=32,
        timestep_shift=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm_config = llm_config
        self.vae_config = vae_config
        self.latent_patch_size = latent_patch_size
        self.max_latent_size = max_latent_size
        self.timestep_shift = timestep_shift


def prepare_prompts(curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids):
    packed_text_ids = list()
    packed_text_position_ids = list()
    text_token_lens = list()
    packed_text_indexes = list()
    packed_key_value_indexes = list()

    curr = 0
    newlens, new_rope = list(), list()
    for prompt, curr_kvlen, curr_position_id in zip(prompts, curr_kvlens, curr_rope):
        packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
        curr += curr_kvlen

        text_ids = tokenizer.encode(prompt)
        text_ids = [new_token_ids["bos_token_id"]] + text_ids + [new_token_ids["eos_token_id"]]
        text_token_lens.append(len(text_ids))
        packed_text_ids.extend(text_ids)
        packed_text_position_ids.extend(range(curr_position_id, curr_position_id + len(text_ids)))
        packed_text_indexes.extend(range(curr, curr + len(text_ids)))
        newlens.append(curr_kvlen + len(text_ids))
        new_rope.append(curr_position_id + len(text_ids))
        curr += len(text_ids)

    generation_input = {
        "text_token_lens": torch.tensor(text_token_lens, dtype=torch.int),
        "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
        "packed_text_position_ids": torch.tensor(packed_text_position_ids, dtype=torch.long),
        "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
        "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
    }

    return generation_input, newlens, new_rope


def prepare_vae_latent(curr_kvlens, curr_rope, image_sizes, new_token_ids, bagel_config: BagelConfig):
    packed_text_ids, packed_text_indexes = list(), list()
    packed_vae_position_ids, packed_vae_token_indexes, packed_init_noises = list(), list(), list()
    packed_position_ids, packed_seqlens, packed_indexes = list(), list(), list()
    packed_key_value_indexes = list()

    latent_downsample = bagel_config.vae_config.downsample * bagel_config.latent_patch_size
    max_latent_size = bagel_config.max_latent_size
    latent_channel = bagel_config.vae_config.z_channels
    latent_patch_size = bagel_config.latent_patch_size

    query_curr = curr = 0
    for (H, W), curr_kvlen, curr_position_id in zip(image_sizes, curr_kvlens, curr_rope):
        packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
        curr += curr_kvlen

        packed_text_ids.append(new_token_ids["start_of_image"])
        packed_text_indexes.append(query_curr)

        packed_indexes.append(curr)
        curr += 1
        query_curr += 1

        vae_position_ids = get_flattened_position_ids_extrapolate(
            H, W, latent_downsample, max_num_patches_per_side=max_latent_size
        )
        packed_vae_position_ids.append(vae_position_ids)

        h, w = H // latent_downsample, W // latent_downsample
        num_image_tokens = h * w

        packed_init_noises.append(torch.randn(num_image_tokens, latent_channel * latent_patch_size**2))
        packed_vae_token_indexes.extend(range(query_curr, query_curr + num_image_tokens))
        packed_seqlens.append(num_image_tokens + 2)

        packed_indexes.extend(range(curr, curr + num_image_tokens))
        curr += num_image_tokens
        query_curr += num_image_tokens

        packed_text_ids.append(new_token_ids["end_of_image"])
        packed_text_indexes.append(query_curr)

        packed_indexes.append(curr)
        curr += 1
        query_curr += 1

        packed_position_ids.extend([curr_position_id] * (num_image_tokens + 2))

    # Construct Output
    generation_input = {
        "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
        "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
        "packed_init_noises": torch.cat(packed_init_noises, dim=0),
        "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
        "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
        "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
        "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
        "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
        "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
    }

    return generation_input
