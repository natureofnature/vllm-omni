# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import PIL.Image


def pil_img2rgb(image: PIL.Image.Image) -> PIL.Image.Image:
    """Match Bagel's behavior: ensure RGB, handling alpha by compositing on white."""
    if image.mode == "RGBA" or image.info.get("transparency", None) is not None:
        image = image.convert("RGBA")
        white = PIL.Image.new(mode="RGB", size=image.size, color=(255, 255, 255))
        white.paste(image, mask=image.split()[3])
        return white
    return image.convert("RGB")


def add_special_tokens(tokenizer) -> tuple[Any, dict[str, int], int]:
    """Add Bagel's special tokens if missing; returns (tokenizer, new_token_ids, num_new_tokens)."""
    all_special_tokens: list[str] = []
    for _, v in getattr(tokenizer, "special_tokens_map", {}).items():
        if isinstance(v, str):
            all_special_tokens.append(v)
        elif isinstance(v, list):
            all_special_tokens += v

    new_tokens: list[str] = []
    for tok in ["<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>"]:
        if tok not in all_special_tokens:
            new_tokens.append(tok)

    num_new_tokens = tokenizer.add_tokens(new_tokens)

    new_token_ids = dict(
        bos_token_id=tokenizer.convert_tokens_to_ids("<|im_start|>"),
        eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
        start_of_image=tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        end_of_image=tokenizer.convert_tokens_to_ids("<|vision_end|>"),
    )
    return tokenizer, new_token_ids, num_new_tokens


@dataclass
class BagelGenParams:
    # Default inference knobs (mirrors Bagel inferencer defaults)
    cfg_text_scale: float = 4.0
    cfg_img_scale: float = 1.5
    cfg_interval: tuple[float, float] = (0.4, 1.0)
    cfg_renorm_min: float = 0.0
    cfg_renorm_type: str = "global"
    num_timesteps: int = 50
    timestep_shift: float = 3.0
