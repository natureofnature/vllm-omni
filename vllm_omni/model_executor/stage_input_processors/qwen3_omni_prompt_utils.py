# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Qwen3 Omni prompt-length helpers shared across runtime stages."""

from __future__ import annotations

IM_START_TOKEN_ID = 151644
SYSTEM_TOKEN_ID = 8948
USER_TOKEN_ID = 872
ASSISTANT_TOKEN_ID = 77091
ASSISTANT_PROMPT_TAIL_LEN = 9


def compute_talker_prompt_ids_length(prompt_ids: list[int]) -> int:
    """Compute the effective talker prompt length from prompt token ids."""
    im_start_indexes = [i for i, token_id in enumerate(prompt_ids) if token_id == IM_START_TOKEN_ID]
    im_start_indexes.append(len(prompt_ids))
    sum_user_len = 0
    assistant_len = 0
    for i in range(len(im_start_indexes) - 1):
        s = im_start_indexes[i]
        e = im_start_indexes[i + 1]
        if s + 1 >= len(prompt_ids):
            continue
        role = prompt_ids[s + 1]
        if role == SYSTEM_TOKEN_ID:
            continue
        if role == USER_TOKEN_ID:
            sum_user_len += e - s
        elif role == ASSISTANT_TOKEN_ID and i == len(im_start_indexes) - 2:
            assistant_len += ASSISTANT_PROMPT_TAIL_LEN
    return sum_user_len + assistant_len
