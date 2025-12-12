# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.distributed.omni_connectors.factory import OmniConnectorFactory
from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec

from .qwen2_navit import NaiveCache

logger = init_logger(__name__)


@dataclass
class BagelPrefillState:
    """All context needed by Bagel gen stage to start flow generation."""

    gen_cache: NaiveCache
    gen_kv_lens: list[int]
    gen_ropes: list[int]

    cfg_text_cache: NaiveCache
    cfg_text_kv_lens: list[int]
    cfg_text_ropes: list[int]

    cfg_img_cache: NaiveCache
    cfg_img_kv_lens: list[int]
    cfg_img_ropes: list[int]


def _cache_from_payload(
    num_layers: int, payload: dict[str, Any], device: torch.device
) -> tuple[NaiveCache, list[int], list[int]]:
    """Rebuild NaiveCache from a connector payload."""
    key_cache = payload["key_cache"]
    value_cache = payload["value_cache"]
    kv_lens = payload["kv_lens"]
    ropes = payload["ropes"]

    cache = NaiveCache(num_layers)
    for i in range(num_layers):
        k = key_cache[i]
        v = value_cache[i]
        # payload may hold CPU tensors; move to target device
        cache.key_cache[i] = k.to(device) if torch.is_tensor(k) else None
        cache.value_cache[i] = v.to(device) if torch.is_tensor(v) else None
    return cache, list(kv_lens), list(ropes)


class BagelKVCacheReceiver:
    """KV cache receiver for Bagel gen stage, backed by OmniConnector (e.g. MooncakeConnector).

    Sender side is intentionally not implemented here.
    """

    def __init__(self, od_config: OmniDiffusionConfig, *, num_layers: int, device: torch.device):
        self._num_layers = num_layers
        self._device = device

        if not od_config.kv_cache_connector_name or not od_config.kv_cache_connector_config:
            raise ValueError("kv_cache_connector_name/kv_cache_connector_config must be set to enable KV receive.")
        self._default_from_stage = od_config.kv_cache_from_stage
        self._default_to_stage = od_config.kv_cache_to_stage

        spec = ConnectorSpec(name=od_config.kv_cache_connector_name, extra=od_config.kv_cache_connector_config)
        self._connector = OmniConnectorFactory.create_connector(spec)

    def try_recv(self, req: OmniDiffusionRequest) -> BagelPrefillState | None:
        if not req.kv_cache_from_connector:
            return None

        rid = req.request_id
        if not rid:
            logger.warning("KV receive requested but request_id is missing; ignoring.")
            return None

        from_stage = req.kv_cache_from_stage or self._default_from_stage
        to_stage = req.kv_cache_to_stage or self._default_to_stage
        if not from_stage or not to_stage:
            logger.warning("KV receive requested but from/to stage not set; ignoring.")
            return None

        payload_tuple = self._connector.get(from_stage, to_stage, rid, metadata=req.kv_cache_connector_metadata)
        if not payload_tuple:
            logger.warning("KV receive timed out or empty for request %s (%s -> %s)", rid, from_stage, to_stage)
            return None

        payload, _nbytes = payload_tuple
        if not isinstance(payload, dict):
            logger.warning("KV payload for %s is not a dict; ignoring.", rid)
            return None

        # Expected schema:
        # {
        #   "gen": {"key_cache": [...], "value_cache": [...], "kv_lens": [...], "ropes": [...]},
        #   "cfg_text": {...},
        #   "cfg_img": {...},
        # }
        gen_cache, gen_kv_lens, gen_ropes = _cache_from_payload(self._num_layers, payload["gen"], self._device)
        cfg_text_cache, cfg_text_kv_lens, cfg_text_ropes = _cache_from_payload(
            self._num_layers, payload["cfg_text"], self._device
        )
        cfg_img_cache, cfg_img_kv_lens, cfg_img_ropes = _cache_from_payload(
            self._num_layers, payload["cfg_img"], self._device
        )

        return BagelPrefillState(
            gen_cache=gen_cache,
            gen_kv_lens=gen_kv_lens,
            gen_ropes=gen_ropes,
            cfg_text_cache=cfg_text_cache,
            cfg_text_kv_lens=cfg_text_kv_lens,
            cfg_text_ropes=cfg_text_ropes,
            cfg_img_cache=cfg_img_cache,
            cfg_img_kv_lens=cfg_img_kv_lens,
            cfg_img_ropes=cfg_img_ropes,
        )
