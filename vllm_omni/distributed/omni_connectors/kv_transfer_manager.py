# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified OmniConnector and KV cache transfer management."""

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import torch
from vllm.logger import init_logger

from .factory import OmniConnectorFactory
from .utils.config import ConnectorSpec
from .utils.kv_utils import normalize_layer_kv

logger = init_logger(__name__)

LayerKV = torch.Tensor | tuple[torch.Tensor, torch.Tensor]


@dataclass
class OmniKVCacheConfig:
    """Configuration for OmniKVTransferManager."""

    connector_config: dict[str, Any] | None = None
    from_stage: str | None = None
    to_stage: str | None = None
    stage_id: str | int | None = None
    engine_input_source: list[str | int] | None = None
    need_recv_cache: bool = False
    need_send_cache: bool = False
    recv_timeout: float = 30.0


@dataclass
class KVCacheTransferData:
    """Container for KV cache transfer data."""

    request_id: str
    layer_blocks: dict[str, Any]
    block_ids: list[int]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class OmniKVTransferManager:
    """Unified management for OmniConnector and KV cache transfer.

    This class encapsulates all KV cache related operations:
    - Connector initialization and lazy creation
    - KV cache extraction from GPU blocks
    - KV cache transfer with retry logic
    - KV cache receiving with timeout
    """

    def __init__(self, config: OmniKVCacheConfig):
        self.config = config
        self._connector = None

        # Pre-calculate send stages (from_stage, to_stage)
        self.send_stages = (
            (str(config.from_stage), str(config.to_stage)) if config.from_stage and config.to_stage else (None, None)
        )

        # Pre-calculate receive stages (from_stage, to_stage)
        recv_from = config.from_stage
        if config.engine_input_source:
            recv_from = config.engine_input_source[0]
        elif isinstance(config.stage_id, int):
            recv_from = config.stage_id - 1

        self.recv_stages = (
            (str(recv_from), str(config.stage_id))
            if recv_from is not None and config.stage_id is not None
            else (None, None)
        )

    @classmethod
    def _create(cls, cfg: dict | None) -> "OmniKVTransferManager":
        """Create manager from raw config dict."""
        if not cfg or not isinstance(cfg, dict):
            return cls(OmniKVCacheConfig())
        return cls(
            OmniKVCacheConfig(
                connector_config=cfg.get("connector_config"),
                from_stage=cfg.get("omni_from_stage"),
                to_stage=cfg.get("omni_to_stage"),
                stage_id=cfg.get("stage_id"),
                engine_input_source=cfg.get("engine_input_source", []),
                need_recv_cache=cfg.get("need_recv_cache", False),
                need_send_cache=cfg.get("need_send_cache", False),
                recv_timeout=cfg.get("recv_timeout", 30.0),
            )
        )

    @classmethod
    def from_model_config(cls, config: Any) -> "OmniKVTransferManager":
        """Create from model config (for AR model runner)."""
        return cls._create(getattr(config, "omni_kv_config", None))

    @classmethod
    def from_od_config(cls, config: Any) -> "OmniKVTransferManager":
        """Create from OmniDiffusion config (for diffusion runner)."""
        return cls._create(getattr(config, "omni_kv_config", None))

    @classmethod
    def from_vllm_config(cls, vllm_config: Any, model_config: Any) -> "OmniKVTransferManager":
        """Create from vllm config with fallback to kv_transfer_config."""
        # Primary: omni_kv_config from model_config
        omni_kv = getattr(model_config, "omni_kv_config", None)
        if isinstance(omni_kv, dict):
            return cls._create(omni_kv)

        # Fallback: check kv_transfer_config
        kv_cfg = getattr(vllm_config, "kv_transfer_config", None)
        if kv_cfg:
            direct = getattr(kv_cfg, "omni_connector_config", None)
            if isinstance(direct, dict) and direct:
                return cls._create({"connector_config": direct})
            extra = getattr(kv_cfg, "kv_connector_extra_config", None)
            if isinstance(extra, dict):
                omni = extra.get("omni_connector_config")
                if isinstance(omni, dict) and omni:
                    return cls._create({"connector_config": omni})

        return cls(OmniKVCacheConfig())

    @property
    def connector(self):
        """Lazy initialization of connector."""
        # If a previous initialization attempt failed, don't retry on every access.
        if self._connector is False:
            return None

        if self._connector is None:
            cfg = self.config.connector_config
            if cfg and (c_type := cfg.get("type")):
                try:
                    logger.info(f"Initializing OmniConnector with config: {cfg}")
                    c_extra = {k: v for k, v in cfg.items() if k != "type"}
                    self._connector = OmniConnectorFactory.create_connector(ConnectorSpec(name=c_type, extra=c_extra))
                except Exception as e:
                    logger.exception("Failed to initialize OmniConnector: %s", e)
                    # Cache failure sentinel to avoid repeated initialization attempts in hot paths.
                    self._connector = False

        return self._connector if self._connector else None

    def get_connector(self):
        """Get connector (compatibility wrapper for existing code)."""
        return self.connector

    def handle_finished_requests_kv_transfer(
        self,
        finished_reqs: dict[str, dict[str, Any]],
        kv_caches: list[LayerKV],
        block_size: int,
        cache_dtype: str,
        request_id_resolver: Callable[[str], str] | None = None,
    ) -> list[str]:
        """Handle KV cache transfer for finished requests.

        This method extracts KV cache from GPU blocks and transfers them
        to the downstream stage via the connector.

        Args:
            finished_reqs: Dict mapping request_id to {block_ids, seq_len}
            kv_caches: List of KV cache (tensor or tuple) per layer
            block_size: Size of each cache block
            cache_dtype: Data type of the cache
            request_id_resolver: Optional function to resolve global request ID

        Returns:
            List of request IDs that were processed
        """
        if not finished_reqs:
            return []

        if not self.config.need_send_cache:
            return list(finished_reqs.keys())

        if not self.connector:
            logger.warning("No connector available, skipping KV transfer but freeing resources")
            return list(finished_reqs.keys())

        logger.debug(f"Processing KV transfer for {len(finished_reqs)} requests")

        extracted_ids = []
        for req_id, data in finished_reqs.items():
            try:
                seq_len = data.get("seq_len", 0)
                block_ids = data.get("block_ids", [])
                if not block_ids:
                    logger.warning(f"Request {req_id} has no block IDs, skipping")
                    continue

                custom_metadata = data.get("custom_metadata")

                # Extract KV cache from GPU blocks -> CPU tensors
                kv_data = self._extract_kv_cache(
                    req_id, block_ids, seq_len, kv_caches, block_size, cache_dtype, custom_metadata
                )
                if kv_data:
                    # Resolve global request ID if available
                    transfer_req_id = request_id_resolver(req_id) if request_id_resolver else req_id

                    # Transfer to downstream stage via connector
                    self._transfer_kv_cache(kv_data, transfer_req_id)

            except Exception as e:
                logger.error(f"Failed KV transfer for {req_id}: {e}")
            finally:
                extracted_ids.append(req_id)

        return extracted_ids

    def _extract_kv_cache(
        self,
        req_id: str,
        block_ids: list[int],
        seq_len: int,
        kv_caches: list[LayerKV],
        block_size: int,
        cache_dtype: str,
        custom_metadata: dict[str, Any] | None = None,
    ) -> KVCacheTransferData | None:
        """Extract KV cache from GPU blocks for a single request.

        Args:
            req_id: Request identifier
            block_ids: List of block IDs to extract
            seq_len: Sequence length
            kv_caches: List of KV cache (tensor or tuple) per layer
            block_size: Size of each cache block
            cache_dtype: Data type of the cache
            custom_metadata: Optional custom metadata to include

        Note: If key/value block counts differ, extraction uses only the overlapping
        block range. Extra key/value blocks are ignored, so returned KV may be partial.

        Returns:
            KVCacheTransferData if extraction successful, None otherwise
        """
        num_layers = len(kv_caches)
        key_cache: list[torch.Tensor | None] = [None] * num_layers
        value_cache: list[torch.Tensor | None] = [None] * num_layers

        for layer_idx, layer_kv in enumerate(kv_caches):
            kv_pair = normalize_layer_kv(layer_kv, req_id=req_id, layer_idx=layer_idx)
            if kv_pair is None:
                continue
            key_blocks, value_blocks = kv_pair

            if key_blocks.shape[0] != value_blocks.shape[0]:
                logger.warning(
                    f"Layer {layer_idx} for request {req_id} has mismatched KV block counts: "
                    f"key={key_blocks.shape[0]}, value={value_blocks.shape[0]}; using shared range"
                )

            # Validate block IDs - shape: [num_blocks, block_size, n_heads, head_dim]
            max_block = min(key_blocks.shape[0], value_blocks.shape[0]) - 1
            valid_ids = [bid for bid in block_ids if 0 <= bid <= max_block]
            if not valid_ids:
                continue

            # Extract and reshape: [n_blocks, block_size, n_heads, head_dim]
            # -> [seq_len, n_heads, head_dim]
            selected_k = key_blocks[valid_ids]
            selected_v = value_blocks[valid_ids]
            flat_k = selected_k.flatten(0, 1)
            flat_v = selected_v.flatten(0, 1)
            if seq_len < flat_k.shape[0]:
                flat_k = flat_k[:seq_len]
                flat_v = flat_v[:seq_len]

            # Move to CPU
            key_cache[layer_idx] = flat_k.detach().cpu().contiguous()
            value_cache[layer_idx] = flat_v.detach().cpu().contiguous()

        if not any(k is not None for k in key_cache):
            return None

        return KVCacheTransferData(
            request_id=req_id,
            layer_blocks={"key_cache": key_cache, "value_cache": value_cache},
            block_ids=block_ids,
            metadata={
                "block_size": block_size,
                "num_layers": num_layers,
                "dtype": str(cache_dtype),
                "seq_len": seq_len,
                **(custom_metadata or {}),
            },
        )

    def _normalize_layer_kv(
        self,
        layer_kv: LayerKV,
        req_id: str,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Normalize one layer KV cache to a `(key_blocks, value_blocks)` tuple.

        Args:
            layer_kv: The raw KV cache (tensor or tuple) for the layer
            req_id: Request ID for logging
            layer_idx: Layer index for logging

        Returns:
            Tuple of (key_blocks, value_blocks) if valid, None otherwise
        """
        if isinstance(layer_kv, torch.Tensor):
            if layer_kv.ndim < 3 or layer_kv.shape[0] != 2:
                logger.warning(
                    f"Layer {layer_idx} for request {req_id} has invalid stacked KV shape: "
                    f"expected [2, blocks, block_size, ...], got {tuple(layer_kv.shape)}"
                )
                return None
            key_blocks = layer_kv[0]
            value_blocks = layer_kv[1]
        elif isinstance(layer_kv, tuple):
            if len(layer_kv) != 2:
                logger.warning(
                    f"Layer {layer_idx} for request {req_id} has KV pair length {len(layer_kv)} (expected 2)"
                )
                return None
            key_blocks, value_blocks = layer_kv
            if not isinstance(key_blocks, torch.Tensor) or not isinstance(value_blocks, torch.Tensor):
                logger.warning(f"Layer {layer_idx} for request {req_id} has non-tensor KV pair entries")
                return None
        else:
            logger.warning(f"Layer {layer_idx} for request {req_id} has unsupported KV type {type(layer_kv).__name__}")
            return None
        # ensure key/value blocks are at least 2D for block indexing
        if key_blocks.ndim < 2 or value_blocks.ndim < 2:
            logger.warning(
                f"Layer {layer_idx} for request {req_id} has invalid KV block shape: "
                f"got key={tuple(key_blocks.shape)} value={tuple(value_blocks.shape)}"
            )
            return None

        return key_blocks, value_blocks

    def _build_rank_aware_send_keys(self, request_id: str, from_stage: str, to_stage: str) -> list[str]:
        key_builder = getattr(self, "kv_send_key_builder", None)
        if callable(key_builder):
            keys = list(key_builder(request_id, from_stage, to_stage))
            if keys:
                return keys
        return [f"omni_{from_stage}_to_{to_stage}_kv_cache_{request_id}"]

    def _build_rank_aware_recv_keys(self, request_id: str, from_stage: str, to_stage: str) -> list[str]:
        key_builder = getattr(self, "kv_recv_key_builder", None)
        if callable(key_builder):
            keys = list(key_builder(request_id, from_stage, to_stage))
            if keys:
                return keys
        return [f"omni_{from_stage}_to_{to_stage}_kv_cache_{request_id}"]

    def _merge_received_rank_shards(self, payloads: list[dict[str, Any]]) -> dict[str, Any] | None:
        merger = getattr(self, "kv_payload_merger", None)
        if callable(merger):
            return merger(payloads)
        return payloads[0] if payloads else None

    def _slice_received_rank_shard(self, payload: dict[str, Any] | None) -> dict[str, Any] | None:
        slicer = getattr(self, "kv_payload_slicer", None)
        if callable(slicer):
            return slicer(payload)
        return payload

    def _transfer_kv_cache(self, kv_data: KVCacheTransferData, transfer_req_id: str) -> None:
        """Transfer KV cache data to downstream stage via OmniConnector."""
        from_stage, to_stage = self.send_stages
        if not from_stage or not to_stage:
            raise ValueError("Transfer stages (omni_from_stage, omni_to_stage) not configured")

        data_dict = kv_data.to_dict()
        data_dict["request_id"] = transfer_req_id
        send_keys = self._build_rank_aware_send_keys(transfer_req_id, from_stage, to_stage)

        total_size = 0
        all_succeeded = True
        for put_key in send_keys:
            success, size, _ = self._transfer_with_retry(from_stage, to_stage, put_key, data_dict)
            total_size += size
            all_succeeded = all_succeeded and success

        if all_succeeded:
            logger.info(f"KV transfer OK: {transfer_req_id}, {total_size} bytes across {len(send_keys)} key(s)")
        else:
            logger.error(f"KV transfer FAILED: {transfer_req_id}")

    def _transfer_with_retry(
        self,
        from_stage: str,
        to_stage: str,
        put_key: str,
        data: dict[str, Any],
        max_retries: int = 3,
    ) -> tuple[bool, int, dict[str, Any] | None]:
        """Transfer data with retry and exponential backoff."""
        for attempt in range(max_retries):
            try:
                success, size, metadata = self.connector.put(
                    from_stage=from_stage, to_stage=to_stage, put_key=put_key, data=data
                )
                if success:
                    return success, size, metadata
                logger.warning(f"Transfer attempt {attempt + 1} failed for {put_key}")
            except Exception as e:
                logger.warning(f"Transfer attempt {attempt + 1} exception for {put_key}: {e}")

            if attempt < max_retries - 1:
                time.sleep(0.1 * (2**attempt))

        return False, 0, None

    @torch.inference_mode()
    def receive_kv_cache_for_request(
        self,
        request_id: str,
        target_device: torch.device | None = None,
    ) -> tuple[dict[str, Any] | None, int]:
        """Receive KV cache for a specific request.

        This implements the receiving logic from gpu_diffusion_model_runner.py.

        Args:
            request_id: The request ID to receive KV cache for
            target_device: Optional device to move tensors to

        Returns:
            Tuple of (data dict, size) if successful, (None, 0) otherwise
        """
        if not self.connector:
            logger.warning("No connector available for receiving KV cache")
            return None, 0

        from_stage, to_stage = self.recv_stages
        if not from_stage or not to_stage:
            logger.warning("Receive stages not configured")
            return None, 0

        # Check if we should receive KV cache based on config
        if not self.config.need_recv_cache:
            logger.info(f"Skip receiving KV cache for {request_id} (need_recv_cache=False)")
            return None, 0

        timeout = self.config.recv_timeout
        start_time = time.time()
        recv_keys = self._build_rank_aware_recv_keys(request_id, from_stage, to_stage)
        pending_keys = list(recv_keys)
        received_payloads: dict[str, tuple[dict[str, Any], int]] = {}

        logger.info(
            "Wait for KV cache for request %s from stage %s to %s via %s key(s)...",
            request_id,
            from_stage,
            to_stage,
            len(recv_keys),
        )

        try:
            while True:
                for get_key in list(pending_keys):
                    result = self.connector.get(
                        from_stage=from_stage,
                        to_stage=to_stage,
                        get_key=get_key,
                    )
                    if not result:
                        continue
                    data, size = result
                    received_payloads[get_key] = (data, size)
                    pending_keys.remove(get_key)

                if not pending_keys and received_payloads:
                    ordered_payloads = [received_payloads[key][0] for key in recv_keys if key in received_payloads]
                    data = self._merge_received_rank_shards(ordered_payloads)
                    data = self._slice_received_rank_shard(data)
                    size = sum(received_payloads[key][1] for key in recv_keys if key in received_payloads)
                    logger.info(
                        "Successfully received KV cache for %s, %s bytes across %s key(s)",
                        request_id,
                        size,
                        len(ordered_payloads),
                    )

                    if target_device is not None and isinstance(data, dict) and "layer_blocks" in data:
                        layer_blocks = data["layer_blocks"]
                        for cache_list in [
                            layer_blocks.get("key_cache", []),
                            layer_blocks.get("value_cache", []),
                        ]:
                            for i, tensor in enumerate(cache_list):
                                if isinstance(tensor, torch.Tensor) and tensor.device != target_device:
                                    cache_list[i] = tensor.to(target_device).contiguous()

                    return data, size

                if time.time() - start_time > timeout:
                    logger.error(
                        "Timeout waiting for KV cache for request %s after %ss; missing keys=%s",
                        request_id,
                        timeout,
                        pending_keys,
                    )
                    return None, 0

                time.sleep(0.5)

        except Exception as e:
            logger.exception("Error receiving KV cache for %s: %s", request_id, e)
            return None, 0

    def apply_kv_cache_to_request(self, req: Any, data: dict[str, Any]) -> None:
        """Apply received KV cache data to a request object.

        Args:
            req: The request object to apply KV cache to
            data: The received KV cache data dictionary
        """
        if isinstance(data, dict) and "layer_blocks" in data:
            layer_blocks = data["layer_blocks"]
            from types import SimpleNamespace

            kv_obj = SimpleNamespace(**layer_blocks)
            req.past_key_values = kv_obj

            # [Omni] Also attach to sampling_params for BagelPipeline compatibility
            # BagelPipeline checks req.sampling_params.past_key_values
            if hasattr(req, "sampling_params") and req.sampling_params is not None:
                req.sampling_params.past_key_values = kv_obj

        if "metadata" in data:
            req.kv_metadata = data["metadata"]
            if hasattr(req, "sampling_params") and req.sampling_params is not None:
                req.sampling_params.kv_metadata = data["metadata"]

    # Legacy compatibility method
    def receive_kv_cache(self, req: Any, target_device: torch.device | None = None) -> bool:
        """Receive KV cache and populate request object (legacy interface).

        Args:
            req: Request object with request_id attribute
            target_device: Optional device to move tensors to

        Returns:
            True if successful, False otherwise
        """
        request_id = getattr(req, "request_id", None)
        if not request_id and hasattr(req, "request_ids") and req.request_ids:
            # Adaptation for new OmniDiffusionRequest which has list of prompts/ids
            request_id = req.request_ids[0]

        if not request_id:
            logger.warning("Request has no ID, cannot receive KV cache")
            return False

        data, size = self.receive_kv_cache_for_request(request_id, target_device)
        if data:
            self.apply_kv_cache_to_request(req, data)
            return True
        return False

    def receive_multi_kv_cache(
        self,
        req: Any,
        cfg_kv_collect_func: Callable | None = None,
        target_device: torch.device | None = None,
    ) -> bool:
        """Receive primary KV cache and optional CFG companion KV caches.

        First receives the primary KV cache (existing logic). Then, if the
        request carries cfg_kv_request_ids and a model-specific
        cfg_kv_collect_func is provided, calls it to fetch and attach the
        companion KV caches to sampling_params.

        Args:
            req: Request object with request_id and sampling_params.
            cfg_kv_collect_func: Model-specific function for collecting
                CFG KV caches. Signature:
                (request_id, cfg_request_ids, kv_transfer_manager, target_device)
                -> dict[str, Any]
            target_device: Device to move tensors to.

        Returns:
            True if primary KV cache was received successfully.
        """
        primary_ok = self.receive_kv_cache(req, target_device)

        cfg_ids = getattr(getattr(req, "sampling_params", None), "cfg_kv_request_ids", None)
        if cfg_ids and cfg_kv_collect_func:
            request_id = getattr(req, "request_id", None) or (
                req.request_ids[0] if hasattr(req, "request_ids") and req.request_ids else None
            )
            try:
                cfg_kvs = cfg_kv_collect_func(
                    request_id,
                    cfg_ids,
                    self,
                    target_device,
                )
                if cfg_kvs and hasattr(req, "sampling_params") and req.sampling_params is not None:
                    for key, value in cfg_kvs.items():
                        setattr(req.sampling_params, key, value)
                    logger.info("Applied CFG KV caches: %s", list(cfg_kvs.keys()))
            except Exception:
                logger.exception("Failed to collect CFG KV caches for %s", request_id)

        return primary_ok
