# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
KV Cache transfer handler.

Handles extraction and application of KV cache data:
- Extract KV cache from GPU blocks
- Transfer between stages
- Apply received KV cache to request objects
"""

from dataclasses import asdict, dataclass
from typing import Any

import torch
from vllm.logger import init_logger

from ..core.config import TransferContext
from .base import TransferHandler

logger = init_logger(__name__)


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


class KVCacheHandler(TransferHandler):
    """Handler for KV cache transfer operations.

    This handler is responsible for:
    - Building KV cache transfer keys
    - Extracting KV cache from GPU blocks (prepare_send_data)
    - Applying KV cache to request objects (process_recv_data)

    Note: The handler itself is stateless. KV cache context (kv_caches, block_size, etc.)
    should be set via set_kv_cache_context() before each use.
    """

    def __init__(self):
        # KV cache context (set before each extraction)
        self._kv_caches: list[torch.Tensor] = []
        self._block_size: int = 0
        self._cache_dtype: str = ""

        # Target device for received tensors
        self._target_device: torch.device | None = None

    def set_kv_cache_context(
        self,
        kv_caches: list[torch.Tensor],
        block_size: int,
        cache_dtype: str,
    ) -> None:
        """Set KV cache context for extraction.

        This should be called before prepare_send_data() in each inference step.

        Args:
            kv_caches: List of KV cache tensors per layer
            block_size: Size of each cache block
            cache_dtype: Data type of the cache
        """
        self._kv_caches = kv_caches
        self._block_size = block_size
        self._cache_dtype = cache_dtype

    def set_target_device(self, device: torch.device | None) -> None:
        """Set target device for received tensors.

        Args:
            device: Target device (e.g., cuda:0)
        """
        self._target_device = device

    def build_key(self, ctx: TransferContext) -> str:
        """Build KV cache transfer key.

        Format: omni_{from_stage}_to_{to_stage}_kv_cache_{request_id}
        """
        return f"omni_{ctx.from_stage}_to_{ctx.to_stage}_kv_cache_{ctx.request_id}"

    def prepare_send_data(self, ctx: TransferContext, raw_input: Any) -> Any | None:
        """Extract KV cache from GPU blocks.

        Args:
            ctx: Transfer context
            raw_input: Dict with {"block_ids": [...], "seq_len": int}

        Returns:
            KVCacheTransferData if extraction successful, None otherwise
        """
        if not isinstance(raw_input, dict):
            logger.warning(f"Invalid raw_input type: {type(raw_input)}")
            return None

        block_ids = raw_input.get("block_ids", [])
        seq_len = raw_input.get("seq_len", 0)

        if not block_ids:
            logger.warning(f"Request {ctx.request_id} has no block IDs, skipping")
            return None

        if not self._kv_caches:
            logger.warning("KV cache context not set, call set_kv_cache_context() first")
            return None

        return self._extract_kv_cache(
            ctx.request_id,
            block_ids,
            seq_len,
        )

    def process_recv_data(self, ctx: TransferContext, data: Any, request: Any) -> None:
        """Apply received KV cache to request object.

        Args:
            ctx: Transfer context
            data: Received KV cache data dict
            request: Request object to update
        """
        if not isinstance(data, dict):
            logger.warning(f"Invalid data type for KV cache: {type(data)}")
            return

        # Move tensors to target device if specified
        if self._target_device is not None and "layer_blocks" in data:
            self._move_to_device(data["layer_blocks"], self._target_device)

        # Apply to request
        if "layer_blocks" in data:
            from types import SimpleNamespace

            kv_obj = SimpleNamespace(**data["layer_blocks"])
            request.past_key_values = kv_obj

            # BagelPipeline compatibility
            if hasattr(request, "sampling_params") and request.sampling_params is not None:
                request.sampling_params.past_key_values = kv_obj

        if "metadata" in data:
            request.kv_metadata = data["metadata"]

    def prepare_recv_data(
        self,
        data: dict[str, Any],
        target_device: torch.device | None = None,
    ) -> dict[str, Any]:
        """Prepare received KV cache data (e.g., move tensors to device).

        Args:
            data: Raw received data dict
            target_device: Optional target device for tensors

        Returns:
            Prepared data dict
        """
        if target_device is not None and "layer_blocks" in data:
            self._move_to_device(data["layer_blocks"], target_device)
        return data

    def on_send_complete(
        self,
        ctx: TransferContext,
        success: bool,
        size: int,
    ) -> None:
        """Log KV cache send completion."""
        if success:
            logger.debug(f"KV cache sent for {ctx.request_id}: {size} bytes")
        else:
            logger.error(f"KV cache send failed for {ctx.request_id}")

    def on_recv_complete(
        self,
        ctx: TransferContext,
        data: Any,
        size: int,
    ) -> None:
        """Log KV cache recv completion."""
        logger.debug(f"KV cache received for {ctx.request_id}: {size} bytes")

    # ============ Internal Methods ============

    def _extract_kv_cache(
        self,
        req_id: str,
        block_ids: list[int],
        seq_len: int,
    ) -> KVCacheTransferData | None:
        """Extract KV cache from GPU blocks.

        Args:
            req_id: Request identifier
            block_ids: List of block IDs to extract
            seq_len: Sequence length

        Returns:
            KVCacheTransferData if extraction successful, None otherwise
        """
        num_layers = len(self._kv_caches)
        key_cache: list[torch.Tensor | None] = [None] * num_layers
        value_cache: list[torch.Tensor | None] = [None] * num_layers

        for layer_idx, kv_tensor in enumerate(self._kv_caches):
            # Validate block IDs
            # Shape: [2, num_blocks, block_size, n_heads, head_dim]
            max_block = kv_tensor.shape[1] - 1
            valid_ids = [bid for bid in block_ids if 0 <= bid <= max_block]
            if not valid_ids:
                continue

            # Extract and reshape
            # [2, n_blocks, block_size, n_heads, head_dim] -> [2, seq_len, n_heads, head_dim]
            selected = kv_tensor[:, valid_ids]
            n_kv, n_blks, blk_sz, n_heads, d_head = selected.shape
            flat = selected.reshape(n_kv, n_blks * blk_sz, n_heads, d_head)

            # Truncate to actual sequence length
            if seq_len < flat.shape[1]:
                flat = flat[:, :seq_len]

            # Move to CPU
            flat_cpu = flat.detach().cpu().contiguous()
            key_cache[layer_idx] = flat_cpu[0]
            value_cache[layer_idx] = flat_cpu[1]

        if not any(k is not None for k in key_cache):
            return None

        return KVCacheTransferData(
            request_id=req_id,
            layer_blocks={"key_cache": key_cache, "value_cache": value_cache},
            block_ids=block_ids,
            metadata={
                "block_size": self._block_size,
                "num_layers": num_layers,
                "dtype": str(self._cache_dtype),
                "seq_len": seq_len,
            },
        )

    def _move_to_device(
        self,
        layer_blocks: dict[str, list],
        device: torch.device,
    ) -> None:
        """Move tensors in layer_blocks to target device."""
        for cache_list in [
            layer_blocks.get("key_cache", []),
            layer_blocks.get("value_cache", []),
        ]:
            for i, tensor in enumerate(cache_list):
                if isinstance(tensor, torch.Tensor) and tensor.device != device:
                    cache_list[i] = tensor.to(device).contiguous()
