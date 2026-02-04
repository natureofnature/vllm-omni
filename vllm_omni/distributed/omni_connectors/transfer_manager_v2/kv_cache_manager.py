# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
KV Cache Transfer Manager facade.

This provides a compatible interface with the original OmniKVTransferManager,
but uses the new unified architecture internally.
"""

from collections.abc import Callable
from typing import Any

import torch
from vllm.logger import init_logger

from .core.config import OmniTransferConfig
from .core.connector_manager import OmniConnectorManager
from .core.transport_engine import OmniTransportEngine
from .handlers.kv_cache_handler import KVCacheHandler
from .transfer_manager import OmniTransferManager

logger = init_logger(__name__)


class OmniKVCacheTransferManager:
    """Facade for KV cache transfer operations.

    This class provides backward compatibility with the original
    OmniKVTransferManager while using the new unified architecture.

    Usage:
        # Create from config dict
        manager = OmniKVCacheTransferManager.from_dict(cfg)

        # Handle finished requests
        manager.handle_finished_requests_kv_transfer(
            finished_reqs, kv_caches, block_size, cache_dtype
        )

        # Receive KV cache
        manager.receive_kv_cache(req, target_device)
    """

    def __init__(self, config: OmniTransferConfig):
        self.config = config

        # Build components
        self._connector_manager = OmniConnectorManager(config)
        self._transport = OmniTransportEngine(self._connector_manager, config)
        self._handler = KVCacheHandler()
        self._core_manager = OmniTransferManager(self._transport, self._handler, config)

        # Transport starts on first submit; call start() if explicit startup is needed

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | None) -> "OmniKVCacheTransferManager":
        """Create from config dictionary."""
        config = OmniTransferConfig.from_dict(cfg)
        return cls(config)

    @classmethod
    def from_model_config(cls, model_config: Any) -> "OmniKVCacheTransferManager":
        """Create from model config."""
        config = OmniTransferConfig.from_model_config(model_config)
        return cls(config)

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: Any,
        model_config: Any,
    ) -> "OmniKVCacheTransferManager":
        """Create from vllm config with fallback."""
        config = OmniTransferConfig.from_vllm_config(vllm_config, model_config)
        return cls(config)

    # ============ Properties ============

    @property
    def connector(self):
        """Get the underlying connector."""
        return self._connector_manager.connector

    @property
    def send_stages(self) -> tuple[str | None, str | None]:
        """Get send stages (from_stage, to_stage)."""
        return self.config.get_send_stages()

    @property
    def recv_stages(self) -> tuple[str | None, str | None]:
        """Get recv stages (from_stage, to_stage)."""
        return self.config.get_recv_stages()

    def get_connector(self):
        """Get connector (compatibility wrapper)."""
        return self.connector

    def get_sender_connection_info(self) -> dict[str, Any] | None:
        """Get sender connector's connection info."""
        return self._connector_manager.get_connection_info()

    def update_sender_info(self, sender_info: dict[str, Any]) -> None:
        """Update sender connection info for receiver."""
        self._connector_manager.update_peer_info(sender_info)

    # ============ Main Interface ============

    def handle_finished_requests_kv_transfer(
        self,
        finished_reqs: dict[str, dict[str, Any]],
        kv_caches: list[torch.Tensor],
        block_size: int,
        cache_dtype: str,
        request_id_resolver: Callable[[str], str] | None = None,
    ) -> list[str]:
        """Handle KV cache transfer for finished requests.

        This method extracts KV cache from GPU blocks and transfers them
        to the downstream stage.

        Args:
            finished_reqs: Dict mapping request_id to {block_ids, seq_len}
            kv_caches: List of KV cache tensors per layer
            block_size: Size of each cache block
            cache_dtype: Data type of the cache
            request_id_resolver: Optional function to resolve global request ID

        Returns:
            List of request IDs that were processed
        """
        if not finished_reqs:
            return []

        if not self.config.need_send:
            return list(finished_reqs.keys())

        if not self._connector_manager.is_ready:
            logger.warning("No connector available, skipping KV transfer")
            return list(finished_reqs.keys())

        # Set KV cache context
        self._handler.set_kv_cache_context(kv_caches, block_size, cache_dtype)

        logger.debug(f"Processing KV transfer for {len(finished_reqs)} requests")

        processed_ids = []
        for req_id, data in finished_reqs.items():
            try:
                # Resolve global request ID if available
                transfer_req_id = request_id_resolver(req_id) if request_id_resolver else req_id

                # Prepare raw input
                raw_input = {
                    "block_ids": data.get("block_ids", []),
                    "seq_len": data.get("seq_len", 0),
                }

                # Sync send
                success, size = self._core_manager.sync_send(
                    req_id,
                    raw_input,
                    external_request_id=transfer_req_id,
                )

                if success:
                    logger.info(f"KV transfer OK: {transfer_req_id}, {size} bytes")
                else:
                    logger.error(f"KV transfer FAILED: {transfer_req_id}")

            except Exception as e:
                logger.error(f"Failed KV transfer for {req_id}: {e}")
            finally:
                processed_ids.append(req_id)

        return processed_ids

    @torch.inference_mode()
    def receive_kv_cache_for_request(
        self,
        request_id: str,
        target_device: torch.device | None = None,
    ) -> tuple[dict[str, Any] | None, int]:
        """Receive KV cache for a specific request.

        Args:
            request_id: The request ID to receive KV cache for
            target_device: Optional device to move tensors to

        Returns:
            (data, size) if successful, (None, 0) otherwise
        """
        if not self._connector_manager.is_ready:
            logger.warning("No connector available for receiving KV cache")
            return None, 0

        if not self.config.need_recv:
            logger.debug(f"Skip receiving KV cache for {request_id} (need_recv=False)")
            return None, 0

        # Sync receive raw data
        data, size = self._core_manager.sync_recv_raw(request_id)
        if data is None:
            return None, 0

        # Prepare data (move to target device if needed)
        if target_device is not None:
            data = self._handler.prepare_recv_data(data, target_device)

        return data, size

    def apply_kv_cache_to_request(self, req: Any, data: dict[str, Any]) -> None:
        """Apply received KV cache data to a request object.

        Args:
            req: The request object to apply KV cache to
            data: The received KV cache data dictionary
        """
        from .core.config import TransferContext

        # Create minimal context
        ctx = TransferContext(
            stage_id=self.config.stage_id or 0,
            from_stage="",
            to_stage="",
            request_id=getattr(req, "request_id", ""),
        )

        self._handler.process_recv_data(ctx, data, req)

    def receive_kv_cache(
        self,
        req: Any,
        target_device: torch.device | None = None,
    ) -> bool:
        """Receive KV cache and populate request object (legacy interface).

        Args:
            req: Request object with request_id attribute
            target_device: Optional device to move tensors to

        Returns:
            True if successful, False otherwise
        """
        # Check if request has sender info
        kv_sender_info = getattr(req, "kv_sender_info", None)
        if kv_sender_info:
            self.update_sender_info(kv_sender_info)

        # Get request ID
        request_id = getattr(req, "request_id", None)
        if not request_id and hasattr(req, "request_ids") and req.request_ids:
            request_id = req.request_ids[0]

        if not request_id:
            logger.warning("Request has no ID, cannot receive KV cache")
            return False

        # Set target device
        self._handler.set_target_device(target_device)

        # Sync receive directly to request
        success, _ = self._core_manager.sync_recv(request_id, req)
        return success

    # ============ Lifecycle ============

    def stop(self) -> None:
        """Stop the transport engine."""
        self._transport.stop()
