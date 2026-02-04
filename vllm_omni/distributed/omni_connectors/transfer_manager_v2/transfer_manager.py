# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unified Transfer Manager.

Combines:
- ConnectorManager (Layer 1)
- TransportEngine (Layer 2)
- TransferHandler (Layer 3)

Into a single interface for managing data transfers.
"""

from typing import Any

from vllm.logger import init_logger

from .core.config import OmniTransferConfig, TransferContext
from .core.connector_manager import OmniConnectorManager
from .core.transport_engine import OmniTransportEngine
from .handlers.base import TransferHandler

logger = init_logger(__name__)


class OmniTransferManager:
    """Unified transfer manager with pluggable handlers.

    This class provides a unified interface for both KV cache and chunk transfers.
    The specific behavior is determined by the handler.

    Usage:
        # Create components
        config = OmniTransferConfig.from_dict(cfg)
        connector_mgr = OmniConnectorManager(config)
        transport = OmniTransportEngine(connector_mgr, config)
        handler = KVCacheHandler()  # or ChunkHandler()

        # Create manager
        manager = OmniTransferManager(transport, handler, config)

        # Submit transfers
        manager.submit_send(request_id, raw_data)
        manager.submit_recv(request_id, request_obj)

        # Poll for results
        manager.poll()
        finished = manager.get_finished_recvs()
    """

    def __init__(
        self,
        transport: OmniTransportEngine,
        handler: TransferHandler,
        config: OmniTransferConfig,
    ):
        self.transport = transport
        self.handler = handler
        self.config = config

        # Request tracking
        # task_id -> TransferContext
        self._pending_send_contexts: dict[str, TransferContext] = {}
        # task_id -> (TransferContext, request)
        self._pending_recv_contexts: dict[str, tuple[TransferContext, Any]] = {}

        # Finished request IDs
        self._finished_sends: set[str] = set()
        self._finished_recvs: set[str] = set()

        # Chunk counters for multi-chunk transfers
        # request_id -> chunk_id
        self._send_chunk_counts: dict[str, int] = {}
        self._recv_chunk_counts: dict[str, int] = {}

        # Pre-calculate stages
        self._send_from, self._send_to = config.get_send_stages()
        self._recv_from, self._recv_to = config.get_recv_stages()

    @property
    def connector_manager(self) -> OmniConnectorManager:
        """Get the connector manager."""
        return self.transport.connector_manager

    @property
    def is_ready(self) -> bool:
        """Check if the manager is ready for transfers."""
        return self.connector_manager.is_ready

    # ============ Send Interface ============

    def submit_send(
        self,
        request_id: str,
        raw_input: Any,
        external_request_id: str | None = None,
    ) -> bool:
        """Submit a send task.

        Args:
            request_id: Internal request ID
            raw_input: Raw input data (handler-specific format)
            external_request_id: External request ID for key building

        Returns:
            True if task was submitted, False if skipped
        """
        if not self._send_from or not self._send_to:
            logger.warning("Send stages not configured")
            return False

        # Get chunk ID for this request
        chunk_id = self._send_chunk_counts.get(request_id, 0)

        # Build context
        ctx = TransferContext(
            stage_id=self.config.stage_id or 0,
            from_stage=self._send_from,
            to_stage=self._send_to,
            request_id=request_id,
            external_request_id=external_request_id,
            chunk_id=chunk_id,
        )

        # Prepare data via handler
        data = self.handler.prepare_send_data(ctx, raw_input)
        if data is None:
            # Handler decided to skip (e.g., buffering)
            return False

        # Build key via handler
        key = self.handler.build_key(ctx)

        # Convert to dict if needed
        if hasattr(data, "to_dict"):
            data = data.to_dict()

        # Submit to transport
        task_id = self.transport.submit_send(
            self._send_from,
            self._send_to,
            key,
            data,
        )

        # Track context
        self._pending_send_contexts[task_id] = ctx

        # Increment chunk counter
        self._send_chunk_counts[request_id] = chunk_id + 1

        logger.debug(f"Submitted send: {key}")
        return True

    def sync_send(
        self,
        request_id: str,
        raw_input: Any,
        external_request_id: str | None = None,
    ) -> tuple[bool, int]:
        """Synchronous send.

        Args:
            request_id: Internal request ID
            raw_input: Raw input data
            external_request_id: External request ID for key building

        Returns:
            (success, size)
        """
        if not self._send_from or not self._send_to:
            logger.warning("Send stages not configured")
            return False, 0

        chunk_id = self._send_chunk_counts.get(request_id, 0)

        ctx = TransferContext(
            stage_id=self.config.stage_id or 0,
            from_stage=self._send_from,
            to_stage=self._send_to,
            request_id=request_id,
            external_request_id=external_request_id,
            chunk_id=chunk_id,
        )

        data = self.handler.prepare_send_data(ctx, raw_input)
        if data is None:
            return False, 0

        key = self.handler.build_key(ctx)

        if hasattr(data, "to_dict"):
            data = data.to_dict()

        success, size = self.transport.sync_send(
            self._send_from,
            self._send_to,
            key,
            data,
        )

        if success:
            self._send_chunk_counts[request_id] = chunk_id + 1
            self.handler.on_send_complete(ctx, True, size)
        else:
            self.handler.on_send_complete(ctx, False, 0)

        return success, size

    # ============ Recv Interface ============

    def submit_recv(
        self,
        request_id: str,
        request: Any,
        external_request_id: str | None = None,
    ) -> str:
        """Submit a recv task.

        Args:
            request_id: Internal request ID
            request: Request object to update after receiving
            external_request_id: External request ID for key building

        Returns:
            task_id for tracking
        """
        if not self._recv_from or not self._recv_to:
            logger.warning("Recv stages not configured")
            return ""

        chunk_id = self._recv_chunk_counts.get(request_id, 0)

        ctx = TransferContext(
            stage_id=self.config.stage_id or 0,
            from_stage=self._recv_from,
            to_stage=self._recv_to,
            request_id=request_id,
            external_request_id=external_request_id,
            chunk_id=chunk_id,
        )

        key = self.handler.build_key(ctx)

        task_id = self.transport.submit_recv(
            self._recv_from,
            self._recv_to,
            key,
        )

        self._pending_recv_contexts[task_id] = (ctx, request)

        logger.debug(f"Submitted recv: {key}")
        return task_id

    def sync_recv(
        self,
        request_id: str,
        request: Any,
        external_request_id: str | None = None,
    ) -> tuple[bool, int]:
        """Synchronous receive.

        Args:
            request_id: Internal request ID
            request: Request object to update after receiving
            external_request_id: External request ID for key building

        Returns:
            (success, size)
        """
        if not self._recv_from or not self._recv_to:
            logger.warning("Recv stages not configured")
            return False, 0

        chunk_id = self._recv_chunk_counts.get(request_id, 0)

        ctx = TransferContext(
            stage_id=self.config.stage_id or 0,
            from_stage=self._recv_from,
            to_stage=self._recv_to,
            request_id=request_id,
            external_request_id=external_request_id,
            chunk_id=chunk_id,
        )

        key = self.handler.build_key(ctx)

        data, size = self.transport.sync_recv(
            self._recv_from,
            self._recv_to,
            key,
        )

        if data is not None:
            self.handler.process_recv_data(ctx, data, request)
            self.handler.on_recv_complete(ctx, data, size)
            self._recv_chunk_counts[request_id] = chunk_id + 1
            return True, size

        return False, 0

    def sync_recv_raw(
        self,
        request_id: str,
        external_request_id: str | None = None,
    ) -> tuple[Any | None, int]:
        """Synchronous receive that returns raw data without applying it.

        Args:
            request_id: Internal request ID
            external_request_id: External request ID for key building

        Returns:
            (data, size) if successful, (None, 0) otherwise
        """
        if not self._recv_from or not self._recv_to:
            logger.warning("Recv stages not configured")
            return None, 0

        chunk_id = self._recv_chunk_counts.get(request_id, 0)

        ctx = TransferContext(
            stage_id=self.config.stage_id or 0,
            from_stage=self._recv_from,
            to_stage=self._recv_to,
            request_id=request_id,
            external_request_id=external_request_id,
            chunk_id=chunk_id,
        )

        key = self.handler.build_key(ctx)
        data, size = self.transport.sync_recv(
            self._recv_from,
            self._recv_to,
            key,
        )
        if data is not None:
            self._recv_chunk_counts[request_id] = chunk_id + 1
        return data, size

    # ============ Polling ============

    def poll(self) -> None:
        """Poll for completed send/recv tasks and process results."""
        # Process send results
        for task_id, success, size in self.transport.poll_send_results():
            if task_id in self._pending_send_contexts:
                ctx = self._pending_send_contexts.pop(task_id)
                self.handler.on_send_complete(ctx, success, size)
                if success:
                    self._finished_sends.add(ctx.request_id)

        # Process recv results
        for task_id, data, size in self.transport.poll_recv_results():
            if task_id in self._pending_recv_contexts:
                ctx, request = self._pending_recv_contexts.pop(task_id)

                if data is not None:
                    self.handler.process_recv_data(ctx, data, request)
                    self.handler.on_recv_complete(ctx, data, size)
                    self._finished_recvs.add(ctx.request_id)

                    # Increment chunk counter
                    self._recv_chunk_counts[ctx.request_id] = (ctx.chunk_id or 0) + 1

    def poll_recv_results_raw(self) -> list[tuple[str, Any, int]]:
        """Poll for completed recv tasks and return raw data.

        This does not call handler.process_recv_data().

        Returns:
            List of (request_id, data, size)
        """
        results: list[tuple[str, Any, int]] = []
        for task_id, data, size in self.transport.poll_recv_results():
            if task_id in self._pending_recv_contexts:
                ctx, _request = self._pending_recv_contexts.pop(task_id)
                if data is not None:
                    results.append((ctx.request_id, data, size))
                    self._recv_chunk_counts[ctx.request_id] = (ctx.chunk_id or 0) + 1
        return results

    def get_finished_sends(self) -> set[str]:
        """Get and clear finished send request IDs."""
        result = self._finished_sends.copy()
        self._finished_sends.clear()
        return result

    def get_finished_recvs(self) -> set[str]:
        """Get and clear finished recv request IDs."""
        result = self._finished_recvs.copy()
        self._finished_recvs.clear()
        return result

    # ============ Lifecycle ============

    def start(self) -> None:
        """Start the transport engine."""
        self.transport.start()

    def stop(self) -> None:
        """Stop the transport engine."""
        self.transport.stop()

    def clear_request_state(self, request_id: str) -> None:
        """Clear all state for a request."""
        self._send_chunk_counts.pop(request_id, None)
        self._recv_chunk_counts.pop(request_id, None)
        self._finished_sends.discard(request_id)
        self._finished_recvs.discard(request_id)

        # Clear handler state if available
        if hasattr(self.handler, "clear_request_state"):
            self.handler.clear_request_state(request_id)
