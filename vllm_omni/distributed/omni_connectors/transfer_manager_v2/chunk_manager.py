# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Chunk Transfer Manager facade.

This provides a compatible interface with the original OmniChunkTransferManager,
but uses the new unified architecture internally.

Also includes scheduler integration methods.
"""

from collections import deque
from collections.abc import Callable
from typing import Any

from vllm.logger import init_logger

from .core.config import OmniTransferConfig
from .core.connector_manager import OmniConnectorManager
from .core.transport_engine import OmniTransportEngine
from .handlers.chunk_handler import ChunkHandler
from .transfer_manager import OmniTransferManager

logger = init_logger(__name__)


class OmniChunkTransferManager:
    """Facade for chunk transfer operations with scheduler integration.

    This class provides backward compatibility with the original
    OmniChunkTransferManager while using the new unified architecture.

    Usage:
        # Create from connector (legacy)
        manager = OmniChunkTransferManager.from_connector(connector)

        # Or create from config
        manager = OmniChunkTransferManager.from_dict(cfg)

        # Load/save chunks
        manager.load(request)
        manager.save(pooling_output, request, custom_func)

        # Scheduler integration
        manager.process_pending_chunks(waiting_queue, running_queue)
        manager.restore_queues(waiting_queue, running_queue)
    """

    def __init__(
        self,
        config: OmniTransferConfig,
        connector_manager: OmniConnectorManager | None = None,
    ):
        self.config = config

        # Build or use provided connector manager
        self._connector_manager = connector_manager or OmniConnectorManager(config)
        self._transport = OmniTransportEngine(self._connector_manager, config)

        # Get stage_id
        stage_id = config.stage_id
        if stage_id is None:
            stage_id = self._connector_manager.stage_id or 0
        self._stage_id = int(stage_id) if isinstance(stage_id, (int, str)) else 0

        self._handler = ChunkHandler(self._stage_id)
        self._core_manager = OmniTransferManager(self._transport, self._handler, config)

        # Request ID mapping (internal -> external)
        self._request_ids_mapping: dict[str, str] = {}

        # Scheduler integration state
        self._waiting_for_chunk_waiting: deque[Any] = deque()
        self._waiting_for_chunk_running: deque[Any] = deque()
        self._requests_with_ready_chunks: set[str] = set()

        # Prompt token IDs cache (for compatibility)
        self._request_prompt_token_ids: dict[str, list[int]] = {}

        # Transport starts on first submit; call start() if explicit startup is needed

    @classmethod
    def from_connector(cls, connector) -> "OmniChunkTransferManager":
        """Create from an existing connector (legacy compatibility).

        Args:
            connector: An OmniConnector instance

        Returns:
            OmniChunkTransferManager instance
        """
        # Extract stage_id from connector
        stage_id = getattr(connector, "stage_id", 0)

        # Build minimal config
        config = OmniTransferConfig(
            stage_id=stage_id,
            from_stage=stage_id,
            to_stage=stage_id + 1 if isinstance(stage_id, int) else None,
            async_mode=True,
        )

        # Create connector manager that wraps existing connector
        conn_manager = OmniConnectorManager(config)
        conn_manager._connector = connector  # Inject existing connector

        return cls(config, conn_manager)

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | None) -> "OmniChunkTransferManager":
        """Create from config dictionary."""
        config = OmniTransferConfig.from_dict(cfg)
        return cls(config)

    # ============ Properties ============

    @property
    def connector(self):
        """Get the underlying connector."""
        return self._connector_manager.connector

    @property
    def stage_id(self) -> int:
        """Get current stage ID."""
        return self._stage_id

    @property
    def request_prompt_token_ids(self) -> dict[str, list[int]]:
        """Get prompt token IDs cache."""
        return self._request_prompt_token_ids

    @property
    def finished_requests(self) -> set[str]:
        """Get finished requests from handler."""
        return self._handler._finished_requests

    # ============ Load/Save Interface ============

    def load(self, request: Any) -> str | None:
        """Request to retrieve a chunk of data for a specific request.

        Args:
            request: The request object needing data.

        Returns:
            task_id if submitted, None if skipped (stage 0)
        """
        if self._stage_id == 0:
            return None

        request_id = request.request_id
        external_req_id = getattr(request, "external_req_id", request_id)
        self._request_ids_mapping[request_id] = external_req_id

        # Ensure additional_information attribute exists
        if not hasattr(request, "additional_information"):
            request.additional_information = None

        # Submit recv task
        task_id = self._core_manager.submit_recv(
            request_id,
            request,
            external_request_id=external_req_id,
        )
        return task_id

    def save(
        self,
        pooling_output: Any,
        request: Any,
        custom_process_input_func: Callable | None = None,
    ) -> bool:
        """Submit a chunk of data to be stored/sent asynchronously.

        Args:
            pooling_output: Partial pooling output dictionary
            request: Request object
            custom_process_input_func: Optional processing function

        Returns:
            True if task was submitted, False if skipped (e.g., buffering)
        """
        request_id = request.request_id
        external_req_id = getattr(request, "external_req_id", request_id)

        # Cache prompt token IDs
        prompt_token_ids = list(getattr(request, "prompt_token_ids", []))
        self._request_prompt_token_ids[request_id] = prompt_token_ids

        # Prepare raw input
        raw_input = {
            "pooling_output": pooling_output,
            "request": request,
            "custom_func": custom_process_input_func,
        }

        # Submit send task
        submitted = self._core_manager.submit_send(
            request_id,
            raw_input,
            external_request_id=external_req_id,
        )
        return submitted

    def get_finished_requests(self) -> set[str]:
        """Get and clear finished recv request IDs.

        Note: Data is automatically applied to the request objects
        passed to load(). Access via request.additional_information.
        """
        # Poll for results first
        self._core_manager.poll()
        return self._core_manager.get_finished_recvs()

    def poll_recv_results(self) -> list[tuple[str, Any, int]]:
        """Poll for completed recv tasks and get raw data.

        Unlike get_finished_requests(), this returns the actual data
        without automatically applying it to request objects.

        Returns:
            List of (request_id, data, size) tuples

        Note: This bypasses the handler's process_recv_data(),
              so you need to manually handle the data.
        """
        return self._core_manager.poll_recv_results_raw()

    # ============ Scheduler Integration ============

    def process_pending_chunks(
        self,
        waiting_queue: Any,
        running_queue: list[Any],
    ) -> int:
        """Process pending chunks for waiting and running queues.

        Args:
            waiting_queue: Waiting request queue (has add_request/remove methods)
            running_queue: Running request list

        Returns:
            Number of running requests waiting for chunks
        """
        if self._stage_id == 0:
            return 0

        # Poll for finished recvs
        finished_reqs = self.get_finished_requests()

        # Import RequestStatus
        try:
            from vllm.v1.request import RequestStatus
        except ImportError:
            logger.warning("Could not import RequestStatus")
            return 0

        # Process waiting queue
        self._process_chunk_queue(
            waiting_queue,
            self._waiting_for_chunk_waiting,
            RequestStatus.WAITING,
            finished_reqs,
        )

        # Process running queue
        self._process_chunk_queue(
            running_queue,
            self._waiting_for_chunk_running,
            RequestStatus.RUNNING,
            finished_reqs,
        )

        return len(self._waiting_for_chunk_running)

    def restore_queues(
        self,
        waiting_queue: Any,
        running_queue: list[Any],
    ) -> None:
        """Restore requests waiting for chunk to the queues.

        Args:
            waiting_queue: Waiting request queue
            running_queue: Running request list
        """
        # Restore waiting requests
        for request in self._waiting_for_chunk_waiting:
            waiting_queue.add_request(request)
        self._waiting_for_chunk_waiting = deque()

        # Restore running requests
        if self._waiting_for_chunk_running:
            running_queue.extend(self._waiting_for_chunk_running)
        self._waiting_for_chunk_running = deque()

    def filter_scheduler_output(self, scheduler_output: Any) -> None:
        """Clean up ready chunks from scheduler output.

        Args:
            scheduler_output: Scheduler output to filter
        """
        if scheduler_output.scheduled_new_reqs:
            for req_data in scheduler_output.scheduled_new_reqs:
                self._requests_with_ready_chunks.discard(req_data.req_id)

        if scheduler_output.scheduled_cached_reqs:
            for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
                self._requests_with_ready_chunks.discard(req_id)

    def _process_chunk_queue(
        self,
        queue: Any,
        waiting_for_chunk_list: deque[Any],
        target_status: Any,
        finished_reqs: set[str],
    ) -> None:
        """Process a queue for chunk waiting status.

        Args:
            queue: The queue to process (waiting or running)
            waiting_for_chunk_list: Deque to hold requests waiting for chunks
            target_status: Status to set when chunk is ready
            finished_reqs: Set of request IDs with finished chunks
        """
        try:
            from vllm.v1.request import RequestStatus
        except ImportError:
            return

        queue_snapshot = list(queue)
        for request in queue_snapshot:
            request_id = request.request_id

            if request.status != RequestStatus.WAITING_FOR_CHUNK:
                # Already has ready chunks?
                if request_id in self._requests_with_ready_chunks:
                    continue

                # Already finished?
                if request_id in self._handler._finished_requests:
                    request.additional_information = None
                    continue

                # Start loading
                self.load(request)
                request.status = RequestStatus.WAITING_FOR_CHUNK
            else:
                # Check if chunk is ready
                if request_id in finished_reqs:
                    request.status = target_status
                    self._requests_with_ready_chunks.add(request_id)
                    continue

            # Remove from original queue and add to waiting list
            queue.remove(request)
            waiting_for_chunk_list.append(request)

    # ============ Lifecycle ============

    def stop(self) -> None:
        """Stop the transport engine."""
        self._transport.stop()

    def clear_request_state(self, request_id: str) -> None:
        """Clear all state for a request."""
        self._request_ids_mapping.pop(request_id, None)
        self._request_prompt_token_ids.pop(request_id, None)
        self._requests_with_ready_chunks.discard(request_id)
        self._core_manager.clear_request_state(request_id)
