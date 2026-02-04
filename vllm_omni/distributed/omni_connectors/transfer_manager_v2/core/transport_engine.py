# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Layer 2: Async transport engine with queues and retry logic.

Responsibilities:
- Async send/recv thread management
- Pending/finished queue management
- put/get atomic operations
- Retry with exponential backoff
- Timeout handling
- Transfer statistics
"""

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from vllm.logger import init_logger

from .config import OmniTransferConfig
from .connector_manager import OmniConnectorManager

logger = init_logger(__name__)


class TaskStatus(Enum):
    """Status of a transfer task."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class TransferTask:
    """A single transfer task."""

    task_id: str
    from_stage: str
    to_stage: str
    key: str
    direction: str  # "send" or "recv"
    data: Any = None  # For send tasks
    status: TaskStatus = TaskStatus.PENDING
    result_data: Any = None  # For recv tasks
    result_size: int = 0
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    attempts: int = 0


@dataclass
class TransferStats:
    """Statistics for transfer operations."""

    total_sends: int = 0
    successful_sends: int = 0
    failed_sends: int = 0
    total_recvs: int = 0
    successful_recvs: int = 0
    failed_recvs: int = 0
    total_bytes_sent: int = 0
    total_bytes_recv: int = 0

    def record_send(self, success: bool, size: int = 0) -> None:
        self.total_sends += 1
        if success:
            self.successful_sends += 1
            self.total_bytes_sent += size
        else:
            self.failed_sends += 1

    def record_recv(self, success: bool, size: int = 0) -> None:
        self.total_recvs += 1
        if success:
            self.successful_recvs += 1
            self.total_bytes_recv += size
        else:
            self.failed_recvs += 1


class OmniTransportEngine:
    """Async transport engine for data transfer.

    Provides both sync and async interfaces for sending/receiving data.

    Async mode:
    - submit_send() / submit_recv() to queue tasks
    - poll_send_results() / poll_recv_results() to get results
    - Background threads handle actual transfers

    Sync mode:
    - sync_send() / sync_recv() for blocking operations
    - With retry and timeout support
    """

    def __init__(
        self,
        connector_manager: OmniConnectorManager,
        config: OmniTransferConfig | None = None,
    ):
        self.connector_manager = connector_manager
        self.config = config or OmniTransferConfig()

        # Task queues
        self._pending_sends: deque[TransferTask] = deque()
        self._pending_recvs: deque[TransferTask] = deque()
        self._finished_sends: deque[TransferTask] = deque()
        self._finished_recvs: deque[TransferTask] = deque()

        # Task lookup for status checking
        self._task_map: dict[str, TransferTask] = {}

        # Thread synchronization
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        # Statistics
        self.stats = TransferStats()

        # Background threads (started on demand)
        self._send_thread: threading.Thread | None = None
        self._recv_thread: threading.Thread | None = None
        self._started = False

    @property
    def connector(self):
        """Get the underlying connector."""
        return self.connector_manager.connector

    # ============ Lifecycle ============

    def start(self) -> None:
        """Start background threads for async mode."""
        if self._started:
            return

        self._stop_event.clear()

        if self.config.async_mode:
            self._send_thread = threading.Thread(target=self._send_loop, daemon=True, name="TransportEngine-Send")
            self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True, name="TransportEngine-Recv")
            self._send_thread.start()
            self._recv_thread.start()
            logger.info("Transport engine started with async threads")

        self._started = True

    def stop(self) -> None:
        """Stop background threads."""
        self._stop_event.set()

        if self._send_thread and self._send_thread.is_alive():
            self._send_thread.join(timeout=1.0)
        if self._recv_thread and self._recv_thread.is_alive():
            self._recv_thread.join(timeout=1.0)

        self._started = False
        logger.info("Transport engine stopped")

    # ============ Async Interface ============

    def submit_send(
        self,
        from_stage: str,
        to_stage: str,
        key: str,
        data: Any,
    ) -> str:
        """Submit a send task to the queue.

        Args:
            from_stage: Source stage identifier
            to_stage: Target stage identifier
            key: Unique key for this transfer
            data: Data to send

        Returns:
            task_id for tracking
        """
        task_id = str(uuid.uuid4())[:8]
        task = TransferTask(
            task_id=task_id,
            from_stage=from_stage,
            to_stage=to_stage,
            key=key,
            direction="send",
            data=data,
        )

        with self._lock:
            self._pending_sends.append(task)
            self._task_map[task_id] = task

        # Auto-start if not started
        if not self._started:
            self.start()

        return task_id

    def submit_recv(
        self,
        from_stage: str,
        to_stage: str,
        key: str,
    ) -> str:
        """Submit a recv task to the queue.

        Args:
            from_stage: Source stage identifier
            to_stage: Target stage identifier (usually current stage)
            key: Unique key for this transfer

        Returns:
            task_id for tracking
        """
        task_id = str(uuid.uuid4())[:8]
        task = TransferTask(
            task_id=task_id,
            from_stage=from_stage,
            to_stage=to_stage,
            key=key,
            direction="recv",
        )

        with self._lock:
            self._pending_recvs.append(task)
            self._task_map[task_id] = task

        # Auto-start if not started
        if not self._started:
            self.start()

        return task_id

    def poll_send_results(self) -> list[tuple[str, bool, int]]:
        """Poll for completed send tasks.

        Returns:
            List of (task_id, success, size)
        """
        results = []
        with self._lock:
            while self._finished_sends:
                task = self._finished_sends.popleft()
                success = task.status == TaskStatus.SUCCESS
                results.append((task.task_id, success, task.result_size))
                # Clean up task map
                self._task_map.pop(task.task_id, None)
        return results

    def poll_recv_results(self) -> list[tuple[str, Any, int]]:
        """Poll for completed recv tasks.

        Returns:
            List of (task_id, data, size)
        """
        results = []
        with self._lock:
            while self._finished_recvs:
                task = self._finished_recvs.popleft()
                results.append((task.task_id, task.result_data, task.result_size))
                # Clean up task map
                self._task_map.pop(task.task_id, None)
        return results

    def get_task_status(self, task_id: str) -> TaskStatus | None:
        """Get status of a task."""
        with self._lock:
            task = self._task_map.get(task_id)
            return task.status if task else None

    # ============ Sync Interface ============

    def sync_send(
        self,
        from_stage: str,
        to_stage: str,
        key: str,
        data: Any,
        max_retries: int | None = None,
        timeout: float | None = None,
    ) -> tuple[bool, int]:
        """Synchronous send with retry.

        Args:
            from_stage: Source stage identifier
            to_stage: Target stage identifier
            key: Unique key for this transfer
            data: Data to send
            max_retries: Override default max retries
            timeout: Total timeout (not per-retry)

        Returns:
            (success, size)
        """
        if max_retries is None:
            max_retries = self.config.max_retries

        start_time = time.time()

        for attempt in range(max_retries):
            try:
                if timeout and (time.time() - start_time) > timeout:
                    logger.warning(f"Send timeout for key {key}")
                    break

                success, size, _ = self._do_send(from_stage, to_stage, key, data)
                if success:
                    self.stats.record_send(True, size)
                    return True, size

                logger.warning(f"Send attempt {attempt + 1}/{max_retries} failed for {key}")

            except Exception as e:
                logger.warning(f"Send attempt {attempt + 1}/{max_retries} exception: {e}")

            if attempt < max_retries - 1:
                time.sleep(self.config.retry_backoff * (2**attempt))

        self.stats.record_send(False)
        return False, 0

    def sync_recv(
        self,
        from_stage: str,
        to_stage: str,
        key: str,
        timeout: float | None = None,
        poll_interval: float | None = None,
    ) -> tuple[Any | None, int]:
        """Synchronous receive with timeout.

        Args:
            from_stage: Source stage identifier
            to_stage: Target stage identifier
            key: Unique key for this transfer
            timeout: Total timeout for waiting
            poll_interval: Interval between poll attempts

        Returns:
            (data, size) if successful, (None, 0) otherwise
        """
        if timeout is None:
            timeout = self.config.recv_timeout
        if poll_interval is None:
            poll_interval = 0.5

        start_time = time.time()

        while True:
            try:
                result = self._do_recv(from_stage, to_stage, key)
                if result:
                    data, size = result
                    elapsed = time.time() - start_time
                    mbps = (size / 1024 / 1024) / elapsed if elapsed > 0 else 0
                    logger.info(f"Received {key}: {size} bytes, {elapsed:.3f}s, {mbps:.1f} MB/s")
                    self.stats.record_recv(True, size)
                    return data, size

            except Exception as e:
                logger.warning(f"Recv error for {key}: {e}")

            if time.time() - start_time > timeout:
                logger.error(f"Recv timeout for key {key} after {timeout}s")
                self.stats.record_recv(False)
                return None, 0

            time.sleep(poll_interval)

    # ============ Background Threads ============

    def _send_loop(self) -> None:
        """Background thread for processing send tasks."""
        while not self._stop_event.is_set():
            task = None
            with self._lock:
                if self._pending_sends:
                    task = self._pending_sends.popleft()
                    task.status = TaskStatus.IN_PROGRESS

            if task:
                self._process_send_task(task)
            else:
                time.sleep(self.config.poll_interval)

    def _recv_loop(self) -> None:
        """Background thread for processing recv tasks."""
        while not self._stop_event.is_set():
            task = None
            with self._lock:
                if self._pending_recvs:
                    task = self._pending_recvs.popleft()
                    task.status = TaskStatus.IN_PROGRESS

            if task:
                self._process_recv_task(task)
            else:
                time.sleep(self.config.poll_interval)

    def _process_send_task(self, task: TransferTask) -> None:
        """Process a single send task with retry."""
        max_retries = self.config.max_retries

        for attempt in range(max_retries):
            task.attempts = attempt + 1
            try:
                success, size, _ = self._do_send(task.from_stage, task.to_stage, task.key, task.data)
                if success:
                    task.status = TaskStatus.SUCCESS
                    task.result_size = size
                    task.completed_at = time.time()
                    self.stats.record_send(True, size)

                    with self._lock:
                        self._finished_sends.append(task)

                    logger.debug(f"Send OK: {task.key}, {size} bytes")
                    return

            except Exception as e:
                task.error = str(e)
                logger.warning(f"Send attempt {attempt + 1} failed for {task.key}: {e}")

            if attempt < max_retries - 1:
                time.sleep(self.config.retry_backoff * (2**attempt))

        # All retries failed
        task.status = TaskStatus.FAILED
        task.completed_at = time.time()
        self.stats.record_send(False)

        with self._lock:
            self._finished_sends.append(task)

        logger.error(f"Send FAILED: {task.key} after {max_retries} attempts")

    def _process_recv_task(self, task: TransferTask) -> None:
        """Process a single recv task with timeout."""
        timeout = self.config.recv_timeout
        start_time = time.time()

        while not self._stop_event.is_set():
            task.attempts += 1
            try:
                result = self._do_recv(task.from_stage, task.to_stage, task.key)
                if result:
                    data, size = result
                    task.status = TaskStatus.SUCCESS
                    task.result_data = data
                    task.result_size = size
                    task.completed_at = time.time()
                    self.stats.record_recv(True, size)

                    with self._lock:
                        self._finished_recvs.append(task)

                    elapsed = time.time() - start_time
                    logger.debug(f"Recv OK: {task.key}, {size} bytes, {elapsed:.3f}s")
                    return

            except Exception as e:
                task.error = str(e)

            if time.time() - start_time > timeout:
                break

            time.sleep(self.config.poll_interval)

        # Timeout
        task.status = TaskStatus.FAILED
        task.completed_at = time.time()
        self.stats.record_recv(False)

        with self._lock:
            self._finished_recvs.append(task)

        logger.error(f"Recv FAILED: {task.key} (timeout after {timeout}s)")

    # ============ Low-level Operations ============

    def _do_send(
        self,
        from_stage: str,
        to_stage: str,
        key: str,
        data: Any,
    ) -> tuple[bool, int, dict[str, Any] | None]:
        """Perform actual send operation via connector."""
        connector = self.connector
        if not connector:
            return False, 0, None

        return connector.put(
            from_stage=from_stage,
            to_stage=to_stage,
            put_key=key,
            data=data,
        )

    def _do_recv(
        self,
        from_stage: str,
        to_stage: str,
        key: str,
    ) -> tuple[Any, int] | None:
        """Perform actual recv operation via connector."""
        connector = self.connector
        if not connector:
            return None

        return connector.get(
            from_stage=from_stage,
            to_stage=to_stage,
            get_key=key,
        )
