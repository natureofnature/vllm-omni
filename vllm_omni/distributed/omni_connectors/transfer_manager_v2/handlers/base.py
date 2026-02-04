# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Layer 3: Transfer handler base class (Strategy Pattern).

TransferHandler defines the interface for business-specific logic:
- Key building
- Data preparation before send
- Data processing after receive

Each handler type (KVCache, Chunk, etc.) implements this interface.
"""

from abc import ABC, abstractmethod
from typing import Any

from ..core.config import TransferContext


class TransferHandler(ABC):
    """Abstract base class for transfer handlers.

    Handlers are responsible for:
    1. Building transfer keys
    2. Preparing data before sending
    3. Processing data after receiving
    4. Applying received data to request objects

    Handlers should be stateless or have minimal state.
    Request-specific state should be managed by the TransferManager.
    """

    @abstractmethod
    def build_key(self, ctx: TransferContext) -> str:
        """Build the transfer key for a given context.

        Args:
            ctx: Transfer context with stage info and request ID

        Returns:
            Unique key string for this transfer
        """
        pass

    @abstractmethod
    def prepare_send_data(self, ctx: TransferContext, raw_input: Any) -> Any | None:
        """Prepare data for sending.

        This method transforms raw input into the format suitable for transfer.
        Return None to skip this send (e.g., buffering chunks).

        Args:
            ctx: Transfer context
            raw_input: Raw input data (format depends on handler type)

        Returns:
            Prepared data to send, or None to skip
        """
        pass

    @abstractmethod
    def process_recv_data(self, ctx: TransferContext, data: Any, request: Any) -> None:
        """Process received data and update request object.

        Args:
            ctx: Transfer context
            data: Received data
            request: Request object to update
        """
        pass

    def on_send_complete(
        self,
        ctx: TransferContext,
        success: bool,
        size: int,
    ) -> None:
        """Called when a send operation completes.

        Override this for custom post-send handling.

        Args:
            ctx: Transfer context
            success: Whether the send succeeded
            size: Number of bytes sent
        """
        pass

    def on_recv_complete(
        self,
        ctx: TransferContext,
        data: Any,
        size: int,
    ) -> None:
        """Called when a recv operation completes.

        Override this for custom post-recv handling.

        Args:
            ctx: Transfer context
            data: Received data
            size: Number of bytes received
        """
        pass

    def get_handler_name(self) -> str:
        """Get the name of this handler for logging."""
        return self.__class__.__name__
