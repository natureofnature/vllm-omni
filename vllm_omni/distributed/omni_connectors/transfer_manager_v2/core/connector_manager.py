# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Layer 1: Connector lifecycle management.

Responsibilities:
- Parse configuration
- Lazy initialize connector
- Auto-configure RDMA role/port
- Dynamic peer info updates
- Connection state management
"""

from typing import Any

from vllm.logger import init_logger

from .config import OmniTransferConfig

logger = init_logger(__name__)

# Port offset for KV transfer to avoid conflicts with request forwarding
KV_TRANSFER_PORT_OFFSET = 100


class OmniConnectorManager:
    """Manages OmniConnector lifecycle with lazy initialization.

    This class handles:
    - Lazy creation of connector (only when first accessed)
    - Auto-configuration of RDMA role (sender/receiver)
    - Port offset calculation for multi-purpose connections
    - Dynamic sender info updates for cross-node RDMA
    - Failure caching to avoid repeated init attempts
    """

    def __init__(self, config: OmniTransferConfig):
        self.config = config
        self._connector = None
        self._init_failed = False

        # Eagerly initialize if sender mode (so connection info is available early)
        if config.need_send and config.connector_type:
            try:
                _ = self.connector  # Trigger lazy init
                logger.info("Sender connector eagerly initialized")
            except Exception as e:
                logger.warning(f"Failed to eagerly initialize sender connector: {e}")

    @property
    def connector(self):
        """Lazy initialization of connector.

        Returns None if:
        - No connector configured
        - Previous initialization failed
        - Initialization fails now
        """
        if self._init_failed:
            return None

        if self._connector is None:
            self._connector = self._create_connector()

        return self._connector

    @property
    def is_ready(self) -> bool:
        """Check if connector is ready for use."""
        return self.connector is not None

    @property
    def stage_id(self) -> str | int | None:
        """Get stage ID from config."""
        return self.config.stage_id

    def _create_connector(self):
        """Create and configure the connector."""
        c_type = self.config.connector_type
        if not c_type:
            return None

        try:
            c_extra = dict(self.config.connector_extra)

            # Auto-configure RDMA connector
            if c_type == "MooncakeRDMAConnector" and c_extra.get("role") == "auto":
                self._configure_rdma_role(c_extra)

            logger.info(
                f"Initializing OmniConnector: type={c_type}, "
                f"role={c_extra.get('role', 'N/A')}, stage={self.config.stage_id}"
            )

            # Import and create connector
            from ..factory import OmniConnectorFactory
            from ..utils.config import ConnectorSpec

            connector = OmniConnectorFactory.create_connector(ConnectorSpec(name=c_type, extra=c_extra))
            return connector

        except Exception as e:
            logger.error(f"Failed to initialize OmniConnector: {e}")
            import traceback

            traceback.print_exc()
            self._init_failed = True
            return None

    def _configure_rdma_role(self, c_extra: dict[str, Any]) -> None:
        """Configure RDMA role and ports based on transfer mode.

        Port offset strategy:
        - request_forwarding: base_port + 0 + from_stage_id
        - kv_transfer: base_port + 100 + from_stage_id
        """
        base_port = c_extra.get("zmq_port", 50051)

        # Pass stage info for dynamic sender discovery
        c_extra["from_stage"] = str(self.config.from_stage) if self.config.from_stage is not None else "0"
        c_extra["to_stage"] = str(self.config.to_stage) if self.config.to_stage is not None else "1"

        if self.config.need_send:
            c_extra["role"] = "sender"
            # Sender port = base_port + KV_TRANSFER_PORT_OFFSET + from_stage_id
            from_stage = self.config.from_stage
            if from_stage is not None:
                try:
                    c_extra["zmq_port"] = base_port + KV_TRANSFER_PORT_OFFSET + int(from_stage)
                except (ValueError, TypeError):
                    c_extra["zmq_port"] = base_port + KV_TRANSFER_PORT_OFFSET

        elif self.config.need_recv:
            c_extra["role"] = "receiver"
            # Receiver connects to sender's port
            from_stage = self.config.from_stage
            sender_port = base_port + KV_TRANSFER_PORT_OFFSET
            if from_stage is not None:
                try:
                    sender_port = base_port + KV_TRANSFER_PORT_OFFSET + int(from_stage)
                except (ValueError, TypeError):
                    pass

            if "sender_host" not in c_extra:
                c_extra["sender_host"] = c_extra.get("host", "127.0.0.1")
            if "sender_zmq_port" not in c_extra:
                c_extra["sender_zmq_port"] = sender_port

    def get_connection_info(self) -> dict[str, Any] | None:
        """Get sender connector's connection info for passing to receiver.

        Returns:
            Dict with 'host', 'zmq_port', 'rpc_port' if sender connector is initialized,
            None otherwise.
        """
        if not self.config.need_send:
            return None
        conn = self.connector
        if conn and hasattr(conn, "get_connection_info"):
            return conn.get_connection_info()
        return None

    def update_peer_info(self, peer_info: dict[str, Any]) -> None:
        """Update peer connection info for receiver connector.

        This should be called before receiving when peer info
        is dynamically provided (e.g., via task from orchestrator).

        Args:
            peer_info: Dict with 'host' and 'zmq_port' keys, or
                       Dict mapping rank_id to {"host": ..., "zmq_port": ...}
        """
        if not self.config.need_recv:
            return

        # Handle nested format: {rank_id: {"host": ..., "zmq_port": ...}}
        actual_info = peer_info
        if peer_info and "host" not in peer_info:
            for rank_id, info in peer_info.items():
                if isinstance(info, dict) and "host" in info:
                    actual_info = info
                    logger.debug(f"Extracted peer info for rank {rank_id}: {info}")
                    break

        if not actual_info or "host" not in actual_info:
            logger.warning(f"Invalid peer_info format: {peer_info}")
            return

        # Update config for new connector creation
        self.config.connector_extra["sender_host"] = actual_info.get("host")
        self.config.connector_extra["sender_zmq_port"] = actual_info.get("zmq_port")
        logger.info(f"Updated peer info: host={actual_info.get('host')}, zmq_port={actual_info.get('zmq_port')}")

        # Update existing connector if possible
        if self._connector and hasattr(self._connector, "sender_host"):
            self._connector.sender_host = actual_info.get("host")
            self._connector.sender_zmq_port = actual_info.get("zmq_port")

    def reset(self) -> None:
        """Reset connector state for re-initialization."""
        self._connector = None
        self._init_failed = False
