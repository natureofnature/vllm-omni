# SPDX-License-Identifier: Apache-2.0

import io
import logging
import time
from typing import Any, Optional

import torch

from .base import OmniConnectorBase

logger = logging.getLogger(__name__)

try:
    from mooncake.store import MooncakeDistributedStore, ReplicateConfig
except ImportError:
    try:
        from mooncake import MooncakeDistributedStore, ReplicateConfig
    except ImportError:
        logger.warning("Mooncake not available, MooncakeOmniConnector will not work")
        MooncakeDistributedStore = None
        ReplicateConfig = None


def k(rid: str, from_stage: str, to_stage: str) -> str:
    """Generate store key for request between stages."""
    return f"{rid}/{from_stage}_to_{to_stage}"


class MooncakeConnector(OmniConnectorBase):
    """Mooncake-based distributed connector for OmniConnector."""

    def __init__(self, config: dict[str, Any]):
        if MooncakeDistributedStore is None:
            raise ImportError("Mooncake not available")

        self.config = config
        self.host = config.get("host", "127.0.0.1")
        self.metadata = config.get("metadata_server", "http://127.0.0.1:8080/metadata")
        self.master = config.get("master", "127.0.0.1:50051")
        self.segment = config.get("segment", 512 * 1024 * 1024)  # 512MB
        self.localbuf = config.get("localbuf", 64 * 1024 * 1024)  # 64MB
        self.proto = config.get("proto", "tcp")
        self.rdma = config.get("rdma", "")

        self.store: Optional[MooncakeDistributedStore] = None
        self.pin: Optional[ReplicateConfig] = None

        self._metrics = {
            "puts": 0,
            "gets": 0,
            "bytes_transferred": 0,
            "errors": 0,
            "timeouts": 0,
        }

        self._init_store()

    def _init_store(self):
        """Initialize Mooncake store."""
        try:
            self.store = MooncakeDistributedStore()
            rc = self.store.setup(
                self.host, self.metadata, self.segment,
                self.localbuf, self.proto, self.rdma, self.master
            )
            if rc != 0:
                raise RuntimeError(f"Mooncake setup failed: {rc}")

            self.pin = ReplicateConfig()
            self.pin.with_soft_pin = True
            logger.info("MooncakeConnector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Mooncake store: {e}")
            raise

    # Use base class serialization methods for consistency

    def put(self, from_stage: str, to_stage: str, request_id: str, data: Any) -> tuple[bool, int]:
        if not self.store:
            logger.error("Store not initialized")
            return False, 0

        try:
            serialized_data = self.serialize_obj(data)
            key = k(request_id, from_stage, to_stage)
            self.store.put(key, serialized_data, self.pin)

            self._metrics["puts"] += 1
            self._metrics["bytes_transferred"] += len(serialized_data)

            logger.debug(f"MooncakeConnector: stored {key}, {len(serialized_data)} bytes")
            print(f"========= mooncake put {from_stage}-{to_stage}-{request_id}, with size {len(serialized_data)}")
            return True, len(serialized_data)

        except Exception as e:
            self._metrics["errors"] += 1
            logger.error(f"MooncakeConnector put failed: {e}")
            return False, 0

    def get(self, from_stage: str, to_stage: str, request_id: str) -> Optional[Any]:
        if not self.store:
            logger.error("Store not initialized")
            return None

        retries = 20
        sleep_s = 0.05
        key = k(request_id, from_stage, to_stage)

        for attempt in range(retries):
            try:
                raw_data = self.store.get(key)

                if raw_data:
                    data = self.deserialize_obj(raw_data)
                    self._metrics["gets"] += 1
                    print(f"========= mooncake get {from_stage}-{to_stage}-{request_id}, with size {len(raw_data)}")
                    return data

            except Exception as e:
                logger.debug(f"MooncakeConnector get attempt {attempt} failed: {e}")

            if attempt < retries - 1:
                time.sleep(sleep_s)

        self._metrics["timeouts"] += 1
        logger.warning(f"MooncakeConnector: timeout waiting for {key}")
        return None

    def cleanup(self, request_id: str) -> None:
        if not self.store:
            return

        # Note: Mooncake doesn't have explicit delete, data will be garbage collected
        # We could implement a cleanup mechanism by storing deletion markers
        logger.debug(f"MooncakeConnector: cleanup requested for {request_id} (no-op)")

    def health(self) -> dict[str, Any]:
        if not self.store:
            return {"status": "unhealthy", "error": "Store not initialized"}

        return {
            "status": "healthy",
            "host": self.host,
            "metadata_server": self.metadata,
            "master": self.master,
            **self._metrics,
        }

    def close(self):
        """Clean shutdown."""
        if self.store:
            try:
                self.store.close()
                self.store = None
                logger.info("MooncakeConnector closed")
            except Exception as e:
                logger.error(f"Error closing Mooncake store: {e}")
