# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Any, Optional

from vllm_omni.distributed.connectors.logging import get_connector_logger

logger = get_connector_logger(__name__)


class OmniConnectorBase(ABC):
    """Base class for all OmniConnectors."""

    @abstractmethod
    def put(self, from_stage: str, to_stage: str, request_id: str, data: Any) -> tuple[bool, int, Optional[dict[str, Any]]]:
        """Store Python object, internal serialization handled by connector.

        Args:
            from_stage: Source stage identifier
            to_stage: Destination stage identifier
            request_id: Unique request identifier
            data: Python object to store

        Returns:
            tuple: (success: bool, serialized_size: int, metadata: Optional[dict])
                   Metadata may contain transport-specific handles or inline data.
        """
        pass

    @abstractmethod
    def get(self, from_stage: str, to_stage: str, request_id: str, metadata: Optional[dict[str, Any]] = None) -> Optional[tuple[Any, int]]:
        """Retrieve Python object and payload size (bytes).

        Args:
            from_stage: Source stage identifier
            to_stage: Destination stage identifier
            request_id: Unique request identifier
            metadata: Optional transport-specific metadata from the put operation

        Returns:
            Tuple of (Python object, serialized byte size) if found, None otherwise
        """
        pass

    @abstractmethod
    def cleanup(self, request_id: str) -> None:
        """Clean up resources for a request."""
        pass

    @abstractmethod
    def health(self) -> dict[str, Any]:
        """Return health status and metrics."""
        pass

    @staticmethod
    def serialize_obj(obj: Any) -> bytes:
        """Serialize a Python object to bytes using centralized serializer."""
        from vllm_omni.distributed.connectors.serialization import OmniSerializer
        return OmniSerializer.serialize(obj)

    @staticmethod
    def deserialize_obj(data: bytes) -> Any:
        """Deserialize bytes to Python object using centralized serializer."""
        from vllm_omni.distributed.connectors.serialization import OmniSerializer
        return OmniSerializer.deserialize(data)


class InMemoryOmniConnector(OmniConnectorBase):
    """Simple in-memory connector for testing and co-located stages."""

    def __init__(self):
        self._storage: dict[str, Any] = {}
        self._metrics = {
            "puts": 0,
            "gets": 0,
            "errors": 0,
        }

    def _make_key(self, from_stage: str, to_stage: str, request_id: str) -> str:
        return f"{from_stage}:{to_stage}:{request_id}"

    def put(self, from_stage: str, to_stage: str, request_id: str, data: Any) -> tuple[bool, int, Optional[dict[str, Any]]]:
        try:
            key = self._make_key(from_stage, to_stage, request_id)
            self._storage[key] = data
            serialized_size = len(self.serialize_obj(data))
            self._metrics["puts"] += 1
            logger.debug(f"InMemoryConnector: stored {key}")
            return True, serialized_size, None
        except Exception as e:
            self._metrics["errors"] += 1
            logger.error(f"InMemoryConnector put failed: {e}")
            return False, 0, None

    def get(self, from_stage: str, to_stage: str, request_id: str, metadata: Optional[dict[str, Any]] = None) -> Optional[tuple[Any, int]]:
        try:
            key = self._make_key(from_stage, to_stage, request_id)
            data = self._storage.get(key)
            if data is not None:
                self._metrics["gets"] += 1
                return data, len(self.serialize_obj(data))
            return None
        except Exception as e:
            logger.error(f"InMemoryConnector get failed: {e}")
            return None

    def cleanup(self, request_id: str) -> None:
        keys_to_remove = [k for k in self._storage.keys() if k.endswith(f":{request_id}")]
        for key in keys_to_remove:
            del self._storage[key]
        logger.debug(f"InMemoryConnector: cleaned up {len(keys_to_remove)} keys for {request_id}")

    def health(self) -> dict[str, Any]:
        return {
            "status": "healthy",
            "storage_size": len(self._storage),
            **self._metrics,
        }
