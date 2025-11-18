# SPDX-License-Identifier: Apache-2.0

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

logger = logging.getLogger(__name__)


class OmniConnectorBase(ABC):
    """Base class for all OmniConnectors."""

    @abstractmethod
    def put(self, from_stage: str, to_stage: str, request_id: str, data: Any) -> tuple[bool, int]:
        """Store Python object, internal serialization handled by connector.

        Args:
            from_stage: Source stage identifier
            to_stage: Destination stage identifier
            request_id: Unique request identifier
            data: Python object to store

        Returns:
            tuple: (success: bool, serialized_size: int)
        """
        pass

    @abstractmethod
    def get(self, from_stage: str, to_stage: str, request_id: str) -> Optional[Any]:
        """Retrieve Python object, internal deserialization handled by connector.

        Args:
            from_stage: Source stage identifier
            to_stage: Destination stage identifier
            request_id: Unique request identifier

        Returns:
            Python object if found, None otherwise
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
        """Serialize a Python object to bytes using torch.save (consistent with stage_utils)."""
        import io
        import torch
        buf = io.BytesIO()
        torch.save(obj, buf)
        return buf.getvalue()

    @staticmethod
    def deserialize_obj(data: bytes) -> Any:
        """Deserialize bytes to Python object using torch.load (consistent with stage_utils)."""
        import io
        import torch
        return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)


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

    def put(self, from_stage: str, to_stage: str, request_id: str, data: Any) -> tuple[bool, int]:
        try:
            key = self._make_key(from_stage, to_stage, request_id)
            self._storage[key] = data
            serialized_size = len(self.serialize_obj(data))
            self._metrics["puts"] += 1
            logger.debug(f"InMemoryConnector: stored {key}")
            return True, serialized_size
        except Exception as e:
            self._metrics["errors"] += 1
            logger.error(f"InMemoryConnector put failed: {e}")
            return False, 0

    def get(self, from_stage: str, to_stage: str, request_id: str) -> Optional[Any]:
        try:
            key = self._make_key(from_stage, to_stage, request_id)
            data = self._storage.get(key)
            if data is not None:
                self._metrics["gets"] += 1
                return data
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
