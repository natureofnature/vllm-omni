# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Callable
from typing import Any

try:
    from .base import OmniConnectorBase
    from .config import ConnectorSpec
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from connectors.base import OmniConnectorBase
    from connectors.config import ConnectorSpec

logger = logging.getLogger(__name__)


class OmniConnectorFactory:
    """Factory for creating OmniConnectors."""

    _registry: dict[str, Callable[[dict[str, Any]], OmniConnectorBase]] = {}

    @classmethod
    def register_connector(
        cls,
        name: str,
        constructor: Callable[[dict[str, Any]], OmniConnectorBase]
    ) -> None:
        """Register a connector constructor."""
        if name in cls._registry:
            raise ValueError(f"Connector '{name}' is already registered.")
        cls._registry[name] = constructor
        logger.debug(f"Registered connector: {name}")

    @classmethod
    def create_connector(cls, spec: ConnectorSpec) -> OmniConnectorBase:
        """Create a connector from specification."""
        if spec.name not in cls._registry:
            raise ValueError(f"Unknown connector: {spec.name}. Available: {list(cls._registry.keys())}")

        constructor = cls._registry[spec.name]
        try:
            connector = constructor(spec.extra)
            logger.info(f"Created connector: {spec.name}")
            return connector
        except Exception as e:
            logger.error(f"Failed to create connector {spec.name}: {e}")
            raise

    @classmethod
    def list_registered_connectors(cls) -> list[str]:
        """List all registered connector names."""
        return list(cls._registry.keys())


# Register built-in connectors with lazy imports
def _create_mooncake_connector(config: dict[str, Any]) -> OmniConnectorBase:
    try:
        from .mooncake_connector import MooncakeConnector
    except ImportError:
        # Fallback import
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from connectors.mooncake_connector import MooncakeConnector
    return MooncakeConnector(config)


def _create_in_memory_connector(config: dict[str, Any]) -> OmniConnectorBase:
    try:
        from .base import InMemoryOmniConnector
    except ImportError:
        # Fallback import
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from connectors.base import InMemoryOmniConnector
    return InMemoryOmniConnector()


# Register connectors
OmniConnectorFactory.register_connector("MooncakeConnector", _create_mooncake_connector)
OmniConnectorFactory.register_connector("InMemoryConnector", _create_in_memory_connector)
