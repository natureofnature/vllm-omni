# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ConnectorSpec:
    """Specification for a connector instance."""
    name: str  # e.g., "MooncakeConnector", "InMemoryConnector"
    extra: dict[str, Any] = field(default_factory=dict)  # backend-specific config


@dataclass
class OmniTransferConfig:
    """Top-level configuration for OmniConnector system."""
    # Direct mapping: (from_stage, to_stage) -> connector
    connectors: dict[tuple[str, str], ConnectorSpec] = field(default_factory=dict)
    default_connector: Optional[ConnectorSpec] = None

    def get_connector_for_edge(self, from_stage: str, to_stage: str) -> Optional[ConnectorSpec]:
        """Get connector spec for a specific edge."""
        edge_key = (from_stage, to_stage)
        return self.connectors.get(edge_key, self.default_connector)

    def has_connector_for_edge(self, from_stage: str, to_stage: str) -> bool:
        """Check if there's a connector configured for the edge."""
        return self.get_connector_for_edge(from_stage, to_stage) is not None
