# SPDX-License-Identifier: Apache-2.0

"""Utilities for OmniConnector configuration and validation."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from vllm_omni.distributed.connectors.config import OmniTransferConfig, ConnectorSpec
from vllm_omni.distributed.connectors.factory import OmniConnectorFactory
from vllm_omni.distributed.connectors.base import OmniConnectorBase

logger = logging.getLogger(__name__)


def initialize_connectors_from_config(
    config_path: Optional[Union[str, Path]] = None
) -> tuple[Optional[OmniTransferConfig], dict[tuple[str, str], OmniConnectorBase]]:
    """
    Initialize connectors from configuration file.

    Returns:
        tuple: (OmniTransferConfig, dict of {(from, to): connector_instance})
    """
    try:
        transfer_config = load_omni_transfer_config(config_path)
    except Exception as e:
        logger.warning(f"Failed to load OmniTransferConfig from {config_path}: {e}")
        return None, {}

    if not transfer_config:
        logger.info("No OmniTransferConfig provided, using default IPC")
        return None, {}

    # 使用统一的连接器创建逻辑
    connectors = create_connectors_from_config(transfer_config.connectors)
    return transfer_config, connectors


def create_connectors_from_config(
    connectors_config: dict[tuple[str, str], ConnectorSpec]
) -> dict[tuple[str, str], OmniConnectorBase]:
    """通用连接器创建逻辑，从配置字典创建连接器实例。"""
    connectors = {}
    try:
        for edge_key, connector_spec in connectors_config.items():
            connector = OmniConnectorFactory.create_connector(connector_spec)
            connectors[edge_key] = connector
            logger.info(f"Created connector for {edge_key[0]} -> {edge_key[1]}: {type(connector).__name__}")
    except Exception as e:
        logger.error(f"Failed to initialize connectors: {e}")

    return connectors


def get_connectors_config_for_stage(
    transfer_config: Optional[OmniTransferConfig],
    stage_id: Union[str, int]
) -> dict[str, Any]:
    """
    Extract connector configurations relevant for a specific stage worker.
    
    Returns a dict compatible with worker initialization:
    {
        "from_stage_X": {
            "spec": {
                "name": "ConnectorName",
                "extra": {...}
            }
        },
        ...
    }
    """
    if not transfer_config:
        return {}
        
    stage_connectors_config = {}
    target_stage = str(stage_id)
    
    # Iterate through all configured edges
    for (from_stage, to_stage), spec in transfer_config.connectors.items():
        # We only care about incoming edges for the worker process
        # (Worker needs to create connectors to receive data)
        if to_stage == target_stage:
            stage_connectors_config[f"from_stage_{from_stage}"] = {
                "spec": {
                    "name": spec.name,
                    "extra": spec.extra
                }
            }
            
    return stage_connectors_config


def load_omni_transfer_config(
    config_path: Optional[Union[str, Path]] = None,
    config_dict: Optional[dict[str, Any]] = None
) -> Optional[OmniTransferConfig]:
    """Load OmniTransferConfig from file or dict."""
    if config_path is None and config_dict is None:
        return None

    if config_path is not None:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() == '.json':
                config_dict = json.load(f)
            elif config_path.suffix.lower() in ['.yaml', '.yml']:
                try:
                    import yaml
                    config_dict = yaml.safe_load(f)
                except ImportError:
                    raise ImportError("PyYAML required for YAML config files")
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")

    if config_dict is None:
        return None

    # Parse connectors
    connectors = {}
    runtime_config = config_dict.get('runtime', {})

    # Parse global connectors (from runtime.connectors)
    global_connectors = runtime_config.get('connectors', {})
    for conn_name, conn_config in global_connectors.items():
        connector = ConnectorSpec(
            name=conn_config['name'],
            extra=conn_config.get('extra', {})
        )
        # Store globally defined connectors for reference
        # These will be resolved to specific edges later

    # Parse stage-level connectors
    for stage_config in config_dict.get('stage_args', []):
        stage_id = str(stage_config['stage_id'])

        # Input connectors
        for input_key, conn_ref in stage_config.get('input_connectors', {}).items():
            if isinstance(conn_ref, str):
                # Reference to global connector
                if conn_ref in global_connectors:
                    conn_config = global_connectors[conn_ref]
                    connector = ConnectorSpec(
                        name=conn_config['name'],
                        extra=conn_config.get('extra', {})
                    )
                else:
                    raise ValueError(f"Undefined connector reference: {conn_ref}")
            else:
                # Inline connector definition
                connector = ConnectorSpec(
                    name=conn_ref['name'],
                    extra=conn_ref.get('extra', {})
                )

            # Parse from_stage from key (e.g., "from_stage_0" -> "0")
            from_stage = input_key.replace("from_stage_", "")
            edge_key = (from_stage, stage_id)
            connectors[edge_key] = connector

        # Output connectors
        for output_key, conn_ref in stage_config.get('output_connectors', {}).items():
            if isinstance(conn_ref, str):
                # Reference to global connector
                if conn_ref in global_connectors:
                    conn_config = global_connectors[conn_ref]
                    connector = ConnectorSpec(
                        name=conn_config['name'],
                        extra=conn_config.get('extra', {})
                    )
                else:
                    raise ValueError(f"Undefined connector reference: {conn_ref}")
            else:
                # Inline connector definition
                connector = ConnectorSpec(
                    name=conn_ref['name'],
                    extra=conn_ref.get('extra', {})
                )

            # Parse to_stage from key (e.g., "to_stage_1" -> "1")
            to_stage = output_key.replace("to_stage_", "")
            edge_key = (stage_id, to_stage)
            connectors[edge_key] = connector

    config = OmniTransferConfig(connectors=connectors)

    logger.info(f"Loaded OmniTransferConfig with {len(connectors)} connector configurations")
    return config


def create_simple_config(
    connector_name: str = "InMemoryConnector",
    connector_config: Optional[dict[str, Any]] = None
) -> OmniTransferConfig:
    """Create a simple OmniTransferConfig for testing."""
    if connector_config is None:
        connector_config = {}

    # Create connectors for typical EPDG pipeline
    connectors = {}
    pipeline_edges = [
        ("0", "1"),  # thinker -> talker
        ("1", "2"),  # talker -> code2wav
    ]

    for from_stage, to_stage in pipeline_edges:
        connectors[(from_stage, to_stage)] = ConnectorSpec(
            name=connector_name,
            extra=connector_config
        )

    return OmniTransferConfig(connectors=connectors)


def create_mooncake_config(
    host: str = "127.0.0.1",
    metadata_server: str = "http://127.0.0.1:8080/metadata",
    master: str = "127.0.0.1:50051",
    segment: int = 512 * 1024 * 1024,
    localbuf: int = 64 * 1024 * 1024,
) -> OmniTransferConfig:
    """Create OmniTransferConfig for Mooncake-based setup."""
    connector_config = {
        "host": host,
        "metadata_server": metadata_server,
        "master": master,
        "segment": segment,
        "localbuf": localbuf,
    }

    return create_simple_config("MooncakeConnector", connector_config)


# High-level management functions (moved from manager.py)

def initialize_orchestrator_connectors(
    config_path: Optional[str],
) -> Tuple[Optional[OmniTransferConfig], Dict[tuple[str, str], OmniConnectorBase]]:
    """Initialize connectors shared at orchestrator level."""
    try:
        transfer_config, connectors = initialize_connectors_from_config(config_path)
        return transfer_config, connectors
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Unexpected error initializing connectors: %s", exc)
        return None, {}


def get_stage_connector_config(
    transfer_config: Optional[OmniTransferConfig],
    stage_id: int,
) -> dict[str, Any]:
    """Return the serialized connector config payload for a specific stage."""
    if transfer_config is None:
        return {}

    try:
        return get_connectors_config_for_stage(transfer_config, stage_id)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning(
            "Failed to build connector config for stage %s: %s. Using IPC fallback.",
            stage_id,
            exc,
        )
        return {}


def build_stage_connectors(
    stage_id: int,
    connectors_config: dict[str, Any],
) -> Optional[dict[tuple[str, str], Any]]:
    """Instantiate OmniConnectors for a stage based on config."""
    if not connectors_config:
        return {}

    logger.info(
        "[Stage-%s] Initializing OmniConnectors with config keys: %s",
        stage_id,
        list(connectors_config.keys()),
    )

    from .factory import OmniConnectorFactory
    from .config import ConnectorSpec

    connectors: dict[tuple[str, str], Any] = {}
    try:
        # 将字典格式的配置转换为ConnectorSpec对象
        stage_connector_specs = {}
        for input_key, config in connectors_config.items():
            if not input_key.startswith("from_stage_"):
                continue

            from_stage = input_key.replace("from_stage_", "")
            spec_dict = config.get("spec", {})
            if not spec_dict:
                continue

            connector_spec = ConnectorSpec(
                name=spec_dict.get("name", "InMemoryConnector"),
                extra=spec_dict.get("extra", {}),
            )
            stage_connector_specs[(str(from_stage), str(stage_id))] = connector_spec

        # 使用统一的连接器创建逻辑
        connectors = create_connectors_from_config(stage_connector_specs)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("[Stage-%s] Failed to initialize connectors: %s", stage_id, exc)
        return None

    return connectors
