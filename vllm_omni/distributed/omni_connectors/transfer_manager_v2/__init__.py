# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Transfer Manager V2 - Unified architecture for data transfer.

Architecture:
    Layer 1: OmniConnectorManager - Connector lifecycle management
    Layer 2: OmniTransportEngine  - Async send/recv + queues + retry
    Layer 3: TransferHandler      - Business logic (KV Cache / Chunk)

Usage:
    from transfer_manager_v2 import OmniKVCacheTransferManager, OmniChunkTransferManager
"""

from .chunk_manager import OmniChunkTransferManager
from .core.config import OmniTransferConfig, TransferContext
from .core.connector_manager import OmniConnectorManager
from .core.transport_engine import OmniTransportEngine
from .handlers.base import TransferHandler
from .handlers.chunk_handler import ChunkHandler
from .handlers.kv_cache_handler import KVCacheHandler
from .kv_cache_manager import OmniKVCacheTransferManager
from .transfer_manager import OmniTransferManager

__all__ = [
    # Facade classes (main entry points)
    "OmniKVCacheTransferManager",
    "OmniChunkTransferManager",
    # Core components
    "OmniTransferManager",
    "OmniConnectorManager",
    "OmniTransportEngine",
    # Config
    "OmniTransferConfig",
    "TransferContext",
    # Handlers
    "TransferHandler",
    "KVCacheHandler",
    "ChunkHandler",
]
