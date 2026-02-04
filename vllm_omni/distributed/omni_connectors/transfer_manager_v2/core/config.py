# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified configuration for transfer manager."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OmniTransferConfig:
    """Unified configuration for OmniTransferManager.

    This config is shared by both KV Cache and Chunk transfer managers.
    It consolidates all connector and transfer settings.
    """

    # Connector settings
    connector_type: str | None = None
    connector_extra: dict[str, Any] = field(default_factory=dict)

    # Stage information
    stage_id: str | int | None = None
    from_stage: str | int | None = None
    to_stage: str | int | None = None

    # Role configuration (for RDMA)
    role: str = "auto"  # "sender" | "receiver" | "auto"

    # Engine input source (for multi-source scenarios)
    engine_input_source: list[str | int] = field(default_factory=list)

    # Transfer mode flags
    need_send: bool = False
    need_recv: bool = False

    # Timeout and retry settings
    recv_timeout: float = 30.0
    max_retries: int = 3
    retry_backoff: float = 0.1

    # Async mode settings
    async_mode: bool = True
    poll_interval: float = 0.001  # 1ms

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | None) -> "OmniTransferConfig":
        """Create config from dictionary."""
        if not cfg or not isinstance(cfg, dict):
            return cls()

        connector_config = cfg.get("connector_config", {})
        connector_type = connector_config.get("type") if connector_config else None
        connector_extra = {k: v for k, v in connector_config.items() if k != "type"} if connector_config else {}

        return cls(
            connector_type=connector_type,
            connector_extra=connector_extra,
            stage_id=cfg.get("stage_id"),
            from_stage=cfg.get("omni_from_stage") or cfg.get("from_stage"),
            to_stage=cfg.get("omni_to_stage") or cfg.get("to_stage"),
            role=cfg.get("role", "auto"),
            engine_input_source=cfg.get("engine_input_source", []),
            need_send=cfg.get("need_send_cache", False) or cfg.get("need_send", False),
            need_recv=cfg.get("need_recv_cache", False) or cfg.get("need_recv", False),
            recv_timeout=cfg.get("recv_timeout", 30.0),
            max_retries=cfg.get("max_retries", 3),
            async_mode=cfg.get("async_mode", True),
        )

    @classmethod
    def from_model_config(cls, model_config: Any) -> "OmniTransferConfig":
        """Create from model config (for AR model runner)."""
        omni_kv = getattr(model_config, "omni_kv_config", None)
        return cls.from_dict(omni_kv)

    @classmethod
    def from_vllm_config(cls, vllm_config: Any, model_config: Any) -> "OmniTransferConfig":
        """Create from vllm config with fallback to kv_transfer_config."""
        # Primary: omni_kv_config from model_config
        omni_kv = getattr(model_config, "omni_kv_config", None)
        if isinstance(omni_kv, dict):
            return cls.from_dict(omni_kv)

        # Fallback: check kv_transfer_config
        kv_cfg = getattr(vllm_config, "kv_transfer_config", None)
        if kv_cfg:
            direct = getattr(kv_cfg, "omni_connector_config", None)
            if isinstance(direct, dict) and direct:
                return cls.from_dict({"connector_config": direct})
            extra = getattr(kv_cfg, "kv_connector_extra_config", None)
            if isinstance(extra, dict):
                omni = extra.get("omni_connector_config")
                if isinstance(omni, dict) and omni:
                    return cls.from_dict({"connector_config": omni})

        return cls()

    def get_send_stages(self) -> tuple[str | None, str | None]:
        """Get (from_stage, to_stage) for sending."""
        if self.from_stage is not None and self.to_stage is not None:
            return (str(self.from_stage), str(self.to_stage))
        return (None, None)

    def get_recv_stages(self) -> tuple[str | None, str | None]:
        """Get (from_stage, to_stage) for receiving."""
        recv_from = self.from_stage
        if self.engine_input_source:
            recv_from = self.engine_input_source[0]
        elif isinstance(self.stage_id, int) and self.stage_id > 0:
            recv_from = self.stage_id - 1

        if recv_from is not None and self.stage_id is not None:
            return (str(recv_from), str(self.stage_id))
        return (None, None)


@dataclass
class TransferContext:
    """Context for a single transfer operation.

    This is passed to TransferHandler methods to provide
    all necessary information about the transfer.
    """

    # Stage information
    stage_id: str | int
    from_stage: str
    to_stage: str

    # Request identification
    request_id: str
    external_request_id: str | None = None

    # For chunked transfers
    chunk_id: int | None = None

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_effective_request_id(self) -> str:
        """Get the request ID to use for key building."""
        return self.external_request_id or self.request_id
