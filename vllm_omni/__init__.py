"""
vLLM-Omni: Multi-modality models inference and serving with
non-autoregressive structures.

This package extends vLLM beyond traditional text-based, autoregressive
generation to support multi-modality models with non-autoregressive
structures and non-textual outputs.

Architecture:
- 🟡 Modified: vLLM components modified for multimodal support
- 🔴 Added: New components for multimodal and non-autoregressive
  processing
"""

# vllm_omni/__init__.py
import vllm
from transformers import AutoConfig, Qwen2Config
from vllm.model_executor.models import ModelRegistry

from vllm_omni.diffusion.models.bagel.bagel_transformer import BagelConfig
from vllm_omni.model_executor.models.bagel.bagel import BagelForConditionalGeneration

from .config import OmniModelConfig
from .entrypoints.async_omni import AsyncOmni

# Main entry points
from .entrypoints.omni import Omni

from .version import __version__, __version_tuple__  # isort:skip


AutoConfig.register("bagel", BagelConfig)
_original_with_hf_config = vllm.config.VllmConfig.with_hf_config


def _patched_with_hf_config(self, hf_config, *args, **kwargs):
    if isinstance(hf_config, dict):
        try:
            hf_config = BagelConfig(**hf_config)
        except Exception:
            hf_config = Qwen2Config(**hf_config)
    if not hasattr(hf_config, "get_text_config"):
        hf_config.get_text_config = lambda: hf_config

    return _original_with_hf_config(self, hf_config, *args, **kwargs)


vllm.config.VllmConfig.with_hf_config = _patched_with_hf_config


ModelRegistry.register_model("BagelForConditionalGeneration", BagelForConditionalGeneration)

try:
    from . import patch  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    if exc.name != "vllm":
        raise
    # Allow importing vllm_omni without vllm (e.g., documentation builds)
    patch = None  # type: ignore


__all__ = [
    "__version__",
    "__version_tuple__",
    # Main components
    "Omni",
    "AsyncOmni",
    # Configuration
    "OmniModelConfig",
    # All other components are available through their respective modules
    # processors.*, schedulers.*, executors.*, etc.
]
