from __future__ import annotations

import inspect
from typing import Any


def kv_cache_manager_new_step_starts(kv_cache_manager: Any) -> None:
    new_step_starts = getattr(kv_cache_manager, "new_step_starts", None)
    if callable(new_step_starts):
        new_step_starts()


def maybe_get_kv_connector_output_compat(
    model_runner: Any,
    scheduler_output: Any,
    *,
    clear_metadata: bool,
):
    maybe_get_kv_connector_output = model_runner.maybe_get_kv_connector_output

    supports_clear_metadata = getattr(
        model_runner,
        "_omni_supports_clear_metadata_arg",
        None,
    )
    if supports_clear_metadata is None:
        try:
            supports_clear_metadata = "clear_metadata" in inspect.signature(maybe_get_kv_connector_output).parameters
        except (TypeError, ValueError):
            supports_clear_metadata = False
        setattr(
            model_runner,
            "_omni_supports_clear_metadata_arg",
            supports_clear_metadata,
        )

    if supports_clear_metadata:
        return maybe_get_kv_connector_output(
            scheduler_output,
            clear_metadata=clear_metadata,
        )
    return maybe_get_kv_connector_output(scheduler_output)
