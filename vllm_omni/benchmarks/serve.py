# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vllm.benchmarks.serve import main_async
from vllm.utils.network_utils import join_host_port

# Import patch to register daily-omni dataset and omni backends
# This monkey-patches vllm.benchmarks.datasets.get_samples before it's used
# Must be imported before any vllm.benchmarks module usage
import vllm_omni.benchmarks.patch.patch  # noqa: F401
from vllm_omni.benchmarks.patch.patch import (
    maybe_enable_stage_metrics,
    set_print_stage,
    should_request_stage_metrics,
)

if TYPE_CHECKING:
    from vllm_omni.benchmarks.omniinteract import OmniInteractBenchmarkConfig


def _omniinteract_config_from_args(args: argparse.Namespace) -> OmniInteractBenchmarkConfig:
    """Map shared ``bench serve`` arguments to the realtime runner config."""
    from vllm_omni.benchmarks.data_modules.omniinteract_dataset import DEFAULT_OMNIINTERACT_REPO
    from vllm_omni.benchmarks.omniinteract import DEFAULT_MODEL, OmniInteractBenchmarkConfig

    backend = getattr(args, "backend", None)
    if backend != "openai-realtime-duplex":
        raise ValueError(
            "OmniInteract requires --backend openai-realtime-duplex because each video is replayed "
            "as one native-duplex Realtime session"
        )

    base_url = getattr(args, "base_url", None)
    if not base_url:
        host = getattr(args, "host", "127.0.0.1")
        port = getattr(args, "port", 8000)
        base_url = f"http://{join_host_port(host, port)}"

    dataset_path = getattr(args, "dataset_path", None)
    data_root = None
    dataset_repo = DEFAULT_OMNIINTERACT_REPO
    if dataset_path:
        candidate = Path(dataset_path).expanduser()
        if candidate.exists() or candidate.is_absolute():
            data_root = str(candidate)
        else:
            dataset_repo = str(dataset_path)

    max_concurrency = getattr(args, "max_concurrency", None) or 1
    output_root = Path(getattr(args, "result_dir", None) or "omniinteract-output")
    endpoint = getattr(args, "endpoint", None)
    explicit_keys = getattr(args, "explicit_keys", None)
    if not endpoint or (explicit_keys is not None and "endpoint" not in explicit_keys):
        endpoint = "/v1/realtime"
    return OmniInteractBenchmarkConfig(
        base_url=base_url,
        endpoint=endpoint,
        model=getattr(args, "model", None) or DEFAULT_MODEL,
        data_root=data_root,
        dataset_repo=dataset_repo,
        subsets=tuple(getattr(args, "omniinteract_subsets")),
        output_root=output_root,
        num_prompts=getattr(args, "num_prompts"),
        max_concurrency=max_concurrency,
        timeout_s=getattr(args, "omniinteract_timeout_s"),
        media_timeout_s=getattr(args, "omniinteract_media_timeout_s"),
        ref_audio=getattr(args, "omniinteract_ref_audio"),
        require_response=getattr(args, "omniinteract_require_response"),
        seed=getattr(args, "seed"),
        disable_shuffle=getattr(args, "disable_shuffle"),
    )


def _run_omniinteract(args: argparse.Namespace) -> dict[str, Any]:
    from vllm_omni.benchmarks.omniinteract import run_omniinteract_benchmark

    benchmark = asyncio.run(run_omniinteract_benchmark(_omniinteract_config_from_args(args)))
    result = benchmark.as_dict()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if benchmark.failed:
        raise SystemExit(1)
    return result


def main(args: argparse.Namespace) -> dict[str, Any]:
    # OmniInteract is selected like other serving datasets, but one sample is
    # a long-lived native-duplex WebSocket session rather than one HTTP request.
    if getattr(args, "dataset_name", None) == "omniinteract":
        return _run_omniinteract(args)

    if getattr(args, "seed_tts_wer_eval", False):
        os.environ["SEED_TTS_WER_EVAL"] = "1"
    if getattr(args, "seed_tts_wer_save_items", False):
        os.environ["SEED_TTS_WER_SAVE_ITEMS"] = "1"
    if getattr(args, "daily_omni_save_eval_items", False):
        os.environ["DAILY_OMNI_SAVE_EVAL_ITEMS"] = "1"
    set_print_stage(getattr(args, "print_stage", False))
    args.extra_body = maybe_enable_stage_metrics(
        getattr(args, "extra_body", None),
        enabled=should_request_stage_metrics(args),
    )
    return asyncio.run(main_async(args))
