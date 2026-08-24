# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Dedicated local OmniInteract benchmark subcommand."""

from __future__ import annotations

import argparse
import asyncio
import json

from vllm_omni.entrypoints.cli.benchmark.base import OmniBenchmarkSubcommandBase
from vllm_omni.entrypoints.cli.benchmark.cli_args import add_omniinteract_cli_args


class OmniInteractBenchmarkSubcommand(OmniBenchmarkSubcommandBase):
    """The ``vllm bench omniinteract --omni`` subcommand."""

    name = "omniinteract"
    help = "Run the OmniInteract realtime benchmark against an existing MiniCPM-o server."

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        add_omniinteract_cli_args(parser)

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        from vllm_omni.benchmarks.omniinteract import (
            OmniInteractBenchmarkConfig,
            run_omniinteract_benchmark,
        )

        config = OmniInteractBenchmarkConfig(
            base_url=args.base_url,
            endpoint=args.endpoint,
            model=args.model,
            data_root=args.data_root,
            dataset_repo=args.dataset_repo,
            subsets=tuple(args.subsets),
            output_root=args.output_dir,
            num_prompts=args.num_prompts,
            max_concurrency=args.max_concurrency,
            timeout_s=args.timeout_s,
            media_timeout_s=args.media_timeout_s,
            ref_audio=args.ref_audio,
            require_response=args.require_response,
            seed=args.seed,
            disable_shuffle=args.disable_shuffle,
        )
        result = asyncio.run(run_omniinteract_benchmark(config))
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
        if result.failed:
            raise SystemExit(1)
