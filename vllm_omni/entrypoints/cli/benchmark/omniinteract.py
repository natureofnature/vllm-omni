# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Dedicated local OmniInteract benchmark subcommand."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from vllm_omni.benchmarks.data_modules.omniinteract import (
    DEFAULT_OMNIINTERACT_REPO,
    OMNIINTERACT_SUBSETS,
)
from vllm_omni.entrypoints.cli.benchmark.base import OmniBenchmarkSubcommandBase


class OmniInteractBenchmarkSubcommand(OmniBenchmarkSubcommandBase):
    """The ``vllm bench omniinteract --omni`` subcommand."""

    name = "omniinteract"
    help = "Run the OmniInteract realtime benchmark against an existing MiniCPM-o server."

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--model", default="openbmb/MiniCPM-o-4_5")
        parser.add_argument("--base-url", default="http://127.0.0.1:8000")
        parser.add_argument("--endpoint", default="/v1/realtime")
        parser.add_argument(
            "--data-root",
            default=None,
            help="Extracted OmniInteract root or local data.tar[.gz]. Downloads from Hugging Face when omitted.",
        )
        parser.add_argument("--dataset-repo", default=DEFAULT_OMNIINTERACT_REPO)
        parser.add_argument(
            "--subsets",
            nargs="+",
            choices=OMNIINTERACT_SUBSETS,
            default=list(OMNIINTERACT_SUBSETS),
        )
        parser.add_argument("--output-dir", type=Path, default=Path("omniinteract-output"))
        parser.add_argument(
            "--num-prompts",
            type=int,
            default=1,
            help="Total videos to run; 0 selects all videos in the requested subsets.",
        )
        parser.add_argument("--max-concurrency", type=int, default=1)
        parser.add_argument("--timeout-s", type=float, default=900.0)
        parser.add_argument(
            "--media-timeout-s",
            type=float,
            default=600.0,
            help="Timeout for each direct ffprobe/ffmpeg media command.",
        )
        parser.add_argument(
            "--ref-audio",
            required=True,
            help="Reference WAV required by MiniCPM-o native-duplex audio output.",
        )
        parser.add_argument(
            "--require-response",
            action="store_true",
            help="Fail LISTEN-only cases; intended for selected functional E2E samples, not accuracy evaluation.",
        )
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--disable-shuffle", action="store_true")

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
