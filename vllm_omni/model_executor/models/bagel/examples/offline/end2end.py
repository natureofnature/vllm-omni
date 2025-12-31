import argparse
import os
import random
import time

import numpy as np
import torch
from vllm.sampling_params import SamplingParams

from vllm_omni.entrypoints.omni_llm import OmniLLM

SEED = 42
# Set all random seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Make PyTorch deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set environment variables for deterministic behavior
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="../models/BAGEL-7B-MoT",
        help="Path to merged model directory.",
    )
    parser.add_argument("--prompts", nargs="+", default=None, help="Input text prompts.")
    parser.add_argument(
        "--txt-prompts",
        type=str,
        default=None,
        help="Path to a .txt file with one prompt per line (preferred).",
    )
    parser.add_argument("--prompt_type", default="text", choices=["text"])

    # OmniLLM init args
    parser.add_argument("--enable-stats", action="store_true", default=False)
    parser.add_argument("--init-sleep-seconds", type=int, default=20)
    parser.add_argument("--batch-timeout", type=int, default=5)
    parser.add_argument("--init-timeout", type=int, default=300)
    parser.add_argument("--shm-threshold-bytes", type=int, default=65536)
    parser.add_argument("--worker-backend", type=str, default="process", choices=["process", "ray"])
    parser.add_argument("--ray-address", type=str, default=None)
    parser.add_argument("--stage-configs-path", type=str, default=None)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model_name = args.model
    try:
        # Preferred: load from txt file (one prompt per line)
        if getattr(args, "txt_prompts", None) and args.prompt_type == "text":
            with open(args.txt_prompts, encoding="utf-8") as f:
                lines = [ln.strip() for ln in f.readlines()]
            args.prompts = [ln for ln in lines if ln != ""]
            print(f"[Info] Loaded {len(args.prompts)} prompts from {args.txt_prompts}")
    except Exception as e:
        print(f"[Error] Failed to load prompts: {e}")
        raise

    if args.prompts is None:
        # Default prompt for text2img test if none provided
        args.prompts = ["<|im_start|>user\ndraw a cat<|im_end|>\n<|im_start|>assistant\n"]
        print(f"[Info] No prompts provided, using default: {args.prompts}")

    omni_llm = OmniLLM(
        model=model_name,
        log_stats=args.enable_stats,
        log_file=("omni_llm_pipeline.log" if args.enable_stats else None),
        init_sleep_seconds=args.init_sleep_seconds,
        batch_timeout=args.batch_timeout,
        init_timeout=args.init_timeout,
        shm_threshold_bytes=args.shm_threshold_bytes,
        worker_backend=args.worker_backend,
        ray_address=args.ray_address,
        stage_configs_path=args.stage_configs_path,
    )
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=20, stop=["<|im_end|>"])

    sampling_params_list = [sampling_params] * len(args.prompts)

    t1 = time.time()
    # Format prompts as required by OmniLLM (list of dicts)
    formatted_prompts = [{"prompt": p} for p in args.prompts]
    omni_outputs = omni_llm.generate(formatted_prompts, sampling_params_list)
    t2 = time.time()
    print(f"==========> time:{t2 - t1}")
    print(omni_outputs)


if __name__ == "__main__":
    main()
