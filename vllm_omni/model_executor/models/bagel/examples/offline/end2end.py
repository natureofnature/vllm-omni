import argparse
import os
import random
import time

import numpy as np
import torch

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
        args.prompts = ["<|im_start|>A cute cat<|im_end|>"]
        print(f"[Info] No prompts provided, using default: {args.prompts}")

    # Load stage configs explicitly to get engine args (optional, Omni handles it)
    # But checking paths is good.
    from vllm_omni.entrypoints.omni import Omni

    # We allow Omni to load configs and manage stages.
    # We don't need to manually extract engine args for OmniLLM anymore.

    omni_kwargs = {}
    if args.stage_configs_path:
        omni_kwargs["stage_configs_path"] = args.stage_configs_path

    # Update with script args
    omni_kwargs.update(
        {
            "log_stats": args.enable_stats,
            "init_sleep_seconds": args.init_sleep_seconds,
            "batch_timeout": args.batch_timeout,
            "init_timeout": args.init_timeout,
            "shm_threshold_bytes": args.shm_threshold_bytes,
            "worker_backend": args.worker_backend,
            "ray_address": args.ray_address,
        }
    )

    omni = Omni(model=model_name, **omni_kwargs)

    t1 = time.time()
    # Format prompts
    formatted_prompts = [{"prompt": p} for p in args.prompts]

    # Omni.generate handles sampling params internally from config if not provided
    # or we can pass overrides. Current end2end.py used hardcoded params for text.
    # But for multi-stage (AR->Diffusion), we rely on YAML defaults or pass explicit list.
    # Let's rely on YAML defaults + kwargs overrides if any.
    # Passing prompt is enough.

    # Only RequestOutput generator is returned? Omni.generate yields.
    # We consume it.
    omni_outputs = list(omni.generate(prompts=formatted_prompts))

    t2 = time.time()
    print(f"==========> time:{t2 - t1}")
    print(f"==========> time:{t2 - t1}")

    # Save images if present
    for i, req_output in enumerate(omni_outputs):
        # req_output is OmniRequestOutput
        print(f"Request {i}: finished={req_output.finished}")
        # Check top-level images
        if req_output.images:
            for j, img in enumerate(req_output.images):
                save_path = f"output_{i}_{j}.png"
                img.save(save_path)
                print(f"[Info] Saved image to {save_path}")

        # Check stage-specific outputs if needed (though top-level should aggregate final output)
        if req_output.request_output:
            for stage_out in req_output.request_output:
                if hasattr(stage_out, "images") and stage_out.images:
                    for k, img in enumerate(stage_out.images):
                        save_path = f"output_{i}_stage_{stage_out.stage_id}_{k}.png"
                        img.save(save_path)
                        print(f"[Info] Saved stage output image to {save_path}")

    print(omni_outputs)


if __name__ == "__main__":
    main()
