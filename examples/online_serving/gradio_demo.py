import argparse
import base64
import io
import os
import os as _os_env_toggle
import random
import signal
import sys
from types import SimpleNamespace
from typing import Optional

import gradio as gr
import numpy as np
import soundfile as sf
import torch

# For HTTP API mode
from openai import OpenAI

# For AsyncOmniLLM mode
from vllm.sampling_params import SamplingParams
from vllm_omni.entrypoints.async_omni_llm import AsyncOmniLLM

# Import utils from offline inference example
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../offline_inference/qwen2_5_omni"))
from utils import make_omni_prompt

_os_env_toggle.environ["VLLM_USE_V1"] = "1"

SEED = 42
ASYNC_INIT_TIMEOUT = 600

SUPPORTED_MODELS: dict[str, dict[str, str]] = {
    "Qwen/Qwen2.5-Omni-7B": {
        "display_name": "Qwen2.5-Omni",
    },
    "Qwen/Qwen3-Omni-30B-A3B-Instruct": {
        "display_name": "Qwen3-Omni",
    },
}


# Ensure deterministic behavior across runs.
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gradio demo for Qwen Omni (Qwen2.5/Qwen3) online inference."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["async_omni_llm", "http_api"],
        default="async_omni_llm",
        help="Inference mode: 'async_omni_llm' for direct AsyncOmniLLM, "
        "'http_api' for vllm serve HTTP API (default: async_omni_llm)",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Omni-7B",
        help="Path to model (for async_omni_llm mode) or model name (for http_api mode).",
    )
    parser.add_argument(
        "--api-base",
        default="http://localhost:8091/v1",
        help="Base URL for vllm serve API (only used in http_api mode).",
    )
    parser.add_argument(
        "--ip",
        default="127.0.0.1",
        help="Host/IP for gradio `launch`.",
    )
    parser.add_argument(
        "--port", type=int, default=7861, help="Port for gradio `launch`."
    )
    parser.add_argument(
        "--share", action="store_true", help="Share the Gradio demo publicly."
    )
    parser.add_argument(
        "--stage-configs-path",
        type=str,
        default=None,
        help="Path to custom stage configs YAML file (optional).",
    )
    return parser.parse_args()


def build_sampling_params(seed: int) -> list[SamplingParams]:
    thinker_sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=2048,
        seed=seed,
        detokenize=True,
        repetition_penalty=1.1,
    )
    talker_sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=2048,
        seed=seed,
        detokenize=True,
        repetition_penalty=1.1,
        stop_token_ids=[8294],
    )
    code2wav_sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=2048,
        seed=seed,
        detokenize=True,
        repetition_penalty=1.1,
    )
    return [
        thinker_sampling_params,
        talker_sampling_params,
        code2wav_sampling_params,
    ]


def build_sampling_params_dict(seed: int) -> list[dict]:
    """Build sampling params as dict for HTTP API mode."""
    thinker_sampling_params = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_tokens": 2048,
        "seed": seed,
        "detokenize": True,
        "repetition_penalty": 1.1,
    }
    talker_sampling_params = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_tokens": 2048,
        "seed": seed,
        "detokenize": True,
        "repetition_penalty": 1.1,
        "stop_token_ids": [8294],
    }
    code2wav_sampling_params = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_tokens": 2048,
        "seed": seed,
        "detokenize": True,
        "repetition_penalty": 1.1,
    }
    return [
        thinker_sampling_params,
        talker_sampling_params,
        code2wav_sampling_params,
    ]


def create_prompt_args(base_args: argparse.Namespace) -> SimpleNamespace:
    # The prompt builder expects a minimal namespace with these attributes.
    return SimpleNamespace(
        model=base_args.model,
        prompt_type="text",
        tokenize=True,
        use_torchvision=True,
        legacy_omni_video=False,
    )


def get_system_prompt():
    """Get system prompt for HTTP API mode."""
    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are Qwen, a virtual human developed by the Qwen Team, "
                    "Alibaba Group, capable of perceiving auditory and visual inputs, "
                    "as well as generating text and speech."
                ),
            }
        ],
    }


async def run_inference_async_omni_llm(
    omni_llm: AsyncOmniLLM,
    sampling_params: list[SamplingParams],
    prompt_args_template: SimpleNamespace,
    user_prompt: str,
):
    """Run inference using AsyncOmniLLM directly."""
    if not user_prompt.strip():
        return "Please provide a valid text prompt.", None

    prompt_args = SimpleNamespace(**prompt_args_template.__dict__)
    omni_prompt = make_omni_prompt(prompt_args, user_prompt)

    try:
        request_id = "0"
        text_outputs: list[str] = []
        audio_output = None

        async for stage_outputs in omni_llm.generate(
            prompt=omni_prompt,
            request_id=request_id,
            sampling_params_list=sampling_params,
        ):
            # stage_outputs.request_output is a RequestOutput object, not a list
            request_output = stage_outputs.request_output
            if stage_outputs.final_output_type == "text":
                if request_output.outputs:
                    for output in request_output.outputs:
                        if output.text:
                            text_outputs.append(output.text)
            elif stage_outputs.final_output_type == "audio":
                # multimodal_output is on the RequestOutput object
                # See vllm_omni/entrypoints/openai/serving_chat.py:680 for reference
                if hasattr(request_output, "multimodal_output") and request_output.multimodal_output:
                    if "audio" in request_output.multimodal_output:
                        audio_tensor = request_output.multimodal_output["audio"]
                        # Ensure audio is 1D (flatten if needed)
                        if hasattr(audio_tensor, "ndim") and audio_tensor.ndim > 1:
                            audio_tensor = audio_tensor.flatten()
                        audio_np = audio_tensor.detach().cpu().numpy()
                        audio_output = (
                            24000,  # sampling rate in Hz
                            audio_np,
                        )

        text_response = "\n\n".join(text_outputs) if text_outputs else "No text output."
        return text_response, audio_output
    except Exception as exc:  # pylint: disable=broad-except
        return f"Inference failed: {exc}", None


def run_inference_http_api(
    client: OpenAI,
    model: str,
    sampling_params_list: list[dict],
    user_prompt: str,
):
    """Run inference using HTTP API (vllm serve)."""
    if not user_prompt.strip():
        return "Please provide a valid text prompt.", None

    try:
        messages = [
            get_system_prompt(),
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt,
                    },
                ],
            },
        ]

        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model,
            extra_body={
                "sampling_params_list": sampling_params_list
            },
        )

        text_outputs: list[str] = []
        audio_output = None

        for choice in chat_completion.choices:
            if choice.message.audio:
                # Decode base64 audio data (already in WAV format)
                audio_data = base64.b64decode(choice.message.audio.data)
                # Use soundfile to read WAV from bytes
                audio_io = io.BytesIO(audio_data)
                audio_np, sample_rate = sf.read(audio_io)
                # Ensure mono audio
                if len(audio_np.shape) > 1:
                    audio_np = audio_np[:, 0]
                audio_output = (sample_rate, audio_np)
            elif choice.message.content:
                text_outputs.append(choice.message.content)

        text_response = "\n\n".join(text_outputs) if text_outputs else "No text output."
        return text_response, audio_output
    except Exception as exc:  # pylint: disable=broad-except
        return f"Inference failed: {exc}", None


def build_interface(
    mode: str,
    omni_llm: Optional[AsyncOmniLLM],
    sampling_params: Optional[list[SamplingParams]],
    prompt_args_template: Optional[SimpleNamespace],
    client: Optional[OpenAI],
    model: str,
    sampling_params_dict: Optional[list[dict]],
    api_base: Optional[str] = None,
):
    """Build Gradio interface based on the selected mode."""

    model_display_name = SUPPORTED_MODELS[model]["display_name"]  # type: ignore[index]
    if mode == "async_omni_llm":
        # AsyncOmniLLM mode - Gradio supports async functions directly
        async def run_inference(user_prompt: str):
            return await run_inference_async_omni_llm(
                omni_llm, sampling_params, prompt_args_template, user_prompt
            )

    else:
        # HTTP API mode
        def run_inference(user_prompt: str):
            return run_inference_http_api(
                client, model, sampling_params_dict, user_prompt
            )

    with gr.Blocks() as demo:
        gr.Markdown(f"# vLLM {model_display_name} Online Serving Demo")
        info_text = f"**Model:** {model} \n\n"
        if mode == "http_api" and api_base:
            info_text += f"**API Base:** {api_base}\n\n"
        gr.Markdown(info_text)
        with gr.Row():
            input_box = gr.Textbox(
                label="Input Prompt",
                placeholder="For example: Please tell me a joke in 30 words.",
                lines=4,
            )
        with gr.Row():
            generate_btn = gr.Button("Generate", variant="primary")
        with gr.Row():
            text_output = gr.Textbox(label="Text Output", lines=6)
            audio_output = gr.Audio(label="Audio Output", interactive=False)

        generate_btn.click(
            fn=run_inference,
            inputs=[input_box],
            outputs=[text_output, audio_output],
        )
        demo.queue()
    return demo


def main():
    args = parse_args()
    omni_llm = None

    # Register signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        print("\nReceived interrupt signal, shutting down...")
        if omni_llm is not None:
            try:
                omni_llm.shutdown()
            except Exception as e:
                print(f"Error during shutdown: {e}")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if args.mode == "async_omni_llm":
        # Initialize AsyncOmniLLM
        print(f"Initializing AsyncOmniLLM with model: {args.model}")
        if args.stage_configs_path:
            print(f"Using custom stage configs: {args.stage_configs_path}")
        
        try:
            sampling_params = build_sampling_params(SEED)
            omni_llm = AsyncOmniLLM(
                model=args.model,
                stage_configs_path=args.stage_configs_path,
                init_timeout=ASYNC_INIT_TIMEOUT,
            )
            print("✓ AsyncOmniLLM initialized successfully")
            prompt_args_template = create_prompt_args(args)
            client = None
            sampling_params_dict = None
        except Exception as e:
            error_msg = str(e)
            print(f"\n✗ Failed to initialize AsyncOmniLLM: {error_msg}")
            
            if omni_llm is not None:
                try:
                    omni_llm.shutdown()
                except Exception:
                    pass
            sys.exit(1)
    else:
        # HTTP API mode
        print(f"Using HTTP API mode with base URL: {args.api_base}")
        print(f"Make sure vllm serve is running: vllm serve {args.model} --omni --port {args.api_base.split(':')[-1].rstrip('/v1')}")
        client = OpenAI(api_key="EMPTY", base_url=args.api_base)
        sampling_params_dict = build_sampling_params_dict(SEED)
        omni_llm = None
        sampling_params = None
        prompt_args_template = None

    demo = build_interface(
        args.mode,
        omni_llm,
        sampling_params,
        prompt_args_template,
        client,
        args.model,
        sampling_params_dict,
        api_base=args.api_base if args.mode == "http_api" else None,
    )
    try:
        demo.launch(
            server_name=args.ip,
            server_port=args.port,
            share=args.share,
        )
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        # Cleanup
        if omni_llm is not None:
            try:
                omni_llm.shutdown()
            except Exception as e:
                print(f"Error during cleanup: {e}")


if __name__ == "__main__":
    main()

