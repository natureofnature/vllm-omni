export PYTHONPATH=/workspace/omni/vllm-omni/:$PYTHONPATH
CUDA_VISIBLE_DEVICES=4 python end2end.py --model /workspace/Qwen2.5-Omni-7B/ \
                                 --voice-type "m02" \
                                 --dit-ckpt none \
                                 --bigvgan-ckpt none \
                                 --output-wav output_audio \
                                 --prompt_type text \
                                 --init-sleep-seconds 0 \
				 --worker-backend ray \
				 --ray-address auto \
                                 --prompts "Explain the system architecture for a scalable audio generation pipeline. Answer in 15 words."
