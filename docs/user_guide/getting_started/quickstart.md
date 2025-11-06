# Quickstart

This guide will help you get started with vLLM-omni.

## Installation

### Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (for GPU acceleration)
- vLLM installed (see installation instructions below)

### Install vLLM

Use Docker to keep consistent basic environment (Optional, Recommended)

```bash
docker run --gpus all --ipc=host --network=host -v $source_dir:$container_dir --rm --name $container_name -it nvcr.io/nvidia/pytorch:25.01-py3 bash
```

Set up basic uv environment

```bash
pip install uv
uv venv --python 3.12 --seed
source .venv/bin/activate
```

Install certain version of vllm with commitid: 808a7b69df479b6b3a16181711cac7ca28a9b941

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout 808a7b69df479b6b3a16181711cac7ca28a9b941
```

Set up environment variables to get pre-built wheels. If there are internet problems, just download the whl file manually. And set VLLM_PRECOMPILED_WHEEL_LOCATION as your local absolute path of whl file.

```bash
export VLLM_COMMIT=808a7b69df479b6b3a16181711cac7ca28a9b941
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
```

Install vllm with command below.

```bash
uv pip install --editable .
```

### Install vLLM-omni

Install additional requirements for vllm-omni

```bash
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm_omni
uv pip install -r requirements/gpu.txt
```

## Quick Start

### Run examples (Qwen2.5-omni)

Get into the example folder

```bash
cd examples/offline_inference/qwen_2_5_omni
```

Modify PYTHONPATH in run.sh as your path of vllm_omni. Then run.

```bash
bash run.sh
```

The output audio is saved in ./output_audio

## Next Steps

- Read the [architecture documentation](../../contributing/design_documents/vllm_omni_design.md)
- Check out the [API reference](../../api/README.md)
- Explore the [examples](../examples/index.md)

