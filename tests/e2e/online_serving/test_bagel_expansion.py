"""
Comprehensive tests of diffusion features that are available in online serving mode
and are supported by the following models:
- ByteDance-Seed/BAGEL-7B-MoT
Coverage:
- TeaCache
- Cache-DiT
- CFG-Parallel
- Tensor-Parallel
- Ulysses-SP
- Ring-Attention

assert_diffusion_response validates successful generation and the expected
512x512 resolution.
"""

from pathlib import Path

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler, dummy_messages_from_mix_data
from tests.helpers.stage_config import modify_stage_config

pytestmark = [pytest.mark.diffusion, pytest.mark.full_model]

# This test uses the default Bagel YAML, and CLI does not control devices.We modify yaml file directly.
_BAGEL_DEFAULT_YAML = str(
    Path(__file__).resolve().parents[3] / "vllm_omni" / "model_executor" / "stage_configs" / "bagel.yaml"
)

PROMPT = "A futuristic city skyline at twilight, cyberpunk style, ultra-detailed, high resolution."
NEGATIVE_PROMPT = "low quality, blurry, distorted, deformed, watermark"

SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"})
PARALLEL_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=2)


def _make_tp_cases(model: str, tp_size: int):
    """Build Bagel TP test cases with devices auto-derived from tp_size.
    Devices can not be set through CLI args, so we set them in the YAML.
    """
    # Dit devices start from 0, due to CI GPU usage constraint,
    # for those GPUs that encountered OOM, adjust the offset accordingly.
    devices = ",".join(str(i) for i in range(tp_size))
    return [
        pytest.param(
            OmniServerParams(
                model=model,
                stage_config_path=modify_stage_config(
                    _BAGEL_DEFAULT_YAML,
                    updates={
                        "stage_args": {
                            1: {
                                "runtime.devices": devices,
                                "engine_args.parallel_config.tensor_parallel_size": tp_size,
                            },
                        },
                    },
                ),
                server_args=["--cache-backend", "cache_dit"],
            ),
            id=f"parallel_tp_{tp_size}",
            marks=PARALLEL_FEATURE_MARKS,
        ),
    ]


def _get_diffusion_feature_cases(model: str):
    """Return L4 diffusion feature cases for Bagel.
    TeaCache, Cache-DiT, CFG-Parallel, Tensor-Parallel,
    Ulysses-SP, Ring-Attention.
    """

    return [
        # TeaCache (single-card)
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--cache-backend",
                    "tea_cache",
                ],
            ),
            id="single_card_teacache",
            marks=SINGLE_CARD_FEATURE_MARKS,
        ),
        # Cache-DiT (single-card)
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--cache-backend",
                    "cache_dit",
                ],
            ),
            id="single_card_cache_dit",
            marks=SINGLE_CARD_FEATURE_MARKS,
        ),
        # CFG-Parallel size 2 (2 GPUs, TeaCache backend)
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--cache-backend",
                    "tea_cache",
                    "--cfg-parallel-size",
                    "2",
                ],
            ),
            id="parallel_cfg_2",
            marks=PARALLEL_FEATURE_MARKS,
        ),
        # Tensor-Parallel size 2 (2 GPUs, Cache-DiT backend)
        # Stage 1 (DiT) needs visible GPUs matching TP size; the default YAML
        # only exposes device "0", so we patch it here.
        *_make_tp_cases(model, tp_size=2),
        # Ulysses-SP degree=2 (2 GPUs)
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--usp",
                    "2",
                ],
            ),
            id="sp_ulysses_2",
            marks=PARALLEL_FEATURE_MARKS,
        ),
        # Ring-Attention degree=2 (2 GPUs)
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--ring",
                    "2",
                ],
            ),
            id="sp_ring_2",
            marks=PARALLEL_FEATURE_MARKS,
        ),
    ]


@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases("ByteDance-Seed/BAGEL-7B-MoT"),
    indirect=True,
)
def test_bagel(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
):
    """L4 diffusion feature coverage for Bagel on H100.

    This test exercises:
    - TeaCache
    - Cache-DiT
    - CFG-Parallel (size=2)
    - Tensor-Parallel (size=2)
    - Ulysses-SP (degree=2)
    - Ring-Attention (degree=2)

    Validation is delegated to assert_diffusion_response in tests/helpers/assertions.py,
    which checks output dimensions and basic correctness.
    """

    messages = dummy_messages_from_mix_data(content_text=PROMPT)

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "height": 512,
            "width": 512,
            "num_inference_steps": 2,
            # Enable CFG for models that use classifier-free guidance
            "negative_prompt": NEGATIVE_PROMPT,
            "true_cfg_scale": 4.0,
            "seed": 42,
        },
    }

    openai_client.send_diffusion_request(request_config)
