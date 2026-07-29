"""
E2E offline tests for MiniCPM-o 4.5 model with multimodal input and audio / text output.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import generate_synthetic_audio, generate_synthetic_image, generate_synthetic_video
from tests.helpers.minicpmo import (
    assert_chunked_audio,
    assert_terminal_output,
    extract_text_chunks,
    generate_delta_outputs,
)
from tests.helpers.stage_config import get_deploy_config_path

models = ["openbmb/MiniCPM-o-4_5"]

_CI_DEPLOY = get_deploy_config_path("minicpmo_4_5_batching.yaml")
_SINGLE_GPU_DEPLOY = get_deploy_config_path("minicpmo_4_5.yaml")


test_params = [(model, None, {"deploy_config": _CI_DEPLOY, "trust_remote_code": True}) for model in models]
async_chunk_test_params = [
    (model, None, {"deploy_config": _SINGLE_GPU_DEPLOY, "trust_remote_code": True}) for model in models
]


def get_question(prompt_type: str = "text") -> str:
    prompts = {
        "text": "What is the capital of China? Answer in 20 words.",
        "audio": "Describe the audio briefly.",
        "image": "What color are the squares in this image?",
        "video": "Describe the video briefly.",
        "mix": "Describe what is in the image and audio.",
    }
    return prompts.get(prompt_type, prompts["text"])


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "npu": "A2"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_text_to_text(omni_runner, omni_runner_handler) -> None:
    """Test processing text, generating text output."""
    request_config = {"prompts": get_question("text"), "modalities": ["text"]}
    omni_runner_handler.send_omni_request(request_config)


@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "npu": "A2"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_audio_to_text(omni_runner, omni_runner_handler) -> None:
    """Test processing audio, generating text output."""
    audio = generate_synthetic_audio(1, 1, 16000)["np_array"]
    if len(audio.shape) == 2:
        audio = audio.squeeze()
    request_config = {"prompts": get_question("audio"), "audios": (audio, 16000), "modalities": ["text"]}
    omni_runner_handler.send_omni_request(request_config)


@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "npu": "A2"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_image_to_text(omni_runner, omni_runner_handler) -> None:
    """Test processing image, generating text output."""
    image = generate_synthetic_image(16, 16)["np_array"]
    request_config = {"prompts": get_question("image"), "images": image, "modalities": ["text"]}
    omni_runner_handler.send_omni_request(request_config)


@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "npu": "A2"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_video_to_text(omni_runner, omni_runner_handler) -> None:
    """Test processing video, generating text output."""
    video = generate_synthetic_video(24, 24, 20)["np_array"]
    request_config = {"prompts": get_question("video"), "videos": video, "modalities": ["text"]}
    omni_runner_handler.send_omni_request(request_config)


@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "npu": "A2"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_text_to_audio(omni_runner, omni_runner_handler) -> None:
    """Test processing text and generating audio through Talker and Code2Wav."""
    request_config = {"prompts": get_question("text"), "modalities": ["audio"]}
    omni_runner_handler.send_omni_request(request_config)


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "npu": "A2"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_mix_to_audio(omni_runner, omni_runner_handler) -> None:
    """Test processing mixed modalities (image + audio), generating audio output."""
    audio = generate_synthetic_audio(1, 1, 16000)["np_array"]
    if len(audio.shape) == 2:
        audio = audio.squeeze()
    image = generate_synthetic_image(16, 16)["np_array"]
    request_config = {
        "prompts": get_question("mix"),
        "audios": (audio, 16000),
        "images": image,
        "modalities": ["audio"],
    }
    omni_runner_handler.send_omni_request(request_config)


@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "npu": "A2"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_video_to_audio(omni_runner, omni_runner_handler) -> None:
    """Test processing video, generating audio output."""
    video = generate_synthetic_video(24, 24, 20)["np_array"]
    request_config = {"prompts": get_question("video"), "videos": video, "modalities": ["audio"]}
    omni_runner_handler.send_omni_request(request_config)


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "npu": "A2"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", async_chunk_test_params, indirect=True)
def test_text_to_text_async_chunk_streaming(omni_runner, run_level) -> None:
    outputs = generate_delta_outputs(
        omni_runner,
        prompt=get_question("text"),
        modalities=["text"],
    )

    chunks = extract_text_chunks(outputs)
    assert len(chunks) >= 2, f"Expected at least two text chunks, got {len(chunks)}"
    assert_terminal_output(outputs, "text")
    if run_level in {"advanced_model", "full_model"}:
        assert "beijing" in "".join(chunks).lower()


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "npu": "A2"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", async_chunk_test_params, indirect=True)
def test_text_to_audio_async_chunk_streaming(omni_runner) -> None:
    outputs = generate_delta_outputs(
        omni_runner,
        prompt=get_question("text"),
        modalities=["audio"],
    )

    assert_chunked_audio(outputs)


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "npu": "A2"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", async_chunk_test_params, indirect=True)
def test_mix_to_text_audio_async_chunk_streaming(omni_runner) -> None:
    audio = generate_synthetic_audio(5, 1, 16000)["np_array"]
    if len(audio.shape) == 2:
        audio = audio.squeeze()
    outputs = generate_delta_outputs(
        omni_runner,
        prompt="What is recited in the audio? What is in this image? Describe the video briefly.",
        audios=(audio, 16000),
        images=generate_synthetic_image(24, 24)["np_array"],
        videos=generate_synthetic_video(24, 24, 20)["np_array"],
        modalities=["text", "audio"],
    )

    chunks = extract_text_chunks(outputs)
    assert chunks, "Expected a final text output"
    assert_terminal_output(outputs, "text")
    assert_chunked_audio(outputs)
