"""MiniCPM-o async-chunk and streaming test helpers."""

import copy

import torch
from vllm.sampling_params import RequestOutputKind


def generate_delta_outputs(
    omni_runner,
    *,
    prompt,
    modalities,
    audios=None,
    images=None,
    videos=None,
):
    omni_inputs = omni_runner.get_omni_inputs(
        prompts=prompt,
        audios=audios,
        images=images,
        videos=videos,
        modalities=modalities,
    )
    sampling_params_list = copy.deepcopy(omni_runner.get_default_sampling_params_list())
    for stage_id, sampling_params in enumerate(sampling_params_list):
        if hasattr(sampling_params, "output_kind"):
            # Audio streaming starts at the Talker. The Thinker must hand off
            # its complete TTS span and aligned hidden states in one output.
            sampling_params.output_kind = (
                RequestOutputKind.FINAL_ONLY if stage_id == 0 and "audio" in modalities else RequestOutputKind.DELTA
            )
    return omni_runner.omni.generate(omni_inputs, sampling_params_list, use_tqdm=False)


def extract_text_chunks(outputs) -> list[str]:
    chunks = []
    for stage_output in outputs:
        if stage_output.final_output_type != "text":
            continue
        text = stage_output.request_output.outputs[0].text
        if text:
            chunks.append(text)
    return chunks


def _audio_chunks(outputs) -> list[torch.Tensor]:
    chunks = []
    for stage_output in outputs:
        if stage_output.final_output_type != "audio":
            continue
        audio = (stage_output.multimodal_output or {}).get("audio")
        if audio is None:
            continue
        values = audio if isinstance(audio, list) else [audio]
        pieces = [torch.as_tensor(value).detach().float().cpu().reshape(-1) for value in values]
        pieces = [piece for piece in pieces if piece.numel()]
        if pieces:
            chunks.append(torch.cat(pieces))
    return chunks


def assert_terminal_output(outputs, output_type: str) -> None:
    assert any(output.final_output_type == output_type and output.finished for output in outputs), (
        f"No terminal {output_type} output received"
    )


def assert_chunked_audio(outputs) -> None:
    chunks = _audio_chunks(outputs)
    assert len(chunks) >= 2, f"Expected at least two audio chunks, got {len(chunks)}"
    waveform = torch.cat(chunks)
    assert waveform.numel() > 0, "Generated audio is empty"
    assert torch.isfinite(waveform).all(), "Generated audio contains non-finite samples"
    assert waveform.abs().max() > 0.01, "Generated audio appears silent"
    assert_terminal_output(outputs, "audio")


def assert_stream_finished(response) -> None:
    finish_reasons = response.finish_reasons or []
    assert finish_reasons, "Stream ended without a finish reason"
    assert finish_reasons[-1] == "stop", f"Unexpected terminal finish reason: {finish_reasons[-1]}"


def assert_text_stream(response, *, min_chunks: int = 2) -> None:
    chunks = response.text_chunks or []
    assert len(chunks) >= min_chunks, f"Expected at least {min_chunks} text chunks, got {len(chunks)}"
    assert response.text_content == "".join(chunks)
    assert_stream_finished(response)


def assert_audio_stream(response) -> None:
    chunks = response.audio_data or []
    assert len(chunks) >= 2, f"Expected at least two audio chunks, got {len(chunks)}"
    assert response.audio_bytes is not None and len(response.audio_bytes) > 44
    assert_stream_finished(response)
