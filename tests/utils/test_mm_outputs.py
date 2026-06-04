import pytest
import torch

from vllm_omni.utils.mm_outputs import build_mm_async_payload, to_payload_element


def test_build_mm_async_payload_snapshots_cuda_tensor_storage():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for async multimodal payload snapshots")

    source = torch.arange(12, device="cuda").reshape(6, 2)
    payload = build_mm_async_payload({"codes.audio": source})

    snap = payload["codes.audio"]
    assert isinstance(snap, torch.Tensor)
    assert snap.is_cuda
    assert snap.data_ptr() != source.data_ptr()
    assert torch.equal(snap.cpu(), source.cpu())

    source.fill_(-1)
    assert torch.equal(snap.cpu(), torch.arange(12).reshape(6, 2))


def test_async_payload_keeps_to_payload_element_semantics_on_cuda_tensor():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for async multimodal payload snapshots")

    source = torch.arange(12, device="cuda").reshape(6, 2)
    payload = build_mm_async_payload({"codes.audio": source})
    sliced = to_payload_element(payload["codes.audio"], idx=0, start=2, end=4, seq_len=6)

    assert isinstance(sliced, torch.Tensor)
    assert sliced.is_cuda
    assert torch.equal(sliced.cpu(), torch.tensor([[4, 5], [6, 7]]))
