"""Real-weight body dispatch and graph timing; not a serving benchmark."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from tests.model_executor.models.minicpmo_4_5.test_dit_fused import _run
from vllm_omni.model_executor.models.minicpmo_4_5 import batched_token2wav as module
from vllm_omni.model_executor.models.minicpmo_4_5.dit_fused import dit_modulation
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav import MiniCPMO45Code2Wav

root = Path(__file__).resolve().parent
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = True
model = MiniCPMO45Code2Wav(vllm_config=SimpleNamespace(
    model_config=SimpleNamespace(
        model=os.environ["MODEL_PATH"],
        stage_connector_config={"extra": {
            "token2wav_float16": False, "enable_hift_graph": False,
            "enable_cfm_graph": False, "enable_code2wav_encoder_graph": False,
            "cfm_graph_precapture": False,
        }},
    ),
    scheduler_config=SimpleNamespace(max_num_seqs=8),
))
model._build_backend()
backend = model.backend
fp32 = backend.flow.decoder.estimator
fp16 = copy.deepcopy(fp32).half()
original_fused = module.blocks_forward_chunk_fused
calls = []


def traced_fused(*args, **kwargs):
    calls.append(str(args[1].dtype))
    return original_fused(*args, **kwargs)


module.blocks_forward_chunk_fused = traced_fused
records = []
with torch.inference_mode():
    for rows in (2, 8):
        outputs = {}
        for label, estimator, fused in (
            ("fp32_fused", fp32, True),
            ("fp16_ragged", fp16, False),
            ("fp16_fused", fp16, True),
        ):
            dtype = next(estimator.parameters()).dtype
            backend._ragged_fused_body = fused
            captured = []
            with torch.autocast("cuda", dtype=torch.float16, enabled=dtype == torch.float16):
                _run(lambda *args: captured.extend(args), estimator,
                     rows=rows, frames=50, cached=200, lengths=[50] * (rows // 2), dtype=dtype)
                captured[-1] = torch.tensor(captured[-1] * 2, device="cuda", dtype=torch.long)
                table = dit_modulation(estimator, captured[2][:1]) if fused else None

                def body():
                    return backend._ragged_body(*captured, modulation=table)

                calls.clear()
                output = body()
                assert bool(calls) == fused, (label, calls)
                outputs[label] = output.detach().float().clone()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        body()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = body()
                for _ in range(5):
                    graph.replay()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(100):
                    graph.replay()
                end.record()
                end.synchronize()
                record = {"rows": rows, "label": label, "milliseconds": start.elapsed_time(end) / 100,
                          "fused_body_called": bool(calls), "finite": bool(output.isfinite().all())}
                records.append(record)
                print(json.dumps(record), flush=True)
                graph.reset()
        difference = outputs["fp16_fused"] - outputs["fp16_ragged"]
        print(json.dumps({"rows": rows, "fp16_output_relative_rmse":
                          (difference.square().mean() / outputs["fp16_ragged"].square().mean()).sqrt().item()}), flush=True)
(root / "dispatch-component-results.json").write_text(json.dumps(records, indent=2) + "\n")
