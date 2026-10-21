# MiniCPM-o 4.5 FP16 CFM benchmark reproduction

Measurements use the unmodified `vllm bench serve --omni` client and its
Seed-TTS/Whisper WER evaluator. `benchmark.sh` only invokes that CLI; it does
not implement request generation, timing, percentile calculation, or WER.
`prepare_inputs.py` downloads the pinned subset and checks reference-audio hashes.

[Measured results](metrics.json) · [PNG comparison](full-comparison.png) ·
[SVG comparison](full-comparison.svg).

## Revisions and workload

- Main: `b20e5bac96a029172a59559438ad0590460aeb3c`.
- Candidate: `618327d1ab01459dc6fea5b04c6b7ab9c63e7746`.
- Model: `openbmb/MiniCPM-o-4_5`, revision
  `073dbbc8c5bc0af2d789e1ce12e7c17a6be746e1`.
- Dataset: `zhaochenyang20/seed-tts-eval`, revision
  `8f5e1aa2a35d42f42e940074c1983358b9491f89`.
- Fixed 64 English evaluation requests, each with its own reference audio;
  32 disjoint warmup requests. IDs and audio hashes are in `input-manifest.json`.
  This is a subset, not a full Seed-TTS evaluation.
- One NVIDIA H800 80 GB, MPS enabled for every arm, vLLM 0.32.0,
  PyTorch `2.13.0+cu130`, Triton 3.7.1, Transformers 5.19.0,
  NVIDIA driver 535.161.08.
- Thinker/Talker BF16, HiFT FP32. Main uses Flow FP32/TF32;
  candidate uses the existing `token2wav_float16: true` opt-in.
- Thinker/Talker/Code2Wav sequence limits: 16/8/8; CFM microbatch: 4;
  CFM graph budget: 32. Client concurrency is independently varied.

Both checkouts receive the same vLLM 0.32.0 compatibility change in
`vllm_omni/model_executor/models/minicpmo_4_5/minicpmo_4_5_omni_llm.py`:
`self.llm.get_input_embeddings(input_ids)` becomes
`self.llm.embed_input_ids(input_ids)`. This is outside the performance patch.
All 32 official benchmark Python files are unchanged in both checkouts.

## Prepare

Install the model dependencies from the checkout's MiniCPM instructions.
Download the pinned model and dataset while online:

```bash
hf download openbmb/MiniCPM-o-4_5 --revision 073dbbc8c5bc0af2d789e1ce12e7c17a6be746e1
hf download openai/whisper-large-v3 --revision 06f233fe06e710322aca913c1bc4249a0d71fce1
python prepare_inputs.py
```

Set `MODEL_PATH` and `WHISPER_PATH` to those local snapshot directories.
Replace `/ABS/PATH/TO/CHECKOUT` in each deploy YAML with the corresponding
checkout path. All other deploy fields are identical between the two files.
Use the same GPU and environment for every server start.

```bash
export CUDA_VISIBLE_DEVICES=0  # Replace with the selected, idle GPU.
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
export HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9
export NO_PROXY=localhost,127.0.0.1,::1
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
```

## Run the official benchmark

Start one server at a time, using main's checkout and `baseline-deploy.yaml`
or the candidate's checkout and `candidate-deploy.yaml`:

```bash
export PYTHONPATH=/ABS/PATH/TO/CHECKOUT
vllm serve "$MODEL_PATH" --omni --deploy-config /ABS/PATH/TO/baseline-deploy.yaml \
  --trust-remote-code --served-model-name crossref-test --host 127.0.0.1 --port 18581
```

After `/health` returns 200, run in another terminal with the same environment:

```bash
export RESULT_DIR=/ABS/PATH/TO/results/baseline_a1
mkdir -p "$RESULT_DIR"
bash benchmark.sh > "$RESULT_DIR.log" 2>&1
```

The recorded order is main A1, candidate B1, candidate B2, main A2, with
a fresh server for each start. Stop the previous server and wait for its GPU
processes to exit before starting the next one. Each start performs 32 warmup
requests at C16; then, at each C1/4/8/16/32, one discarded run and two timed
runs of 64 requests. Run the client from the main checkout for every arm;
only the server checkout and Flow precision change.

WER runs separately after timing: 64 requests at C1 and 64 at C16, using
the official Whisper large-v3 evaluator and saved waveforms. The recorded
WER is the mean of per-request WER values. SIM and UTMOS are not measured.

A same-precision control uses main with the same compatibility change and
`token2wav_float16: true`, followed by:

```bash
export CONCURRENCIES=16 RESULT_DIR=/ABS/PATH/TO/results/fp16_control
mkdir -p "$RESULT_DIR"
bash benchmark.sh > "$RESULT_DIR.log" 2>&1
```

This control distinguishes the fusion implementation from merely changing
Flow precision. Do not combine its results with the FP32/TF32 baseline.

An additional C4 quality check repeats two runs of 64 requests on a fresh
main server and a fresh candidate server. Each server receives the same
32-request warmup, three 64-request C1 priming runs, and one 64-request C4
discard run before the two C4 quality runs. It uses the same official CLI
arguments, enabling the WER/audio-save environment variables for both C4
runs and a separate audio directory per run. These results are excluded
from the performance aggregates.

```bash
export C4_QUALITY_ONLY=1 RESULT_DIR=/ABS/PATH/TO/results/c4_quality_baseline
mkdir -p "$RESULT_DIR"
bash benchmark.sh > "$RESULT_DIR.log" 2>&1
```

Repeat on a fresh candidate server with a different result directory;
unset `C4_QUALITY_ONLY` before using the main sweep again.

## Metric interpretation

Audio throughput is generated audio seconds per wall-clock second. TTFA is
the official `audio_ttfp` metric. Mean RTF is the official
`mean_audio_rtf` metric. Aggregate cells are arithmetic means of four runs
(two server starts, two runs per start); reported p50/p95 values are means
of per-run percentiles, not percentiles pooled across requests. Warmup,
discard, and quality runs are excluded from the performance table.

These measurements cover the MRV2 turn-serving configuration above. FP16
changes rounding; WER and finite-waveform checks do not establish equivalent
speaker similarity or perceptual quality on all workloads. Default Flow
precision remains FP32/TF32.

## CFM body attribution

[Component measurements](component-attribution.json) isolate the actual CFM
`_ragged_body` dispatch with checkpoint weights, 50 frames and 200 cached frames.
Two/eight classifier-free-guidance rows represent one/four requests. At the same
FP16 precision, ragged/fused body times were 3.856/1.214 ms and 4.423/1.355 ms,
respectively. Each value is a CUDA-event mean over 100 graph replays after three
eager and five graph warmups. These are isolated component timings, not served
throughput gains; the official C16 FP16 control above supplies the end-to-end
ablation. The component probe does not measure the entire Code2Wav stage or
establish that CFM is the only serving bottleneck.

To repeat the diagnostic with the measured candidate checkout and the pinned
model snapshot, set `MODEL_PATH` and run:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/path/to/vllm-omni \
  python /path/to/evidence/probe_dispatch.py
```

The probe imports a test fixture from the candidate checkout and records
`dispatch-component-results.json` beside the script.
