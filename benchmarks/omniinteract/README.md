# OmniInteract MiniCPM Duplex Benchmark

This benchmark runs each OmniInteract video as one continuous MiniCPM-o 4.5
Realtime Session. Audio is sent in 200 ms PCM16 chunks and video at 1 FPS.

## Run

Start MiniCPM native duplex serving, then run:

```bash
vllm bench serve --omni \
  --backend minicpmo-realtime \
  --base-url http://127.0.0.1:8000 \
  --endpoint /v1/realtime \
  --model openbmb/MiniCPM-o-4_5 \
  --trust-remote-code \
  --dataset-name omniinteract \
  --dataset-path lucky-lance/OmniInteract \
  --omniinteract-subsets 1q1a,1q1a_math,1qna \
  --omniinteract-realtime-ref-audio /path/to/ref.wav \
  --omniinteract-official-output-dir ./omniinteract-output \
  --no-oversample \
  --num-prompts 2 \
  --max-concurrency 2
```

Use `--omniinteract-root` for an extracted local dataset. Official accuracy
runs require the default 200 ms chunks, 1 FPS video, and realtime pacing.

One benchmark request represents a complete video Session, so generic token
TPOT/ITL and token goodput are not reported. Session latency and generated text
remain in the normal benchmark result. Per-response TTFT, TPOT, audio RTF, and
detailed Session results are reported under `omniinteract_realtime_turn_*` and
`omniinteract_sessions`.

## Official evaluation

Successful Sessions produce:

```text
omniinteract-output/
├── batch_summary.json
├── official_eval_manifest.jsonl
└── <subset>/<video>/
    ├── .done
    ├── output.wav
    └── wav_transcript.json
```

These are the artifacts consumed by the official evaluator; the compact
benchmark does not emit the older diagnostic `events.jsonl`, response dumps, or
per-second audio directories.

Failed or incomplete Sessions produce `.failed.json` and are excluded from the
manifest. Existing ASR, alignment, and judge outputs are removed before a new
run so stale scores cannot be reused.

Run the upstream evaluator with:

```bash
python benchmarks/omniinteract/run_official_eval.py \
  --official-repo /path/to/OmniInteract \
  --output-root ./omniinteract-output \
  --asr-model /path/to/asr-checkpoint \
  --align-model /path/to/forced-aligner
```

The wrapper performs fresh ASR and precise alignment by default. Set
`JUDGE_API_KEY` in the environment; it is not passed on the command line.

## Completion rule

An official Session succeeds only after the server reports that its final
accepted input sequence was processed. A `speak` decision must also have a
matching `response.done(status=completed)`. Missing, stale, reordered, or failed
events produce `.failed.json` instead of a scoreable sample.
