# OmniInteract MiniCPM Duplex Benchmark

```bash
vllm bench serve --omni \
  --backend minicpmo-realtime --base-url http://127.0.0.1:8000 \
  --endpoint /v1/realtime --model openbmb/MiniCPM-o-4_5 --trust-remote-code \
  --dataset-name omniinteract --dataset-path lucky-lance/OmniInteract \
  --omniinteract-subsets 1q1a,1q1a_math,1qna \
  --omniinteract-realtime-ref-audio /path/to/ref.wav \
  --omniinteract-official-output-dir ./omniinteract-output \
  --no-oversample --num-prompts 2 --max-concurrency 2
```

Use `--omniinteract-root` for an extracted local dataset. Official accuracy runs require realtime pacing.
Successful Sessions write `.done`, `output.wav`, `wav_transcript.json`, and an
`official_eval_manifest.jsonl` entry. Failures write `.failed.json` and are not
scoreable. Run the upstream evaluator with:

```bash
python benchmarks/omniinteract/run_official_eval.py --official-repo /path/to/OmniInteract \
  --output-root ./omniinteract-output --asr-model /path/to/asr-checkpoint \
  --align-model /path/to/forced-aligner
```

Set `JUDGE_API_KEY` in the environment. A Session completes only after its final
accepted input is processed; `speak` also requires a completed response.
