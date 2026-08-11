# OmniInteract Full-Duplex Serving Benchmark

OmniInteract evaluates continuous audio-visual interaction over full videos
([Lucky-Lance/OmniInteract](https://github.com/Lucky-Lance/OmniInteract)).
vLLM-Omni integrates it as a **full-duplex only** path:

- one benchmark request = one continuous full-video duplex session
- `1q1a` / `1q1a_math`: full source video from `video_json_map.json`
- `1qna`: one continuous session per `videos_bench/**/*.mp4` with matching
  annotation slots (same layout as official MiniCPM-o batch inference)
- media is streamed as paced PCM + sampled frames over `--backend minicpmo-realtime`
- generated audio is acknowledged progressively at a simulated realtime playback cursor
- static/clip OpenAI-chat evaluation is removed

QA annotations stay attached to the Session artifacts. Accuracy is computed by
the upstream OmniInteract ASR, alignment, slot matching, and LLM-judge tools;
the older per-request proxy metric is not used for continuous Sessions.

## Requirements

Install vLLM-Omni and prepare
[`lucky-lance/OmniInteract`](https://huggingface.co/datasets/lucky-lance/OmniInteract).
`--dataset-path` may be:

- the Hugging Face dataset ID;
- a local directory containing `data.tar.gz`; or
- an extracted directory containing `1q1a/`, `1q1a_math/`, and `1qna/`.

Server must expose MiniCPM-o full duplex over WebSocket `/v1/realtime`.

## Example: MiniCPM-o 4.5 full duplex

```bash
DATASET_ROOT=/data/models/datasets/OmniInteract

vllm bench serve --omni \
  --trust-remote-code \
  --host 127.0.0.1 \
  --port 8099 \
  --backend minicpmo-realtime \
  --endpoint /v1/realtime \
  --model openbmb/MiniCPM-o-4_5 \
  --dataset-name omniinteract \
  --dataset-path "${DATASET_ROOT}" \
  --omniinteract-subsets 1q1a,1q1a_math,1qna \
  --omniinteract-realtime-chunk-ms 200 \
  --omniinteract-realtime-video-fps 1 \
  --omniinteract-realtime-ref-audio /path/to/ref.wav \
  --omniinteract-official-output-dir ./omniinteract_official/minicpmo \
  --num-prompts 32 \
  --max-concurrency 1 \
  --no-oversample \
  --save-result \
  --result-dir ./omniinteract_results
```

`1qna`-only:

```bash
vllm bench serve --omni \
  ... \
  --omniinteract-subsets 1qna \
  --num-prompts 8
```

Result JSON includes `omniinteract_realtime_turn_metrics` with per-response
`ttft_s`, `tpot_s`, and audio `rtf`, plus absolute-pacing lag. Paper accuracy
metrics come only from the official evaluator below.

`--omniinteract-official-output-dir` additionally writes the upstream
OmniInteract interchange format for every video:

```text
<output-dir>/<subset>/<video>/
  output.wav
  wav_transcript.json
  model_output.jsonl
  responses.jsonl
  events.jsonl
  audio_per_second/
  .done                    # successful sample only
  .failed.json             # failed attempt; never emitted with .done
<output-dir>/batch_summary.json
<output-dir>/official_eval_manifest.jsonl
```

Run the official ASR, forced-alignment, LLM-judge, and paper metrics from an
upstream OmniInteract checkout:

```bash
python benchmarks/omniinteract/run_official_eval.py \
  --official-repo /path/to/OmniInteract \
  --output-root ./omniinteract_official/minicpmo \
  --asr-model /path/to/Qwen3-ASR-1.7B \
  --align-model /path/to/Qwen3-ForcedAligner-0.6B
```

Set `JUDGE_API_URL`, `JUDGE_API_MODEL`, and `JUDGE_API_KEY` for the official
judge. `--skip-data-prep` is available for a quick native-transcript check; its
result does not replace the recommended ASR truncation and forced alignment.
The wrapper reruns derived ASR/alignment/judge work by default and fails when
any requested benchmark sample, data-prep item, or judge item fails. Use
`--resume` or `--allow-partial` only for explicit diagnostic runs.

Accuracy runs require 200 ms PCM chunks, 1 FPS midpoint video sampling,
realtime pacing, and `--no-oversample`. Replaying media
without pacing is useful only for load/debug experiments: it changes the
audio-video/model timing and therefore cannot produce comparable accuracy.
MiniCPM Stage0 currently consumes one queued frame per roughly one-second model
unit, so this benchmark accepts at most 1 FPS and defaults to the official
one-midpoint-frame-per-second cadence.

## Important options

| Option | Description |
|---|---|
| `--omniinteract-root` | Explicit local extracted dataset root |
| `--omniinteract-subsets` | Comma-separated subset list (`1q1a`, `1q1a_math`, `1qna`) |
| `--omniinteract-realtime-chunk-ms` | PCM append chunk size; official accuracy uses 200 ms, load/debug accepts 1–1000 ms |
| `--omniinteract-realtime-video-fps` | Frame sampling rate while streaming; MiniCPM accuracy runs currently support at most 1 FPS |
| `--omniinteract-realtime-ref-audio` | Reference WAV for voice cloning |
| `--omniinteract-official-output-dir` | Save official-compatible audio, transcript, event, batch, and manifest artifacts |
| `--omniinteract-realtime-no-pace` | Disable realtime pacing for load/debug runs; incompatible with accuracy and official output |
| `--omniinteract-realtime-timeout-s` | Per-session timeout |
| `--omniinteract-eval` | Legacy proxy; rejected for continuous realtime Sessions because it does not implement all official slot/judge rules |
| `--no-oversample` | Do not duplicate sessions when fewer samples exist |

## Result interpretation

- **Session latency / request throughput:** one value per continuous client Session.
- **TTFT / TPOT / audio RTF:** computed per model response and reported only in
  the `OmniInteract Duplex Result` fields. Generic token, TPOT, ITL, and output
  throughput fields are omitted because a continuous Session contains zero or
  more independently generated responses.
- **Official `inference_sec` / `paced_e2e_ratio`:** paced media stream through final response drain;
  video/audio preprocessing and artifact serialization are excluded.
- **Official `preprocess_sec`:** media probing, PCM extraction, and frame sampling
  before the measured Session starts.
- **IA-QTF1 / IDS / NCCS:** emitted only by the upstream official evaluator.

## Runtime completion requirement

Official accuracy output requires a Session/epoch-scoped sequence proving that
the final accepted model unit completed its `listen`/`speak` decision. The
MiniCPM native-duplex runtime emits:

- `input_audio_buffer.committed`: `session_id`, `epoch`, and
  `accepted_input_seq`;
- `input_audio_buffer.processed`: the same Session/epoch,
  `processed_input_seq`, `outcome=listen|speak|failed`, and `response_id` for a
  speak outcome.

The official benchmark matches the two events by exact Session, epoch, and
input sequence before writing `.done`. Older runtimes without these fields fail
the sample rather than scoring a possibly incomplete final video unit.
Non-official load/debug runs retain the observable response-drain fallback. A
fixed sleep is not a correctness substitute.

## Troubleshooting

- **No sessions sampled:** verify subset maps/videos exist. `1qna` needs
  `videos_bench/**/*.mp4` plus matching `annotations/**/*.json`.
- **Backend rejected:** OmniInteract requires `--backend minicpmo-realtime`.
- **Empty audio extract:** ensure `ffmpeg` can decode the source MP4 audio track.
- **Timeouts on long 1qna videos:** raise `--omniinteract-realtime-timeout-s`.
