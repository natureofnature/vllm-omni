# OmniInteract Full-Duplex Serving Benchmark

OmniInteract evaluates continuous audio-visual interaction over full videos
([Lucky-Lance/OmniInteract](https://github.com/Lucky-Lance/OmniInteract)).
vLLM-Omni integrates it as a **full-duplex only** path:

- one benchmark request = one continuous full-video duplex session
- `1q1a` / `1q1a_math`: full source video from `video_json_map.json`
- `1qna`: one continuous session per `videos_bench/**/*.mp4` with matching
  annotation slots (same layout as official MiniCPM-o batch inference)
- media is streamed as paced PCM + sampled frames over `--backend minicpmo-realtime`
- static/clip OpenAI-chat evaluation is removed

QA slots stay attached for optional proxy metrics (`--omniinteract-eval`).
Official OmniInteract LLM-judge scoring remains external.

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
  --num-prompts 32 \
  --max-concurrency 1 \
  --no-oversample \
  --omniinteract-eval \
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
`ttft_s`, `tpot_s`, and `rtf`, plus optional OmniInteract QA metrics when
`--omniinteract-eval` is enabled. Eval flattens matched/unmatched slots inside
each continuous session.

## Important options

| Option | Description |
|---|---|
| `--omniinteract-root` | Explicit local extracted dataset root |
| `--omniinteract-subsets` | Comma-separated subset list (`1q1a`, `1q1a_math`, `1qna`) |
| `--omniinteract-realtime-chunk-ms` | PCM append chunk size |
| `--omniinteract-realtime-video-fps` | Frame sampling rate while streaming |
| `--omniinteract-realtime-ref-audio` | Reference WAV for voice cloning |
| `--omniinteract-realtime-no-pace` | Disable realtime pacing |
| `--omniinteract-realtime-timeout-s` | Per-session timeout |
| `--omniinteract-eval` | Compute proxy QA / interaction metrics |
| `--omniinteract-save-eval-items` | Include per-slot evaluation rows |
| `--no-oversample` | Do not duplicate sessions when fewer samples exist |

## Result interpretation

- **Session latency / TTFT / TPOT / RTF:** duplex session and per-response metrics.
- **Exact/soft match / IA-QTF1 / IDS / NCCS:** proxy metrics after flattening
  session responses onto annotation slots by video time. Not identical to
  official OmniInteract LLM-judge scoring.

## Troubleshooting

- **No sessions sampled:** verify subset maps/videos exist. `1qna` needs
  `videos_bench/**/*.mp4` plus matching `annotations/**/*.json`.
- **Backend rejected:** OmniInteract requires `--backend minicpmo-realtime`.
- **Empty audio extract:** ensure `ffmpeg` can decode the source MP4 audio track.
- **Timeouts on long 1qna videos:** raise `--omniinteract-realtime-timeout-s`.
