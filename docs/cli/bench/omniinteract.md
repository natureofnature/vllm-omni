# OmniInteract Realtime Benchmark

This runner replays the official OmniInteract videos against an already running
MiniCPM-o native-duplex Realtime endpoint. It is intentionally separate from
`vllm bench serve`: one benchmark case is a long-lived WebSocket session, not a
single token request.

## Run locally

Start `openbmb/MiniCPM-o-4_5` with its duplex recipe, then run:

```bash
vllm bench omniinteract --omni \
  --model openbmb/MiniCPM-o-4_5 \
  --base-url http://127.0.0.1:8000 \
  --endpoint /v1/realtime \
  --ref-audio /path/to/reference.wav \
  --data-root /path/to/OmniInteract \
  --subsets 1q1a 1q1a_math 1qna \
  --output-dir ./omniinteract-output \
  --num-prompts 3 \
  --max-concurrency 2
```

When `--data-root` is omitted, the runner downloads
`lucky-lance/OmniInteract` from Hugging Face. `--num-prompts` is the total
across all selected subsets, not a per-subset count; `0` runs every selected
video. Audio is sent as 16 kHz PCM16 in 200 ms chunks,
video is sampled at 1 FPS, and input is paced in real time. These official
replay semantics are fixed by the public CLI.
`--ref-audio` is required because the MiniCPM-o native-duplex runtime uses it
to condition audio output. `--media-timeout-s` bounds each direct
`ffprobe`/`ffmpeg` command; preprocessing and WebSocket concurrency share the
`--max-concurrency` bound.

Use `--require-response` only for response-required functional E2E samples. A
normal OmniInteract model may choose LISTEN for an entire video; that is a valid
benchmark result with zero answer accuracy, so the default runner still writes
a silent WAV and empty transcript for it.

## Completion and artifacts

A case is transport-complete after the final commit is acknowledged, every
observed `response.created` identity has exactly one `response.done`, the
response set remains quiet for the settle window, playback is acknowledged,
and `session.close` is confirmed. Server errors, failed responses, duplicate or
orphan response identities, expired sessions, malformed PCM, and early
WebSocket closure fail the case.

Each successful case writes:

- `output.wav` (24 kHz mono PCM16, silence-padded or clipped to the
  ceil-second input horizon used by the official MiniCPM video writer);
- `wav_transcript.json`;
- `events.json` with audio payloads removed;
- `result.json`;
- `.done`, written last.

Failures remove stale success markers and write `.failed.json`. The output root
also contains `batch_summary.json` (with per-result `status`) and an
official-compatible `official_eval_manifest.jsonl` (`sample_id`, `gt_json`,
`model_json`, `scene_type`) for a later accuracy workflow. `result.json`
records `audio_clipped_bytes` at the official horizon and
`audio_overwritten_bytes` in the WAV; a transport-successful case with either
count above zero remains in the batch summary but is excluded from the
evaluator manifest.

The current Realtime protocol does not expose accepted/processed input identity
on `main`. Consequently, `.done` proves transport, response-lifecycle, and
artifact completeness; it does not prove that the final commit was attributed
to a particular model decision or that the answer is accurate. ASR, forced
alignment, LLM judging, and Nightly orchestration are separate follow-ups.

## E2E reuse

Tests can call `run_omniinteract_case()` or
`run_omniinteract_benchmark()` directly. The same runner supports deterministic
subset selection, `--num-prompts`, bounded preprocessing/WebSocket concurrency,
per-case artifacts, and a batch manifest, so a Nightly test can select four
response-required videos from one subset per invocation with concurrency two
without duplicating the client implementation.
