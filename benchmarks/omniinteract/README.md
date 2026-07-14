# OmniInteract Serving Benchmark

OmniInteract evaluates audio-visual question answering over video clips. The
vLLM-Omni integration uses `vllm bench serve --omni` and keeps dataset timing
and labels independent from model-specific request formats.

The current implementation is a **clip profile**:

- `1q1a` and `1q1a_math` use `subvideos/<video>_<qa>.mp4`.
- Models that consume a spoken question use the matching
  `audios/<video>_<qa>.wav`.
- `1qna` is flattened into independent requests; it is not a continuous
  session.
- IA-QTF1, IDS, and NCCS are estimates from independent requests. Results are
  emitted with `omniinteract_profiles=["clip"]` and
  `omniinteract_official_compatible=false`.

Continuous media-clock streaming, interruption, and nested-session evaluation
are separate follow-up work.

## Requirements

Install vLLM-Omni and prepare the
[`lucky-lance/OmniInteract`](https://huggingface.co/datasets/lucky-lance/OmniInteract)
dataset. `--dataset-path` may be:

- the Hugging Face dataset ID;
- a local directory containing `data.tar.gz`; or
- an extracted directory containing `1q1a/`, `1q1a_math/`, and `1qna/`.

Profiles whose `content_order` contains `audio` require synthesized question
WAV files. Missing WAV rows are skipped with a warning.

By default requests contain `file://` media URLs. Start the server with an
appropriate `--allowed-local-media-path`, or pass
`--omniinteract-inline-local-video` to embed MP4/WAV data in each request.

## Model special config

All request differences are selected with:

```text
--omniinteract-model-special-config <preset | inline-json | json-file>
```

Built-in presets:

| Preset | User content | Important request fields |
|---|---|---|
| `video` | video + text question | `use_audio_in_video=true` |
| `minicpmo_4_5` | question audio + video | text/audio output, `use_audio_in_video=false`, MiniCPM TTS template |
| `aura` | question audio + video | AURA sampling stages and TTS `additional_information` |

The JSON schema is:

```json
{
  "preset": "video",
  "name": "my-model",
  "content_order": ["audio", "video"],
  "system_prompt": "You are a helpful audio-visual assistant.",
  "extra_body": {
    "modalities": ["text", "audio"],
    "mm_processor_kwargs": {
      "use_audio_in_video": false
    }
  }
}
```

Rules:

- `preset` supplies defaults; all other fields are optional overrides.
- Nested `extra_body` objects are deep-merged with preset defaults.
- `content_order` must include `video` and at least one question carrier:
  `audio` or `question`.
- Model config cannot mark a clip run as official-compatible.
- `--omniinteract-input-mode` and `--omniinteract-aura-tts-*` remain
  deprecated AURA compatibility options. Prefer the unified config.

## Example: MiniCPM-o 4.5

Start the default two-GPU server:

```bash
DATASET_ROOT=/data/models/datasets/OmniInteract

vllm serve openbmb/MiniCPM-o-4_5 --omni \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8099 \
  --allowed-local-media-path "${DATASET_ROOT}"
```

Run a performance benchmark:

```bash
vllm bench serve --omni \
  --host 127.0.0.1 \
  --port 8099 \
  --backend openai-chat-omni \
  --endpoint /v1/chat/completions \
  --model openbmb/MiniCPM-o-4_5 \
  --dataset-name omniinteract \
  --dataset-path "${DATASET_ROOT}" \
  --omniinteract-subsets 1q1a,1q1a_math \
  --omniinteract-model-special-config minicpmo_4_5 \
  --num-prompts 32 \
  --num-warmups 2 \
  --max-concurrency 1 \
  --no-oversample \
  --percentile-metrics ttft,tpot,itl,e2el,audio_ttfp,audio_rtf,audio_duration \
  --print-stage \
  --save-result \
  --result-dir ./omniinteract_results \
  --result-filename minicpmo_4_5_c1.json
```

Add `--omniinteract-eval` for proxy QA metrics:

```bash
vllm bench serve --omni \
  --host 127.0.0.1 \
  --port 8099 \
  --backend openai-chat-omni \
  --endpoint /v1/chat/completions \
  --model openbmb/MiniCPM-o-4_5 \
  --dataset-name omniinteract \
  --dataset-path "${DATASET_ROOT}" \
  --omniinteract-subsets 1q1a,1q1a_math \
  --omniinteract-model-special-config minicpmo_4_5 \
  --num-prompts 32 \
  --max-concurrency 1 \
  --no-oversample \
  --omniinteract-eval \
  --omniinteract-save-eval-items \
  --save-result \
  --save-detailed \
  --result-dir ./omniinteract_results \
  --result-filename minicpmo_4_5_eval.json
```

The MiniCPM preset sends each spoken question exactly once as `audio_url`,
followed by `video_url`. It does not duplicate the question transcript in the
user message. Its merged request body contains:

```json
{
  "modalities": ["text", "audio"],
  "mm_processor_kwargs": {
    "use_audio_in_video": false
  },
  "chat_template_kwargs": {
    "use_tts_template": true
  }
}
```

## Example: AURA CustomVoice

The AURA preset defaults to Base TTS. For CustomVoice, override
`additional_information` in the unified config:

```bash
AURA_CONFIG='{
  "preset": "aura",
  "extra_body": {
    "additional_information": {
      "tts_task_type": "CustomVoice",
      "tts_language": "English",
      "tts_speaker": "Vivian"
    }
  }
}'

vllm bench serve --omni \
  --host 127.0.0.1 \
  --port 8666 \
  --backend openai-chat-omni \
  --endpoint /v1/chat/completions \
  --model /data/models/AURA \
  --dataset-name omniinteract \
  --dataset-path "${DATASET_ROOT}" \
  --omniinteract-subsets 1q1a,1q1a_math \
  --omniinteract-model-special-config "${AURA_CONFIG}" \
  --num-prompts 32 \
  --max-concurrency 1 \
  --no-oversample \
  --percentile-metrics ttft,tpot,itl,e2el,audio_ttfp,audio_rtf,audio_duration \
  --print-stage
```

For Base TTS, supply `tts_ref_audio` and `tts_ref_text` in the same
`additional_information` object. A JSON file is easier to maintain:

```json
{
  "preset": "aura",
  "extra_body": {
    "additional_information": {
      "tts_task_type": "Base",
      "tts_language": "English",
      "tts_ref_audio": "/data/voices/reference.wav",
      "tts_ref_text": "Reference transcript."
    }
  }
}
```

```bash
vllm bench serve --omni \
  ... \
  --omniinteract-model-special-config ./aura-base.json
```

Paths inside `extra_body` are passed to the server unchanged. Use paths that
are valid in the server process or use URLs understood by that model.

## Example: native video-audio model

For a model that reads the original audio track from the video and receives
the question as text:

```bash
vllm bench serve --omni \
  --host 127.0.0.1 \
  --port 8000 \
  --backend openai-chat-omni \
  --endpoint /v1/chat/completions \
  --model "${MODEL}" \
  --dataset-name omniinteract \
  --dataset-path "${DATASET_ROOT}" \
  --omniinteract-subsets 1q1a,1q1a_math \
  --omniinteract-model-special-config video \
  --num-prompts 32 \
  --max-concurrency 1 \
  --no-oversample
```

## Example: custom model JSON file

Create `my-model.json`:

```json
{
  "preset": "minicpmo_4_5",
  "name": "my-audio-video-model",
  "system_prompt": "Answer the spoken question using the video.",
  "extra_body": {
    "chat_template_kwargs": {
      "custom_template_flag": true
    },
    "custom_request_option": "value"
  }
}
```

Then invoke:

```bash
vllm bench serve --omni \
  ... \
  --omniinteract-model-special-config ./my-model.json
```

The custom fields are delivered as top-level OpenAI request fields after
merging with any run-level `--extra-body`.

## Important options

| Option | Description |
|---|---|
| `--omniinteract-root` | Explicit local extracted dataset root |
| `--omniinteract-subsets` | Comma-separated subset list |
| `--omniinteract-inline-local-video` | Embed local video and audio as data URLs |
| `--omniinteract-model-special-config` | Preset, inline JSON, or JSON file |
| `--omniinteract-eval` | Compute proxy QA and estimated interaction metrics |
| `--omniinteract-save-eval-items` | Include per-request evaluation rows |
| `--no-oversample` | Do not duplicate rows when fewer samples are available |

## Result interpretation

- **TTFT/TPOT/ITL:** text-generation latency.
- **E2EL:** complete request latency.
- **Audio TTFP:** first audio packet observed by the benchmark client. A model
  that returns one complete WAV does not provide true streaming TTFP.
- **Audio RTF:** inference time divided by produced audio duration.
- **Exact/soft match:** clip-level QA proxy.
- **IA-QTF1/IDS/NCCS:** estimated for clip runs and not directly comparable to
  official continuous-session OmniInteract results.

Use performance-only runs for latency/throughput comparisons. Enable evaluation
only when the additional per-request scoring output is needed.

## Troubleshooting

- **No requests sampled:** verify the selected subset contains generated
  `subvideos/`; audio profiles also require matching `audios/` WAV files.
- **Server cannot open `file://` media:** configure
  `--allowed-local-media-path` or use `--omniinteract-inline-local-video`.
- **AURA Base TTS fails:** provide both `tts_ref_audio` and `tts_ref_text`.
- **Empty MiniCPM audio:** confirm the server uses the MiniCPM-o 4.5 deploy
  config and the request contains `chat_template_kwargs.use_tts_template=true`.
- **Result marked non-official:** expected for the current clip profile.
