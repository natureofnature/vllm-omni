#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Invoke the unchanged official client. The server is started separately.
set -euo pipefail
: "${MODEL_PATH:?Set MODEL_PATH to the local MiniCPM-o-4_5 snapshot}"
: "${RESULT_DIR:?Set RESULT_DIR for this server start}"
task_evidence_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
task_base_url=${BASE_URL:-http://127.0.0.1:18581}
export SEED_TTS_WER_EVAL=0 SEED_TTS_UTMOS_EVAL=0 SEED_TTS_SIM_EVAL=0

bench() {
  local task_split=$1 task_count=$2 task_c=$3 task_name=$4 task_ready_timeout=${5:-0}
  vllm bench serve --omni \
    --backend openai-chat-omni --base-url "$task_base_url" \
    --endpoint /v1/chat/completions --model crossref-test \
    --tokenizer "$MODEL_PATH" --trust-remote-code \
    --dataset-name seed-tts --dataset-path "$task_evidence_dir/inputs/$task_split" \
    --seed-tts-locale en \
    --seed-tts-system-prompt 'Please read the following text out loud in English.' \
    --disable-shuffle --num-prompts "$task_count" --output-len 256 --seed 42 \
    --request-rate inf --max-concurrency "$task_c" --num-warmups 0 \
    --ready-check-timeout-sec "$task_ready_timeout" --extra-body "$(< "$task_evidence_dir/extra-body.json")" \
    --percentile-metrics audio_ttfp,audio_rtf,audio_duration,e2el \
    --metric-percentiles 50,95 --disable-tqdm --save-result --save-detailed \
    --result-dir "$RESULT_DIR/c$task_c" --result-filename "$task_name.json"
}

# Run after /health returns 200. The warmup and discard measurements are excluded.
bench warmup 32 16 warmup 60
if [[ ${C4_QUALITY_ONLY:-0} == 1 ]]; then
  for task_name in discard r1 r2; do
    bench evaluation 64 1 "$task_name"
  done
  bench evaluation 64 4 discard
  : "${WHISPER_PATH:?Set WHISPER_PATH to the local whisper-large-v3 snapshot}"
  export SEED_TTS_WER_EVAL=1 SEED_TTS_WER_SAVE_ITEMS=1
  export SEED_TTS_EVAL_DEVICE=cuda:0 SEED_TTS_HF_WHISPER_MODEL="$WHISPER_PATH"
  for task_name in quality1 quality2; do
    export SEED_TTS_WER_SAVE_AUDIO_DIR="$RESULT_DIR/audio-c4-$task_name"
    bench evaluation 64 4 "$task_name"
  done
  exit 0
fi
for task_c in ${CONCURRENCIES:-1 4 8 16 32}; do
  bench evaluation 64 "$task_c" discard
  bench evaluation 64 "$task_c" r1
  bench evaluation 64 "$task_c" r2
done

# ASR evaluation runs separately, after timing, and uses the official evaluator.
if [[ ${RUN_QUALITY:-1} == 1 ]]; then
  : "${WHISPER_PATH:?Set WHISPER_PATH to the local whisper-large-v3 snapshot}"
  export SEED_TTS_WER_EVAL=1 SEED_TTS_WER_SAVE_ITEMS=1
  export SEED_TTS_EVAL_DEVICE=cuda:0 SEED_TTS_HF_WHISPER_MODEL="$WHISPER_PATH"
  for task_c in 1 16; do
    export SEED_TTS_WER_SAVE_AUDIO_DIR="$RESULT_DIR/audio-c$task_c"
    bench evaluation 64 "$task_c" quality
  done
fi
