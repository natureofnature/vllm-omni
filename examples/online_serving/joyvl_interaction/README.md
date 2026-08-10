# JoyAI-VL Interaction Serving

This example supports two deployment modes.

## In-process vLLM-Omni mode

The phase-one path keeps JoyAI session policy in the API Server and sends every
model inference through the ordinary AsyncOmni and Orchestrator request path.
It does not use the MiniCPM resident-request control plane, KV leases, or a
custom scheduler/runtime.

```bash
vllm-omni serve jdopensource/JoyAI-VL-Interaction-Preview \
  --omni \
  --deploy-config vllm_omni/deploy/joyvl_interaction.yaml \
  --served-model-name JoyAI-VL-Interaction-Preview \
  --port 8061
```

Send each tick to `POST /v1/chat/completions`. The body remains OpenAI-compatible
and may include an image plus an optional text query. The in-process mode
requires a stable Session ID. Session metadata can be provided as headers:

- `x-session-id`: required stable business session identifier.
- `x-operation-id`: idempotency key for one input tick.
- `x-session-epoch`: required for reset and subsequent ticks; use `0` before
  the first reset, then use the epoch returned by the server.

The response includes an `interaction` object with the selected action,
session ID, operation ID, and current epoch.

Reset a session:

```bash
curl -X POST http://localhost:8061/v1/session/reset \
  -H 'x-session-id: demo-session' \
  -H 'x-session-epoch: 0'
```

Phase one accepts non-streaming Chat Completions. It preserves session
isolation, operation idempotency, atomic state updates after successful
inference, epoch-based stale-result filtering, and the existing JoyAI
Brain/Memory/WorkingChunk behavior.

## External sidecar mode

The original deployment remains available:

```bash
bash examples/online_serving/joyvl_interaction/scripts/start_all.sh
```

In this mode the interaction server owns JoyAI session state and calls a normal
OpenAI-compatible model server over HTTP. The model server intentionally does
not use the vLLM-Omni pipeline.

The sidecar still accepts a missing Session ID as the legacy `default` Session
and keeps `/v1/streaming/persona`. Its reset protocol now requires
`session_epoch`: send `0` for the first reset and then keep the epoch returned
by the server. The reset response sets `advanced=false` when it recognizes an
idempotent retry; a client that intentionally starts a new run must reset again
with the returned epoch. After a reset, chat and persona requests must send the
current epoch. Older clients that omit it receive `400 missing_session_epoch`
on reset or `409 missing_session_epoch` on later requests; update them before
deployment.

The two modes share the same JoyAI policy implementation. Choose in-process mode
to remove the model HTTP hop; choose sidecar mode when independent scaling and
failure isolation are more important.
