# Shared Realtime protocol boundary

This is the P0a protocol extraction for
[#6592](https://github.com/vllm-project/vllm-omni/issues/6592), adapted to the
engine-owned session architecture in
[#7181](https://github.com/vllm-project/vllm-omni/issues/7181).
It is stacked on the unmerged unified-duplex prototype, not a claim that
Qwen Realtime or streaming video is already supported.

## Ownership

The framework has one session owner: the engine. Realtime protocol state is
part of that session, not a second serving-side session or execution loop.

```text
WebSocket envelope -> Realtime command translation -> typed DuplexCommand
                         Engine session runner -> DuplexModelPlugin -> stages
                    internal output -> Realtime projection -> typed DuplexEvent
WebSocket writer <- event.to_realtime()
```

| Component | Responsibility |
| --- | --- |
| `entrypoints/duplex/realtime_input.py` | Connection handshake, resume envelope and declared wire defaults; delegates command translation |
| `engine/realtime/commands.py` | Wire-format validation and conversion into existing framework commands; does not inspect a live session |
| `engine/realtime/projection.py` | Conversation/item/response bookkeeping, session-dependent command resolution, and output projection; state is held by the engine session |
| `engine/duplex/commands.py`, `events.py` | Framework command/event types and mailbox/wire serialization |
| `engine/duplex/session_runner.py` | Executes commands and owns ordering, cancellation and lifecycle policy |
| `engine/duplex/plugin.py` | Existing model integration boundary: model configuration, append planning and output decisions |

The protocol package lives under `engine/` so engine-side command processing
does not import a serving entrypoint. It deliberately uses the unified
framework's command/event types instead of introducing parallel contracts.

The older RFC calls the model boundary `RealtimeModelAdapter`. On the #7181
architecture, that role belongs to `DuplexModelPlugin`; adding a second model
adapter and session runner would duplicate ownership. The current extraction
does not change the plugin interface or implement a Qwen plugin.

## Preserved behavior

The extraction moves the prototype's `engine/duplex/realtime_commands.py`
and `realtime_events.py` into the shared package and updates their callers.
It preserves the existing protocol, including its compatibility extensions:

- Audio append decoding, session format defaults and speech/overlap hints.
- Conversation items and explicit response create/cancel commands.
- Commit validation against engine-owned input-buffer state. Parsing a commit
  does not itself create a response or invent a model decision.
- Response and item identity, terminal events, and duplicate-done suppression.
- MiniCPM LISTEN/SPEAK events, terminal audio/transcript pairing and the current
  `video_frames` validation.
- The existing handshake, resume fields and legacy wire event names.

This is not a switch to a strict GA-only schema. In particular, preserving
`response.audio.delta` and the MiniCPM `video_frames` extension does not make
them the required schema for future models. The current audio normalization
also remains unchanged: PCM16/G.711 input is normalized to the framework's
16 kHz float-PCM path. Qwen's input-rate and protocol-profile decisions must
be made explicitly during its integration.

## Follow-up boundaries

- **P0b:** Implement the Qwen text/audio model plugin on this session path,
  including the agreed GA/compatibility profile, input audio policy and real
  text/audio lifecycle tests. The shared codec alone does not enable Qwen.
- **Video P1:** Add the `input_image` session store, retention/sampling and
  legacy video-route translation described by
  [#7223](https://github.com/vllm-project/vllm-omni/issues/7223), after or stacked
  on P0b. Do not implement another connection-local model execution loop.

## Validation

`tests/engine/test_realtime_protocol.py` uses the real command translator and
event projector without model weights. It covers text-only response completion
without native LISTEN/SPEAK, MiniCPM audio terminal ordering, input/response
identity, session-local defaults, and malformed input.

Run in the repository's vLLM/vLLM-Omni development container (CPU sufficient;
no model download required):

```bash
cd tests
pytest -q engine/test_realtime_protocol.py -m 'core_model and cpu'
```

These contract tests do not substitute for a MiniCPM WebSocket E2E on the
unified-duplex prototype. That integration validation remains a readiness gate
before this stacked draft can become merge-ready.
