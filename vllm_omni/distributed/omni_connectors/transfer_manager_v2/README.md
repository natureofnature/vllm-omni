# Transfer Manager V2

Unified transfer manager architecture that combines Chunk Transfer and KV Cache Transfer into a single, extensible framework.

## Architecture Overview

### Layer Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                      Facade Layer                               │
│  ┌─────────────────────────┐  ┌─────────────────────────────┐  │
│  │ OmniChunkTransferManager│  │ OmniKVCacheTransferManager  │  │
│  │     (chunk_manager.py)  │  │   (kv_cache_manager.py)     │  │
│  └───────────┬─────────────┘  └─────────────┬───────────────┘  │
└──────────────┼──────────────────────────────┼──────────────────┘
               │                              │
               ▼                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Core Manager Layer                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              OmniTransferManager                         │   │
│  │              (transfer_manager.py)                       │   │
│  │  • submit_send/recv  • sync_send/recv  • poll()         │   │
│  │  • Chunk counting    • State management                  │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
               │                              │
               ▼                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Handler Layer                               │
│  ┌─────────────────────────┐  ┌─────────────────────────────┐  │
│  │      ChunkHandler       │  │      KVCacheHandler         │  │
│  │  (chunk_handler.py)     │  │  (kv_cache_handler.py)      │  │
│  │  • Stage 0/1 logic      │  │  • Block extraction         │  │
│  │  • Embeddings merge     │  │  • KV cache apply           │  │
│  │  • Code chunking        │  │  • Device transfer          │  │
│  └───────────┬─────────────┘  └─────────────┬───────────────┘  │
│              └──────────┬───────────────────┘                   │
│                         ▼                                       │
│              ┌─────────────────────┐                            │
│              │   TransferHandler   │  (base.py)                 │
│              │   Abstract Base     │                            │
│              └─────────────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Transport Layer                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              OmniTransportEngine                         │   │
│  │              (transport_engine.py)                       │   │
│  │  • Async send/recv threads  • Pending/Finished queues   │   │
│  │  • Retry with backoff       • Timeout handling          │   │
│  │  • Transfer statistics                                   │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Connector Layer                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              OmniConnectorManager                        │   │
│  │              (connector_manager.py)                      │   │
│  │  • Lazy initialization      • RDMA role config          │   │
│  │  • Peer info updates        • Connection state          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                         │                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              OmniTransferConfig                          │   │
│  │              (config.py)                                 │   │
│  │  • Stage settings           • Timeout/retry config      │   │
│  │  • Connector type           • Async mode                │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Infrastructure                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              OmniConnector                               │   │
│  │              (RDMA / ZMQ / Redis / etc.)                 │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Layer | Component | Responsibilities |
|-------|-----------|------------------|
| **Facade** | `ChunkManager` / `KVCacheManager` | Backward-compatible API, scheduler integration |
| **Manager** | `OmniTransferManager` | Unified send/recv dispatch, chunk counting, state management |
| **Handler** | `ChunkHandler` / `KVCacheHandler` | Business logic: key generation, data packing/unpacking |
| **Transport** | `OmniTransportEngine` | Async queues, background threads, retry, statistics |
| **Connector** | `OmniConnectorManager` | Connector lifecycle, RDMA role, peer configuration |
| **Config** | `OmniTransferConfig` / `TransferContext` | Configuration parsing, per-transfer context |

---

## Data Flow

### Send Flow

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│   App    │───▶│  Facade  │───▶│ Manager  │───▶│ Handler  │───▶│Transport │
│          │    │          │    │          │    │          │    │  Engine  │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └────┬─────┘
                                                                     │
    save(data)   submit_send()   prepare_send_data()   submit_send() │
                                 build_key()                         │
                                                                     ▼
                                                              ┌──────────┐
                                                              │ Connector│
                                                              │  put()   │
                                                              └──────────┘
```

**Detailed Steps:**

1. **Application** calls `facade.save(data, request)` or `facade.handle_finished_requests()`
2. **Facade** prepares raw input and calls `manager.submit_send(request_id, raw_input)`
3. **Manager** invokes `handler.prepare_send_data(ctx, raw_input)` for business processing
4. **Handler** performs stage-specific logic (merge embeddings / extract KV cache)
5. **Handler** returns processed data (or `None` to buffer)
6. **Manager** calls `handler.build_key(ctx)` to generate transfer key
7. **Manager** submits to `transport.submit_send(from, to, key, data)`
8. **Transport** adds task to `pending_sends` queue
9. **Background thread** picks up task and calls `connector.put()`
10. On completion, task moves to `finished_sends` queue
11. **Application** calls `manager.poll()` to retrieve results

### Recv Flow

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│   App    │───▶│  Facade  │───▶│ Manager  │───▶│ Handler  │───▶│Transport │
│          │    │          │    │          │    │          │    │  Engine  │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └────┬─────┘
                                                                     │
    load(req)    submit_recv()   build_key()       submit_recv()     │
                                                                     ▼
                                                              ┌──────────┐
                                                              │ Connector│
                                                              │  get()   │
                                                              └──────────┘
                                                                     │
                                                                     ▼
                                 process_recv_data()  ◀──────────────┘
                                 (update request)
```

**Detailed Steps:**

1. **Application** calls `facade.load(request)` or `facade.receive_kv_cache(request)`
2. **Facade** calls `manager.submit_recv(request_id, request)`
3. **Manager** invokes `handler.build_key(ctx)` to generate transfer key
4. **Manager** submits to `transport.submit_recv(from, to, key)`
5. **Transport** adds task to `pending_recvs` queue
6. **Background thread** polls `connector.get()` until data available
7. On data received, task moves to `finished_recvs` queue
8. **Application** calls `manager.poll()`
9. **Manager** invokes `handler.process_recv_data(ctx, data, request)` to update request
10. **Manager** returns finished request IDs

---

## Usage Examples

### 1. KV Cache Transfer Manager

#### Basic Usage

```python
from vllm_omni.distributed.omni_connectors.transfer_manager_v2 import (
    OmniKVCacheTransferManager,
)

# Create from config dict
config = {
    "connector_config": {
        "type": "MooncakeRDMAConnector",
        "host": "192.168.1.100",
        "zmq_port": 50051,
    },
    "stage_id": 0,
    "omni_from_stage": 0,
    "omni_to_stage": 1,
    "need_send_cache": True,
}
manager = OmniKVCacheTransferManager.from_dict(config)

# Send KV cache for finished requests
finished_reqs = {
    "req_001": {"block_ids": [0, 1, 2], "seq_len": 128},
    "req_002": {"block_ids": [3, 4], "seq_len": 64},
}

processed = manager.handle_finished_requests_kv_transfer(
    finished_reqs=finished_reqs,
    kv_caches=model.kv_caches,  # List of KV cache tensors
    block_size=16,
    cache_dtype="float16",
)
print(f"Processed requests: {processed}")

# Cleanup
manager.stop()
```

#### Receiving KV Cache

```python
# Receiver side config
config = {
    "connector_config": {
        "type": "MooncakeRDMAConnector",
        "sender_host": "192.168.1.100",
        "sender_zmq_port": 50151,
    },
    "stage_id": 1,
    "need_recv_cache": True,
}
manager = OmniKVCacheTransferManager.from_dict(config)

# Receive and apply KV cache to request
success = manager.receive_kv_cache(
    req=request,  # Request object with request_id
    target_device=torch.device("cuda:0"),
)

if success:
    # request.past_key_values is now populated
    print(f"KV cache received for {request.request_id}")
```

#### Alternative: Receive Raw Data

```python
# Receive raw data without auto-apply
data, size = manager.receive_kv_cache_for_request(
    request_id="req_001",
    target_device=torch.device("cuda:0"),
)

if data:
    # Manually apply to request
    manager.apply_kv_cache_to_request(request, data)
```

---

### 2. Chunk Transfer Manager

#### Basic Usage

```python
from vllm_omni.distributed.omni_connectors.transfer_manager_v2 import (
    OmniChunkTransferManager,
)

# Create from existing connector (legacy compatibility)
manager = OmniChunkTransferManager.from_connector(connector)

# Or create from config
config = {
    "connector_config": {"type": "RedisConnector", "host": "localhost"},
    "stage_id": 1,
    "async_mode": True,
}
manager = OmniChunkTransferManager.from_dict(config)
```

#### Sending Chunks

```python
def custom_process_func(pooling_output, request):
    """Extract payload from pooling output."""
    return {
        "thinker_embeddings": pooling_output.get("embeddings"),
        "thinker_hidden_states": pooling_output.get("hidden_states"),
        "thinker_input_ids": list(request.prompt_token_ids),
        "finished": request.is_finished,
    }

# Send chunk data
submitted = manager.save(
    pooling_output=model_output,
    request=request,
    custom_process_input_func=custom_process_func,
)

if submitted:
    print(f"Chunk submitted for {request.request_id}")
```

#### Receiving Chunks

```python
# Request to load chunk (async)
task_id = manager.load(request)

# Later, check for finished requests
finished = manager.get_finished_requests()

for req_id in finished:
    # Data is automatically applied to request.additional_information
    print(f"Chunk ready for {req_id}")
```

#### Scheduler Integration

```python
# In scheduler loop
class Scheduler:
    def __init__(self, chunk_manager):
        self.chunk_manager = chunk_manager

    def schedule(self, waiting_queue, running_queue):
        # Process pending chunks - moves requests to waiting_for_chunk state
        num_waiting = self.chunk_manager.process_pending_chunks(
            waiting_queue=waiting_queue,
            running_queue=running_queue,
        )

        # ... do scheduling logic ...

        # Restore requests that received chunks back to queues
        self.chunk_manager.restore_queues(
            waiting_queue=waiting_queue,
            running_queue=running_queue,
        )

        # Clean up ready chunks from scheduler output
        self.chunk_manager.filter_scheduler_output(scheduler_output)

        return scheduler_output
```

---

### 3. Creating Custom Handler

To add a new transfer type, implement `TransferHandler`:

```python
from vllm_omni.distributed.omni_connectors.transfer_manager_v2.handlers.base import (
    TransferHandler,
)
from vllm_omni.distributed.omni_connectors.transfer_manager_v2.core.config import (
    TransferContext,
)

class MyCustomHandler(TransferHandler):
    """Custom handler for new data type."""

    def build_key(self, ctx: TransferContext) -> str:
        """Build unique transfer key."""
        return f"custom_{ctx.from_stage}_to_{ctx.to_stage}_{ctx.request_id}"

    def prepare_send_data(self, ctx: TransferContext, raw_input: Any) -> Any | None:
        """Process data before sending."""
        # Return None to skip/buffer
        # Return processed data to send
        return {"processed": raw_input, "timestamp": time.time()}

    def process_recv_data(self, ctx: TransferContext, data: Any, request: Any) -> None:
        """Apply received data to request."""
        request.custom_data = data.get("processed")

    def on_send_complete(self, ctx: TransferContext, success: bool, size: int) -> None:
        """Called after send completes."""
        if success:
            logger.info(f"Sent {size} bytes for {ctx.request_id}")

    def on_recv_complete(self, ctx: TransferContext, data: Any, size: int) -> None:
        """Called after recv completes."""
        logger.info(f"Received {size} bytes for {ctx.request_id}")


# Use with OmniTransferManager
from vllm_omni.distributed.omni_connectors.transfer_manager_v2 import (
    OmniTransferManager,
    OmniTransportEngine,
    OmniConnectorManager,
    OmniTransferConfig,
)

config = OmniTransferConfig.from_dict(cfg)
connector_mgr = OmniConnectorManager(config)
transport = OmniTransportEngine(connector_mgr, config)
handler = MyCustomHandler()

manager = OmniTransferManager(transport, handler, config)

# Now use manager.submit_send(), manager.submit_recv(), etc.
```

---

## Configuration Reference

### OmniTransferConfig Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `connector_type` | `str` | `None` | Connector class name (e.g., "MooncakeRDMAConnector") |
| `connector_extra` | `dict` | `{}` | Extra connector config (host, port, etc.) |
| `stage_id` | `int/str` | `None` | Current stage identifier |
| `from_stage` | `int/str` | `None` | Source stage for transfers |
| `to_stage` | `int/str` | `None` | Target stage for transfers |
| `role` | `str` | `"auto"` | RDMA role: "sender", "receiver", or "auto" |
| `need_send` | `bool` | `False` | Whether this instance sends data |
| `need_recv` | `bool` | `False` | Whether this instance receives data |
| `recv_timeout` | `float` | `30.0` | Receive timeout in seconds |
| `max_retries` | `int` | `3` | Max retry attempts |
| `async_mode` | `bool` | `True` | Enable async background threads |
| `poll_interval` | `float` | `0.001` | Poll interval in seconds (1ms) |

### Example Config Dict

```python
config = {
    "connector_config": {
        "type": "MooncakeRDMAConnector",
        "host": "0.0.0.0",
        "zmq_port": 50051,
        "role": "auto",
    },
    "stage_id": 0,
    "omni_from_stage": 0,
    "omni_to_stage": 1,
    "need_send_cache": True,
    "need_recv_cache": False,
    "recv_timeout": 30.0,
    "max_retries": 3,
    "async_mode": True,
}
```

---

## File Structure

```
transfer_manager_v2/
├── __init__.py              # Public exports
├── README.md                # This file
├── transfer_manager.py      # OmniTransferManager (core)
├── chunk_manager.py         # OmniChunkTransferManager (facade)
├── kv_cache_manager.py      # OmniKVCacheTransferManager (facade)
├── core/
│   ├── __init__.py
│   ├── config.py            # OmniTransferConfig, TransferContext
│   ├── connector_manager.py # OmniConnectorManager
│   └── transport_engine.py  # OmniTransportEngine
└── handlers/
    ├── __init__.py
    ├── base.py              # TransferHandler (abstract)
    ├── chunk_handler.py     # ChunkHandler
    └── kv_cache_handler.py  # KVCacheHandler
```

---

## Design Principles

1. **Separation of Concerns**: Each layer handles a specific responsibility
2. **Pluggable Handlers**: Add new transfer types by implementing `TransferHandler`
3. **Backward Compatibility**: Facade classes maintain original API
4. **Async by Default**: Background threads for non-blocking transfers
5. **Unified State Management**: Single source of truth for request state
6. **Extensible Configuration**: `OmniTransferConfig` supports multiple config sources
