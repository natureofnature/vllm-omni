# MooncakeTransferEngineConnector

## When to Use

Best for high-performance multi-node data transfer between stages using Mooncake
Transfer Engine. Supports both RDMA and TCP protocols with a managed memory pool,
zero-copy deserialization, and optional GPUDirect RDMA. Applicable to any
inter-stage data (KV caches, request payloads, etc.), not limited to KV cache transfer.

Compared to `MooncakeStoreConnector` (TCP key-value store), this connector
provides **~60x faster** data transfer via RDMA direct memory access.

## Installation

```bash
pip install mooncake-transfer-engine
```

Ensure RDMA drivers are installed on all nodes (e.g., Mellanox OFED for
InfiniBand/RoCE NICs).

## Configuration

Define the connector in runtime:

```yaml
runtime:
  connectors:
    rdma_connector:
      name: MooncakeTransferEngineConnector
      extra:
        role: "auto"                  # Auto-detect role based on context (sender/receiver)
        host: "auto"                  # Auto-detect local RDMA IP
        zmq_port: 50051               # Base ZMQ port for sender listener
        sender_host: "auto"           # Resolved dynamically from orchestrator
        sender_zmq_port: 50051        # Sender's ZMQ port
        protocol: "rdma"              # "rdma" or "tcp"
        device_name: ""               # RDMA device (e.g., "mlx5_0"), empty for auto-detect
        memory_pool_size: 2147483648  # 2GB memory pool
        memory_pool_device: "cpu"     # "cpu" for pinned memory, "cuda" for GPUDirect RDMA
```

Wire stages to the connector:

```yaml
stage_args:
  - stage_id: 0
    output_connectors:
      to_stage_1: rdma_connector

  - stage_id: 1
    input_connectors:
      from_stage_0: rdma_connector
```

## Parameters

### Required

| Parameter | Description |
|---|---|
| `role` | `"auto"` (recommended) or `"receiver"`. Auto-injected by the initialization layer based on `output_connectors` (sender) / `input_connectors` (receiver). Manual override is rarely needed. |
| `host` | Local IP address for RDMA. `"auto"` detects from network interfaces. |
| `protocol` | Transport protocol: `"rdma"` (InfiniBand/RoCE) or `"tcp"`. |

### Memory Pool

| Parameter | Default | Description |
|---|---|---|
| `memory_pool_size` | 1 GB | Total size of the RDMA-registered memory pool in bytes. |
| `memory_pool_device` | `"cpu"` | `"cpu"`: pinned host memory (recommended). `"cuda"`: GPU VRAM for GPUDirect RDMA (requires NIC-GPU direct PCIe connectivity). |

### Networking

| Parameter | Default | Description |
|---|---|---|
| `zmq_port` | 50051 | Base ZMQ port for the sender's metadata listener. |
| `sender_host` | `"auto"` | Sender's IP, resolved from orchestrator. |
| `sender_zmq_port` | 50051 | Sender's ZMQ port. |
| `device_name` | `""` | RDMA device name (e.g., `"mlx5_0"`). Empty for auto-detect. Can also be set via `RDMA_DEVICE_NAME` env var. |

## Memory Pool Modes

| Mode | Config | Data Flow | Best For |
|---|---|---|---|
| CPU Pinned | `memory_pool_device: "cpu"` | GPU → CPU pool → RDMA → CPU pool → GPU | Most hardware topologies (recommended) |
| GPUDirect | `memory_pool_device: "cuda"` | GPU → GPU pool → RDMA (NIC reads GPU BAR1) → GPU pool | NIC-GPU direct PCIe (PIX topology) |

> **Note**: GPUDirect RDMA requires the NIC and GPU to share a direct PCIe
> switch (PIX topology). On systems where they are connected via PXB or NODE,
> CPU pinned memory is faster due to GPU BAR1 bandwidth limitations.

## Environment Variables

| Variable | Description |
|---|---|
| `RDMA_DEVICE_NAME` | Override RDMA device name (e.g., `mlx5_0`). |
| `MC_IB_PCI_RELAXED_ORDERING` | Set to `1` to enable PCIe relaxed ordering for GPUDirect. |

## Docker / Container Setup

RDMA requires host-level device access:

```bash
docker run -it \
    --cap-add=SYS_PTRACE \
    --cap-add=IPC_LOCK \
    --security-opt seccomp=unconfined \
    --network=host \
    --device=/dev/infiniband \
    -v /sys/class/infiniband:/sys/class/infiniband:ro \
    your-image:tag
```

## Performance

Benchmark results on H800 GPUs with mlx5_0 RDMA NIC (~186 MB KV cache):

| Metric | MooncakeStoreConnector | MooncakeTransferEngineConnector (CPU) |
|---|---|---|
| KV transfer wall time | ~810 ms | **~14 ms** |
| RDMA throughput | N/A (TCP) | ~22 GB/s |
| Speedup | 1x | **58x** |

## Troubleshooting

### Quick Diagnostics

```bash
# 1. Check RDMA devices and link status
ibdev2netdev
# Expected: "mlx5_X port 1 ==> <iface> (Up)"
# RoCE devices map to Ethernet interfaces (e.g., enp75s0f0)
# IB devices map to ib0, ib1, etc.

# 2. Check InfiniBand device details
ibstat

# 3. Verify /dev/infiniband is accessible (required in containers)
ls /dev/infiniband/

# 4. Check Mooncake installation
python -c "from mooncake.engine import TransferEngine; print('OK')"

# 5. Check environment variables
echo "RDMA_DEVICE_NAME=${RDMA_DEVICE_NAME:-<not set>}"
echo "MC_IB_PCI_RELAXED_ORDERING=${MC_IB_PCI_RELAXED_ORDERING:-<not set>}"
```

### Common Issues

| Symptom | Cause | Fix |
|---|---|---|
| `Failed to modify QP to RTR` | Cross-NIC QP handshake failure (multi-NIC DGX) | Set `device_name` to a single RoCE NIC (e.g., `mlx5_2`) or set `RDMA_DEVICE_NAME` env var |
| `transport retry counter exceeded` | RDMA path between incompatible NICs | Same as above — restrict to one NIC |
| `zmq.error.Again: Resource temporarily unavailable` | ZMQ recv timeout (transfer took too long) | Check NIC selection; increase data may need longer timeout |
| `Mooncake Engine initialization failed` | Missing RDMA drivers or `/dev/infiniband` | Install Mellanox OFED; in Docker add `--device=/dev/infiniband` |
| `MemoryError` in allocator | Memory pool too small for payload | Increase `memory_pool_size` |
| GPU transfer slower than CPU | GPU BAR1 bandwidth limitation (PXB/NODE topology) | Use `memory_pool_device: "cpu"` instead of `"cuda"` |

### Multi-NIC Environments (DGX)

On DGX machines with 12+ RDMA NICs, only RoCE NICs (with a bound network
interface) reliably support loopback. IB-only NICs may fail cross-NIC QP
handshakes. To identify RoCE NICs:

```bash
ibdev2netdev | grep -v "ib[0-9]"
# RoCE devices show Ethernet interface names like enp75s0f0
```

Then configure the connector:
```yaml
device_name: "mlx5_2"  # or set RDMA_DEVICE_NAME=mlx5_2
```

See the [RDMA Test README](../../../../tests/distributed/omni_connectors/README.md)
for test-specific setup instructions.

For more details on the underlying engine, refer to the
[Mooncake repository](https://github.com/kvcache-ai/Mooncake).
