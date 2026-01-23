# RDMA Test Configuration Guide

This document explains how to configure RDMA environment for single-node and multi-node testing.

## Table of Contents

- [Docker Container Permissions](#docker-container-permissions)
- [Single-Node Testing](#single-node-testing)
- [Multi-Node Testing](#multi-node-testing)
- [Running Tests](#running-tests)
- [Troubleshooting](#troubleshooting)

---

## Docker Container Permissions

RDMA tests require access to InfiniBand/RoCE devices and system topology. Add the following permissions when running `docker run`.

### Option 1: Minimal Permissions (Recommended)

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

Parameter explanation:
- `--cap-add=SYS_PTRACE`: Allow reading system topology information
- `--cap-add=IPC_LOCK`: Allow memory locking (required for RDMA memory registration)
- `--security-opt seccomp=unconfined`: Disable seccomp restrictions
- `--network=host`: Use host network (required for RDMA)
- `--device=/dev/infiniband`: Mount InfiniBand devices
- `-v /sys/class/infiniband`: Mount IB device info (read-only)

### Option 2: Full Permissions (Quick but not recommended for production)

```bash
docker run -it \
    --privileged \
    --network=host \
    your-image:tag
```

`--privileged` grants full host permissions. Suitable for quick testing but not recommended for production.

---

## Single-Node Testing

When running single-node tests (producer and consumer on the same machine), ensure they use the **same RDMA device**.

### Problem Background

InfiniBand devices use LID (Local Identifier) for routing. Different devices have different LIDs and cannot communicate directly. If no device is specified, Mooncake may assign different devices to connectors, causing handshake failures.

Common error:
```
[Handshake] Failed to modify QP to RTR, check mtu, gid, peer lid, peer qp num: Invalid argument [22]
```

### Solution

**Method 1: Set Environment Variable (Recommended)**

```bash
# List available RDMA devices
ibstat

# Select a device (e.g., mlx5_0)
export RDMA_DEVICE_NAME='mlx5_0'

# Run tests
pytest test_mooncake_rdma.py -v -s
```

**Method 2: Use RoCE Devices**

If the system has RoCE devices (using IPv4 routing), the test code will automatically detect and prefer them. RoCE device GIDs start with `00:00:00:00:00:00:00:00:00:00:ff:ff` (IPv4-mapped).

**Method 3: Ensure MTU Consistency**

Make sure both endpoints use the same MTU:

```bash
# Check device MTU
ibstatus mlx5_0
```

---

## Multi-Node Testing

For multi-node tests, producer and consumer run on different machines connected via InfiniBand switch.

### Prerequisites

1. Both machines have Mooncake and RDMA drivers installed
2. Both machines are in the same InfiniBand subnet
3. Switch is properly configured

### Configuration

**Machine A (Producer):**

```bash
# Set RDMA host IP (InfiniBand interface IP)
export RDMA_TEST_HOST='10.0.0.1'

# Optional: Specify device
export RDMA_DEVICE_NAME='mlx5_0'
```

**Machine B (Consumer):**

```bash
# Set RDMA host IP
export RDMA_TEST_HOST='10.0.0.2'

# Optional: Specify device
export RDMA_DEVICE_NAME='mlx5_0'
```

### Verify Connectivity

```bash
# Ping IB interface
ping 10.0.0.2

# Test RDMA connectivity with ibping
# On Machine B (server)
ibping -S

# On Machine A (client)
ibping -G <Machine_B_GID>
```

---

## Running Tests

### Run All RDMA Tests

```bash
cd tests/distributed/omni_connectors
pytest test_mooncake_rdma.py -v -s
```

### Run Specific Tests

```bash
# Host-to-Host E2E tests
pytest test_mooncake_rdma.py::TestMooncakeRDMAEndToEnd -v -s

# GPU-to-GPU E2E tests
pytest test_mooncake_rdma.py::TestMooncakeRDMAGPUEndToEnd -v -s

# Single test
pytest test_mooncake_rdma.py::TestMooncakeRDMAEndToEnd::test_put_get_tensor_e2e -v -s
```

### Run Diagnostic Script

```bash
cd tests/distributed/omni_connectors/rdma_tests
python diagnose_rdma.py
```

The diagnostic script checks:
- Mooncake TransferEngine availability
- RDMA device discovery and status
- Network interface configuration
- Environment variables
- Docker/container permissions

### Run Correctness Tests (Single-Node)

```bash
cd tests/distributed/omni_connectors/rdma_tests
pytest test_rdma_correctness.py -v -s
```

This runs stress tests including:
- Large payload transfer (100MB, 500MB)
- Concurrent put/get operations
- Multi-threaded allocator stress
- Mixed data types
- Edge cases

---

## Cross-Node Testing

The `test_cross_node.py` script enables testing RDMA transfers between two separate machines.

### Prerequisites

1. Both machines have Mooncake installed
2. Both machines are connected via InfiniBand/RoCE switch
3. Firewall allows ZMQ ports (default: 15500)
4. Same RDMA device name on both nodes (if multiple devices exist)

### Running Cross-Node Tests

**On Machine A (Producer - 10.0.0.1):**

```bash
cd tests/distributed/omni_connectors/rdma_tests

# Optional: specify device if multiple exist
export RDMA_DEVICE_NAME='mlx5_0'

python test_cross_node.py \
    --role producer \
    --local-host 10.0.0.1 \
    --remote-host 10.0.0.2 \
    --tensor-size-mb 100 \
    --num-transfers 3
```

**On Machine B (Consumer - 10.0.0.2):**

```bash
cd tests/distributed/omni_connectors/rdma_tests

export RDMA_DEVICE_NAME='mlx5_0'

python test_cross_node.py \
    --role consumer \
    --local-host 10.0.0.2 \
    --remote-host 10.0.0.1 \
    --tensor-size-mb 100 \
    --num-transfers 3
```

### Cross-Node Test Options

| Option | Description | Default |
|--------|-------------|---------|
| `--role` | `producer` or `consumer` | Required |
| `--local-host` | Local RDMA IP address | Required |
| `--remote-host` | Remote RDMA IP address | Required |
| `--local-port` | Local ZMQ port | 15500 |
| `--remote-port` | Remote ZMQ port | 15500 |
| `--tensor-size-mb` | Tensor size in MB | 100 |
| `--num-transfers` | Number of transfers | 3 |

---

## Troubleshooting

### 1. "Failed to modify QP to RTR" Error

**Cause**: QP handshake failed, usually due to device configuration mismatch.

**Solution**:
```bash
# Force using the same device
export RDMA_DEVICE_NAME='mlx5_0'
```

### 2. "Mooncake TransferEngine is not available"

**Cause**: Mooncake not installed or import failed.

**Solution**:
```bash
# Check Mooncake installation
python -c "from mooncake.engine import TransferEngine; print('OK')"

# Reinstall if needed
pip install mooncake
```

### 3. "Permission denied" accessing /dev/infiniband

**Cause**: Container lacks IB device access permissions.

**Solution**:
```bash
docker run --device=/dev/infiniband --cap-add=IPC_LOCK ...
```

### 4. Test Timeout

**Cause**: RDMA connection establishment failed or network latency.

**Solution**:
```bash
# Check network status
ibstat
ibstatus
```

### 5. GPU Test Failed "CUDA is not available"

**Cause**: CUDA environment not configured or GPU unavailable.

**Solution**:
```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Docker needs NVIDIA runtime
docker run --gpus all ...
```

---

## Environment Variables Reference

| Variable | Description | Example |
|----------|-------------|---------|
| `RDMA_DEVICE_NAME` | Specify RDMA device name | `mlx5_0` |
| `RDMA_TEST_HOST` | Specify test host IP | `10.0.0.1` |
| `MC_TE_METRIC` | Enable Mooncake metrics | `1` |
| `MC_IB_PCI_RELAXED_ORDERING` | Enable PCIe relaxed ordering | `1` |

---

## Test Files Overview

| File | Description |
|------|-------------|
| `test_mooncake_rdma.py` | Comprehensive unit tests for MooncakeRDMAConnector |
| `test_rdma_correctness.py` | Stress tests and correctness verification |
| `test_buffer_management.py` | Memory pool and buffer management tests |
| `test_cross_node.py` | Cross-node (multi-machine) testing script |
| `diagnose_rdma.py` | RDMA environment diagnostic tool |

---

## Test Classes Overview

### test_mooncake_rdma.py

| Test Class | Memory Pool | Description |
|------------|-------------|-------------|
| `TestMooncakeRDMAConnector` | CPU | Basic functionality (put/get/cleanup) |
| `TestMooncakeRDMAEndToEnd` | CPU | Host-to-Host E2E transfer tests |
| `TestMooncakeRDMAConcurrency` | CPU | Concurrency tests |
| `TestMooncakeRDMALifecycle` | CPU | Resource management tests |
| `TestMooncakeRDMAGPUPool` | GPU | GPU memory pool tests |
| `TestMooncakeRDMAGPUEndToEnd` | GPU | GPU-to-GPU E2E transfer tests |

### test_rdma_correctness.py

| Test Class | Description |
|------------|-------------|
| `TestRDMALargePayload` | Large tensor transfer (100MB, 500MB) with MD5 verification |
| `TestRDMAConcurrencyStress` | 10-thread concurrent put/get, allocator stress test |
| `TestRDMAMixedWorkload` | Mixed data types (tensors, bytes, objects) |
| `TestRDMAEdgeCases` | Empty tensor, small tensor, pool exhaustion |
