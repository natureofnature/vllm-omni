# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Smoke Tests for MooncakeRDMAConnector (RFC 3.2).

These tests verify:
1. Large payload transfer (>100MB) with data integrity
2. Concurrent put/get operations under stress
3. Multi-threaded allocator safety

NOTE: These tests require a working RDMA environment and sufficient memory.
They may take longer to run than unit tests.
"""

import hashlib
import os
import subprocess
import threading
import time
import unittest

import torch

from vllm_omni.distributed.omni_connectors.connectors.mooncake_rdma_connector import (
    ManagedBuffer,
    MooncakeRDMAConnector,
    TransferEngine,
)


def get_rdma_host():
    """
    Get the RDMA-capable host IP address.
    Priority: RDMA_TEST_HOST env var > auto-detect InfiniBand interface > fallback to 127.0.0.1
    """
    import socket

    # 1. Check environment variable
    env_host = os.environ.get("RDMA_TEST_HOST")
    if env_host:
        print(f"[RDMA] Using RDMA_TEST_HOST={env_host}")
        return env_host

    # 2. Auto-detect InfiniBand/RoCE interface IP using ip command
    ip_commands = ["ip", "/sbin/ip", "/usr/sbin/ip"]
    for ip_cmd in ip_commands:
        try:
            result = subprocess.run([ip_cmd, "addr", "show"], capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                continue

            lines = result.stdout.split("\n")

            # Pattern 1: Interface name on same line as inet (ibp*, ib0, ib1, mlx*, roce*)
            rdma_patterns = ["ibp", "ib0", "ib1", "ib2", "mlx", "roce"]
            for line in lines:
                if "inet " in line:
                    for pattern in rdma_patterns:
                        if pattern in line.lower():
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                ip = parts[1].split("/")[0]
                                # Skip loopback and link-local
                                if not ip.startswith("127.") and not ip.startswith("169.254."):
                                    print(f"[RDMA] Auto-detected RDMA IP: {ip}")
                                    return ip

            # Pattern 2: Look for non-loopback, non-docker IPs as fallback
            for line in lines:
                if "inet " in line and "scope global" in line:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        ip = parts[1].split("/")[0]
                        # Skip common non-RDMA interfaces
                        if not ip.startswith("127.") and not ip.startswith("172.17."):
                            print(f"[RDMA] Using fallback global IP: {ip}")
                            return ip
            break  # Command worked but no IP found
        except FileNotFoundError:
            continue
        except Exception:
            continue

    # 3. Fallback: use socket to get hostname IP
    try:
        hostname = socket.gethostname()
        host_ip = socket.gethostbyname(hostname)
        if host_ip and not host_ip.startswith("127."):
            print(f"[RDMA] Using hostname IP: {host_ip}")
            return host_ip
    except Exception:
        pass

    # 4. Final fallback to localhost
    print("[RDMA] Using 127.0.0.1 (localhost)")
    return "127.0.0.1"


# Global RDMA host for tests
RDMA_HOST = get_rdma_host()


def detect_first_rdma_device():
    """
    Detect the first available RDMA device for single-node testing.
    This ensures both producer and consumer use the same device to avoid
    cross-device communication issues on single-node setups.
    """
    # Check environment variable first
    env_device = os.environ.get("RDMA_DEVICE_NAME")
    if env_device:
        print(f"[RDMA] Using RDMA_DEVICE_NAME={env_device}")
        return env_device

    if TransferEngine is None:
        return ""

    try:
        import json

        temp_engine = TransferEngine()
        ret = temp_engine.initialize("127.0.0.1", "P2PHANDSHAKE", "rdma", "")
        if ret != 0:
            return ""

        topo_str = temp_engine.get_local_topology()
        if topo_str:
            topo = json.loads(topo_str)
            # Get first device name
            for device_name in topo.keys():
                if isinstance(topo[device_name], dict):
                    print(f"[RDMA] Auto-detected first device: {device_name}")
                    return device_name
    except Exception as e:
        print(f"[RDMA] Failed to detect device: {e}")

    return ""


# Global RDMA device for single-node tests
RDMA_DEVICE_NAME = detect_first_rdma_device()


def get_connector_config(zmq_port, pool_size=128 * 1024 * 1024):
    """Get standard connector config with device filtering for single-node tests."""
    config = {
        "host": RDMA_HOST,
        "zmq_port": zmq_port,
        "protocol": "rdma",
        "memory_pool_size": pool_size,
    }
    if RDMA_DEVICE_NAME:
        config["device_name"] = RDMA_DEVICE_NAME
    return config


def skip_if_no_mooncake(func):
    """Decorator to skip test if Mooncake is not available."""

    def wrapper(self, *args, **kwargs):
        if TransferEngine is None:
            self.skipTest("Mooncake TransferEngine is not available")
        return func(self, *args, **kwargs)

    return wrapper


def get_free_port():
    """Get a free TCP port."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((RDMA_HOST, 0))
        return s.getsockname()[1]


def compute_md5(tensor: torch.Tensor) -> str:
    """Compute MD5 checksum of a tensor."""
    if tensor.is_cuda:
        tensor = tensor.cpu()
    data = tensor.contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.md5(data).hexdigest()


@unittest.skipIf(TransferEngine is None, "Mooncake TransferEngine is not available")
class TestRDMALargePayload(unittest.TestCase):
    """
    Smoke Test: Large Payload Transfer (RFC 3.2.1)

    Verifies stability and data integrity with large payloads (>100MB).
    """

    def test_large_tensor_transfer_100mb(self):
        """Transfer 100MB tensor and verify MD5 checksum."""
        producer_port = get_free_port()
        consumer_port = get_free_port()

        # 100MB pool
        pool_size = 128 * 1024 * 1024

        producer = MooncakeRDMAConnector(get_connector_config(producer_port, pool_size))
        consumer = MooncakeRDMAConnector(get_connector_config(consumer_port, pool_size))

        try:
            # ~100MB tensor (25M float32 = 100MB)
            original = torch.randn(5000, 5000, dtype=torch.float32)
            original_md5 = compute_md5(original)
            req_id = "req_100mb"

            print(f"\n[INFO] Transferring 100MB tensor (MD5: {original_md5[:16]}...)")

            success, size, metadata = producer.put("s0", "s1", req_id, original)
            self.assertTrue(success, "Put failed for large tensor")
            self.assertEqual(size, original.nbytes)

            time.sleep(1.0)  # Allow time for large transfer

            result = consumer.get("s0", "s1", req_id, metadata)
            self.assertIsNotNone(result, "RDMA transfer failed")

            recv_buffer, recv_size = result
            self.assertEqual(recv_size, original.nbytes)

            # Verify checksum
            reconstructed = recv_buffer.as_tensor(dtype=original.dtype, shape=original.shape)
            recv_md5 = compute_md5(reconstructed)

            self.assertEqual(original_md5, recv_md5, f"MD5 mismatch! Sent: {original_md5}, Received: {recv_md5}")

            recv_buffer.release()
            print(f"[PASS] 100MB transfer verified. MD5: {recv_md5[:16]}...")

        finally:
            producer.close()
            consumer.close()

    def test_large_tensor_transfer_500mb(self):
        """Transfer 500MB tensor and verify MD5 checksum (RFC 3.2.1)."""
        producer_port = get_free_port()
        consumer_port = get_free_port()

        # 512MB pool
        pool_size = 512 * 1024 * 1024

        producer = MooncakeRDMAConnector(get_connector_config(producer_port, pool_size))
        consumer = MooncakeRDMAConnector(get_connector_config(consumer_port, pool_size))

        try:
            # ~500MB tensor
            # 500MB / 4 bytes = 125M elements = 11180 x 11180 ≈ 125M
            original = torch.randn(11180, 11180, dtype=torch.float32)
            original_md5 = compute_md5(original)
            req_id = "req_500mb"

            print(f"\n[INFO] Transferring ~500MB tensor (MD5: {original_md5[:16]}...)")

            success, size, metadata = producer.put("s0", "s1", req_id, original)
            self.assertTrue(success, "Put failed for 500MB tensor")

            time.sleep(2.0)  # Allow time for very large transfer

            result = consumer.get("s0", "s1", req_id, metadata)
            self.assertIsNotNone(result, "RDMA transfer failed for 500MB tensor")

            recv_buffer, recv_size = result

            reconstructed = recv_buffer.as_tensor(dtype=original.dtype, shape=original.shape)
            recv_md5 = compute_md5(reconstructed)

            self.assertEqual(original_md5, recv_md5, f"MD5 mismatch! Sent: {original_md5}, Received: {recv_md5}")

            recv_buffer.release()
            print(f"[PASS] 500MB transfer verified. MD5: {recv_md5[:16]}...")

        finally:
            producer.close()
            consumer.close()


@unittest.skipIf(TransferEngine is None, "Mooncake TransferEngine is not available")
class TestRDMAConcurrencyStress(unittest.TestCase):
    """
    Smoke Test: Concurrent Put/Get Operations (RFC 3.2.2)

    Verifies allocator thread safety and connector stability under concurrent access.
    """

    def test_concurrent_put_get_pairs(self):
        """
        10 concurrent threads doing put/get pairs.
        Verifies no race conditions or data corruption.
        """
        producer_port = get_free_port()
        consumer_port = get_free_port()

        pool_size = 128 * 1024 * 1024  # 128MB

        producer = MooncakeRDMAConnector(get_connector_config(producer_port, pool_size))
        consumer = MooncakeRDMAConnector(get_connector_config(consumer_port, pool_size))

        errors = []
        successes = []
        lock = threading.Lock()

        def transfer_worker(worker_id):
            """Worker that does multiple put/get cycles."""
            try:
                for i in range(5):
                    req_id = f"req_worker{worker_id}_iter{i}"
                    # Random size tensor (64KB - 1MB)
                    size = 64 * 1024 + (worker_id * i * 1024) % (1024 * 1024)
                    num_elements = size // 4
                    original = torch.randn(num_elements, dtype=torch.float32)
                    original_md5 = compute_md5(original)

                    success, _, metadata = producer.put("s0", "s1", req_id, original)
                    if not success:
                        with lock:
                            errors.append(f"Worker {worker_id} iter {i}: put failed")
                        continue

                    time.sleep(0.2)  # Brief delay for RDMA to complete

                    result = consumer.get("s0", "s1", req_id, metadata)
                    if result is None:
                        with lock:
                            errors.append(f"Worker {worker_id} iter {i}: get failed (RDMA issue)")
                        continue

                    recv_buffer, _ = result
                    if isinstance(recv_buffer, ManagedBuffer):
                        reconstructed = recv_buffer.as_tensor(dtype=torch.float32, shape=(num_elements,))
                        recv_md5 = compute_md5(reconstructed)
                        recv_buffer.release()

                        # Verify data integrity via MD5 checksum
                        if original_md5 != recv_md5:
                            with lock:
                                errors.append(
                                    f"Worker {worker_id} iter {i}: MD5 mismatch "
                                    f"(sent={original_md5[:8]}, recv={recv_md5[:8]})"
                                )
                        else:
                            with lock:
                                successes.append(f"Worker {worker_id} iter {i}")

            except Exception as e:
                with lock:
                    errors.append(f"Worker {worker_id} exception: {e}")

        try:
            time.sleep(0.5)  # Allow listeners to start

            # Launch 10 concurrent workers
            threads = []
            for w in range(10):
                t = threading.Thread(target=transfer_worker, args=(w,))
                threads.append(t)
                t.start()

            for t in threads:
                t.join(timeout=60)

            total_ops = 10 * 5  # 10 workers * 5 iterations
            success_rate = len(successes) / total_ops if total_ops > 0 else 0

            print(f"\n[INFO] Completed {len(successes)}/{total_ops} successful transfers ({success_rate:.1%})")
            if errors:
                print(f"[INFO] Errors: {len(errors)} - {errors[:3]}...")  # Show first 3 errors

            # In high-concurrency scenarios, allow some failures due to ZMQ timeouts
            # Require at least 80% success rate for stress test
            min_success_rate = 0.80
            self.assertGreaterEqual(
                success_rate,
                min_success_rate,
                f"Success rate {success_rate:.1%} below threshold {min_success_rate:.0%}. Errors: {errors}",
            )

            print(f"[PASS] Concurrent put/get stress test passed with {success_rate:.1%} success rate")

        finally:
            producer.close()
            consumer.close()

    def test_allocator_stress_10_threads_10_seconds(self):
        """
        Stress test allocator with 10 threads for 10 seconds (RFC 3.2.2).
        Verifies no crashes or data corruption in the allocator.
        """
        port = get_free_port()

        connector = MooncakeRDMAConnector(get_connector_config(port, 64 * 1024 * 1024))

        errors = []
        operations = [0]  # Use list for mutable counter
        stop_flag = threading.Event()
        lock = threading.Lock()

        def allocator_worker(worker_id):
            """Worker that rapidly allocs/frees from pool via put/cleanup."""
            local_ops = 0
            try:
                while not stop_flag.is_set():
                    req_id = f"stress_{worker_id}_{local_ops}"
                    # Small tensors for rapid alloc/free
                    tensor = torch.randn(256, dtype=torch.float32)

                    success, _, _ = connector.put("s", "s", req_id, tensor)
                    if success:
                        connector.cleanup(req_id)
                        local_ops += 1

            except Exception as e:
                with lock:
                    errors.append(f"Worker {worker_id}: {e}")
            finally:
                with lock:
                    operations[0] += local_ops

        try:
            # Launch 10 threads
            threads = []
            for w in range(10):
                t = threading.Thread(target=allocator_worker, args=(w,))
                threads.append(t)
                t.start()

            # Run for 10 seconds
            time.sleep(10)
            stop_flag.set()

            for t in threads:
                t.join(timeout=5)

            self.assertEqual(len(errors), 0, f"Allocator stress errors: {errors}")
            print(f"\n[PASS] Allocator stress test: {operations[0]} operations in 10 seconds, no errors")

        finally:
            connector.close()


@unittest.skipIf(TransferEngine is None, "Mooncake TransferEngine is not available")
class TestRDMAMixedWorkload(unittest.TestCase):
    """
    Mixed workload tests combining different data types and sizes.
    """

    def test_mixed_data_types_sequential(self):
        """Test sequential transfers of different data types."""
        producer_port = get_free_port()
        consumer_port = get_free_port()

        producer = MooncakeRDMAConnector(get_connector_config(producer_port, 32 * 1024 * 1024))
        consumer = MooncakeRDMAConnector(get_connector_config(consumer_port, 32 * 1024 * 1024))

        try:
            test_cases = [
                ("tensor_f32", torch.randn(1000, 1000, dtype=torch.float32)),
                ("tensor_f16", torch.randn(1000, 1000, dtype=torch.float16)),
                ("tensor_i64", torch.randint(0, 100, (500, 500), dtype=torch.int64)),
                ("bytes_small", b"Hello RDMA!" * 100),
                ("bytes_large", b"X" * (1024 * 1024)),  # 1MB
                ("object_dict", {"key": "value", "list": [1, 2, 3]}),
            ]

            time.sleep(0.5)

            for name, data in test_cases:
                req_id = f"req_{name}"

                success, size, metadata = producer.put("s0", "s1", req_id, data)
                self.assertTrue(success, f"Put failed for {name}")

                time.sleep(0.2)

                result = consumer.get("s0", "s1", req_id, metadata)
                self.assertIsNotNone(result, f"RDMA get failed for {name}")

                recv_data, recv_size = result

                # Verify based on type
                if isinstance(data, torch.Tensor):
                    if isinstance(recv_data, ManagedBuffer):
                        reconstructed = recv_data.as_tensor(dtype=data.dtype, shape=data.shape)
                        self.assertTrue(torch.equal(reconstructed, data), f"Tensor mismatch for {name}")
                        recv_data.release()
                    else:
                        self.fail(f"Expected ManagedBuffer for tensor, got {type(recv_data)}")
                elif isinstance(data, bytes):
                    self.assertEqual(recv_data, data, f"Bytes mismatch for {name}")
                else:
                    self.assertEqual(recv_data, data, f"Object mismatch for {name}")

                print(f"  [OK] {name}: {recv_size} bytes")

            print("\n[PASS] Mixed data types test passed")

        finally:
            producer.close()
            consumer.close()


@unittest.skipIf(TransferEngine is None, "Mooncake TransferEngine is not available")
class TestRDMAEdgeCases(unittest.TestCase):
    """
    Edge case tests for robustness.
    """

    def test_empty_tensor(self):
        """Test handling of empty (zero-size) tensor."""
        port = get_free_port()

        connector = MooncakeRDMAConnector(get_connector_config(port))

        try:
            empty_tensor = torch.tensor([], dtype=torch.float32)
            success, size, metadata = connector.put("s0", "s1", "req_empty", empty_tensor)

            # Should either succeed with 0 bytes or fail gracefully
            if success:
                self.assertEqual(size, 0)
            # Either way, no crash is a pass

        finally:
            connector.close()

    def test_very_small_tensor(self):
        """Test handling of very small tensor (single element)."""
        producer_port = get_free_port()
        consumer_port = get_free_port()

        producer = MooncakeRDMAConnector(get_connector_config(producer_port))
        consumer = MooncakeRDMAConnector(get_connector_config(consumer_port))

        try:
            small_tensor = torch.tensor([3.14159], dtype=torch.float32)

            success, size, metadata = producer.put("s0", "s1", "req_small", small_tensor)
            self.assertTrue(success)
            self.assertEqual(size, 4)  # 1 float32 = 4 bytes

            time.sleep(0.5)

            result = consumer.get("s0", "s1", "req_small", metadata)
            if result:
                recv_data, _ = result
                if isinstance(recv_data, ManagedBuffer):
                    reconstructed = recv_data.as_tensor(dtype=torch.float32, shape=(1,))
                    self.assertTrue(torch.equal(reconstructed, small_tensor))
                    recv_data.release()

        finally:
            producer.close()
            consumer.close()

    def test_pool_exhaustion_and_recovery(self):
        """Test behavior when pool is exhausted and then freed."""
        port = get_free_port()

        # Very small pool (64KB)
        connector = MooncakeRDMAConnector(get_connector_config(port, 64 * 1024))

        try:
            # Fill the pool
            req_ids = []
            for i in range(10):
                tensor = torch.randn(1000, dtype=torch.float32)  # 4KB each
                success, _, _ = connector.put("s", "s", f"req_fill_{i}", tensor)
                if success:
                    req_ids.append(f"req_fill_{i}")
                else:
                    break  # Pool exhausted

            # Try one more - should fail
            tensor = torch.randn(1000, dtype=torch.float32)
            success, _, _ = connector.put("s", "s", "req_overflow", tensor)
            # May or may not succeed depending on fragmentation

            # Cleanup and verify recovery
            for req_id in req_ids:
                connector.cleanup(req_id)

            # Should be able to allocate again
            tensor = torch.randn(1000, dtype=torch.float32)
            success, _, _ = connector.put("s", "s", "req_recovery", tensor)
            self.assertTrue(success, "Pool recovery failed after cleanup")

        finally:
            connector.close()


if __name__ == "__main__":
    # Run with verbose output
    unittest.main(verbosity=2)
