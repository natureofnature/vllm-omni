# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Integration tests for MooncakeRDMAConnector (RFC 3.1.3, 3.2).
These tests require Mooncake TransferEngine and RDMA environment.
"""

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


def skip_if_no_mooncake(func):
    """Decorator to skip test if Mooncake is not available."""

    def wrapper(self, *args, **kwargs):
        if TransferEngine is None:
            self.skipTest("Mooncake TransferEngine is not available")
        return func(self, *args, **kwargs)

    return wrapper


class TestMooncakeRDMAConnector(unittest.TestCase):
    """Integration tests for MooncakeRDMAConnector (RFC 3.1.3)"""

    @staticmethod
    def _get_free_port():
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((RDMA_HOST, 0))
            return s.getsockname()[1]

    @skip_if_no_mooncake
    def test_initialization(self):
        """Verify connector initialization with memory pool."""
        port = self._get_free_port()
        config = {
            "host": RDMA_HOST,
            "zmq_port": port,
            "protocol": "rdma",
            "memory_pool_size": 1024 * 1024,  # 1MB
            "memory_pool_device": "cpu",
        }

        connector = MooncakeRDMAConnector(config)
        try:
            self.assertIsNotNone(connector.rpc_port)
            self.assertNotEqual(connector.rpc_port, 0)
            self.assertEqual(connector.pool_size, 1024 * 1024)
            self.assertEqual(connector.pool_device, "cpu")
            self.assertTrue(connector.pool.is_pinned())
        finally:
            connector._stop_event.set()
            time.sleep(0.1)

    @skip_if_no_mooncake
    def test_put_tensor(self):
        """Test put with tensor data (copy path)."""
        port = self._get_free_port()
        config = {"host": "127.0.0.1", "zmq_port": port, "protocol": "rdma"}
        connector = MooncakeRDMAConnector(config)

        try:
            tensor = torch.randn(100, dtype=torch.float32)
            req_id = "req_tensor_1"

            success, size, metadata = connector.put("stage0", "stage1", req_id, tensor)

            self.assertTrue(success)
            self.assertEqual(size, tensor.nbytes)
            internal_key = MooncakeRDMAConnector._make_key(req_id, "stage0", "stage1")
            self.assertIn(internal_key, connector._local_buffers)
            self.assertTrue(metadata["is_fast_path"])
        finally:
            connector._stop_event.set()
            time.sleep(0.1)

    @skip_if_no_mooncake
    def test_put_bytes(self):
        """Test put with bytes data."""
        port = self._get_free_port()
        config = {"host": "127.0.0.1", "zmq_port": port, "protocol": "rdma"}
        connector = MooncakeRDMAConnector(config)

        try:
            data = b"hello world rdma test"
            req_id = "req_bytes_1"

            success, size, metadata = connector.put("stage0", "stage1", req_id, data)

            self.assertTrue(success)
            self.assertEqual(size, len(data))
            self.assertTrue(metadata["is_fast_path"])
        finally:
            connector._stop_event.set()
            time.sleep(0.1)

    @skip_if_no_mooncake
    def test_put_python_object(self):
        """Test put with arbitrary Python object (serialization path)."""
        port = self._get_free_port()
        config = {"host": "127.0.0.1", "zmq_port": port, "protocol": "rdma"}
        connector = MooncakeRDMAConnector(config)

        try:
            data = {"key": "value", "list": [1, 2, 3], "nested": {"a": 1}}
            req_id = "req_obj_1"

            success, size, metadata = connector.put("stage0", "stage1", req_id, data)

            self.assertTrue(success)
            self.assertGreater(size, 0)
            self.assertFalse(metadata["is_fast_path"])  # Needs deserialization
        finally:
            connector._stop_event.set()
            time.sleep(0.1)

    @skip_if_no_mooncake
    def test_cleanup_releases_internal_buffer(self):
        """Verify cleanup releases internally created buffers."""
        port = self._get_free_port()
        config = {"host": "127.0.0.1", "zmq_port": port, "protocol": "rdma"}
        connector = MooncakeRDMAConnector(config)

        try:
            tensor = torch.randn(100, dtype=torch.float32)
            req_id = "req_cleanup_test"

            success, _, _ = connector.put("stage0", "stage1", req_id, tensor)
            self.assertTrue(success)
            internal_key = MooncakeRDMAConnector._make_key(req_id, "stage0", "stage1")
            self.assertIn(internal_key, connector._local_buffers)

            # Manual cleanup using raw key + stage info (tests user-friendly API)
            connector.cleanup(req_id, from_stage="stage0", to_stage="stage1")

            self.assertNotIn(internal_key, connector._local_buffers)
        finally:
            connector._stop_event.set()
            time.sleep(0.1)

    @skip_if_no_mooncake
    def test_health_check(self):
        """Verify health check returns correct status."""
        port = self._get_free_port()
        config = {
            "host": RDMA_HOST,
            "zmq_port": port,
            "protocol": "rdma",
            "memory_pool_size": 2 * 1024 * 1024,
            "memory_pool_device": "cpu",
        }
        connector = MooncakeRDMAConnector(config)

        try:
            health = connector.health()
            self.assertEqual(health["status"], "healthy")
            self.assertEqual(health["protocol"], "rdma")
            self.assertEqual(health["pool_device"], "cpu")
            self.assertEqual(health["pool_size"], 2 * 1024 * 1024)
        finally:
            connector._stop_event.set()
            time.sleep(0.1)


def detect_roce_devices():
    """
    Detect RoCE devices that use IPv4/RoCEv2 (compatible for same-node loopback).
    Returns a comma-separated list of device names, or empty string for all devices.
    """
    try:
        # Try to get topology from a temporary TransferEngine instance
        temp_engine = TransferEngine()
        # Initialize with P2P and get topology
        ret = temp_engine.initialize("127.0.0.1", "P2PHANDSHAKE", "rdma", "")
        if ret != 0:
            return ""

        import json

        topo_str = temp_engine.get_local_topology()
        if topo_str:
            topo = json.loads(topo_str)
            # Find devices with IPv4 GID (RoCE devices)
            roce_devices = []
            for device_name, device_info in topo.items():
                if isinstance(device_info, dict):
                    gid = device_info.get("gid", "")
                    # IPv4-mapped IPv6 or RoCEv2 GID starts with "00:00:00:00:00:00:00:00:00:00:ff:ff"
                    if gid.startswith("00:00:00:00:00:00:00:00:00:00:ff:ff"):
                        roce_devices.append(device_name)
            if roce_devices:
                return ",".join(roce_devices)
    except Exception as e:
        print(f"[RDMA] Failed to detect RoCE devices: {e}")

    return ""


# Detect compatible devices for same-node testing
RDMA_DEVICE_NAME = os.environ.get("RDMA_DEVICE_NAME", "")
if not RDMA_DEVICE_NAME and TransferEngine is not None:
    # Auto-detect RoCE devices for better same-node loopback compatibility
    RDMA_DEVICE_NAME = detect_roce_devices()
    if RDMA_DEVICE_NAME:
        print(f"[RDMA] Auto-detected RoCE devices: {RDMA_DEVICE_NAME}")


@unittest.skipIf(TransferEngine is None, "Mooncake TransferEngine is not available")
class TestMooncakeRDMAEndToEnd(unittest.TestCase):
    """
    End-to-End RDMA transfer tests (RFC 3.1.3, 3.2.1).

    NOTE: These tests require a working RDMA environment.
    For same-node testing, RoCE devices are preferred over InfiniBand
    to avoid cross-NIC routing issues.
    """

    @staticmethod
    def _get_free_port():
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((RDMA_HOST, 0))
            return s.getsockname()[1]

    @staticmethod
    def _get_connector_config(zmq_port, pool_size=16 * 1024 * 1024):
        """Get standard connector config with proper device filtering."""
        config = {
            "host": RDMA_HOST,
            "zmq_port": zmq_port,
            "protocol": "rdma",
            "memory_pool_size": pool_size,
        }
        # Add device filtering if available
        if RDMA_DEVICE_NAME:
            config["device_name"] = RDMA_DEVICE_NAME
        return config

    def test_put_get_tensor_e2e(self):
        """End-to-End data integrity test for tensor transfer."""
        producer_port = self._get_free_port()
        consumer_port = self._get_free_port()

        producer = MooncakeRDMAConnector(self._get_connector_config(producer_port))
        consumer = MooncakeRDMAConnector(self._get_connector_config(consumer_port))

        try:
            # 1. Prepare Data
            original_tensor = torch.randn(1024, 1024, dtype=torch.float32)
            req_id = "req_e2e_tensor"

            # 2. Producer Put
            success, size, metadata = producer.put("stage0", "stage1", req_id, original_tensor)
            self.assertTrue(success)
            self.assertEqual(size, original_tensor.nbytes)

            # Wait for listener startup
            time.sleep(0.5)

            # 3. Consumer Get
            result = consumer.get("stage0", "stage1", req_id, metadata)
            self.assertIsNotNone(result, "RDMA transfer failed")

            recv_buffer, recv_size = result
            self.assertEqual(recv_size, original_tensor.nbytes)

            # 4. Verify Data Integrity
            self.assertIsInstance(recv_buffer, ManagedBuffer)

            reconstructed = recv_buffer.as_tensor(dtype=original_tensor.dtype, shape=original_tensor.shape)
            self.assertTrue(torch.equal(reconstructed, original_tensor), "Data mismatch!")

            # 5. Release buffer
            recv_buffer.release()

            print(f"\n[PASS] E2E tensor transfer of {recv_size} bytes successful.")

        finally:
            producer._stop_event.set()
            consumer._stop_event.set()
            time.sleep(0.1)

    def test_put_get_bytes_e2e(self):
        """End-to-End test for bytes transfer."""
        producer_port = self._get_free_port()
        consumer_port = self._get_free_port()

        producer = MooncakeRDMAConnector(self._get_connector_config(producer_port))
        consumer = MooncakeRDMAConnector(self._get_connector_config(consumer_port))

        try:
            original_data = b"Hello RDMA World! " * 1000  # ~18KB
            req_id = "req_e2e_bytes"

            success, size, metadata = producer.put("stage0", "stage1", req_id, original_data)
            self.assertTrue(success)

            time.sleep(0.5)

            result = consumer.get("stage0", "stage1", req_id, metadata)
            self.assertIsNotNone(result, "RDMA transfer failed")

            recv_data, recv_size = result
            # fast_path returns ManagedBuffer for zero-copy access;
            # convert to bytes for comparison, then release the buffer.
            if hasattr(recv_data, "tensor") and hasattr(recv_data, "release"):
                recv_bytes = recv_data.to_bytes()
                recv_data.release()
            else:
                recv_bytes = recv_data
            self.assertIsInstance(recv_bytes, bytes)
            self.assertEqual(recv_bytes, original_data)

            print(f"\n[PASS] E2E bytes transfer of {recv_size} bytes successful.")

        finally:
            producer._stop_event.set()
            consumer._stop_event.set()
            time.sleep(0.1)

    def test_put_get_tensor_zero_copy_e2e(self):
        """
        End-to-End test for ZERO COPY tensor transfer using ManagedBuffer.

        Zero Copy Flow:
        1. Allocate ManagedBuffer directly from producer's memory pool
        2. Write data directly into the buffer (no intermediate copy)
        3. Put the ManagedBuffer (uses address directly, no copy to pool)
        4. RDMA transfer happens from the pre-registered memory
        5. Consumer receives data in its own ManagedBuffer

        This is the optimal path for high-performance data transfer.
        """
        producer_port = self._get_free_port()
        consumer_port = self._get_free_port()

        producer = MooncakeRDMAConnector(self._get_connector_config(producer_port))
        consumer = MooncakeRDMAConnector(self._get_connector_config(consumer_port))

        try:
            # Test parameters
            tensor_shape = (1024, 1024)
            tensor_dtype = torch.float32
            tensor_numel = tensor_shape[0] * tensor_shape[1]
            tensor_size = tensor_numel * 4  # float32 = 4 bytes
            req_id = "req_zero_copy_tensor"

            # ========== ZERO COPY PATH ==========
            # Step 1: Allocate ManagedBuffer directly from producer's pool
            offset = producer.allocator.alloc(tensor_size)
            send_buffer = ManagedBuffer(producer.allocator, offset, tensor_size, producer.pool)

            # Step 2: Create reference tensor for comparison
            reference_tensor = torch.randn(*tensor_shape, dtype=tensor_dtype)

            # Step 3: Write data DIRECTLY into the ManagedBuffer (zero copy write)
            # Convert buffer's view to the target shape and copy reference data
            buffer_tensor = send_buffer.as_tensor(dtype=tensor_dtype, shape=tensor_shape)
            buffer_tensor.copy_(reference_tensor)

            # Verify data was written correctly to the buffer
            self.assertTrue(torch.equal(buffer_tensor, reference_tensor), "Data not correctly written to ManagedBuffer")

            # Step 4: Put using ManagedBuffer - this is ZERO COPY!
            # The put() method detects ManagedBuffer and uses its address directly
            success, size, metadata = producer.put("stage0", "stage1", req_id, send_buffer)
            self.assertTrue(success, "Zero copy put failed")
            self.assertEqual(size, tensor_size, "Size mismatch")
            self.assertTrue(metadata.get("is_fast_path", False), "Should use fast path for ManagedBuffer")

            print(f"\n[ZERO COPY] Producer put: size={size}, metadata={metadata}")

            # Wait for listener startup
            time.sleep(0.5)

            # Step 5: Consumer Get - receives data via RDMA into its own pool
            result = consumer.get("stage0", "stage1", req_id, metadata)
            self.assertIsNotNone(result, "RDMA zero copy transfer failed")

            recv_buffer, recv_size = result
            self.assertEqual(recv_size, tensor_size, "Received size mismatch")
            self.assertIsInstance(recv_buffer, ManagedBuffer, "Should receive ManagedBuffer for zero copy")

            # Step 6: Verify Data Integrity
            reconstructed = recv_buffer.as_tensor(dtype=tensor_dtype, shape=tensor_shape)
            self.assertTrue(torch.equal(reconstructed, reference_tensor), "Zero copy data mismatch!")

            # Step 7: Release buffers
            recv_buffer.release()
            # Note: send_buffer ownership was transferred to producer._local_buffers
            # It will be auto-released after transfer completion

            print(f"\n[PASS] ZERO COPY E2E tensor transfer of {recv_size} bytes successful!")
            print("       - No intermediate copy on producer side")
            print("       - Data transferred directly from RDMA-registered memory")

        finally:
            producer._stop_event.set()
            consumer._stop_event.set()
            time.sleep(0.1)

    def test_zero_copy_vs_copy_path_comparison(self):
        """
        Compare zero copy path vs regular copy path performance characteristics.
        This test verifies both paths work correctly and documents the difference.
        """
        producer_port = self._get_free_port()
        consumer_port = self._get_free_port()

        producer = MooncakeRDMAConnector(self._get_connector_config(producer_port, pool_size=32 * 1024 * 1024))
        consumer = MooncakeRDMAConnector(self._get_connector_config(consumer_port, pool_size=32 * 1024 * 1024))

        try:
            tensor_shape = (512, 512)
            tensor_dtype = torch.float32
            tensor_size = 512 * 512 * 4

            # ========== TEST 1: Regular Copy Path ==========
            regular_tensor = torch.randn(*tensor_shape, dtype=tensor_dtype)
            req_id_copy = "req_copy_path"

            success_copy, size_copy, meta_copy = producer.put("stage0", "stage1", req_id_copy, regular_tensor)
            self.assertTrue(success_copy, "Copy path put failed")
            # Copy path also uses fast_path=True (raw transfer), just with an extra copy
            print(f"\n[COPY PATH] Put success: size={size_copy}, is_fast_path={meta_copy.get('is_fast_path')}")

            time.sleep(0.3)

            result_copy = consumer.get("stage0", "stage1", req_id_copy, meta_copy)
            self.assertIsNotNone(result_copy, "Copy path transfer failed")
            recv_buf_copy, _ = result_copy
            reconstructed_copy = recv_buf_copy.as_tensor(dtype=tensor_dtype, shape=tensor_shape)
            self.assertTrue(torch.equal(reconstructed_copy, regular_tensor))
            recv_buf_copy.release()

            # ========== TEST 2: Zero Copy Path ==========
            offset = producer.allocator.alloc(tensor_size)
            zero_copy_buffer = ManagedBuffer(producer.allocator, offset, tensor_size, producer.pool)

            # Write different data to distinguish
            reference_tensor = torch.randn(*tensor_shape, dtype=tensor_dtype)
            buffer_view = zero_copy_buffer.as_tensor(dtype=tensor_dtype, shape=tensor_shape)
            buffer_view.copy_(reference_tensor)

            req_id_zero = "req_zero_copy_path"

            success_zero, size_zero, meta_zero = producer.put("stage0", "stage1", req_id_zero, zero_copy_buffer)
            self.assertTrue(success_zero, "Zero copy path put failed")
            print(f"[ZERO COPY] Put success: size={size_zero}, is_fast_path={meta_zero.get('is_fast_path')}")

            time.sleep(0.3)

            result_zero = consumer.get("stage0", "stage1", req_id_zero, meta_zero)
            self.assertIsNotNone(result_zero, "Zero copy path transfer failed")
            recv_buf_zero, _ = result_zero
            reconstructed_zero = recv_buf_zero.as_tensor(dtype=tensor_dtype, shape=tensor_shape)
            self.assertTrue(torch.equal(reconstructed_zero, reference_tensor))
            recv_buf_zero.release()

            print("\n[PASS] Both copy paths verified:")
            print("       - Regular path: tensor -> copy to pool -> RDMA transfer")
            print("       - Zero copy:    ManagedBuffer (already in pool) -> RDMA transfer")

        finally:
            producer._stop_event.set()
            consumer._stop_event.set()
            time.sleep(0.1)

    def test_put_get_object_e2e(self):
        """End-to-End test for Python object (serialization path)."""
        producer_port = self._get_free_port()
        consumer_port = self._get_free_port()

        producer = MooncakeRDMAConnector(self._get_connector_config(producer_port))
        consumer = MooncakeRDMAConnector(self._get_connector_config(consumer_port))

        try:
            original_obj = {
                "message": "hello",
                "numbers": [1, 2, 3, 4, 5],
                "nested": {"key": "value"},
            }
            req_id = "req_e2e_object"

            success, size, metadata = producer.put("stage0", "stage1", req_id, original_obj)
            self.assertTrue(success, f"Put failed, metadata: {metadata}")
            self.assertFalse(metadata["is_fast_path"])  # Object requires deserialization

            print(f"\n[DEBUG] Object put success, size={size}, metadata={metadata}")

            # Longer wait for serialized object transfer
            time.sleep(1.0)

            result = consumer.get("stage0", "stage1", req_id, metadata)
            self.assertIsNotNone(result, f"RDMA transfer failed, metadata was: {metadata}")

            recv_obj, recv_size = result
            # is_fast_path=False means we get deserialized object
            self.assertEqual(recv_obj, original_obj)

            print("\n[PASS] E2E object transfer successful.")

        finally:
            producer._stop_event.set()
            consumer._stop_event.set()
            time.sleep(0.1)

    def test_auto_cleanup_after_transfer(self):
        """Verify producer buffer is auto-cleaned after successful transfer."""
        producer_port = self._get_free_port()
        consumer_port = self._get_free_port()

        producer = MooncakeRDMAConnector(self._get_connector_config(producer_port))
        consumer = MooncakeRDMAConnector(self._get_connector_config(consumer_port))

        try:
            tensor = torch.randn(100, dtype=torch.float32)
            req_id = "req_auto_cleanup"

            success, _, metadata = producer.put("stage0", "stage1", req_id, tensor)
            self.assertTrue(success)
            internal_key = MooncakeRDMAConnector._make_key(req_id, "stage0", "stage1")
            self.assertIn(internal_key, producer._local_buffers)

            time.sleep(0.5)

            # Get triggers RDMA transfer, which should auto-cleanup on producer
            result = consumer.get("stage0", "stage1", req_id, metadata)
            self.assertIsNotNone(result, "RDMA transfer failed")

            # Give some time for cleanup to propagate
            time.sleep(0.2)

            # Producer should have cleaned up
            self.assertNotIn(internal_key, producer._local_buffers)

            # Release consumer buffer
            recv_buffer, _ = result
            if isinstance(recv_buffer, ManagedBuffer):
                recv_buffer.release()

        finally:
            producer._stop_event.set()
            consumer._stop_event.set()
            time.sleep(0.1)


@unittest.skipIf(TransferEngine is None, "Mooncake TransferEngine is not available")
class TestMooncakeRDMAConcurrency(unittest.TestCase):
    """Concurrency tests for MooncakeRDMAConnector (RFC 3.2.2)"""

    @staticmethod
    def _get_free_port():
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((RDMA_HOST, 0))
            return s.getsockname()[1]

    @staticmethod
    def _get_connector_config(zmq_port, pool_size=16 * 1024 * 1024):
        """Get standard connector config with proper device filtering."""
        config = {
            "host": RDMA_HOST,
            "zmq_port": zmq_port,
            "protocol": "rdma",
            "memory_pool_size": pool_size,
        }
        if RDMA_DEVICE_NAME:
            config["device_name"] = RDMA_DEVICE_NAME
        return config

    def test_concurrent_put(self):
        """Verify connector handles concurrent put without race conditions."""
        port = self._get_free_port()

        connector = MooncakeRDMAConnector(self._get_connector_config(port, pool_size=64 * 1024 * 1024))

        errors = []
        results = []
        lock = threading.Lock()

        def put_worker(req_id, data):
            try:
                success, size, metadata = connector.put("s0", "s1", req_id, data)
                if success:
                    with lock:
                        results.append((req_id, metadata, data))
            except Exception as e:
                with lock:
                    errors.append(f"Put {req_id}: {e}")

        try:
            # Launch concurrent puts
            threads = []
            for i in range(10):
                data = torch.randn(256, 256, dtype=torch.float32)
                t = threading.Thread(target=put_worker, args=(f"req_{i}", data))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            self.assertEqual(len(errors), 0, f"Concurrency errors: {errors}")
            self.assertEqual(len(results), 10, "Not all puts succeeded")

            print(f"\n[PASS] Concurrent put test with {len(results)} transfers.")

        finally:
            connector.close()


@unittest.skipIf(TransferEngine is None, "Mooncake TransferEngine is not available")
class TestMooncakeRDMALifecycle(unittest.TestCase):
    """Lifecycle and resource management tests for MooncakeRDMAConnector"""

    @staticmethod
    def _get_free_port():
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((RDMA_HOST, 0))
            return s.getsockname()[1]

    @staticmethod
    def _get_connector_config(zmq_port, pool_size=16 * 1024 * 1024):
        """Get standard connector config with proper device filtering."""
        config = {
            "host": RDMA_HOST,
            "zmq_port": zmq_port,
            "protocol": "rdma",
            "memory_pool_size": pool_size,
        }
        if RDMA_DEVICE_NAME:
            config["device_name"] = RDMA_DEVICE_NAME
        return config

    def test_close_releases_resources(self):
        """Verify close() properly releases all resources."""
        port = self._get_free_port()
        connector = MooncakeRDMAConnector(self._get_connector_config(port, pool_size=1024 * 1024))

        # Put some data to create internal buffers
        tensor = torch.randn(100, dtype=torch.float32)
        connector.put("s0", "s1", "req_close_test", tensor)
        internal_key = MooncakeRDMAConnector._make_key("req_close_test", "s0", "s1")
        self.assertIn(internal_key, connector._local_buffers)

        # Close should clean up
        connector.close()

        # Verify cleanup
        self.assertTrue(connector._stop_event.is_set())
        self.assertEqual(len(connector._local_buffers), 0)
        self.assertFalse(connector._listener_thread.is_alive())

    def test_context_manager(self):
        """Verify connector works as context manager."""
        port = self._get_free_port()

        with MooncakeRDMAConnector(self._get_connector_config(port)) as connector:
            tensor = torch.randn(50, dtype=torch.float32)
            success, _, _ = connector.put("s0", "s1", "req_ctx", tensor)
            self.assertTrue(success)

        # After context exit, connector should be closed
        self.assertTrue(connector._stop_event.is_set())

    def test_double_close_safe(self):
        """Verify calling close() twice is safe."""
        port = self._get_free_port()
        connector = MooncakeRDMAConnector(self._get_connector_config(port))

        connector.close()
        # Second close should not raise
        connector.close()


@unittest.skipIf(TransferEngine is None, "Mooncake TransferEngine is not available")
@unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
class TestMooncakeRDMAGPUPool(unittest.TestCase):
    """GPU memory pool tests for MooncakeRDMAConnector"""

    @staticmethod
    def _get_free_port():
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((RDMA_HOST, 0))
            return s.getsockname()[1]

    @staticmethod
    def _get_gpu_connector_config(zmq_port, pool_size=16 * 1024 * 1024, device="cuda:0"):
        """Get GPU connector config with proper device filtering."""
        config = {
            "host": RDMA_HOST,
            "zmq_port": zmq_port,
            "protocol": "rdma",
            "memory_pool_size": pool_size,
            "memory_pool_device": device,
        }
        if RDMA_DEVICE_NAME:
            config["device_name"] = RDMA_DEVICE_NAME
        return config

    def test_gpu_pool_initialization(self):
        """Verify connector initializes with GPU memory pool."""
        port = self._get_free_port()

        connector = MooncakeRDMAConnector(self._get_gpu_connector_config(port))

        try:
            self.assertEqual(connector.pool_device, "cuda:0")
            self.assertTrue(connector.pool.is_cuda)
            self.assertEqual(connector.pool.device.index, 0)

            health = connector.health()
            self.assertEqual(health["pool_device"], "cuda:0")
        finally:
            connector.close()

    def test_gpu_pool_put_cpu_tensor(self):
        """Test putting CPU tensor into GPU pool (H2D copy)."""
        port = self._get_free_port()

        connector = MooncakeRDMAConnector(self._get_gpu_connector_config(port))

        try:
            # CPU tensor should be copied to GPU pool
            cpu_tensor = torch.randn(256, 256, dtype=torch.float32)
            success, size, metadata = connector.put("s0", "s1", "req_h2d", cpu_tensor)

            self.assertTrue(success)
            self.assertEqual(size, cpu_tensor.nbytes)
            self.assertTrue(metadata["is_fast_path"])
        finally:
            connector.close()

    def test_gpu_pool_put_gpu_tensor(self):
        """Test putting GPU tensor into GPU pool (D2D copy or same device)."""
        port = self._get_free_port()

        connector = MooncakeRDMAConnector(self._get_gpu_connector_config(port))

        try:
            # GPU tensor
            gpu_tensor = torch.randn(256, 256, dtype=torch.float32, device="cuda:0")
            success, size, metadata = connector.put("s0", "s1", "req_d2d", gpu_tensor)

            self.assertTrue(success)
            self.assertEqual(size, gpu_tensor.nbytes)
            self.assertTrue(metadata["is_fast_path"])
        finally:
            connector.close()


@unittest.skipIf(TransferEngine is None, "Mooncake TransferEngine is not available")
@unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
class TestMooncakeRDMAGPUEndToEnd(unittest.TestCase):
    """GPU E2E transfer tests for MooncakeRDMAConnector"""

    @staticmethod
    def _get_free_port():
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((RDMA_HOST, 0))
            return s.getsockname()[1]

    @staticmethod
    def _get_gpu_connector_config(zmq_port, pool_size=32 * 1024 * 1024, device="cuda:0"):
        """Get GPU connector config with proper device filtering."""
        config = {
            "host": RDMA_HOST,
            "zmq_port": zmq_port,
            "protocol": "rdma",
            "memory_pool_size": pool_size,
            "memory_pool_device": device,
        }
        if RDMA_DEVICE_NAME:
            config["device_name"] = RDMA_DEVICE_NAME
        return config

    def test_gpu_pool_e2e_transfer(self):
        """End-to-End GPU pool RDMA transfer test."""
        producer_port = self._get_free_port()
        consumer_port = self._get_free_port()

        producer = MooncakeRDMAConnector(self._get_gpu_connector_config(producer_port))
        consumer = MooncakeRDMAConnector(self._get_gpu_connector_config(consumer_port))

        try:
            # Create GPU tensor
            original = torch.randn(512, 512, dtype=torch.float32, device="cuda:0")
            req_id = "req_gpu_e2e"

            success, size, metadata = producer.put("s0", "s1", req_id, original)
            self.assertTrue(success)

            time.sleep(0.5)

            result = consumer.get("s0", "s1", req_id, metadata)
            self.assertIsNotNone(result, "GPU RDMA transfer failed")

            recv_buffer, recv_size = result
            self.assertIsInstance(recv_buffer, ManagedBuffer)

            # Verify data on GPU
            reconstructed = recv_buffer.as_tensor(dtype=original.dtype, shape=original.shape)
            self.assertTrue(reconstructed.is_cuda)
            # Compare on CPU to avoid device mismatch error
            self.assertTrue(torch.equal(reconstructed.cpu(), original.cpu()), "GPU data mismatch!")

            recv_buffer.release()
            print(f"\n[PASS] GPU E2E transfer of {recv_size} bytes successful.")

        finally:
            producer.close()
            consumer.close()


if __name__ == "__main__":
    unittest.main()
