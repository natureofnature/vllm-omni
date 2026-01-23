# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests for BufferAllocator and ManagedBuffer (RFC 3.1.1, 3.1.2).
These tests do NOT require Mooncake or RDMA environment.
"""

import threading
import unittest

import torch

from vllm_omni.distributed.omni_connectors.connectors.mooncake_rdma_connector import (
    BufferAllocator,
    ManagedBuffer,
)


class TestBufferAllocator(unittest.TestCase):
    """Unit tests for BufferAllocator (RFC 3.1.1)"""

    def test_allocator_basic(self):
        """Verify allocator logic (alloc, free, fragmentation handling)."""
        allocator = BufferAllocator(total_size=4096, alignment=64)

        # Alloc 512 bytes
        offset1 = allocator.alloc(512)
        self.assertEqual(offset1, 0)

        # Alloc another 512 bytes
        offset2 = allocator.alloc(512)
        self.assertGreater(offset2, 0)

        # Free first block
        allocator.free(offset1, 512)

        # Should be able to alloc from freed space
        offset3 = allocator.alloc(512)
        self.assertEqual(offset3, 0)  # Reuses freed block

    def test_allocator_full_then_free(self):
        """Test that full allocation fails, then succeeds after free."""
        allocator = BufferAllocator(total_size=1024, alignment=64)

        # Alloc all space (aligned to 64)
        offset1 = allocator.alloc(1024)
        self.assertEqual(offset1, 0)

        # Should fail - no space left
        with self.assertRaises(MemoryError):
            allocator.alloc(64)

        # Free and retry
        allocator.free(offset1, 1024)
        offset2 = allocator.alloc(1024)
        self.assertEqual(offset2, 0)

    def test_allocator_alignment(self):
        """Verify allocation respects alignment."""
        allocator = BufferAllocator(total_size=4096, alignment=128)

        # Alloc 100 bytes - should be aligned to 128
        _offset1 = allocator.alloc(100)  # noqa: F841
        offset2 = allocator.alloc(100)

        # Second allocation should start at aligned boundary
        self.assertEqual(offset2 % 128, 0)
        self.assertEqual(offset2, 128)  # First alloc uses 128 bytes (aligned)

    def test_allocator_double_free_detection(self):
        """Verify double-free is detected and ignored."""
        allocator = BufferAllocator(total_size=4096, alignment=64)

        offset = allocator.alloc(512)
        allocator.free(offset, 512)

        # Double free should be ignored (warning logged)
        # After first free, the block may be merged with remaining space
        # So the second free should detect it's already within a free block
        allocator.free(offset, 512)  # Should not raise

    def test_allocator_corruption_detection(self):
        """Verify partial overlap (corruption) raises RuntimeError."""
        allocator = BufferAllocator(total_size=4096, alignment=64)

        # Scenario: Create a free block, then try to free a block that
        # partially overlaps (extends beyond the free block boundary)

        # Alloc three consecutive blocks to prevent auto-merging
        offset1 = allocator.alloc(512)  # 0-512
        _offset2 = allocator.alloc(512)  # 512-1024  # noqa: F841
        _offset3 = allocator.alloc(512)  # 1024-1536  # noqa: F841

        # Free only the first block
        # After this: free_blocks = [(0, 512), (1536, remaining)]
        allocator.free(offset1, 512)

        # Now try to free a block that starts in the free region (0-512)
        # but extends into the allocated region (512-1024)
        # Range 256-768 partially overlaps with free block 0-512 but extends beyond
        with self.assertRaises(RuntimeError):
            allocator.free(256, 512)  # 256-768 partially overlaps with 0-512

    def test_allocator_merge_adjacent(self):
        """Verify adjacent free blocks are merged."""
        allocator = BufferAllocator(total_size=4096, alignment=64)

        # Alloc three blocks
        o1 = allocator.alloc(512)
        o2 = allocator.alloc(512)
        o3 = allocator.alloc(512)

        # Free middle, then first, then third
        allocator.free(o2, 512)
        allocator.free(o1, 512)
        allocator.free(o3, 512)

        # Should be able to alloc large block (merged)
        offset = allocator.alloc(1536)
        self.assertEqual(offset, 0)

    def test_allocator_thread_safety(self):
        """Verify allocator is thread-safe under concurrent access."""
        allocator = BufferAllocator(total_size=1024 * 1024, alignment=64)
        errors = []

        def worker(worker_id):
            try:
                for i in range(100):
                    size = 1024 + (i % 10) * 64
                    offset = allocator.alloc(size)
                    # Simulate work
                    allocator.free(offset, size)
            except Exception as e:
                errors.append(f"Worker {worker_id}: {e}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Thread safety errors: {errors}")


class TestManagedBuffer(unittest.TestCase):
    """Unit tests for ManagedBuffer (RFC 3.1.2)"""

    def setUp(self):
        self.allocator = BufferAllocator(total_size=4096, alignment=64)
        self.pool = torch.zeros(4096, dtype=torch.uint8)

    def test_managed_buffer_tensor_view(self):
        """Verify tensor property returns correct view."""
        offset = self.allocator.alloc(256)
        buf = ManagedBuffer(self.allocator, offset, 256, self.pool)

        # Write to pool directly
        self.pool[offset : offset + 256] = 42

        # View should reflect the data
        view = buf.tensor
        self.assertEqual(view.shape[0], 256)
        self.assertTrue(torch.all(view == 42))

        buf.release()

    def test_managed_buffer_as_tensor(self):
        """Verify as_tensor returns correctly typed and shaped view."""
        # Allocate enough for 16 float32 values (64 bytes)
        offset = self.allocator.alloc(64)
        buf = ManagedBuffer(self.allocator, offset, 64, self.pool)

        # Write float32 data
        src = torch.arange(16, dtype=torch.float32)
        self.pool[offset : offset + 64] = src.view(torch.uint8)

        # as_tensor should return correct view
        typed_view = buf.as_tensor(dtype=torch.float32, shape=(4, 4))
        self.assertEqual(typed_view.shape, (4, 4))
        self.assertEqual(typed_view.dtype, torch.float32)
        self.assertTrue(torch.equal(typed_view.flatten(), src))

        buf.release()

    def test_managed_buffer_as_tensor_invalid_size(self):
        """Verify as_tensor raises ValueError for size mismatch."""
        offset = self.allocator.alloc(64)
        buf = ManagedBuffer(self.allocator, offset, 64, self.pool)

        # Try to create view with wrong size
        with self.assertRaises(ValueError):
            buf.as_tensor(dtype=torch.float32, shape=(100,))  # 400 bytes != 64

        buf.release()

    def test_managed_buffer_to_bytes(self):
        """Verify to_bytes returns correct data."""
        offset = self.allocator.alloc(10)
        buf = ManagedBuffer(self.allocator, offset, 10, self.pool)

        # Write data
        test_data = b"helloworld"
        self.pool[offset : offset + 10] = torch.tensor(list(test_data), dtype=torch.uint8)

        # to_bytes should return the same
        result = buf.to_bytes()
        self.assertEqual(result, test_data)

        buf.release()

    def test_managed_buffer_context_manager(self):
        """Verify context manager releases buffer."""
        offset = self.allocator.alloc(128)

        with ManagedBuffer(self.allocator, offset, 128, self.pool) as buf:
            self.assertFalse(buf._released)

        self.assertTrue(buf._released)

        # Should be able to realloc the same space
        new_offset = self.allocator.alloc(128)
        self.assertEqual(new_offset, offset)

    def test_managed_buffer_auto_release_on_del(self):
        """Verify buffer is released on garbage collection."""
        offset = self.allocator.alloc(128)
        buf = ManagedBuffer(self.allocator, offset, 128, self.pool)

        del buf  # Should trigger __del__ -> release()

        # Should be able to realloc
        new_offset = self.allocator.alloc(128)
        self.assertEqual(new_offset, offset)


if __name__ == "__main__":
    unittest.main()
