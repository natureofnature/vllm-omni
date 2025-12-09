# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector
from vllm_omni.distributed.omni_connectors.factory import OmniConnectorFactory
from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec
from vllm_omni.distributed.omni_connectors.utils.serialization import OmniSerializer


class TestOmniSerializer(unittest.TestCase):
    def test_pickle_serialization(self):
        """Test basic pickle serialization."""
        data = {"key": "value", "list": [1, 2, 3]}
        serialized = OmniSerializer.serialize(data, method="cloudpickle")
        self.assertIsInstance(serialized, bytes)

        deserialized = OmniSerializer.deserialize(serialized, method="cloudpickle")
        self.assertEqual(data, deserialized)


class TestOmniConnectorFactory(unittest.TestCase):
    def test_create_shm_connector(self):
        """Test creating SharedMemoryConnector via Factory."""
        spec = ConnectorSpec(name="SharedMemoryConnector", extra={"shm_threshold_bytes": 1024})
        connector = OmniConnectorFactory.create_connector(spec)
        self.assertIsInstance(connector, SharedMemoryConnector)
        self.assertEqual(connector.threshold, 1024)

    def test_create_unknown_connector(self):
        """Test error when creating unknown connector."""
        spec = ConnectorSpec(name="UnknownConnector")
        with self.assertRaises(ValueError):
            OmniConnectorFactory.create_connector(spec)


class TestSharedMemoryConnector(unittest.TestCase):
    def setUp(self):
        self.config = {"shm_threshold_bytes": 100}  # Small threshold for testing
        self.connector = SharedMemoryConnector(self.config)

    def test_put_get_inline(self):
        """Test inline transfer for small data."""
        data = {"small": "data"}
        # Ensure data is smaller than threshold (100 bytes)

        success, size, metadata = self.connector.put("stage_0", "stage_1", "req_1", data)
        self.assertTrue(success)
        self.assertIn("inline_bytes", metadata)
        self.assertNotIn("shm", metadata)

        # Retrieve
        retrieved_data, ret_size = self.connector.get("stage_0", "stage_1", "req_1", metadata)
        self.assertEqual(data, retrieved_data)
        self.assertEqual(size, ret_size)

    @patch("vllm_omni.distributed.omni_connectors.connectors.shm_connector.shm_write_bytes")
    @patch("vllm_omni.distributed.omni_connectors.connectors.shm_connector.shm_read_bytes")
    def test_put_get_shm(self, mock_read, mock_write):
        """Test SHM transfer logic for large data (Mocked)."""
        # Create data larger than 100 bytes
        data = {"large": "x" * 200}

        # Mock SHM return values
        mock_handle = {"name": "test_shm", "size": 200}
        mock_write.return_value = mock_handle

        # When reading, return the serialized bytes of the data
        serialized_data = self.connector.serialize_obj(data)
        mock_read.return_value = serialized_data

        # Put
        success, size, metadata = self.connector.put("stage_0", "stage_1", "req_2", data)

        self.assertTrue(success)
        # Should use SHM because data > threshold
        self.assertIn("shm", metadata)
        self.assertEqual(metadata["shm"], mock_handle)
        self.assertNotIn("inline_bytes", metadata)

        mock_write.assert_called_once()

        # Get
        retrieved_data, ret_size = self.connector.get("stage_0", "stage_1", "req_2", metadata)

        self.assertEqual(data, retrieved_data)
        mock_read.assert_called_once_with(mock_handle)

    def test_get_invalid_metadata(self):
        """Test get with invalid metadata."""
        result = self.connector.get("stage_0", "stage_1", "req_3", {})
        self.assertIsNone(result)

        result = self.connector.get("stage_0", "stage_1", "req_3", {"unknown": "format"})
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
