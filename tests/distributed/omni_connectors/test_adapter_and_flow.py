# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import MagicMock

from vllm_omni.distributed.omni_connectors.adapter import try_recv_via_connector, try_send_via_connector
from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector
from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec, OmniTransferConfig
from vllm_omni.distributed.omni_connectors.utils.initialization import get_connectors_config_for_stage


class TestAdapter(unittest.TestCase):
    def setUp(self):
        self.mock_connector = MagicMock()
        self.mock_metrics = MagicMock()
        self.mock_queue_fn = MagicMock()

    def test_send_success(self):
        """Test try_send_via_connector success path."""
        # Setup
        stage_id = 0
        next_stage_id = 1
        req_id = "req_123"
        inputs = {"input_ids": [1, 2, 3]}
        sampling_params = {"temperature": 0.7}
        prompt = "test prompt"

        # Mock connector.put return
        # Returns: (success, size, metadata)
        mock_metadata = {"handle": "xyz"}
        self.mock_connector.put.return_value = (True, 100, mock_metadata)

        # Execute
        result = try_send_via_connector(
            connector=self.mock_connector,
            stage_id=stage_id,
            next_stage_id=next_stage_id,
            req_id=req_id,
            next_inputs=inputs,
            sampling_params=sampling_params,
            original_prompt=prompt,
            next_stage_queue_submit_fn=self.mock_queue_fn,
            metrics=self.mock_metrics,
        )

        # Verify
        self.assertTrue(result)

        # 1. Verify connector.put called correctly
        self.mock_connector.put.assert_called_once()
        args, _ = self.mock_connector.put.call_args
        self.assertEqual(args[0], "0")  # from_stage
        self.assertEqual(args[1], "1")  # to_stage
        self.assertEqual(args[2], req_id)
        # Verify payload structure in put
        payload = args[3]
        self.assertEqual(payload["engine_inputs"], inputs)
        self.assertEqual(payload["sampling_params"], sampling_params)

        # 2. Verify queue notification submitted
        self.mock_queue_fn.assert_called_once()
        notify_payload = self.mock_queue_fn.call_args[0][0]
        self.assertEqual(notify_payload["request_id"], req_id)
        self.assertTrue(notify_payload["from_connector"])
        self.assertEqual(notify_payload["connector_metadata"], mock_metadata)

        # 3. Verify metrics recorded
        self.mock_metrics.on_forward.assert_called_once()

    def test_send_fail(self):
        """Test try_send_via_connector when connector fails."""
        self.mock_connector.put.return_value = (False, 0, None)

        result = try_send_via_connector(
            connector=self.mock_connector,
            stage_id=0,
            next_stage_id=1,
            req_id="req_fail",
            next_inputs={},
            sampling_params={},
            original_prompt="",
            next_stage_queue_submit_fn=self.mock_queue_fn,
            metrics=self.mock_metrics,
        )

        self.assertFalse(result)
        self.mock_queue_fn.assert_not_called()

    def test_recv_success(self):
        """Test try_recv_via_connector success path."""
        # Setup task received from queue
        task = {
            "request_id": "req_recv",
            "from_connector": True,
            "from_stage": "0",
            "connector_metadata": {"handle": "xyz"},
        }

        # Setup connectors dict
        connectors = {("0", "1"): self.mock_connector}

        # Mock connector.get return
        expected_data = {"engine_inputs": {"ids": [1]}}
        # get returns: (data_obj, size)
        self.mock_connector.get.return_value = (expected_data, 50)
        # serialize_obj needed for metrics calculation if size not returned directly
        self.mock_connector.serialize_obj.return_value = b"bytes"

        # Execute
        # We are stage 1 receiving from stage 0
        inputs, rx_metrics = try_recv_via_connector(task, connectors, stage_id=1)

        # Verify
        self.assertEqual(inputs, expected_data["engine_inputs"])
        self.assertIsNotNone(rx_metrics)
        self.mock_connector.get.assert_called_once_with("0", "1", "req_recv", metadata={"handle": "xyz"})

    def test_recv_no_connector(self):
        """Test recv fails when no connector exists for edge."""
        task = {"request_id": "req_missing", "from_connector": True, "from_stage": "0"}
        connectors = {}  # Empty connectors

        inputs, _ = try_recv_via_connector(task, connectors, stage_id=1)
        self.assertIsNone(inputs)


class TestEndToEndFlow(unittest.TestCase):
    def test_shm_connector_flow(self):
        """
        Verify the full flow: Send -> Adapter -> Connector -> Adapter -> Recv.
        Using real SharedMemoryConnector (inline mode for simplicity).
        """
        # 1. Setup Connector
        config = {"shm_threshold_bytes": 1024}  # Large threshold to use inline
        connector = SharedMemoryConnector(config)
        connectors_map = {("0", "1"): connector}

        # 2. Setup Data
        stage_id = 0
        next_stage_id = 1
        req_id = "flow_req"
        inputs = {"tokens": [10, 20, 30]}
        sampling_params = {"n": 1}

        # Queue capture mechanism
        queue_capture = []

        def mock_submit(payload):
            queue_capture.append(payload)

        mock_metrics = MagicMock()

        # 3. Send
        success = try_send_via_connector(
            connector=connector,
            stage_id=stage_id,
            next_stage_id=next_stage_id,
            req_id=req_id,
            next_inputs=inputs,
            sampling_params=sampling_params,
            original_prompt="prompt",
            next_stage_queue_submit_fn=mock_submit,
            metrics=mock_metrics,
        )
        self.assertTrue(success)
        self.assertEqual(len(queue_capture), 1)

        # 4. Recv
        # The 'task' is what would be popped from the queue
        received_task = queue_capture[0]

        # Verify queue payload contains what we expect
        self.assertTrue(received_task["from_connector"])
        self.assertEqual(received_task["from_stage"], "0")

        # Decode
        decoded_inputs, _ = try_recv_via_connector(received_task, connectors_map, stage_id=1)

        # 5. Verify Data Integrity
        self.assertEqual(decoded_inputs, inputs)


class TestInitialization(unittest.TestCase):
    def test_get_connectors_for_stage(self):
        """Test filtering logic for stage config."""
        # Config has edges: 0->1, 1->2
        config = OmniTransferConfig(
            connectors={("0", "1"): ConnectorSpec(name="C1"), ("1", "2"): ConnectorSpec(name="C2")}
        )

        # Get config for Stage 1
        # Stage 1 receives from 0 (input) and sends to 2 (output)
        # get_connectors_config_for_stage ONLY returns INPUT connectors for the worker to initialize

        stage_config = get_connectors_config_for_stage(config, stage_id=1)

        # Should contain "from_stage_0"
        self.assertIn("from_stage_0", stage_config)
        self.assertEqual(stage_config["from_stage_0"]["spec"]["name"], "C1")

        # Should NOT contain "from_stage_1" or related to output
        self.assertNotIn("from_stage_1", stage_config)

        # Verify Stage 2
        stage_2_config = get_connectors_config_for_stage(config, stage_id=2)
        self.assertIn("from_stage_1", stage_2_config)
        self.assertEqual(stage_2_config["from_stage_1"]["spec"]["name"], "C2")


class TestEdgeCases(unittest.TestCase):
    def test_recv_with_missing_metadata(self):
        """Test recv when queue payload is malformed (missing metadata)."""
        # Connector expects metadata but task doesn't have it
        task = {
            "request_id": "req_bad",
            "from_connector": True,
            "from_stage": "0",
            # Missing "connector_metadata"
        }
        mock_conn = MagicMock()
        # If get is called with None metadata, connector usually handles it or adapter handles exception
        mock_conn.get.side_effect = Exception("Get failed")

        connectors = {("0", "1"): mock_conn}

        inputs, _ = try_recv_via_connector(task, connectors, stage_id=1)
        self.assertIsNone(inputs)


if __name__ == "__main__":
    unittest.main()
