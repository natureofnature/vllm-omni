# SPDX-License-Identifier: Apache-2.0

import unittest
from pathlib import Path

# Use the new import path for initialization utilities
from vllm_omni.distributed.omni_connectors.utils.initialization import load_omni_transfer_config


class TestYamlConfigs(unittest.TestCase):
    def test_load_qwen_yaml_configs(self):
        """
        Scan and test loading of all qwen*.yaml config files.
        This ensures that existing stage configs are compatible with the OmniConnector system.
        """
        # Define the path to stage_configs relative to this test file
        # Assuming test is in vllm-omni/tests/connectors/
        # Configs are in vllm-omni/vllm_omni/model_executor/stage_configs/

        # Go up two levels from 'tests/connectors' to 'vllm-omni' root
        base_dir = Path(__file__).resolve().parent.parent.parent.parent
        config_dir = base_dir / "vllm_omni" / "model_executor" / "stage_configs"

        if not config_dir.exists():
            self.skipTest(f"Config directory not found at {config_dir}")

        # Find all yaml files starting with 'qwen'
        yaml_files = list(config_dir.glob("qwen*.yaml"))

        print(f"\nFound {len(yaml_files)} config files to test.")

        # Fail if no configs found - this likely means path resolution or filtering is wrong
        if not yaml_files:
            self.fail(f"No config files found in {config_dir}. Check directory path or file naming.")

        for yaml_file in yaml_files:
            with self.subTest(config_file=yaml_file.name):
                print(f"Testing config load: {yaml_file.name}")
                try:
                    # Attempt to load the config
                    # default_shm_threshold doesn't matter much for loading correctness, using default
                    config = load_omni_transfer_config(yaml_file)

                    self.assertIsNotNone(config, "Config should not be None")

                    # Basic validation
                    # Note: Some configs might not have 'runtime' or 'connectors' section if they rely on auto-shm
                    # but the load function should succeed regardless.

                    # If the config defines stages, we expect connectors to be populated (either explicit or auto SHM)
                    # We can't strictly assert len(config.connectors) > 0 because a single stage pipeline might have 0 edges.

                    print(f"  -> Successfully loaded. Connectors: {len(config.connectors)}")

                except Exception as e:
                    self.fail(f"Failed to load config {yaml_file.name}: {e}")


if __name__ == "__main__":
    unittest.main()
