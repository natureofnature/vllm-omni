#!/usr/bin/env python3
"""
Test OmniConnector integration with OmniLLM.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from vllm_omni.distributed.connectors import create_mooncake_config

def test_connector_config():
    """Test connector configuration creation."""
    print("Testing connector configuration...")

    # Create Mooncake config
    config = create_mooncake_config(
        host="127.0.0.1",
        metadata_server="http://127.0.0.1:8080/metadata",
        master="127.0.0.1:50051"
    )

    print(f"Created config with {len(config.connectors)} connectors")
    for edge, connector_spec in config.connectors.items():
        print(f"  {edge}: {connector_spec.name}")

    return config

def test_omni_llm_initialization():
    """Test OmniLLM with connector config."""
    print("\nTesting OmniLLM initialization with connectors...")

    try:
        from vllm_omni.entrypoints.omni_llm import OmniLLM

        # Create connector config
        connector_config = {
            "connectors": {
                "0_to_1": {
                    "name": "InMemoryConnector",  # Use InMemory for testing
                    "extra": {}
                }
            }
        }

        # Test connector initialization logic
        omni = OmniLLM.__new__(OmniLLM)  # Create without calling __init__
        omni.omni_transfer_config = connector_config
        omni.connectors = {}
        omni._initialize_connectors()

        # Check if connector was created
        if ("0", "1") in omni.connectors:
            print(f"✅ Connector created: {type(omni.connectors[('0', '1')]).__name__}")
            return True
        else:
            print("❌ Connector not created")
            return False

    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("OmniConnector Integration Test")
    print("=" * 40)

    # Test config creation
    config = test_connector_config()

    # Test OmniLLM integration
    success = test_omni_llm_initialization()

    if success:
        print("\n✅ Integration test passed!")
        print("Connector system is ready for use with OmniLLM")
    else:
        print("\n❌ Integration test failed!")
