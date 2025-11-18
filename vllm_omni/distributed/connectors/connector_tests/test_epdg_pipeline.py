#!/usr/bin/env python3
"""
Test EPD (Encode-Prefill-Decode) 3-stage pipeline with connectors.
Tests data flow between stages using configured connectors.
"""

import os
import sys
from typing import Any, Dict, Optional
from dataclasses import dataclass
import uuid

try:
    import yaml
except ImportError:
    print("PyYAML not available, using basic YAML parsing")
    yaml = None

# Add the connectors package to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import from parent package
try:
    from connectors.config import ConnectorSpec, OmniTransferConfig
    from connectors.base import OmniConnectorBase, InMemoryOmniConnector
    from connectors.factory import OmniConnectorFactory
    from connectors.utils import load_omni_transfer_config
except ImportError:
    # Fallback for direct execution
    sys.path.insert(0, current_dir)
    from config import ConnectorSpec, OmniTransferConfig
    from base import OmniConnectorBase, InMemoryOmniConnector
    from factory import OmniConnectorFactory
    from utils import load_omni_transfer_config


@dataclass
class StageResult:
    """Result from a stage processing."""
    stage_name: str
    request_id: str
    output_data: Any
    metadata: Dict[str, Any]


class EncodeStage:
    """Simple Encode stage for testing."""

    def __init__(self, stage_id: int, output_connector: Optional[OmniConnectorBase] = None):
        self.stage_id = stage_id
        self.stage_name = "encode"
        self.output_connector = output_connector

    def process(self, request_id: str, input_data: Any) -> StageResult:
        """Process input data and optionally send to next stage."""
        print(f"[{self.stage_name}] Processing request {request_id}")

        # Simple encoding simulation
        encoded_data = {
            "original_text": input_data,
            "encoded_tokens": [ord(c) for c in str(input_data)][:10],  # Simple tokenization
            "embeddings": [0.1 * i for i in range(10)],  # Mock embeddings
            "stage": self.stage_name
        }

        result = StageResult(
            stage_name=self.stage_name,
            request_id=request_id,
            output_data=encoded_data,
            metadata={"tokens_count": len(encoded_data["encoded_tokens"])}
        )

        # Send to next stage if connector is available
        if self.output_connector:
            success = self.output_connector.put(
                str(self.stage_id), "1", request_id, encoded_data
            )
            print(f"[{self.stage_name}] Sent data to stage 1 via connector: {success}")

        return result


class PrefillStage:
    """Simple Prefill stage for testing."""

    def __init__(self, stage_id: int,
                 input_connector: Optional[OmniConnectorBase] = None,
                 output_connector: Optional[OmniConnectorBase] = None):
        self.stage_id = stage_id
        self.stage_name = "prefill"
        self.input_connector = input_connector
        self.output_connector = output_connector

    def process(self, request_id: str) -> Optional[StageResult]:
        """Process data from previous stage."""
        print(f"[{self.stage_name}] Processing request {request_id}")

        # Try to get data from input connector
        input_data = None
        if self.input_connector:
            input_data = self.input_connector.get("0", str(self.stage_id), request_id)

        if input_data is None:
            print(f"[{self.stage_name}] No input data received for {request_id}")
            return None

        print(f"[{self.stage_name}] Received data from encode")

        # Simple prefill simulation
        prefills_data = {
            **input_data,
            "kv_cache": {"layer_0": [0.2 * i for i in range(20)]},  # Mock KV cache
            "stage": self.stage_name
        }

        result = StageResult(
            stage_name=self.stage_name,
            request_id=request_id,
            output_data=prefills_data,
            metadata={"kv_size": len(prefills_data["kv_cache"])}
        )

        # Send to next stage if connector is available
        if self.output_connector:
            success = self.output_connector.put(
                str(self.stage_id), "2", request_id, prefills_data
            )
            print(f"[{self.stage_name}] Sent data to stage 2 via connector: {success}")

        return result


class DecodeStage:
    """Simple Decode stage for testing (final stage in 3-stage pipeline)."""

    def __init__(self, stage_id: int,
                 input_connector: Optional[OmniConnectorBase] = None):
        self.stage_id = stage_id
        self.stage_name = "decode"
        self.input_connector = input_connector

    def process(self, request_id: str) -> Optional[StageResult]:
        """Process data from previous stage (final output)."""
        print(f"[{self.stage_name}] Processing request {request_id}")

        # Try to get data from input connector
        input_data = None
        if self.input_connector:
            input_data = self.input_connector.get("1", str(self.stage_id), request_id)

        if input_data is None:
            print(f"[{self.stage_name}] No input data received for {request_id}")
            return None

        print(f"[{self.stage_name}] Received data from prefill")

        # Simple decode simulation (final output)
        decoded_data = {
            **input_data,
            "hidden_states": [0.3 * i for i in range(15)],  # Mock hidden states
            "final_output": f"Decoded text from: {input_data.get('original_text', 'unknown')}",
            "stage": self.stage_name
        }

        result = StageResult(
            stage_name=self.stage_name,
            request_id=request_id,
            output_data=decoded_data,
            metadata={"hidden_size": len(decoded_data["hidden_states"]), "final": True}
        )

        return result


class GeneratorStage:
    """Simple Generator stage for testing (removed for 3-stage pipeline)."""
    pass  # Not used in current 3-stage setup


def load_config_from_yaml(yaml_path: str) -> OmniTransferConfig:
    """Load connector configuration from YAML file."""
    if yaml is None:
        # Fallback: create a simple test config
        print("Using fallback InMemoryConnector config for testing")
        return create_test_config()

    with open(yaml_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    try:
        config = load_omni_transfer_config(config_dict=config_dict)
        print(f"Successfully loaded config from YAML with {len(config.connectors)} connectors")
        return config
    except Exception as e:
        print(f"Failed to load YAML config: {e}, using fallback config")
        return create_test_config()


def create_test_config() -> OmniTransferConfig:
    """Create a test configuration with InMemoryConnectors."""
    from .config import ConnectorSpec

    connectors = {}
    # Create connectors for EPD (3-stage) pipeline: 0->1->2
    pipeline_stages = [("0", "1"), ("1", "2")]

    for from_stage, to_stage in pipeline_stages:
        connectors[(from_stage, to_stage)] = ConnectorSpec(
            name="InMemoryConnector",
            extra={}
        )

    return OmniTransferConfig(connectors=connectors)


def create_stages_with_connectors(config: OmniTransferConfig) -> Dict[str, Any]:
    """Create EPD (3-stage) pipeline with appropriate connectors."""

    # Create connector instances based on config
    connectors = {}
    for edge_key, connector_spec in config.connectors.items():
        from_stage, to_stage = edge_key
        connectors[edge_key] = OmniConnectorFactory.create_connector(connector_spec)

    # Create stages with connectors (3-stage pipeline: 0->1->2)
    stages = {}

    # Encode stage (stage 0)
    encode_output_connector = connectors.get(("0", "1"))  # to prefill
    stages["encode"] = EncodeStage(0, encode_output_connector)

    # Prefill stage (stage 1)
    prefill_input_connector = connectors.get(("0", "1"))   # from encode
    prefill_output_connector = connectors.get(("1", "2"))  # to decode
    stages["prefill"] = PrefillStage(1, prefill_input_connector, prefill_output_connector)

    # Decode stage (stage 2) - final stage
    decode_input_connector = connectors.get(("1", "2"))    # from prefill
    stages["decode"] = DecodeStage(2, decode_input_connector)

    return stages


def test_pipeline_with_config(yaml_path: str):
    """Test the EPDG pipeline with connector configuration."""

    print("🔧 Loading configuration from YAML...")
    try:
        config = load_config_from_yaml(yaml_path)
        print(f"✓ Loaded config with {len(config.connectors)} connector configurations")
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return False

    print("\n🏗️  Creating EPDG stages with connectors...")
    try:
        stages = create_stages_with_connectors(config)
        print("✓ Created all stages with connectors")
    except Exception as e:
        print(f"✗ Failed to create stages: {e}")
        return False

    print("\n🚀 Testing data flow through pipeline...")

    # Test request
    request_id = f"test_req_{uuid.uuid4()}"
    input_text = "Hello, EPDG pipeline!"

    try:
        # Stage 1: Encode
        print(f"\n--- Stage 1: Encode ---")
        encode_result = stages["encode"].process(request_id, input_text)
        print(f"✓ Encode completed: {encode_result.metadata}")

        # Stage 2: Prefill
        print(f"\n--- Stage 2: Prefill ---")
        prefill_result = stages["prefill"].process(request_id)
        if prefill_result:
            print(f"✓ Prefill completed: {prefill_result.metadata}")
        else:
            print("✗ Prefill failed - no input data")
            return False

        # Stage 3: Decode
        print(f"\n--- Stage 3: Decode ---")
        decode_result = stages["decode"].process(request_id)
        if decode_result:
            print(f"✓ Decode completed: {decode_result.metadata}")
        else:
            print("✗ Decode failed - no input data")
            return False

        # Decode stage produces final output in 3-stage pipeline

    except Exception as e:
        print(f"✗ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n🎉 Pipeline test completed successfully!")
    return True


def main():
    """Main test function."""
    # Path to the YAML config file
    yaml_path = "/workspace/omni/vllm-omni/vllm_omni/model_executor/stage_configs/qwen2_5_omni.yaml"

    if not os.path.exists(yaml_path):
        print(f"✗ YAML config file not found: {yaml_path}")
        return False

    print("Testing EPDG Pipeline with Connectors")
    print("=" * 50)
    print(f"Config file: {yaml_path}")

    success = test_pipeline_with_config(yaml_path)

    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Tests failed!")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
