# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""JoyAI interaction pipeline: Session policy above one ordinary VLM stage."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

JOYVL_INTERACTION_PIPELINE = PipelineConfig(
    model_type="joyvl_interaction",
    default_deploy_config_name="joyvl_interaction.yaml",
    session_serving_adapter=("vllm_omni.experimental.fullduplex.joyvl.serving.omni_adapter.JoyVLSessionServingAdapter"),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="vlm",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="text",
            sampling_constraints={"detokenize": True},
        ),
    ),
)
