from typing import Any


def ensure_list(value: Any) -> Any:
    """Convert ConstantList-like values to plain Python lists."""
    if hasattr(value, "_x"):
        return list(value._x)
    if not isinstance(value, list):
        return value
    return list(value)


def validate_stage_inputs(stage_list: list[Any], engine_input_source: list[int]) -> Any:
    """Return engine outputs for the source stage after basic validation."""
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    stage_id = engine_input_source[0]
    if stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {stage_id}")

    stage = stage_list[stage_id]
    if stage.engine_outputs is None:
        raise RuntimeError(f"Stage {stage_id} has no outputs yet")

    return stage.engine_outputs
