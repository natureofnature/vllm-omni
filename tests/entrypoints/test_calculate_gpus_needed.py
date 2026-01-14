from dataclasses import dataclass

from omegaconf import OmegaConf

from vllm_omni.entrypoints.utils import calculate_gpus_needed


@dataclass
class ParallelConfigDataclass:
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    prefill_context_parallel_size: int = 1
    sequence_parallel_size: int = 1


def test_calculate_gpus_needed_with_dataclass_parallel_config() -> None:
    engine_args = {
        "parallel_config": ParallelConfigDataclass(
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            data_parallel_size=1,
            prefill_context_parallel_size=3,
            sequence_parallel_size=1,
        ),
    }

    assert calculate_gpus_needed(engine_args) == 12


def test_calculate_gpus_needed_includes_pcp() -> None:
    engine_args = {
        "parallel_config": {
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
            "prefill_context_parallel_size": 4,
        },
    }

    assert calculate_gpus_needed(engine_args) == 8


class _NoGetParallelConfig:
    def __init__(self) -> None:
        self.tensor_parallel_size = 8


def test_calculate_gpus_needed_ignores_non_dict_parallel_config() -> None:
    engine_args = {
        "parallel_config": _NoGetParallelConfig(),
        "tensor_parallel_size": 4,
    }

    assert calculate_gpus_needed(engine_args) == 4


def test_calculate_gpus_needed_uses_ulysses_ring_for_sp() -> None:
    engine_args = {
        "parallel_config": {
            "ulysses_degree": 2,
            "ring_degree": 2,
        },
    }

    assert calculate_gpus_needed(engine_args) == 4


def test_calculate_gpus_needed_includes_cfg_and_ep() -> None:
    engine_args = {
        "parallel_config": {
            "tensor_parallel_size": 2,
            "cfg_parallel_size": 3,
            "expert_parallel_size": 2,
        },
    }

    assert calculate_gpus_needed(engine_args) == 12


def test_calculate_gpus_needed_with_omegaconf_parallel_config() -> None:
    parallel_config = OmegaConf.create(
        {
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 2,
            "data_parallel_size": 2,
        }
    )
    engine_args = {"parallel_config": parallel_config}

    assert calculate_gpus_needed(engine_args) == 8
