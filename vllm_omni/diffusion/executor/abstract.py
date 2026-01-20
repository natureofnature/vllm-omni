from abc import ABC, abstractmethod
from typing import Any

from vllm.utils.import_utils import resolve_obj_by_qualname

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest


class DiffusionExecutor(ABC):
    """Abstract base class for Diffusion executors."""

    uses_multiproc: bool = False

    @staticmethod
    def get_class(od_config: OmniDiffusionConfig) -> type["DiffusionExecutor"]:
        backend = od_config.distributed_executor_backend
        if isinstance(backend, type):
            if not issubclass(backend, DiffusionExecutor):
                raise TypeError(f"distributed_executor_backend must be a subclass of DiffusionExecutor. Got {backend}.")
            return backend

        if isinstance(backend, str):
            if backend == "mp":
                from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor

                return MultiprocDiffusionExecutor

            try:
                executor_class = resolve_obj_by_qualname(backend)
            except (ImportError, ValueError) as e:
                # If backend is not a valid python path, raise error
                raise ValueError(
                    f"Failed to load executor backend '{backend}'. "
                    f"Ensure it is a valid python path, 'mp', or a DiffusionExecutor subclass. Error: {e}"
                ) from e

            if not issubclass(executor_class, DiffusionExecutor):
                raise TypeError(
                    f"distributed_executor_backend must be a subclass of DiffusionExecutor. Got {executor_class}."
                )
            return executor_class

        raise TypeError(
            f"distributed_executor_backend must be a string or a subclass of DiffusionExecutor. Got {type(backend)}."
        )

    def __init__(self, od_config: OmniDiffusionConfig):
        self.od_config = od_config
        self._init_executor()

    @abstractmethod
    def _init_executor(self) -> None:
        """Initialize the executor (e.g., launch workers, setup IPC)."""
        pass

    @abstractmethod
    def add_req(self, requests: list[OmniDiffusionRequest]):
        """Add requests to the execution queue."""
        pass

    @abstractmethod
    def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        unique_reply_rank: int | None = None,
    ) -> Any:
        """Execute a method on workers."""
        pass

    @abstractmethod
    def check_health(self) -> None:
        """Check if the executor and workers are healthy."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the executor and release resources."""
        pass
