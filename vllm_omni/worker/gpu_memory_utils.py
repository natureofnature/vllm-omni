"""NVML-based GPU memory accounting utilities.

Shared across worker types (OmniGPUWorkerBase, DiffusionWorker, etc.)
for process-aware memory accounting, with device-scoped fallback when
container PID namespaces hide the host PID from NVML.
"""

from __future__ import annotations

import os

from vllm.logger import init_logger
from vllm.third_party.pynvml import (
    nvmlDeviceGetComputeRunningProcesses,
    nvmlDeviceGetHandleByIndex,
    nvmlInit,
    nvmlShutdown,
)

logger = init_logger(__name__)


def is_process_scoped_memory_available() -> bool:
    """Check if NVML process-scoped memory tracking is available.

    When True, concurrent stage initialization is safe because each
    process can accurately measure its own GPU memory via NVML.
    When False, sequential initialization (file locks) is needed.
    """
    try:
        nvmlInit()
        nvmlShutdown()
        return True
    except Exception:
        return False


def parse_cuda_visible_devices() -> list[str | int]:
    """Parse CUDA_VISIBLE_DEVICES into a list of device identifiers.

    Returns list of integers (physical indices) or strings (UUIDs/MIG IDs).
    """
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible_devices:
        return []

    result: list[str | int] = []
    for item in visible_devices.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            result.append(int(item))
        except ValueError:
            # UUID (GPU-xxx) or MIG ID (MIG-xxx)
            result.append(item)
    return result


def get_device_handle(device_id: str | int):
    """Get NVML device handle by index or UUID."""
    if isinstance(device_id, int):
        return nvmlDeviceGetHandleByIndex(device_id)
    else:
        from vllm.third_party.pynvml import nvmlDeviceGetHandleByUUID

        return nvmlDeviceGetHandleByUUID(device_id)


def get_process_gpu_memory(local_rank: int) -> int | None:
    """Get GPU memory for the current workload on the target device via pynvml.

    Supports CUDA_VISIBLE_DEVICES with integer indices, UUIDs, or MIG IDs.

    Returns:
        Memory in bytes attributed to the current process when NVML can match
        the PID directly. In non-host-PID containers, falls back to the total
        compute-process memory on the target device. Returns None when NVML is
        unavailable.

    Raises:
        RuntimeError: If device validation fails (invalid index or UUID).
    """
    from vllm.third_party.pynvml import nvmlDeviceGetCount

    my_pid = os.getpid()
    visible_devices = parse_cuda_visible_devices()

    try:
        nvmlInit()
    except Exception as e:
        logger.warning("NVML init failed, will use profiling fallback: %s", e)
        return None

    try:
        if visible_devices and local_rank < len(visible_devices):
            device_id = visible_devices[local_rank]
            try:
                handle = get_device_handle(device_id)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to get NVML handle for device '{device_id}' (local_rank={local_rank}). "
                    f"Check CUDA_VISIBLE_DEVICES or stage config 'devices' setting."
                ) from e
        else:
            # No CUDA_VISIBLE_DEVICES or local_rank out of range: use index directly
            device_count = nvmlDeviceGetCount()
            if local_rank >= device_count:
                raise RuntimeError(
                    f"Invalid GPU device {local_rank}. Only {device_count} GPU(s) available. "
                    f"Check CUDA_VISIBLE_DEVICES or stage config 'devices' setting."
                )
            device_id = local_rank
            handle = nvmlDeviceGetHandleByIndex(local_rank)

        processes = list(nvmlDeviceGetComputeRunningProcesses(handle))
        for proc in processes:
            if proc.pid == my_pid:
                return proc.usedGpuMemory

        if processes:
            device_memory = sum(proc.usedGpuMemory for proc in processes)
            logger.warning(
                "NVML PID mismatch for local pid %d on device %r; "
                "using device-scoped compute memory across %d process(es). "
                "This commonly happens inside non-host PID containers.",
                my_pid,
                device_id,
                len(processes),
            )
            return device_memory
        return 0
    except RuntimeError:
        raise
    except Exception as e:
        logger.warning("NVML query failed, will use profiling fallback: %s", e)
        return None
    finally:
        try:
            nvmlShutdown()
        except Exception:
            pass
