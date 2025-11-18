
import torch
import os
import functools
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

def is_ray_initialized():
    """Check if Ray is initialized without hard dependency on Ray."""
    # 1. Try standard API
    try:
        import ray
        if ray.is_initialized():
            return True
    except ImportError:
        pass
    
    # 2. Fallback: Check environment variables typical for Ray Workers
    # RAY_RAYLET_PID is always set in Ray workers
    if "RAY_RAYLET_PID" in os.environ:
        return True
        
    return False

def calculate_total_bytes(size_args, dtype):
    """
    Calculate total bytes for a tensor allocation, handling nested tuples in size args.
    """
    num_elements = 1
    for s in size_args:
        if isinstance(s, (tuple, list)):
            for inner in s:
                num_elements *= inner
        else:
            num_elements *= s
            
    element_size = torch.tensor([], dtype=dtype).element_size()
    return num_elements * element_size

@contextmanager
def maybe_disable_pin_memory_for_ray(obj, size_bytes, threshold=32 * 1024 * 1024):
    """
    Context manager to temporarily disable pin_memory if running in Ray and 
    the allocation size exceeds the threshold.
    
    This is a workaround for Ray workers often having low ulimit -l (locked memory),
    causing OS call failed errors when allocating large pinned buffers.
    """
    should_disable = False
    old_pin = False
    
    # Check 1: Are we in a Ray-like environment?
    in_ray = is_ray_initialized()
    
    # Check 2: Is the size large enough to worry?
    is_large = size_bytes > threshold
    
    # Check 3: Is pinning currently enabled?
    is_pinned = getattr(obj, "pin_memory", False)
    
    if in_ray and is_large and is_pinned:
        should_disable = True
        old_pin = obj.pin_memory
        obj.pin_memory = False
        # logger.info(f"Disabling pin_memory for large allocation ({size_bytes/1024/1024:.2f} MB) in Ray environment.")
            
    try:
        yield
    finally:
        if should_disable:
            obj.pin_memory = old_pin
