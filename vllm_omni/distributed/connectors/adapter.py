# SPDX-License-Identifier: Apache-2.0
# temporary for compatibility with vllm_omni.entrypoints.omni_stage.py 
# and vllm_omni.entrypoints.omni_llm.py

import time
import logging
from typing import Any, Optional, Dict, Callable

logger = logging.getLogger(__name__)


def try_send_via_connector(
    connector: Any,
    stage_id: int,
    next_stage_id: int,
    req_id: str,
    next_inputs: Any,
    sampling_params: Any,
    original_prompt: Any,
    next_stage_queue_submit_fn: Callable[[Dict[str, Any]], None],
    metrics: Any
) -> bool:
    """
    Attempts to send data via OmniConnector.
    Returns True if successful, False otherwise.
    Encapsulates the logic of preparing payload, sending via connector, 
    sending notification, and recording metrics.
    """
    try:
        t0 = time.time()

        # Prepare data for connector
        payload_data = {
            "engine_inputs": next_inputs,
            "sampling_params": sampling_params,
            "metadata": {
                "original_prompt": original_prompt,
                "stage_transition": f"{stage_id}->{next_stage_id}",
                "timestamp": time.time()
            }
        }

        # Send data via connector
        success, serialized_size = connector.put(str(stage_id), str(next_stage_id), str(req_id), payload_data)
        
        if success:
            # Send lightweight notification via queue
            notify_payload = {
                "request_id": req_id,
                "sampling_params": sampling_params,
                "from_connector": True,
                "from_stage": str(stage_id),
                "to_stage": str(next_stage_id),
                "sent_ts": time.time(),
            }
            next_stage_queue_submit_fn(notify_payload)

            t1 = time.time()
            tx_ms = (t1 - t0) * 1000.0
            
            metrics.on_forward(
                stage_id,
                next_stage_id,
                req_id,
                serialized_size,  # Use size from connector
                float(tx_ms),
                True,  # Mark as using connector
            )
            return True
        else:
            # If put returned False, we let the caller handle fallback
            return False

    except Exception as e:
        logger.warning(
            "[Orchestrator] OmniConnector failed for req %s: %s; falling back to queue",
            req_id,
            e,
        )
        return False


def try_recv_via_connector(
    task: Dict[str, Any], 
    connectors: Dict[Any, Any], 
    stage_id: int,
) -> tuple[Any, Optional[Dict[str, Any]]]:
    """
    Attempts to resolve input data from either connector or IPC.
    Returns (engine_inputs, rx_metrics) or (None, None) if failed/skipped.
    """
    rid = task["request_id"]
    
    if task.get("from_connector"):
        from_stage = task.get("from_stage")
        to_stage = str(stage_id)

        if not from_stage:
            logger.error(
                "[Stage-%s] 'from_connector' is true but 'from_stage' is missing for request %s", stage_id, rid
            )
            return None, None

        # Get connector for this edge
        connector_key = (from_stage, to_stage)
        connector = connectors.get(connector_key)

        if connector:
            try:
                # Get data from connector with timeout
                _t_start = time.time()
                payload_data = connector.get(from_stage, to_stage, str(rid))
                _t_end = time.time()
                
                if payload_data and isinstance(payload_data, dict):
                    ein = payload_data.get("engine_inputs")
                    # Use connector's serialization size for consistency
                    serialized_size = len(connector.serialize_obj(payload_data))
                    decode_ms = (_t_end - _t_start) * 1000.0
                    
                    rx_metrics = {
                        "rx_decode_time_ms": decode_ms,
                        "rx_transfer_bytes": serialized_size
                    }
                    return ein, rx_metrics
                else:
                    logger.error(
                        "[Stage-%s] Failed to get data from connector for request %s or payload is empty", stage_id, rid
                    )
                    return None, None
            except Exception as e:
                logger.error(
                    "[Stage-%s] Error retrieving data from connector for request %s: %s", stage_id, rid, e
                )
                return None, None
        else:
            logger.error(
                "[Stage-%s] No connector found for edge %s -> %s for request %s", stage_id, from_stage, to_stage, rid
            )
            return None, None
    else:
        # Data comes from queue as usual
        #return maybe_load_from_ipc_with_metrics_fn(task, obj_key="engine_inputs", shm_key="engine_inputs_shm")
        return None, None

