# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Chunk transfer handler.

Handles chunk-based data transfer between stages:
- Process payload data (thinker embeddings, code predictor codes)
- Merge/buffer chunks before sending
- Update request objects after receiving
"""

from collections import defaultdict
from collections.abc import Callable
from typing import Any

import torch
from vllm.logger import init_logger

from ..core.config import TransferContext
from .base import TransferHandler

logger = init_logger(__name__)


class ChunkHandler(TransferHandler):
    """Handler for chunk-based transfer operations.

    This handler manages:
    - Payload processing with custom functions
    - Stage-specific logic (thinker merge at Stage 0, code chunking at Stage 1)
    - Request field updates after receiving

    Note: This handler has some state for buffering (request_payload, code_prompt_token_ids).
    Consider moving this state to TransferManager if it causes issues.
    """

    def __init__(self, stage_id: int):
        self.stage_id = stage_id

        # Buffering state for payload merging
        # Key: request_id, Value: partial payload
        self._request_payload: dict[str, Any] = {}

        # Buffering state for code chunking
        # Key: request_id, Value: list of code chunks
        self._code_prompt_token_ids: dict[str, list[list[int]]] = defaultdict(list)

        # Track finished requests
        self._finished_requests: set[str] = set()

        # Chunking parameters (TODO: make configurable)
        self._chunk_size = 25
        self._left_context_size = 25

    def build_key(self, ctx: TransferContext) -> str:
        """Build chunk transfer key.

        Format: {external_request_id}_{stage_id}_{chunk_id}
        """
        ext_id = ctx.get_effective_request_id()
        return f"{ext_id}_{ctx.stage_id}_{ctx.chunk_id}"

    def prepare_send_data(self, ctx: TransferContext, raw_input: Any) -> Any | None:
        """Process payload for sending.

        Args:
            ctx: Transfer context
            raw_input: Dict with:
                - "pooling_output": Partial pooling output
                - "request": Request object
                - "custom_func": Optional processing function

        Returns:
            Processed payload to send, or None to skip/buffer
        """
        if not isinstance(raw_input, dict):
            logger.warning(f"Invalid raw_input type: {type(raw_input)}")
            return None

        pooling_output = raw_input.get("pooling_output")
        request = raw_input.get("request")
        custom_func: Callable | None = raw_input.get("custom_func")

        # Apply custom processing function
        payload_data = None
        if custom_func:
            try:
                payload_data = custom_func(
                    pooling_output=pooling_output,
                    request=request,
                )
            except Exception as e:
                logger.error(f"Failed to process payload: {e}")
                return None

        if not payload_data:
            logger.debug(f"No payload data for request {ctx.request_id}")
            return None

        # Stage-specific processing
        if self.stage_id == 0:
            return self._process_stage_0(ctx, payload_data)
        elif self.stage_id == 1:
            return self._process_stage_1(ctx, payload_data)
        else:
            return payload_data

    def process_recv_data(self, ctx: TransferContext, data: Any, request: Any) -> None:
        """Process received chunk and update request.

        Args:
            ctx: Transfer context
            data: Received payload data
            request: Request object to update
        """
        if not isinstance(data, dict):
            logger.warning(f"Invalid data type for chunk: {type(data)}")
            return

        if self.stage_id != 2:
            # Stage 0, 1: Set additional_information
            request.additional_information = data
            if data.get("finished"):
                self._finished_requests.add(ctx.request_id)
        else:
            # Stage 2: Append to prompt_token_ids
            if ctx.chunk_id == 0:
                request.prompt_token_ids = data.get("code_predictor_codes", [])
            else:
                request.prompt_token_ids += data.get("code_predictor_codes", [])

            if data.get("finished"):
                self._finished_requests.add(ctx.request_id)
                # Import here to avoid circular dependency
                try:
                    from vllm.v1.request import RequestStatus

                    request.status = RequestStatus.FINISHED_STOPPED
                except ImportError:
                    pass

    def on_send_complete(
        self,
        ctx: TransferContext,
        success: bool,
        size: int,
    ) -> None:
        """Log chunk send completion."""
        if success:
            logger.info(f"[Stage-{self.stage_id}] Sent chunk {ctx.chunk_id} for {ctx.request_id}")
        else:
            logger.error(f"[Stage-{self.stage_id}] Failed to send chunk for {ctx.request_id}")

    def on_recv_complete(
        self,
        ctx: TransferContext,
        data: Any,
        size: int,
    ) -> None:
        """Log chunk recv completion."""
        logger.info(f"[Stage-{self.stage_id}] Received chunk {ctx.chunk_id} for {ctx.request_id}")

    def is_request_finished(self, request_id: str) -> bool:
        """Check if a request is finished."""
        return request_id in self._finished_requests

    def clear_request_state(self, request_id: str) -> None:
        """Clear buffered state for a request."""
        self._request_payload.pop(request_id, None)
        self._code_prompt_token_ids.pop(request_id, None)
        self._finished_requests.discard(request_id)

    # ============ Stage-specific Processing ============

    def _process_stage_0(self, ctx: TransferContext, payload_data: dict) -> Any | None:
        """Stage 0: Merge thinker embeddings and hidden states.

        At Stage 0, we need to wait for both parts of the embeddings
        before sending. The first call buffers the data, the second
        call merges and returns.
        """
        chunk_id = ctx.chunk_id or 0

        if chunk_id == 0:
            if self._request_payload.get(ctx.request_id) is None:
                if not payload_data.get("finished"):
                    # Buffer first part
                    self._request_payload[ctx.request_id] = payload_data
                    return None  # Don't send yet
            else:
                # Merge with buffered data
                saved = self._request_payload.pop(ctx.request_id)

                if "thinker_embeddings" in saved and "thinker_embeddings" in payload_data:
                    payload_data["thinker_embeddings"] = torch.cat(
                        (saved["thinker_embeddings"], payload_data["thinker_embeddings"]),
                        dim=0,
                    )

                if "thinker_hidden_states" in saved and "thinker_hidden_states" in payload_data:
                    payload_data["thinker_hidden_states"] = torch.cat(
                        (saved["thinker_hidden_states"], payload_data["thinker_hidden_states"]),
                        dim=0,
                    )

                logger.info(f"[Stage-0] Merged embeddings for {ctx.request_id}")

        return payload_data

    def _process_stage_1(self, ctx: TransferContext, payload_data: dict) -> Any | None:
        """Stage 1: Code predictor chunking.

        Accumulate code predictor codes and send in chunks of _chunk_size.
        Include left context for continuity.
        """
        req_id = ctx.request_id

        # Accumulate codes
        self._code_prompt_token_ids[req_id].append(payload_data.get("code_predictor_codes", []))

        length = len(self._code_prompt_token_ids[req_id])
        chunk_length = length % self._chunk_size

        # Don't send until we have a full chunk (unless finished)
        if chunk_length != 0 and not payload_data.get("finished"):
            return None

        # Calculate context window
        context_length = chunk_length if chunk_length != 0 else self._chunk_size
        end_index = min(length, self._left_context_size + context_length)

        # Build chunked codes
        codes = self._code_prompt_token_ids[req_id][-end_index:]
        payload_data["code_predictor_codes"] = torch.tensor(codes).transpose(0, 1).reshape(-1).tolist()

        return payload_data
