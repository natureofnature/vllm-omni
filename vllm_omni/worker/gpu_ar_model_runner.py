"""AR GPU Model Runner for vLLM-Omni.

Exposes per-request hidden representations via ModelRunnerOutput.pooler_output
and also outputs sampled tokens.
"""

from __future__ import annotations

from contextlib import nullcontext
from copy import copy
from dataclasses import replace
from typing import Any, NamedTuple

import numpy as np
import torch
from vllm.config import CUDAGraphMode
from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
)
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.outputs import AsyncModelRunnerOutput, make_empty_encoder_model_runner_output
from vllm.v1.spec_decode.draft_model import DraftModelProposer
from vllm.v1.spec_decode.eagle import EagleProposer

try:
    from vllm.v1.spec_decode.extract_hidden_states import ExtractHiddenStatesProposer
except ImportError:

    class ExtractHiddenStatesProposer:  # type: ignore[no-redef]
        pass


from vllm.v1.structured_output.utils import apply_grammar_bitmask
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.gpu_model_runner import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncGPUModelRunnerOutput,
    IntermediateTensors,
)
from vllm.v1.worker.ubatch_utils import maybe_create_ubatch_slices
from vllm.v1.worker.utils import is_residual_scattered_for_sp

from vllm_omni.distributed.omni_connectors.kv_transfer_manager import OmniKVTransferManager
from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.utils.mm_outputs import build_mm_cpu, to_payload_element
from vllm_omni.v1_compat import maybe_get_kv_connector_output_compat
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner
from vllm_omni.worker.omni_connector_model_runner_mixin import OmniConnectorModelRunnerMixin

logger = init_logger(__name__)


class _AsyncChunkRequestAdapter:
    """Thin adapter that wraps ``CachedRequestState`` to expose the
    attributes expected by async-chunk processor functions
    (``external_req_id``, ``all_token_ids``, ``is_finished()``).

    ``CachedRequestState`` uses ``req_id`` whereas
    ``OmniEngineCoreRequest`` uses ``request_id`` / ``external_req_id``.
    This adapter bridges the gap without mutating the original object.
    """

    __slots__ = ("_inner", "_external_req_id", "_finished")

    def __init__(self, cached_state: Any, external_req_id: str, finished: bool):
        self._inner = cached_state
        self._external_req_id = external_req_id
        self._finished = finished

    # ---- attributes expected by process functions ----
    @property
    def external_req_id(self) -> str:
        return self._external_req_id

    @property
    def request_id(self) -> str:  # for send_chunk
        return self._inner.req_id

    @property
    def req_id(self) -> str:
        return self._inner.req_id

    @property
    def all_token_ids(self) -> list[int]:
        prompt = self._inner.prompt_token_ids or []
        return list(prompt) + list(self._inner.output_token_ids)

    @property
    def prompt_token_ids(self) -> list[int] | None:
        return self._inner.prompt_token_ids

    @property
    def output_token_ids(self) -> list[int]:
        return self._inner.output_token_ids

    def is_finished(self) -> bool:
        return self._finished

    def __getattr__(self, name: str) -> Any:
        # Delegate everything else to the inner CachedRequestState
        return getattr(self._inner, name)


class _FinishSentinelAdapter:
    """Minimal adapter for sending finish sentinels without a live request.

    Unlike ``_AsyncChunkRequestAdapter`` this does NOT require a
    ``CachedRequestState`` inner object.  It only carries the identifiers
    needed by ``send_chunk`` and the process functions to emit a
    ``finished=True`` sentinel payload.
    """

    __slots__ = ("_req_id", "_external_req_id")

    def __init__(self, req_id: str, external_req_id: str):
        self._req_id = req_id
        self._external_req_id = external_req_id

    @property
    def external_req_id(self) -> str:
        return self._external_req_id

    @property
    def request_id(self) -> str:
        return self._req_id

    @property
    def req_id(self) -> str:
        return self._req_id

    @property
    def all_token_ids(self) -> list[int]:
        return []

    @property
    def prompt_token_ids(self) -> list[int] | None:
        return []

    @property
    def output_token_ids(self) -> list[int]:
        return []

    def is_finished(self) -> bool:
        return True


class ExecuteModelState(NamedTuple):
    scheduler_output: SchedulerOutput
    logits: torch.Tensor | None
    spec_decode_metadata: Any
    spec_decode_common_attn_metadata: Any
    hidden_states: torch.Tensor
    sample_hidden_states: torch.Tensor
    aux_hidden_states: list[torch.Tensor] | None
    ec_connector_output: Any
    cudagraph_stats: Any
    # OMNI: multimodal_outputs field for omni-specific multimodal handling
    multimodal_outputs: Any
    # slot_mappings for attention/drafter (aligned with upstream v1 API)
    slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None = None


class GPUARModelRunner(OmniConnectorModelRunnerMixin, OmniGPUModelRunner):
    """Autoregressive GPU model runner that returns hidden states per request.

    Follows the v0.12 two-phase execute/sample flow from GPUModelRunner, and
    reuses Omni hooks for model_intermediate_buffer / multimodal outputs. This
    class only overrides sample_tokens to expose hidden states + multimodal
    outputs per request while keeping Async output semantics.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ar_log_counter: int = 0
        self.input_ids = self._make_buffer(self.max_num_tokens, dtype=torch.int32)
        # each model stage has their own hidden size
        self.hidden_size = self.model_config.hf_text_config.hidden_size
        self.inputs_embeds = self._make_buffer(self.max_num_tokens, self.hidden_size, dtype=self.dtype, numpy=False)
        # Initialize KV cache manager (preserve vllm_config fallback behavior)
        self.kv_transfer_manager = OmniKVTransferManager.from_vllm_config(self.vllm_config, self.model_config)
        # Initialize unified connector mixin (delegates KV to kv_transfer_manager)
        self.init_omni_connectors(
            vllm_config=self.vllm_config,
            model_config=self.model_config,
            kv_transfer_manager=self.kv_transfer_manager,
        )

    def _make_buffer(self, *size, dtype, numpy=True):
        # Prevent ray from pinning the buffer due to large size
        from vllm_omni.distributed.ray_utils.utils import (
            calculate_total_bytes,
            maybe_disable_pin_memory_for_ray,
        )

        total_bytes = calculate_total_bytes(size, dtype)

        # Use the context manager to temporarily disable pinning if needed
        with maybe_disable_pin_memory_for_ray(self, total_bytes):
            return super()._make_buffer(*size, dtype=dtype, numpy=numpy)

    def _build_model_sampler_output_token_ids(self) -> list[list[int]]:
        """Build decoded-token history for custom model samplers.

        vLLM only populates sampling_metadata.output_token_ids when penalties or
        logits processors require it. CosyVoice3's custom RAS sampler also
        depends on this history, so we reconstruct it directly from the input
        batch for prefer_model_sampler models.
        """
        req_output_token_ids = getattr(self.input_batch, "req_output_token_ids", [])
        req_ids = list(getattr(self.input_batch, "req_ids", []))
        output_token_ids = [list(req_output_token_ids[idx] or []) for idx in range(len(req_ids))]

        sampled_token_ids_cpu = getattr(self.input_batch, "sampled_token_ids_cpu", None)
        async_copy_ready_event = getattr(self.input_batch, "async_copy_ready_event", None)
        prev_req_id_to_index = getattr(self.input_batch, "prev_req_id_to_index", None)
        if sampled_token_ids_cpu is None or not output_token_ids or prev_req_id_to_index is None:
            return output_token_ids

        sampled_token_ids: list[list[int]] | None = None
        for index, req_id in enumerate(req_ids):
            prev_index = prev_req_id_to_index.get(req_id)
            if prev_index is None:
                continue
            req_history = output_token_ids[index]
            if not req_history or req_history[-1] != -1:
                continue
            if sampled_token_ids is None:
                assert async_copy_ready_event is not None
                async_copy_ready_event.synchronize()
                sampled_token_ids = sampled_token_ids_cpu.tolist()
            new_ids = list(sampled_token_ids[prev_index])
            if not new_ids:
                continue
            num_sampled_ids = len(new_ids) if new_ids[-1] != -1 else new_ids.index(-1)
            first_placeholder = req_history.index(-1)
            num_placeholders = len(req_history) - first_placeholder
            num_to_replace = min(num_sampled_ids, num_placeholders)
            req_history[first_placeholder : first_placeholder + num_to_replace] = new_ids[:num_to_replace]

        return output_token_ids

    def _sampling_metadata_for_model_sampler(self, sampling_metadata):
        output_token_ids = self._build_model_sampler_output_token_ids()
        if output_token_ids == sampling_metadata.output_token_ids:
            return sampling_metadata
        return replace(sampling_metadata, output_token_ids=output_token_ids)

    def capture_model(self) -> int:
        result = super().capture_model()
        self._capture_talker_mtp_graphs()
        return result

    def _capture_talker_mtp_graphs(self) -> None:
        from vllm_omni.worker.gpu_model_runner import CUDAGraphWrapper

        if not self.has_talker_mtp or not isinstance(self.talker_mtp, CUDAGraphWrapper):
            return

        from vllm.compilation.monitor import set_cudagraph_capturing_enabled
        from vllm.distributed.parallel_state import graph_capture

        capture_sizes = self.compilation_config.cudagraph_capture_sizes
        num_warmups = self.compilation_config.cudagraph_num_of_warmups
        capture_sizes = sorted(capture_sizes, reverse=True)
        logger.info("Capturing talker_mtp graphs for sizes %s", capture_sizes)

        set_cudagraph_capturing_enabled(True)
        try:
            with torch.inference_mode(), graph_capture(device=self.device):
                for bsz in capture_sizes:
                    _, batch_desc, _, _, _ = self._determine_batch_execution_and_padding(
                        num_tokens=bsz,
                        num_reqs=bsz,
                        num_scheduled_tokens_np=np.ones(bsz, dtype=np.int32),
                        max_num_scheduled_tokens=1,
                        use_cascade_attn=False,
                    )
                    n = batch_desc.num_tokens
                    ids = self.talker_mtp_input_ids.gpu[:n]
                    emb = self.talker_mtp_inputs_embeds.gpu[:n]
                    hid = self.last_talker_hidden.gpu[:n]
                    ts = self.text_step.gpu[:n]

                    for _ in range(num_warmups):
                        with set_forward_context(
                            None,
                            self.vllm_config,
                            cudagraph_runtime_mode=CUDAGraphMode.NONE,
                            batch_descriptor=batch_desc,
                        ):
                            self.talker_mtp(ids, emb, hid, ts)

                    with set_forward_context(
                        None,
                        self.vllm_config,
                        cudagraph_runtime_mode=CUDAGraphMode.FULL,
                        batch_descriptor=batch_desc,
                    ):
                        self.talker_mtp(ids, emb, hid, ts)
                    torch.cuda.synchronize()

            logger.info("Captured talker_mtp graphs for %d sizes", len(capture_sizes))
        except RuntimeError as e:
            raise RuntimeError(
                f"talker_mtp graph capture failed for a model that declared talker_mtp_graph_safe=True: {e}"
            ) from e
        finally:
            set_cudagraph_capturing_enabled(False)

    def _maybe_update_prefix_cache(
        self,
        hidden_states: torch.Tensor,
        multimodal_outputs: dict,
        num_tokens_unpadded: int,
        num_tokens_padded: int,
    ):
        """If prefix caching is enabled and it's the last pipeline parallelism rank,
        retrieve the hidden states & multimodal outputs from the prefix cache based
        on our batch slot mappings.
        """
        # Cache hidden states if we've enabled hidden state prefix caching
        # unless this isn't the last pipeline parallelism rank.
        if self.omni_prefix_cache is not None and get_pp_group().is_last_rank:
            # If this happens, it generally means the model is not following the correct
            # interface yet and is therefore currently not compatible with prefix cache.
            if multimodal_outputs is not None and not isinstance(multimodal_outputs, dict):
                logger.warning_once(
                    "prefix caching expects mm outputs to be a dict, but got %s",
                    type(multimodal_outputs),
                )

            self.omni_prefix_cache.update_omni_tensor_prefix_cache(
                hidden_states=hidden_states,
                multimodal_outputs=multimodal_outputs,
                num_tokens_unpadded=num_tokens_unpadded,
                slot_mapping=self.input_batch.block_table[0].slot_mapping.cpu,
                num_tokens_padded=num_tokens_padded,
            )

    def _maybe_get_combined_prefix_cache_tensors(
        self,
        hidden_states: torch.Tensor,
        multimodal_outputs: dict,
        num_scheduled_tokens: dict[str, int],
    ) -> tuple[dict[str, torch.Tensor] | None, dict | None]:
        """If prefix caching is enabled, extract the merged hidden states and multimodal outputs for
        all requests in the batch (including those that aren't a hit on Prefix cache).
        """
        # Prior to applying the post-processing func, extract
        # the prefix cached hidden states and multimodal states.
        combined_hidden_states, combined_multimodal_outputs = None, None
        if self.omni_prefix_cache is not None:
            combined_hidden_states = self.omni_prefix_cache.get_merged_hidden_states(
                query_start_loc=self.query_start_loc.cpu,
                input_batch=self.input_batch,
                hidden_states=hidden_states,
                num_scheduled_tokens=num_scheduled_tokens,
            )
            combined_multimodal_outputs = self.omni_prefix_cache.get_merged_multimodal_states(
                query_start_loc=self.query_start_loc.cpu,
                input_batch=self.input_batch,
                multimodal_outputs=multimodal_outputs,
                num_scheduled_tokens=num_scheduled_tokens,
            )
        return combined_hidden_states, combined_multimodal_outputs

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> OmniModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors | None:
        if self.execute_model_state is not None:
            raise RuntimeError("State error: sample_tokens() must be called after execute_model() returns None.")

        if not getattr(self, "_warmup_state_cleared", False):
            self._warmup_state_cleared = True
            model = getattr(self, "model", None)
            if model is not None and hasattr(model, "_clear_warmup_state"):
                model._clear_warmup_state()

        chunk_registrations = list(getattr(scheduler_output, "pending_chunk_registrations", []))
        input_registrations = list(getattr(scheduler_output, "pending_input_registrations", []))

        # [Omni] Register requests that need connector-backed recv work.
        for request in [*chunk_registrations, *input_registrations]:
            self.register_chunk_recv(request)

        # [Omni] Consume full_payload_mode stage inputs that arrived via bg thread
        self.recv_full_payload_inputs(scheduler_output)

        self._ar_log_counter += 1
        _finished = list(getattr(scheduler_output, "finished_req_ids", set()))
        if _finished or self._ar_log_counter % 5000 == 1:
            logger.info(
                "[Stage-%s AR] execute_model: scheduled=%s, "
                "pending_full_payload_send=%s, req_count=%s, "
                "finished_req_ids=%s (call#%s)",
                getattr(self, "_stage_id", "?"),
                scheduler_output.total_num_scheduled_tokens,
                list(self._pending_full_payload_send.keys()),
                len(self.requests),
                _finished[:5],
                self._ar_log_counter,
            )

        # [Omni] Drive mixin-owned KV lifecycle around the existing transfer manager.
        finished_reqs = getattr(scheduler_output, "finished_requests_needing_kv_transfer", {})
        if finished_reqs and hasattr(self.model, "get_kv_transfer_metadata"):
            for req_id, data in finished_reqs.items():
                try:
                    req_idx = self.input_batch.req_id_to_index.get(req_id)
                    num_computed = (
                        int(self.input_batch.num_computed_tokens_cpu[req_idx]) if req_idx is not None else None
                    )
                    model_meta = self.model.get_kv_transfer_metadata(
                        req_id,
                        num_computed_tokens=num_computed,
                    )
                    if model_meta:
                        existing = data.get("custom_metadata") or {}
                        existing.update(model_meta)
                        data["custom_metadata"] = existing
                except Exception as e:
                    logger.warning(f"Failed to get custom metadata from model for {req_id}: {e}")
        for req_id, kv_meta in finished_reqs.items():
            self.mark_kv_transfer(
                req_id,
                seq_len=kv_meta.get("seq_len", 0),
                block_ids=kv_meta.get("block_ids", []),
                custom_metadata=kv_meta.get("custom_metadata"),
            )
        self.kv_extracted_req_ids = self.send_kv_cache(
            finished_reqs=self.drain_pending_kv_transfers(),
            kv_caches=self.kv_caches,
            block_size=self.cache_config.block_size,
            cache_dtype=str(self.cache_config.cache_dtype),
            request_id_resolver=self._resolve_transfer_request_id,
        )
        if self.kv_extracted_req_ids:
            self.ack_kv_transfers(self.kv_extracted_req_ids)

        # [Omni] Async-chunk: send a final finished=True sentinel for requests
        # that completed in the previous engine-core cycle.  The last real
        # send_chunk call (in sample_tokens) had finished=False because
        # stop-token detection runs *after* the model runner returns.
        if self._async_chunk and _finished:
            self._send_async_chunk_finish_sentinels(_finished)

        # [Omni] Flush accumulated full_payload_mode outputs for requests that finished.
        # Use both scheduler_output.finished_req_ids (requests freed this cycle)
        # and any stale entries in _pending_full_payload_send whose request is no
        # longer tracked by the model runner.
        if self._pending_full_payload_send:
            flush_ids = set(getattr(scheduler_output, "finished_req_ids", set()))
            stale = {rid for rid in self._pending_full_payload_send if rid not in self.requests}
            flush_ids.update(stale)
            if flush_ids:
                logger.info(
                    "[Stage-%s AR] flush_full_payload_outputs: flushing %s (from finished=%s, stale=%s)",
                    getattr(self, "_stage_id", "?"),
                    flush_ids,
                    set(getattr(scheduler_output, "finished_req_ids", set())),
                    stale,
                )
                self.flush_full_payload_outputs(flush_ids)

        # [Omni] Clean up per-request mixin state for finished requests.
        # Must happen AFTER sentinel sending and batch flushing (which need
        # _put_req_chunk / _request_ids_mapping), but timing relative to
        # _update_states doesn't matter since cleanup uses its own dicts.
        if _finished:
            for rid in _finished:
                self.cleanup_finished_request(rid)

        if self.routed_experts_initialized:
            capturer = RoutedExpertsCapturer.get_instance()
            if capturer is not None:
                capturer.clear_buffer()  # noqa
            else:
                logger.error("RoutedExpertsCapturer not initialized.")

        if has_kv_transfer_group():
            kv_connector_metadata = getattr(scheduler_output, "kv_connector_metadata", None)
            if kv_connector_metadata is not None:
                get_kv_transfer_group().handle_preemptions(kv_connector_metadata)

        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        with (
            record_function_or_nullcontext("gpu_model_runner: preprocess"),
            self.synchronize_input_prep(),
        ):
            # Update persistent batch states.
            deferred_state_corrections_fn = self._update_states(scheduler_output)
            protected_req_ids = set(self.requests.keys())
            protected_req_ids.update(
                req_id
                for req_id in (
                    getattr(request, "request_id", None) for request in chunk_registrations + input_registrations
                )
                if req_id is not None
            )
            self.prune_inactive_requests(protected_req_ids)

            # [Omni] Post-update stale flush: requests removed by
            # _update_states are now detectable as stale.
            if self._pending_full_payload_send:
                stale = {rid for rid in self._pending_full_payload_send if rid not in self.requests}
                if stale:
                    logger.info("[Stage-%s AR] post-update stale flush: %s", getattr(self, "_stage_id", "?"), stale)
                    self.flush_full_payload_outputs(stale)

            if has_ec_transfer() and not get_ec_transfer().is_consumer:
                with self.maybe_get_ec_connector_output(
                    scheduler_output,
                    encoder_cache=self.encoder_cache,
                ) as ec_connector_output:
                    self._execute_mm_encoder(scheduler_output)

                    kv_ids = self.kv_extracted_req_ids
                    self.kv_extracted_req_ids = None

                    output = make_empty_encoder_model_runner_output(scheduler_output)
                    if kv_ids:
                        output = copy(output)
                        output.kv_extracted_req_ids = kv_ids
                    return output

            if not num_scheduled_tokens:
                if (
                    self.parallel_config.distributed_executor_backend == "external_launcher"
                    and self.parallel_config.data_parallel_size > 1
                ):
                    self._dummy_run(1)

                # Capture KV extraction results before early return;
                # sample_tokens() is skipped on this path so the IDs
                # would otherwise be silently overwritten next step.
                kv_ids = self.kv_extracted_req_ids
                self.kv_extracted_req_ids = None

                if not has_kv_transfer_group():
                    return self._empty_output_with_connector_signals()
                result = self.kv_connector_no_forward(scheduler_output, self.vllm_config)
                return self.attach_omni_connector_output(result)

            if self.cache_config.kv_sharing_fast_prefill:
                assert not self.num_prompt_logprobs, (
                    "--kv-sharing-fast-prefill produces incorrect "
                    "logprobs for prompt tokens, tokens, please disable "
                    "it when the requests need prompt logprobs"
                )

            num_reqs = self.input_batch.num_reqs
            req_ids = self.input_batch.req_ids
            tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
            num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)
            max_num_scheduled_tokens = int(num_scheduled_tokens_np.max())
            num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens

            logits_indices, spec_decode_metadata = self._prepare_inputs(
                scheduler_output,
                num_scheduled_tokens_np,
            )

            cascade_attn_prefix_lens = None
            # Disable cascade attention when using microbatching (DBO)
            if self.cascade_attn_enabled and not self.parallel_config.use_ubatching:
                # Pre-compute cascade attention prefix lengths
                cascade_attn_prefix_lens = self._compute_cascade_attn_prefix_lens(
                    num_scheduled_tokens_np,
                    self.input_batch.num_computed_tokens_cpu[:num_reqs],
                    scheduler_output.num_common_prefix_blocks,
                )

            (
                cudagraph_mode,
                batch_desc,
                should_ubatch,
                num_tokens_across_dp,
                cudagraph_stats,
            ) = self._determine_batch_execution_and_padding(
                num_tokens=num_tokens_unpadded,
                num_reqs=num_reqs,
                num_scheduled_tokens_np=num_scheduled_tokens_np,
                max_num_scheduled_tokens=max_num_scheduled_tokens,
                use_cascade_attn=cascade_attn_prefix_lens is not None,
                num_encoder_reqs=len(scheduler_output.scheduled_encoder_inputs),
            )

            num_tokens_padded = batch_desc.num_tokens
            num_reqs_padded = batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
            ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
                should_ubatch,
                num_scheduled_tokens_np,
                num_tokens_padded,
                num_reqs_padded,
                self.parallel_config.num_ubatches,
            )

            pad_attn = cudagraph_mode == CUDAGraphMode.FULL

            use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
            ubatch_slices_attn = ubatch_slices_padded if pad_attn else ubatch_slices

            # True if any attention backend handles KV cache update separately
            # from forward() (i.e., forward_includes_kv_cache_update=False). When true,
            # slot_mappings must use padded dimensions to match the key/value tensors.
            from vllm.v1.kv_cache_interface import EncoderOnlyAttentionSpec

            has_separate_kv_update = not all(
                all(g.backend.forward_includes_kv_cache_update for g in self.attn_groups[id])
                for id, spec in enumerate(self.kv_cache_config.kv_cache_groups)
                if not isinstance(spec.kv_cache_spec, EncoderOnlyAttentionSpec)
            )

            slot_mappings_by_group, slot_mappings = self._get_slot_mappings(
                num_tokens_padded=num_tokens_padded if pad_attn or has_separate_kv_update else num_tokens_unpadded,
                num_reqs_padded=(num_reqs_padded if pad_attn or has_separate_kv_update else num_reqs),
                num_tokens_unpadded=num_tokens_unpadded,
                ubatch_slices=ubatch_slices_padded,
            )

            attn_metadata, spec_decode_common_attn_metadata = self._build_attention_metadata(
                num_tokens=num_tokens_unpadded,
                num_tokens_padded=num_tokens_padded if pad_attn else None,
                num_reqs=num_reqs,
                num_reqs_padded=num_reqs_padded if pad_attn else None,
                max_query_len=max_num_scheduled_tokens,
                ubatch_slices=ubatch_slices_attn,
                logits_indices=logits_indices,
                use_spec_decode=use_spec_decode,
                num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
                cascade_attn_prefix_lens=cascade_attn_prefix_lens,
                slot_mappings=slot_mappings_by_group,
            )

            (
                input_ids,
                inputs_embeds,
                positions,
                intermediate_tensors,
                model_kwargs,
                ec_connector_output,
            ) = self._preprocess(scheduler_output, num_tokens_padded, intermediate_tensors)

        # Let the model adjust inputs before forward (e.g. restore input_ids
        # for multimodal position detection, fix decode position offsets).
        if hasattr(self.model, "prepare_runner_inputs"):
            input_ids, positions = self.model.prepare_runner_inputs(
                input_ids=input_ids,
                positions=positions,
                inputs_embeds=inputs_embeds,
                req_ids=req_ids[:num_reqs],
                num_computed_tokens=[int(self.input_batch.num_computed_tokens_cpu[i]) for i in range(num_reqs)],
                num_scheduled_tokens=[int(num_scheduled_tokens_np[i]) for i in range(num_reqs)],
                input_ids_buffer=self.input_ids.gpu[:num_tokens_padded],
            )

        # Set cudagraph mode to none if calc_kv_scales is true.
        # KV scales calculation involves dynamic operations that are incompatible
        # with CUDA graph capture.
        if self.calculate_kv_scales:
            cudagraph_mode = CUDAGraphMode.NONE
            # Mark KV scales as calculated after the first forward pass
            self.calculate_kv_scales = False

        # Run the model.
        # Use persistent buffers for CUDA graphs.
        # When spec decode is enabled, defer connector finalization
        # (wait_for_save + clear metadata) until after draft model runs.
        defer_kv_connector_finalize = self.speculative_config is not None
        with (
            nullcontext(),
            set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens_padded,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_mode,
                batch_descriptor=batch_desc,
                ubatch_slices=ubatch_slices_padded,
                slot_mapping=slot_mappings,  # OMNI: required for KV cache operations
            ),
            record_function_or_nullcontext("gpu_model_runner: forward"),
            maybe_get_kv_connector_output_compat(
                self,
                scheduler_output,
                clear_metadata=not defer_kv_connector_finalize,
            ) as kv_connector_output,
        ):
            model_output = self._model_forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
                sampling_metadata=self.input_batch.sampling_metadata,
                logits_index=logits_indices,
                sampler=self.sampler,
            )

            # [Omni] Map pending ropes metadata to req_ids.
            if hasattr(self.model, "flush_pending_metadata"):
                self.model.flush_pending_metadata(list(req_ids))

        with record_function_or_nullcontext("gpu_model_runner: postprocess"):
            if self.use_aux_hidden_state_outputs:
                # True when EAGLE 3 is used.
                hidden_states, aux_hidden_states = model_output
            else:
                # Common case.
                hidden_states = model_output
                aux_hidden_states = None

            hidden_states, multimodal_outputs = self.extract_multimodal_outputs(model_output)

            # Cache hidden states & multimodal outputs if we've enabled hidden state
            # prefix caching unless this isn't the last pipeline parallelism rank.
            self._maybe_update_prefix_cache(
                hidden_states=hidden_states,
                multimodal_outputs=multimodal_outputs,
                num_tokens_unpadded=num_tokens_unpadded,
                num_tokens_padded=num_tokens_padded,
            )

            if not self.broadcast_pp_output:
                # Common case.
                if not get_pp_group().is_last_rank:
                    # Return the intermediate tensors.
                    assert isinstance(hidden_states, IntermediateTensors)
                    hidden_states.kv_connector_output = kv_connector_output
                    self.kv_connector_output = kv_connector_output
                    return hidden_states

                if self.is_pooling_model:
                    # Return the pooling output.
                    return self._pool(
                        hidden_states,
                        num_scheduled_tokens,
                        num_scheduled_tokens_np,
                        kv_connector_output,
                    )

                sample_hidden_states = hidden_states[logits_indices]
                # Try with sampling_metadata first; fall back to without for models that don't support it
                try:
                    logits = self.model.compute_logits(
                        sample_hidden_states, sampling_metadata=self.input_batch.sampling_metadata
                    )
                except TypeError:
                    logits = self.model.compute_logits(sample_hidden_states)
            else:
                # Rare case.
                assert not self.is_pooling_model

                sample_hidden_states = hidden_states[logits_indices]
                if not get_pp_group().is_last_rank:
                    all_gather_tensors = {
                        "residual": not is_residual_scattered_for_sp(self.vllm_config, num_tokens_padded)
                    }
                    get_pp_group().send_tensor_dict(
                        hidden_states.tensors,
                        all_gather_group=get_tp_group(),
                        all_gather_tensors=all_gather_tensors,
                    )
                    logits = None
                else:
                    # Try with sampling_metadata first; fall back to without for models that don't support it
                    try:
                        logits = self.model.compute_logits(
                            sample_hidden_states, sampling_metadata=self.input_batch.sampling_metadata
                        )
                    except TypeError:
                        logits = self.model.compute_logits(sample_hidden_states)

                model_output_broadcast_data: dict[str, Any] = {}
                if logits is not None:
                    model_output_broadcast_data["logits"] = logits.contiguous()

                broadcasted = get_pp_group().broadcast_tensor_dict(
                    model_output_broadcast_data, src=len(get_pp_group().ranks) - 1
                )
                assert broadcasted is not None
                logits = broadcasted["logits"]

        self.execute_model_state = ExecuteModelState(
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            ec_connector_output,
            cudagraph_stats,
            multimodal_outputs,
            slot_mappings,  # OMNI: pass slot_mappings for drafter
        )
        self.kv_connector_output = kv_connector_output

        if deferred_state_corrections_fn:
            deferred_state_corrections_fn()

        return None

    def _sample(
        self,
        logits: torch.Tensor | None,
        spec_decode_metadata: Any,
    ):
        sampling_metadata = self.input_batch.sampling_metadata
        if spec_decode_metadata is None:
            model_sample = getattr(self.model, "sample", None)
            if logits is not None and callable(model_sample) and getattr(self.model, "prefer_model_sampler", False):
                # Apply logit bias (min_tokens, allowed_token_ids) before
                # the custom model sampler — the standard GPU sampler does
                # this internally, but prefer_model_sampler bypasses it.
                if hasattr(self.sampler, "logit_bias_state"):
                    self.sampler.logit_bias_state.apply_logit_bias(
                        logits,
                        self.input_batch.expanded_idx_mapping,
                        self.input_batch.idx_mapping_np,
                        self.input_batch.positions[self.input_batch.logits_indices],
                    )
                sampler_output = model_sample(
                    logits,
                    self._sampling_metadata_for_model_sampler(sampling_metadata),
                )
                if sampler_output is not None:
                    return sampler_output
            self.input_batch.update_async_output_token_ids()
            return self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )

        return super()._sample(logits, spec_decode_metadata)

    @staticmethod
    def _resolve_req_hidden_states(
        hidden_states_cpu: torch.Tensor,
        combined_hidden_states: dict[str, torch.Tensor] | None,
        rid: str,
        start: int,
        end: int,
    ):
        if combined_hidden_states is not None:
            # We always have all request IDs for prefix cache, even for
            # partial cache misses, so this should never happen.
            if rid not in combined_hidden_states:
                raise RuntimeError("Request IDs in the batch are missing from the merged states!")
            return combined_hidden_states[rid]
        # Prefix caching is disabled
        return hidden_states_cpu[start:end]

    @torch.inference_mode()
    def sample_tokens(
        self,
        grammar_output: GrammarOutput | None,
    ) -> OmniModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
        kv_extracted_req_ids = getattr(self, "kv_extracted_req_ids", None)
        self.kv_extracted_req_ids = None

        # Used for prefix cache
        combined_hidden_states = None
        combined_multimodal_outputs = None
        # Used when we don't use prefix cache; prefix cache builds the payloads
        # internally since it already needs to do this for the cached tensors
        mm_cpu = {}

        if self.execute_model_state is None:
            kv_connector_output = self.kv_connector_output
            self.kv_connector_output = None
            # Nothing to do (PP non-final rank case), output isn't used.
            if not kv_connector_output:
                return None  # type: ignore[return-value]

            # In case of PP with kv transfer, we need to pass through the
            # kv_connector_output
            if kv_connector_output.is_empty():
                return copy(EMPTY_MODEL_RUNNER_OUTPUT)

            output = copy(EMPTY_MODEL_RUNNER_OUTPUT)
            output.kv_connector_output = kv_connector_output
            return output

        # Unpack ephemeral state.
        (
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            ec_connector_output,
            cudagraph_stats,
            multimodal_outputs,
            slot_mappings,  # OMNI: unpack slot_mappings for drafter
        ) = self.execute_model_state
        self.execute_model_state = None
        seq_len = hidden_states.shape[0]

        # Apply structured output bitmasks if present.
        if grammar_output is not None:
            apply_grammar_bitmask(scheduler_output, grammar_output, self.input_batch, logits)

        # Correct padding values of prompt_token_ids to match the logits vocabulary size
        if logits is not None and not self.input_batch.sampling_metadata.no_penalties:
            smd = self.input_batch.sampling_metadata
            if smd.prompt_token_ids is not None:
                logits_vocab = logits.shape[-1]
                if self.input_batch.vocab_size > logits_vocab:
                    smd.prompt_token_ids = smd.prompt_token_ids.clamp(max=logits_vocab)

        with record_function_or_nullcontext("gpu_model_runner: sample"):
            sampler_output = self._sample(logits, spec_decode_metadata)

        self._update_states_after_model_execute(sampler_output.sampled_token_ids, scheduler_output)

        self._draft_token_ids = None
        self._draft_token_req_ids = None
        self.valid_sampled_token_count_gpu = None
        self.input_batch.prev_sampled_token_ids = None

        def propose_draft_token_ids(sampled_token_ids):
            assert spec_decode_common_attn_metadata is not None
            with record_function_or_nullcontext("gpu_model_runner: draft"):
                self._draft_token_ids = self.propose_draft_token_ids(
                    scheduler_output,
                    sampled_token_ids,
                    self.input_batch.sampling_metadata,
                    hidden_states,
                    sample_hidden_states,
                    aux_hidden_states,
                    spec_decode_metadata,
                    spec_decode_common_attn_metadata,
                    slot_mappings,  # OMNI: pass slot_mappings to drafter (upstream v1 API)
                )
                self._copy_draft_token_ids_to_cpu(scheduler_output)

        spec_config = self.speculative_config
        propose_drafts_after_bookkeeping = False
        if spec_config is not None:
            input_fits_in_drafter = spec_decode_common_attn_metadata is not None and (
                spec_decode_common_attn_metadata.max_seq_len + self.num_spec_tokens
                <= self.effective_drafter_max_model_len
            )
            use_gpu_toks = (
                spec_config.use_eagle() or spec_config.uses_draft_model() or spec_config.uses_extract_hidden_states()
            ) and not spec_config.disable_padded_drafter_batch
            if use_gpu_toks:
                assert isinstance(
                    self.drafter,
                    EagleProposer | DraftModelProposer | ExtractHiddenStatesProposer,
                )
                sampled_token_ids = sampler_output.sampled_token_ids
                if input_fits_in_drafter:
                    propose_draft_token_ids(sampled_token_ids)
                elif self.valid_sampled_token_count_event is not None:
                    assert spec_decode_common_attn_metadata is not None
                    next_token_ids, valid_sampled_tokens_count = self.drafter.prepare_next_token_ids_padded(
                        self.optimistic_seq_lens_cpu,
                        sampled_token_ids,
                        self.requests,
                        self.input_batch,
                        self.discard_request_mask.gpu,
                    )
                    self._copy_valid_sampled_token_count(next_token_ids, valid_sampled_tokens_count)
                    # Since we couldn't run the drafter,
                    # just use zeros for the draft tokens.
                    self._draft_token_ids = torch.zeros(1, device=self.device, dtype=torch.int32).expand(
                        len(self.input_batch.req_ids), self.num_spec_tokens
                    )
                    self._copy_draft_token_ids_to_cpu(scheduler_output, zeros_only=True)
            else:
                propose_drafts_after_bookkeeping = input_fits_in_drafter

        with record_function_or_nullcontext("gpu_model_runner: bookkeep"):
            (
                num_nans_in_logits,
                logprobs_lists,
                valid_sampled_token_ids,
                prompt_logprobs_dict,
                req_ids_output_copy,
                req_id_to_index_output_copy,
                invalid_req_indices,
            ) = self._bookkeeping_sync(
                scheduler_output,
                sampler_output,
                logits,
                hidden_states,
                scheduler_output.total_num_scheduled_tokens,
                spec_decode_metadata,
            )

        if propose_drafts_after_bookkeeping:
            # ngram and other speculative decoding methods use the sampled
            # tokens on the CPU, so they are run after bookkeeping.
            propose_draft_token_ids(valid_sampled_token_ids)

        # Finalize KV connector (wait_for_save + clear metadata) after
        # draft model runs. Deferred from target model forward to allow
        # draft model to also save its KV cache.
        if self.speculative_config is not None:
            if hasattr(self, "clear_kv_connector_metadata"):
                self.clear_kv_connector_metadata()

        with record_function_or_nullcontext("gpu_model_runner: eplb"):
            self.eplb_step()

        # kv_connector_output may be modified during drafting
        kv_connector_output = self.kv_connector_output
        self.kv_connector_output = None

        hidden_states_cpu = hidden_states.detach().to("cpu").contiguous()
        num_scheduled_tokens_np = getattr(self, "_omni_num_scheduled_tokens_np", None)
        if num_scheduled_tokens_np is None:
            req_ids = self.input_batch.req_ids
            num_scheduled_tokens_np = np.array(
                [scheduler_output.num_scheduled_tokens[rid] for rid in req_ids],
                dtype=np.int32,
            )

        # Prior to applying the post-processing func, extract
        # the prefix cached hidden states and multimodal states.
        if self.omni_prefix_cache is not None:
            (
                combined_hidden_states,
                combined_multimodal_outputs,
            ) = self._maybe_get_combined_prefix_cache_tensors(
                hidden_states,
                multimodal_outputs,
                scheduler_output.num_scheduled_tokens,
            )
        # Otherwise we don't have the mm CPU data yet, so we still need to build it
        if self.omni_prefix_cache is None:
            mm_cpu = build_mm_cpu(multimodal_outputs)

        self._process_model_intermediate_buffer_updates(
            hidden_states, multimodal_outputs, num_scheduled_tokens_np, scheduler_output
        )

        pooler_output: list[dict[str, object]] = []
        for rid in req_ids_output_copy:
            idx = req_id_to_index_output_copy[rid]
            start = int(self.query_start_loc.cpu[idx])
            sched = int(num_scheduled_tokens_np[idx])
            end = start + sched
            # If prefix cache is enabled, we have already split everything
            # by request and converted the states to CPU tensors
            req_hidden_states = self._resolve_req_hidden_states(
                hidden_states_cpu,
                combined_hidden_states,
                rid,
                start,
                end,
            )
            payload: dict[str, object] = {"hidden": req_hidden_states}

            mm_payload: dict[str, object] = {}
            if combined_multimodal_outputs or mm_cpu:
                if combined_multimodal_outputs:
                    # Prefix cache enabled; all items have already been processed
                    # and split apart for each request as needed, and all tensors
                    # have already been detached to the CPU. The only exception is
                    # lists, which we keep as passthrough data for consistent behavior
                    # in postprocess.
                    for mm_key in combined_multimodal_outputs.keys():
                        value = combined_multimodal_outputs[mm_key][rid]
                        if isinstance(value, list):
                            mm_payload[mm_key] = value[idx] if idx < len(value) else value[0]
                        else:
                            mm_payload[mm_key] = value

                else:
                    # Prefix cache disabled; we still need to process the data
                    for mm_key, mm_val in mm_cpu.items():
                        mm_payload[mm_key] = to_payload_element(
                            element=mm_val,
                            idx=idx,
                            start=start,
                            end=end,
                            pass_lists_through=False,
                            seq_len=seq_len,
                        )
                payload.update(mm_payload)
            pooler_output.append(payload)

        if self._async_chunk and self._custom_process_func is not None:
            _chunk_sent = 0
            _finished_ids = set(getattr(scheduler_output, "finished_req_ids", set()))
            for i, rid in enumerate(req_ids_output_copy):
                req_state = self.requests.get(rid)
                if req_state is not None and pooler_output[i]:
                    ext_id = self._resolve_transfer_request_id(rid)
                    wrapped = _AsyncChunkRequestAdapter(
                        req_state,
                        external_req_id=ext_id,
                        finished=(rid in _finished_ids),
                    )
                    self.send_chunk(request=wrapped, pooling_output=pooler_output[i])
                    _chunk_sent += 1
            if _chunk_sent and self._ar_log_counter % 5000 == 1:
                logger.info(
                    "[Stage-%s AR] sample_tokens: sent %s chunks (async_chunk)",
                    getattr(self, "_stage_id", "?"),
                    _chunk_sent,
                )
        elif self._async_chunk:
            if self._ar_log_counter == 1:
                logger.warning(
                    "[Stage-%s AR] sample_tokens: async_chunk=True but custom_process_func=%s",
                    getattr(self, "_stage_id", "?"),
                    self._custom_process_func,
                )

        if not self._async_chunk and self._custom_process_func is not None:
            for i, rid in enumerate(req_ids_output_copy):
                req_state = self.requests.get(rid)
                if pooler_output[i] and req_state is not None:
                    self.accumulate_full_payload_output(rid, pooler_output[i], req_state)
            if self._ar_log_counter % 5000 == 1:
                logger.info(
                    (
                        "[Stage-%s AR] sample_tokens: accumulated full_payload payloads "
                        "for %s reqs, pending_full_payload_send=%s"
                    ),
                    getattr(self, "_stage_id", "?"),
                    len(req_ids_output_copy),
                    list(self._pending_full_payload_send.keys()),
                )

        with record_function_or_nullcontext("gpu_model_runner: ModelRunnerOutput"):
            if self.routed_experts_initialized:
                capturer = RoutedExpertsCapturer.get_instance()
                if capturer is not None:
                    capturer.save_captured_experts(indices=self.slot_mapping)  # noqa
                else:
                    logger.error("RoutedExpertsCapturer not initialized.")
            output = OmniModelRunnerOutput(
                req_ids=req_ids_output_copy,
                req_id_to_index=req_id_to_index_output_copy,
                sampled_token_ids=valid_sampled_token_ids,
                logprobs=logprobs_lists,
                prompt_logprobs_dict=prompt_logprobs_dict,
                pooler_output=(pooler_output if self.vllm_config.model_config.engine_output_type != "text" else None),
                kv_connector_output=kv_connector_output,
                ec_connector_output=ec_connector_output if self.supports_mm_inputs else None,
                num_nans_in_logits=num_nans_in_logits,
                cudagraph_stats=cudagraph_stats,
            )
            output.kv_extracted_req_ids = kv_extracted_req_ids
            output.omni_connector_output = self.get_omni_connector_output()

        if not self.use_async_scheduling:
            return output
        with record_function_or_nullcontext("gpu_model_runner: AsyncGPUModelRunnerOutput"):
            async_output = AsyncGPUModelRunnerOutput(
                model_runner_output=output,
                sampled_token_ids=sampler_output.sampled_token_ids,
                logprobs_tensors=sampler_output.logprobs_tensors,
                invalid_req_indices=invalid_req_indices,
                async_output_copy_stream=self.async_output_copy_stream,
                vocab_size=self.input_batch.vocab_size,
            )
        with record_function_or_nullcontext("gpu_model_runner: set_async_sampled_token_ids"):
            # Save ref of sampled_token_ids CPU tensor if the batch contains
            # any requests with sampling params that require output ids.
            self.input_batch.set_async_sampled_token_ids(
                async_output.sampled_token_ids_cpu,
                async_output.async_copy_ready_event,
            )

        return async_output

    def _resolve_transfer_request_id(self, req_id: str) -> str:
        """Resolve cross-stage request ID from connector mappings or request state."""
        mapped = self._request_ids_mapping.get(req_id)
        if mapped is not None:
            return mapped

        req_state = self.requests.get(req_id)
        if req_state is None:
            return req_id
        external_req_id = getattr(req_state, "external_req_id", None)
        if external_req_id is not None:
            return str(external_req_id)
        return req_id

    def _send_async_chunk_finish_sentinels(self, finished_req_ids: list[str]) -> None:
        """Send a final ``finished=True`` sentinel chunk for completed requests.

        In async-chunk mode, ``send_chunk`` is called from ``sample_tokens``
        where the model runner doesn't yet know whether a stop token was
        generated (that decision is made by the engine core *after*
        ``sample_tokens`` returns).  As a result, the last real chunk is
        always sent with ``finished=False``.

        This helper is called at the start of the *next* ``execute_model``
        cycle, when ``scheduler_output.finished_req_ids`` lists the requests
        that completed in the previous cycle.  For each such request that
        was active in async-chunk sending, we enqueue one final sentinel
        payload whose ``finished`` flag is ``True`` so the downstream
        consumer knows the stream is complete.

        The sentinel is constructed using ``_FinishSentinelAdapter`` which
        does NOT require the ``CachedRequestState`` to still be alive in
        ``self.requests``.  This is critical because ``_update_states``
        from the previous cycle may have already freed the request.
        """
        if not self._custom_process_func:
            return
        for rid in finished_req_ids:
            # Resolve external ID without depending on self.requests.
            # Try _request_ids_mapping first (populated by register_chunk_recv),
            # then fall back to self.requests if still available, then rid itself.
            ext_id = self._request_ids_mapping.get(rid)
            if ext_id is None:
                ext_id = self._resolve_transfer_request_id(rid)

            # Check that we've actually been sending chunks for this request.
            # _put_req_chunk is keyed by external_req_id (set by send_chunk).
            if ext_id not in self._put_req_chunk:
                continue

            wrapped = _FinishSentinelAdapter(
                req_id=rid,
                external_req_id=ext_id,
            )
            # Send the finish sentinel with an empty pooling_output.
            # The process function should recognise finished=True and
            # flush/emit accordingly.
            self.send_chunk(request=wrapped, pooling_output={})
            logger.info(
                "[Stage-%s AR] sent async-chunk finish sentinel for req=%s (ext=%s)",
                getattr(self, "_stage_id", "?"),
                rid,
                ext_id,
            )
