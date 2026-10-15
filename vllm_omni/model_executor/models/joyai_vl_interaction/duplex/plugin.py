# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Turn-commit JoyAI observations on the existing three-stage pipeline."""

from __future__ import annotations

import asyncio
import io
import math
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy

import pybase64
from PIL import Image
from vllm.config import ModelConfig
from vllm.sampling_params import RequestOutputKind, SamplingParams

from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import DuplexAppendPlan, DuplexFence, DuplexOutputAction, DuplexOutputDecision
from vllm_omni.engine.duplex.plugin import (
    DuplexAuxiliaryPolicy,
    DuplexAuxiliaryRequest,
    DuplexInputCapacityError,
    DuplexModelPlugin,
    DuplexModelSessionState,
    DuplexRuntimeConfigError,
    duplex_history_digest,
    duplex_text_history,
)
from vllm_omni.experimental.fullduplex.joyvl.decision.output_parser import Action
from vllm_omni.experimental.fullduplex.joyvl.decision.prompts import (
    LONG_TERM_COMPRESS_PROMPT,
    MID_TERM_SUMMARY_PROMPT,
    QA_HISTORY_HEADER,
    SILENT_SYSTEM_PROMPT,
    USER_QUERY_HEADER,
    VIDEO_HISTORY_HEADER,
)
from vllm_omni.model_executor.models.joyai_vl_interaction.duplex.data_plane import JoyAIDataPlane, JoyAISessionState
from vllm_omni.model_executor.models.joyai_vl_interaction.duplex.memory import (
    RECENT_MESSAGES,
    SUMMARY_CHUNK_MESSAGES,
    SUMMARY_MAX_TOKENS,
    VisualFrame,
    VisualMemoryCapacityError,
    VisualMemorySource,
    VisualSummary,
)


class JoyAIDuplexPlugin(DuplexModelPlugin):
    plugin_id = "joyai-vl-interaction"
    supports_auxiliary_requests = True
    joyai_stage_id = 0

    def __init__(self, encode_audio: Callable[..., str]) -> None:
        self.data_plane = JoyAIDataPlane(encode_audio)
        self.processor = None
        self.max_model_len = 0
        self.max_images = 32
        self._mid_prompt_tokens = 0
        self._long_prompt_tokens = 0
        self._timestamp_tokens = 0
        self._processor_lock = asyncio.Lock()

    def capabilities(self, *, max_sessions: int) -> DuplexCapabilities:
        return DuplexCapabilities(
            supports_input_append=True,
            supports_replace_latest_chunk=False,
            supports_reencode_context=True,
            supports_turn_commit_only=True,
            supports_core_resumable_request=False,
            required_input_modalities=frozenset(),
            optional_input_modalities=frozenset({"video"}),
            supports_independent_io_streams=True,
            supports_realtime_endpoint=True,
            supports_multi_session=max_sessions > 1,
            supports_multi_session_same_replica=max_sessions > 1,
            supports_session_lease=True,
            supports_session_resume=True,
            session_admission_mode="engine_managed",
            supports_image_input=True,
            supports_text_only_turn=True,
            adapter_patterns=["turn_commit"],
            signal_sources=["client_event", "server_policy"],
            stage_handoff_transport="stage_connector",
            target_barge_in_latency_ms=None,
        )

    def create_session_state(self) -> JoyAISessionState:
        return JoyAISessionState()

    def validate_client_extra_body(self, extra_body: object) -> None:
        super().validate_client_extra_body(extra_body)
        if isinstance(extra_body, Mapping) and (extra_body.get("auto_response") or extra_body.get("realtime_tools")):
            raise DuplexRuntimeConfigError(
                "JoyAI video observations use explicit commits; tools use the follow-up adapter"
            )

    async def prepare_runtime_config(
        self, config: DuplexSessionConfig, *, model_config: ModelConfig | None
    ) -> dict[str, object]:
        self.validate_client_extra_body(config.extra_body)
        async with self._processor_lock:
            if self.processor is None:
                if model_config is None:
                    raise DuplexRuntimeConfigError("JoyAI requires the stage-0 model configuration")
                from transformers import AutoProcessor

                self.processor = await asyncio.to_thread(
                    AutoProcessor.from_pretrained,
                    model_config.model,
                    trust_remote_code=model_config.trust_remote_code,
                    revision=getattr(model_config, "revision", None),
                )
                self.max_model_len = model_config.max_model_len
                self.max_images = model_config.multimodal_config.get_limit_per_prompt("image")
                await asyncio.to_thread(self._prepare_memory_budget)
        return self.runtime_config_for_update(config, {})

    def runtime_config_for_update(
        self, config: DuplexSessionConfig, current: Mapping[str, object]
    ) -> dict[str, object]:
        self.validate_client_extra_body(config.extra_body)
        if config.ref_audio:
            raise DuplexRuntimeConfigError("JoyAI's current TTS profile uses CustomVoice speakers")
        extra = config.extra_body or {}
        return {
            "instructions": config.instructions,
            "initial_user_text": config.initial_user_text,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            **{
                key: extra.get(key, default)
                for key, default in (("tts_speaker", "Vivian"), ("tts_language", "Auto"), ("tts_instruct", ""))
            },
        }

    def configure_sampling_params(
        self, *, runtime_config: dict[str, object], defaults: tuple[object, ...]
    ) -> tuple[object, ...]:
        configured = []
        for index, params in enumerate(defaults):
            if isinstance(params, SamplingParams):
                params = params.clone()
                params.output_kind = RequestOutputKind.FINAL_ONLY
                if index == 0:
                    params.detokenize = True
                    params.skip_special_tokens = False
                    for key in ("temperature", "max_tokens"):
                        if runtime_config.get(key) is not None:
                            setattr(params, key, runtime_config[key])
            configured.append(params)
        return tuple(configured)

    @staticmethod
    def _sync_visual_history(state: JoyAISessionState, history: Sequence[dict[str, object]]) -> tuple[str, ...]:
        images = tuple(
            part["image_url"]["url"]
            for message in history
            if message.get("role") == "user" and isinstance(message.get("content"), list)
            for part in message["content"]
            if part.get("type") == "image_url"
        )
        previous = state.conversation_images
        if images[: len(previous)] != previous:
            # Conservatively revoke derived blocks as well as raw images. An
            # image edit can arrive after its raw source has been compressed.
            state.visual_memory.clear()
            state.frames = []
            state.conversation_images = ()
        return images

    def _visual_source(self, state: JoyAISessionState) -> VisualMemorySource | None:
        memory = state.visual_memory
        if state.visual_source is not None and memory.matches(state.visual_source):
            return state.visual_source
        state.visual_source = self._choose_visual_source(state)
        return state.visual_source

    def _choose_visual_source(self, state: JoyAISessionState) -> VisualMemorySource | None:
        memory = state.visual_memory
        mid_tokens = sum(entry.token_count + self._timestamp_tokens for entry in memory.mid_term)
        long_budget = self.max_model_len - memory.long_max_tokens - self._long_prompt_tokens
        if len(memory.mid_term) >= memory.mid_group or (
            len(memory.mid_term) >= 2 and mid_tokens + memory.mid_max_tokens > long_budget
        ):
            return memory.long_source(closure_reason="count" if len(memory.mid_term) >= memory.mid_group else "context")
        frame_tokens = sum(frame.token_count + self._timestamp_tokens for frame in memory.pending)
        if len(memory.pending) >= memory.chunk_frames:
            return memory.frame_source()
        if memory.pending and len(memory.pending) >= self.max_images:
            return memory.frame_source(closure_reason="image_limit")
        if memory.pending and frame_tokens + self._mid_prompt_tokens + memory.mid_max_tokens >= self.max_model_len:
            return memory.frame_source(closure_reason="context")
        return None

    def auxiliary_policy(
        self,
        *,
        state: DuplexModelSessionState,
        history_revision: int,
        history: tuple[dict[str, object], ...],
    ) -> DuplexAuxiliaryPolicy:
        if not isinstance(state, JoyAISessionState):
            raise TypeError("JoyAI requires JoyAISessionState")
        self._sync_visual_history(state, history)
        source = self._visual_source(state)
        return DuplexAuxiliaryPolicy(source_key=source or history_revision, allow_during_response=source is not None)

    def auxiliary_source_matches(
        self,
        *,
        request: DuplexAuxiliaryRequest,
        state: DuplexModelSessionState,
        history: tuple[dict[str, object], ...],
    ) -> bool:
        if not isinstance(state, JoyAISessionState):
            raise TypeError("JoyAI requires JoyAISessionState")
        self._sync_visual_history(state, history)
        if isinstance(request.model_source, VisualMemorySource):
            return state.visual_memory.matches(request.model_source)
        return super().auxiliary_source_matches(request=request, state=state, history=history)

    def auxiliary_preempted_by_input(
        self,
        *,
        payload: Mapping[str, object],
        state: DuplexModelSessionState,
        has_new_user_input: bool,
    ) -> bool:
        return bool(has_new_user_input or payload.get("audio") or not payload.get("video_frames"))

    def prepare_prompt_config(
        self, config: dict[str, object], *, state: DuplexModelSessionState, payload: dict[str, object]
    ) -> dict[str, object]:
        if not isinstance(state, JoyAISessionState):
            raise TypeError("JoyAI requires JoyAISessionState")
        state.memory.validate(duplex_text_history(config.get("conversation", [])))
        images = self._sync_visual_history(state, config.get("conversation", []))
        previous = state.conversation_images
        new_images = [*images[len(previous) :], *payload.get("video_frames", [])]
        # Append each conversation image once, alongside newly committed camera
        # frames, so retained camera history cannot displace a newer user image.
        frames = [*state.frames, *new_images][-4:]
        return {
            **config,
            "joyai_state": state,
            "joyai_frames": frames,
            "joyai_conversation_images": images,
            "joyai_new_frames": new_images,
            "joyai_pending_frames": tuple(state.visual_memory.pending),
            "joyai_visual_summaries": tuple([*state.visual_memory.long_term, *state.visual_memory.mid_term]),
            "joyai_summary": state.memory.text,
            "joyai_summary_messages": state.memory.covered_messages,
        }

    async def prepare_auxiliary_request(
        self,
        *,
        history: tuple[tuple[str, str], ...],
        runtime_config: Mapping[str, object],
        state: DuplexModelSessionState,
    ) -> DuplexAuxiliaryRequest | None:
        if not isinstance(state, JoyAISessionState):
            raise TypeError("JoyAI requires JoyAISessionState")
        visual = state.visual_memory
        source = self._visual_source(state)
        if source is not None:
            return await asyncio.to_thread(
                self._prepare_visual_summary,
                source,
                tuple(visual.pending if source.kind == "frames" else visual.mid_term),
                visual.mid_max_tokens if source.kind == "frames" else visual.long_max_tokens,
            )
        # Snapshot model-owned memory before yielding; preparation never mutates
        # it. Prompt invalidation and result application stay on the mailbox.
        current = state.memory.matches(history)
        covered = state.memory.covered_messages if current else 0
        previous_summary = state.memory.text if current else ""
        # Camera-only commits keep their ledger positions for fingerprints but
        # must not displace text from the verbatim recent-history window.
        cutoff = len(history)
        recent = 0
        while cutoff and recent < RECENT_MESSAGES:
            cutoff -= 1
            role, text = history[cutoff]
            if role in {"user", "assistant"} and text:
                recent += 1
        # Textless observations retain their ledger indices but consume no
        # summary budget, so a silent camera span cannot stall consolidation.
        end = covered
        pending_text = 0
        while end < cutoff and pending_text < SUMMARY_CHUNK_MESSAGES:
            role, text = history[end]
            if role in {"user", "assistant"} and text:
                pending_text += 1
            end += 1
        # Preserve the current query and unresolved user turn outside the summary.
        while end > covered and history[end - 1][0] != "assistant":
            end -= 1
        if end <= covered:
            return None

        def prepare() -> DuplexAuxiliaryRequest | None:
            target = end
            while target > covered:
                transcript = "\n".join(
                    f"[{index + 1}] {role}: {text}"
                    for index, (role, text) in enumerate(history[covered:target], start=covered)
                    if role in {"user", "assistant"} and text
                )
                content = LONG_TERM_COMPRESS_PROMPT.format(
                    merged_range=f"conversation turns 1-{target}",
                    summaries_text=(
                        f"Previous committed summary:\n{previous_summary}\n\nCommitted conversation:\n{transcript}"
                    ),
                    length_instruction=f" (at most {SUMMARY_MAX_TOKENS} tokens) ",
                )
                prompt = self.processor.apply_chat_template(
                    [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
                )
                if (
                    len(self.processor.tokenizer(text=[prompt])["input_ids"][0]) + SUMMARY_MAX_TOKENS
                    <= self.max_model_len
                ):
                    return DuplexAuxiliaryRequest(
                        stage_id=self.joyai_stage_id,
                        prompt={"prompt": prompt},
                        sampling_params=SamplingParams(
                            temperature=0.0,
                            max_tokens=SUMMARY_MAX_TOKENS,
                            output_kind=RequestOutputKind.FINAL_ONLY,
                            detokenize=True,
                        ),
                        history_length=target,
                        history_digest=duplex_history_digest(history[:target]),
                    )
                target -= 1
                while target > covered and history[target - 1][0] != "assistant":
                    target -= 1
            return None

        return await asyncio.to_thread(prepare)

    def _prepare_visual_summary(
        self,
        source: VisualMemorySource,
        entries: tuple[VisualFrame, ...] | tuple[VisualSummary, ...],
        output_limit: int,
    ) -> DuplexAuxiliaryRequest | None:
        # The caller snapshots before offloading; image objects are borrowed
        # read-only, never re-encoded or copied into a second memory queue.
        entries = entries[: len(source.covered_ids)]
        closure_reason = source.closure_reason
        if source.kind == "frames" and len(entries) > self.max_images:
            entries = entries[: self.max_images]
            closure_reason = "image_limit"
        while entries:
            images = []
            if source.kind == "frames":
                first, last = entries[0].time_s, entries[-1].time_s
                content = [
                    {
                        "type": "text",
                        "text": MID_TERM_SUMMARY_PROMPT.format(
                            chunk_index=source.chunk_index,
                            frame_range=f"{first:.3f}s ~ {last:.3f}s",
                            length_instruction="",
                            preferred_time_span="10 seconds",
                        ),
                    }
                ]
                for entry in entries:
                    content.extend([{"type": "text", "text": f"<{entry.time_s:.3f}s>"}, {"type": "image"}])
                    images.append(entry.image)
                visual_tokens = sum(entry.token_count for entry in entries) - len(images)
            else:
                if len(entries) < 2:
                    return None
                content = LONG_TERM_COMPRESS_PROMPT.format(
                    merged_range=f"{entries[0].start_s:.3f}s-{entries[-1].end_s:.3f}s",
                    summaries_text="\n\n".join(
                        f"<{entry.start_s:.3f}s ~ {entry.end_s:.3f}s>\n{entry.text}" for entry in entries
                    ),
                    length_instruction="",
                )
                visual_tokens = 0
            prompt = self.processor.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
            )
            if (
                len(self.processor.tokenizer(text=[prompt])["input_ids"][0]) + visual_tokens + output_limit
                <= self.max_model_len
            ):
                covered = source.covered_ids[: len(entries)]
                return DuplexAuxiliaryRequest(
                    stage_id=self.joyai_stage_id,
                    prompt={"prompt": prompt, "multi_modal_data": {"image": images} if images else {}},
                    sampling_params=SamplingParams(
                        temperature=0.8 if images else 0.3,
                        top_p=0.9 if images else 0.7,
                        top_k=40 if images else 30,
                        repetition_penalty=1.1,
                        presence_penalty=1.0 if images else 0.5,
                        max_tokens=output_limit,
                        output_kind=RequestOutputKind.FINAL_ONLY,
                        detokenize=True,
                    ),
                    history_length=0,
                    history_digest="",
                    model_source=VisualMemorySource(
                        source.kind,
                        source.generation,
                        covered,
                        source.chunk_index,
                        closure_reason,
                    ),
                    timeout_s=300.0,
                    retry_delays_s=(1.0, 5.0, 30.0),
                )
            entries = entries[:-1]
            closure_reason = "context"
        return None

    def apply_auxiliary_result(
        self, *, request: DuplexAuxiliaryRequest, output: object, state: DuplexModelSessionState
    ) -> bool:
        if not isinstance(state, JoyAISessionState):
            raise TypeError("JoyAI requires JoyAISessionState")
        completion = next(iter(getattr(output, "outputs", ()) or ()), None)
        text = (getattr(completion, "text", "") or "").strip()
        # A truncated or action-only completion is not a usable memory replacement.
        if not text or getattr(completion, "finish_reason", None) == "length":
            return False
        if any(marker in text for marker in ("</silence>", "</delegation>")):
            return False
        text = text.removeprefix("</response>").strip()
        tokens = len(self.processor.tokenizer(text=[text])["input_ids"][0])
        if isinstance(request.model_source, VisualMemorySource):
            return state.visual_memory.apply(request.model_source, text=text, token_count=tokens)
        if not text or tokens > SUMMARY_MAX_TOKENS:
            return False
        state.memory.text = text
        state.memory.covered_messages = request.history_length
        state.memory.covered_digest = request.history_digest
        return True

    async def prepare_append_plan(
        self,
        *,
        request_id: str,
        fence: DuplexFence,
        session_config: dict[str, object],
        runtime_config: dict[str, object],
        seq: int,
        turn_seq: int,
        payload: object,
        final: bool,
        sampling_params: object,
    ) -> DuplexAppendPlan:
        # Bind on the session loop before yielding. Close/cancel may remove this
        # request while image decoding runs; the worker never mutates live state.
        config = dict(session_config)
        state = config.pop("joyai_state")
        config["conversation"] = deepcopy(config.get("conversation", []))
        self.data_plane.begin_request(request_id)
        request = self.data_plane.requests[request_id]
        request.state = state
        request.frames = config["joyai_frames"]
        request.conversation_images = config.pop("joyai_conversation_images")
        request.modalities = tuple(config["modalities"])
        request.previous_response = state.last_response_text
        request.visual_generation = state.visual_memory.generation

        def prepare():
            new_images = self._decode_images(config.get("joyai_new_frames", []))
            counts = self._image_token_counts(new_images)
            config["joyai_new_images"] = new_images
            config["joyai_new_image_tokens"] = counts
            plan = self.plan_append(
                request_id=request_id,
                fence=fence,
                session_config=config,
                runtime_config=runtime_config,
                seq=seq,
                turn_seq=turn_seq,
                payload=payload,
                final=final,
                sampling_params=sampling_params,
            )
            return plan, list(zip(new_images, counts))

        try:
            plan, new_frames = await asyncio.to_thread(prepare)
            try:
                state.visual_memory.check_capacity([image for image, _ in new_frames])
            except VisualMemoryCapacityError as exc:
                raise DuplexInputCapacityError(str(exc), code="visual_memory_capacity") from exc
            request.new_frames = new_frames
            return plan
        except BaseException:
            self.data_plane.close_stream(request_id)
            raise

    def commit_append_plan(self, request_id: str) -> None:
        request = self.data_plane.requests[request_id]
        request.state.frames = request.frames
        request.state.conversation_images = request.conversation_images
        if request.visual_generation == request.state.visual_memory.generation:
            for image, tokens in request.new_frames:
                request.state.visual_memory.append(source_id=request_id, image=image, token_count=tokens)

    @staticmethod
    def _decode_images(frames: list[str]) -> list[Image.Image]:
        images = []
        for encoded in frames:
            if encoded.startswith("data:"):
                encoded = encoded.split(",", 1)[1]
            with Image.open(io.BytesIO(pybase64.b64decode(encoded, validate=True))) as image:
                if image.width * image.height > 262144:
                    ratio = math.sqrt(262144 / (image.width * image.height))
                    image = image.resize((max(1, int(image.width * ratio)), max(1, int(image.height * ratio))))
                images.append(image.convert("RGB").copy())
        return images

    def _image_token_counts(self, images: list[Image.Image]) -> list[int]:
        processor = self.processor.image_processor
        return [
            processor.get_number_of_image_patches(image.height, image.width, {}) // processor.merge_size**2
            for image in images
        ]

    def _prepare_memory_budget(self) -> None:
        memory = self.create_session_state().visual_memory

        def overhead(content: str) -> int:
            prompt = self.processor.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
            )
            return len(self.processor.tokenizer(text=[prompt])["input_ids"][0])

        self._mid_prompt_tokens = overhead(
            MID_TERM_SUMMARY_PROMPT.format(
                chunk_index=1,
                frame_range="0.000s ~ 999999999.999s",
                length_instruction="",
                preferred_time_span="10 seconds",
            )
        )
        self._long_prompt_tokens = overhead(
            LONG_TERM_COMPRESS_PROMPT.format(
                merged_range="0.000s-999999999.999s",
                summaries_text="",
                length_instruction="",
            )
        )
        # Conservative scheduling hint for each timestamp and template boundary.
        # The actual full prompt is tokenized again before every submission.
        self._timestamp_tokens = overhead("<999999999.999s ~ 999999999.999s>")
        image_tokens = self._image_token_counts([Image.new("RGB", (512, 512))])[0]
        if self.max_images < 1 or (
            self._mid_prompt_tokens + self._timestamp_tokens + image_tokens + memory.mid_max_tokens > self.max_model_len
            or self._long_prompt_tokens + 2 * (memory.mid_max_tokens + self._timestamp_tokens) + memory.long_max_tokens
            > self.max_model_len
        ):
            raise DuplexRuntimeConfigError(
                "JoyAI visual memory requires a context that fits one image summary and two mid-term summaries; "
                "increase the JoyAI stage max_model_len."
            )

    def plan_append(
        self,
        *,
        request_id: str,
        fence: DuplexFence,
        session_config: dict[str, object],
        runtime_config: dict[str, object],
        seq: int,
        turn_seq: int,
        payload: object,
        final: bool,
        sampling_params: object,
    ) -> DuplexAppendPlan:
        if not final or not isinstance(payload, Mapping) or payload.get("audio"):
            raise DuplexRuntimeConfigError("JoyAI video profile requires committed text/image/video input")
        history = []
        query = runtime_config.get("initial_user_text") or ""
        for index, (role, text) in enumerate(duplex_text_history(session_config.get("conversation", []))):
            if role not in {"user", "assistant"}:
                continue
            if text:
                if role == "user":
                    query = text
                if index >= session_config.get("joyai_summary_messages", 0):
                    history.append({"role": role, "content": ("</response> " if role == "assistant" else "") + text})
        # Keep interactive prefill bounded even while summary generation fails
        # or falls behind. The complete unsummarized ledger stays available to
        # prepare_auxiliary_request for eventual consolidation.
        history = history[-(RECENT_MESSAGES + SUMMARY_CHUNK_MESSAGES) :]
        while history and history[0]["role"] == "assistant":
            history.pop(0)
        # The active instruction is attached to the latest observation below.
        if history and history[-1]["role"] == "user":
            history.pop()
        unsummarized_history = history.copy()
        pending_frames = session_config.get("joyai_pending_frames", ())
        images = [entry.image for entry in pending_frames] + session_config.get("joyai_new_images", [])
        image_token_counts = [entry.token_count for entry in pending_frames] + session_config.get(
            "joyai_new_image_tokens", []
        )
        if not images:
            images = self._decode_images(session_config.get("joyai_frames", []))
            image_token_counts = self._image_token_counts(images)
        images = images[-self.max_images :] if self.max_images else []
        image_token_counts = image_token_counts[-self.max_images :] if self.max_images else []
        system = SILENT_SYSTEM_PROMPT
        if runtime_config.get("instructions"):
            system += "\n\n" + runtime_config["instructions"]
        base_system = system
        summary = session_config.get("joyai_summary")
        visual_summaries = [
            entry
            for entry in session_config.get("joyai_visual_summaries", ())
            if query or entry.chunk_index is not None
        ]
        if not isinstance(sampling_params, SamplingParams):
            raise DuplexRuntimeConfigError("JoyAI requires autoregressive sampling parameters")
        budget = self.max_model_len - (sampling_params.max_tokens or 256)
        while True:
            system = base_system
            if visual_summaries:
                system += (
                    "\n\n"
                    + VIDEO_HISTORY_HEADER
                    + "\n\n".join(
                        f"<{entry.start_s:.3f}s ~ {entry.end_s:.3f}s>\n{entry.text}" for entry in visual_summaries
                    )
                )
            if summary and query:
                system += "\n\n" + QA_HISTORY_HEADER + summary
            content = [{"type": "image"} for _ in images]
            content.append(
                {
                    "type": "text",
                    "text": f"{USER_QUERY_HEADER}\n{query}"
                    if query
                    else "Observe the current scene and choose whether to speak or stay silent.",
                }
            )
            messages = [{"role": "system", "content": system}, *history, {"role": "user", "content": content}]
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            text_tokens = self.processor.tokenizer(text=[prompt])["input_ids"][0]
            if len(text_tokens) + sum(image_token_counts) - len(images) <= budget:
                break
            if history:
                history.pop(0)
                while history and history[0]["role"] == "assistant":
                    history.pop(0)
            elif summary:
                # Derived memory must not displace standing instructions or
                # make a previously admissible current observation fail.
                summary = ""
                # Rebudget the recent turns now that the summary no longer
                # consumes their space. Covered history remains summarized.
                history = unsummarized_history
            elif visual_summaries:
                # Current observations/query take priority, as in the original
                # working chunk. Omitted blocks remain in bounded model memory.
                visual_summaries.pop(0)
            elif len(images) > 1:
                images.pop(0)
                image_token_counts.pop(0)
            else:
                raise DuplexRuntimeConfigError("JoyAI instruction and current observation exceed the context budget")
        return DuplexAppendPlan(
            prompt={
                "prompt": prompt,
                "multi_modal_data": {"image": images} if images else {},
                "additional_information": {
                    key: runtime_config[key] for key in ("tts_speaker", "tts_language", "tts_instruct")
                },
            }
        )

    def decide_output(
        self,
        *,
        stage_id: int,
        final_stage_id: int,
        segment_finished: bool,
        segment_token_ids: tuple[int, ...],
        segment_output_metadata: dict[str, object],
        output: object,
    ) -> DuplexOutputDecision | None:
        if stage_id != 0 or not segment_finished:
            return None
        observation = self.data_plane.requests.get(output.request_id)
        if observation is None:
            return DuplexOutputDecision(action=DuplexOutputAction.DIRECT_RESPONSE)
        action = observation.parse(output)
        if (
            "audio" not in observation.modalities
            or action.action is not Action.RESPONSE
            or not action.text
            or observation.is_repeat(action)
        ):
            return DuplexOutputDecision(action=DuplexOutputAction.DIRECT_RESPONSE)
        return None

    def project_intermediate_output(self, *, stage_id: int, output: object, context: object) -> bool:
        return stage_id == 0 and context.segment_finished
