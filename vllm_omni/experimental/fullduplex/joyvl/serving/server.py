# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from vllm_omni.experimental.fullduplex.joyvl.bridges.backend import ModelBackend, OpenAIBackend
from vllm_omni.experimental.fullduplex.joyvl.bridges.delegation import (
    DelegationBridge,
    ImageEditDelegationBridge,
    ImageGenDelegationBridge,
    OpenAIDelegationBridge,
    RoutingDelegationBridge,
    StubDelegationBridge,
)
from vllm_omni.experimental.fullduplex.joyvl.decision.output_parser import to_token_form
from vllm_omni.experimental.fullduplex.joyvl.memory.memory import Summarizer
from vllm_omni.experimental.fullduplex.joyvl.serving.config import InteractionConfig
from vllm_omni.experimental.fullduplex.joyvl.serving.session import InteractionSession, StepResult


class SessionOperationConflictError(ValueError):
    pass


class StaleSessionEpochError(RuntimeError):
    pass


@dataclass
class _CompletedOperation:
    digest: str
    result: StepResult


@dataclass
class _InFlightOperation:
    digest: str
    task: asyncio.Task[StepResult]


@dataclass
class _SessionSlot:
    session: InteractionSession
    epoch: int
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    commit_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    completed: OrderedDict[str, _CompletedOperation] = field(default_factory=OrderedDict)
    in_flight: dict[str, _InFlightOperation] = field(default_factory=dict)


class SessionManager:
    def __init__(
        self,
        config: InteractionConfig,
        *,
        backend: ModelBackend | None = None,
        summarizer_backend: ModelBackend | None = None,
    ) -> None:
        self.config = config
        self._backend = backend or OpenAIBackend(
            config.main_backend_url, config.main_model, config.api_key, config.request_timeout_seconds
        )
        self._summarizer: Summarizer | None = None
        if config.enable_memory:
            memory_backend = summarizer_backend or OpenAIBackend(
                config.resolved_summarizer_url,
                config.resolved_summarizer_model,
                config.api_key,
                config.request_timeout_seconds,
            )
            self._summarizer = Summarizer(
                memory_backend,
                key_frames_per_chunk=config.mid_term_key_frames,
                mid_term_max_tokens=config.mid_term_max_tokens,
                long_term_max_tokens=config.long_term_max_tokens,
                max_pixels=config.max_pixels,
            )
        self._delegation = self._build_delegation(config) if config.enable_delegation else None
        self._slots: dict[str, _SessionSlot] = {}
        self._epochs: dict[str, int] = {}
        self._retired_tasks: set[asyncio.Task[None]] = set()

    @staticmethod
    def _build_delegation(config: InteractionConfig) -> DelegationBridge | None:
        # Returns None (delegation off) unless a real backend is configured. The stub is
        # only returned on explicit opt-in (--delegation-kind stub) for tests/demos — it
        # would otherwise fold fake answers into session memory.
        kind = config.delegation_kind
        url = config.delegation_backend_url

        def chat_bridge(backend_url: str) -> OpenAIDelegationBridge:
            return OpenAIDelegationBridge(
                backend_url,
                config.resolved_delegation_model,
                config.resolved_delegation_api_key,
                max_tokens=config.delegation_max_tokens,
                timeout=config.request_timeout_seconds,
            )

        if kind == "stub":
            return StubDelegationBridge()
        if kind == "router":
            if not (url or config.delegation_image_url or config.delegation_edit_url):
                return None
            return RoutingDelegationBridge(
                chat=chat_bridge(url) if url else None,
                image=ImageGenDelegationBridge(config.delegation_image_url, timeout=config.request_timeout_seconds)
                if config.delegation_image_url
                else None,
                edit=ImageEditDelegationBridge(
                    config.delegation_edit_url,
                    config.delegation_edit_model or config.main_model,
                    timeout=config.request_timeout_seconds,
                )
                if config.delegation_edit_url
                else None,
            )
        if url and kind == "image":
            return ImageGenDelegationBridge(url, timeout=config.request_timeout_seconds)
        if url and kind == "edit":
            return ImageEditDelegationBridge(
                url, config.resolved_delegation_model, timeout=config.request_timeout_seconds
            )
        if url:
            return chat_bridge(url)
        return None

    def _get(self, session_id: str) -> _SessionSlot:
        slot = self._slots.get(session_id)
        if slot is None:
            session = InteractionSession(
                session_id,
                self.config,
                self._backend,
                summarizer=self._summarizer,
                delegation=self._delegation,
            )
            slot = _SessionSlot(session=session, epoch=self._epochs.setdefault(session_id, 0))
            self._slots[session_id] = slot
        return slot

    async def step(
        self,
        session_id: str,
        frames: list[str],
        query: str | None,
        *,
        operation_id: str | None = None,
        epoch: int | None = None,
    ) -> StepResult:
        await self._evict_expired()
        slot = self._get(session_id)
        if epoch is not None and epoch != slot.epoch:
            raise StaleSessionEpochError(f"Session {session_id!r} is at epoch {slot.epoch}, received {epoch}")

        def require_current() -> None:
            if self._slots.get(session_id) is not slot or self._epochs.get(session_id, 0) != slot.epoch:
                raise StaleSessionEpochError(f"Session {session_id!r} epoch {slot.epoch} completed after reset")

        digest = self._operation_digest(frames, query)
        if operation_id is not None:
            completed = slot.completed.get(operation_id)
            if completed is not None:
                if completed.digest != digest:
                    raise SessionOperationConflictError(
                        f"operation_id {operation_id!r} was reused with different input"
                    )
                return completed.result

            in_flight = slot.in_flight.get(operation_id)
            if in_flight is not None:
                if in_flight.digest != digest:
                    raise SessionOperationConflictError(
                        f"operation_id {operation_id!r} was reused with different input"
                    )
                result = await asyncio.shield(in_flight.task)
                require_current()
                return result

            task = asyncio.create_task(
                self._run_step(session_id, slot, frames, query, operation_id, digest),
                name=f"joyvl-step-{session_id}-{slot.epoch}-{operation_id}",
            )
            slot.in_flight[operation_id] = _InFlightOperation(digest, task)
            task.add_done_callback(
                lambda done, current_slot=slot, current_operation=operation_id: self._discard_in_flight(
                    current_slot, current_operation, done
                )
            )
            result = await asyncio.shield(task)
            require_current()
            return result

        return await self._run_step(session_id, slot, frames, query, None, digest)

    async def reset(self, session_id: str, *, expected_epoch: int) -> int:
        epoch, _ = await self.reset_with_status(session_id, expected_epoch=expected_epoch)
        return epoch

    async def reset_with_status(self, session_id: str, *, expected_epoch: int) -> tuple[int, bool]:
        if expected_epoch < 0:
            raise ValueError("expected_epoch must be non-negative")

        while True:
            current_epoch = self._epochs.get(session_id, 0)
            if expected_epoch + 1 == current_epoch:
                return current_epoch, False
            if expected_epoch != current_epoch:
                raise StaleSessionEpochError(
                    f"Session {session_id!r} is at epoch {current_epoch}, cannot reset epoch {expected_epoch}"
                )

            old_slot = self._slots.get(session_id)
            if old_slot is None:
                new_epoch = current_epoch + 1
                self._epochs[session_id] = new_epoch
                return new_epoch, True

            async with old_slot.commit_lock:
                if self._slots.get(session_id) is not old_slot:
                    continue
                current_epoch = self._epochs.get(session_id, 0)
                if expected_epoch + 1 == current_epoch:
                    return current_epoch, False
                if expected_epoch != current_epoch:
                    raise StaleSessionEpochError(
                        f"Session {session_id!r} is at epoch {current_epoch}, cannot reset epoch {expected_epoch}"
                    )

                new_epoch = current_epoch + 1
                self._epochs[session_id] = new_epoch
                self._slots.pop(session_id, None)
                task = asyncio.create_task(
                    self._retire_slot(old_slot),
                    name=f"joyvl-retire-{session_id}-{old_slot.epoch}",
                )
                self._retired_tasks.add(task)
                task.add_done_callback(self._retired_tasks.discard)
                return new_epoch, True

    async def set_persona(self, session_id: str, persona: str, *, epoch: int | None = None) -> bool:
        slot = self._get(session_id)
        if epoch is not None and epoch != slot.epoch:
            raise StaleSessionEpochError(f"Session {session_id!r} is at epoch {slot.epoch}, received {epoch}")
        async with slot.lock:
            if self._slots.get(session_id) is not slot:
                raise StaleSessionEpochError(f"Session {session_id!r} was reset while persona was queued")
            return slot.session.set_persona(persona)

    def current_epoch(self, session_id: str) -> int:
        return self._epochs.get(session_id, 0)

    @staticmethod
    def _operation_digest(frames: list[str], query: str | None) -> str:
        digest = hashlib.sha256()
        query_bytes = (query or "").encode()
        digest.update(len(query_bytes).to_bytes(8, "big"))
        digest.update(query_bytes)
        for frame in frames:
            frame_bytes = frame.encode()
            digest.update(len(frame_bytes).to_bytes(8, "big"))
            digest.update(frame_bytes)
        return digest.hexdigest()

    async def _run_step(
        self,
        session_id: str,
        slot: _SessionSlot,
        frames: list[str],
        query: str | None,
        operation_id: str | None,
        digest: str,
    ) -> StepResult:
        def require_current() -> None:
            if self._slots.get(session_id) is not slot or self._epochs.get(session_id, 0) != slot.epoch:
                raise StaleSessionEpochError(f"Session {session_id!r} epoch {slot.epoch} was reset while queued")

        @contextlib.asynccontextmanager
        async def commit_scope() -> AsyncIterator[None]:
            async with slot.commit_lock:
                require_current()
                yield

        async with slot.lock:
            require_current()
            result = await slot.session.step(frames, query, commit_scope=commit_scope)
            require_current()
            if operation_id is not None:
                slot.completed[operation_id] = _CompletedOperation(digest, result)
                slot.completed.move_to_end(operation_id)
                while len(slot.completed) > 256:
                    slot.completed.popitem(last=False)
            return result

    @staticmethod
    def _discard_in_flight(slot: _SessionSlot, operation_id: str, task: asyncio.Task[StepResult]) -> None:
        current = slot.in_flight.get(operation_id)
        if current is not None and current.task is task:
            slot.in_flight.pop(operation_id, None)

    @staticmethod
    async def _retire_slot(slot: _SessionSlot) -> None:
        async with slot.lock:
            await slot.session.reset()

    async def _evict_expired(self) -> None:
        ttl = self.config.session_timeout_seconds
        if ttl <= 0:
            return
        now = time.monotonic()
        expired = [
            sid
            for sid, slot in self._slots.items()
            if now - slot.session.last_access > ttl and not slot.lock.locked() and not slot.in_flight
        ]
        for sid in expired:
            slot = self._slots.get(sid)
            if slot is not None:
                await self.reset(sid, expected_epoch=slot.epoch)

    async def aclose(self) -> None:
        for sid in list(self._slots):
            slot = self._slots.get(sid)
            if slot is not None:
                await self.reset(sid, expected_epoch=slot.epoch)
        if self._retired_tasks:
            await asyncio.gather(*self._retired_tasks, return_exceptions=True)
            self._retired_tasks.clear()
        await self._backend.aclose()
        if self._summarizer is not None:
            await self._summarizer.aclose()
        if self._delegation is not None:
            await self._delegation.aclose()


async def _request_payload(request: Request, *, allow_empty: bool = False) -> dict[str, Any]:
    body = await request.body()
    if not body and allow_empty:
        return {}
    payload = await request.json()
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    return payload


def _extract_frames_and_query(payload: dict[str, Any]) -> tuple[list[str], str | None]:
    messages = payload.get("messages") or []
    frames: list[str] = []
    texts: list[str] = []
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
            continue
        for part in content or []:
            ptype = part.get("type")
            if ptype == "image_url":
                url = (part.get("image_url") or {}).get("url")
                if url:
                    frames.append(url)
            elif ptype == "text" and part.get("text"):
                texts.append(part["text"])
    query = "\n".join(t.strip() for t in texts if t.strip()) or None
    return frames, query


def _explicit_session_id(request: Request, payload: dict[str, Any]) -> str | None:
    value = request.headers.get("x-streaming-session")
    if value is None:
        value = request.headers.get("x-session-id")
    if value is None:
        value = payload.get("session_id")
    if value is None:
        value = payload.get("user")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("session_id must be a non-empty string or integer")
    normalized = str(value).strip()
    return normalized or None


def _session_id(request: Request, payload: dict[str, Any]) -> str:
    return _explicit_session_id(request, payload) or "default"


def _operation_id(request: Request, payload: dict[str, Any]) -> str | None:
    value = request.headers.get("x-operation-id")
    if value is None:
        value = payload.get("operation_id")
    if value is None:
        value = payload.get("input_seq")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("operation_id must be a non-empty string or integer")
    normalized = str(value).strip()
    if not normalized:
        raise ValueError("operation_id must be a non-empty string or integer")
    return normalized


def _session_epoch(request: Request, payload: dict[str, Any]) -> int | None:
    value = request.headers.get("x-session-epoch")
    if value is None:
        value = payload.get("session_epoch", payload.get("epoch"))
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("session_epoch must be a non-negative integer")
    epoch = int(value)
    if epoch < 0:
        raise ValueError("session_epoch must be a non-negative integer")
    return epoch


def _completion_response(
    model: str,
    result: StepResult,
    *,
    session_id: str | None = None,
    epoch: int | None = None,
    operation_id: str | None = None,
) -> dict[str, Any]:
    action = result.action
    memory = {"long_term_memory": result.long_term_memory, "mid_term_summaries": result.mid_term_summaries}
    return {
        "id": f"intchat-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": to_token_form(action)},
                "finish_reason": "stop",
            }
        ],
        "usage": None,
        "interaction": {
            "session_id": session_id,
            "epoch": epoch,
            "operation_id": operation_id,
            "action": action.action.value,
            "spoke": action.spoke,
            "text": action.text,
            "delegated_question": action.delegated_question,
            "delegation": result.delegation,
            "chunk_index": result.chunk_index,
            "frame_index": result.frame_index,
            "inference_skipped": result.inference_skipped,
            "latency_ms": result.latency_ms,
            "memory": memory,
        },
    }


def create_app(config: InteractionConfig) -> FastAPI:
    manager = SessionManager(config)

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            # close httpx/AsyncOpenAI clients and cancel+await pending tasks on shutdown/reload
            await manager.aclose()

    app = FastAPI(title="vLLM-Omni Interaction Server", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": config.main_model, "object": "model", "owned_by": "vllm-omni"}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        try:
            payload = await _request_payload(request)
            frames, query = _extract_frames_and_query(payload)
            if not frames:
                return JSONResponse(
                    {"error": "interaction server requires at least one image_url frame"}, status_code=400
                )
            session_id = _session_id(request, payload)
            operation_id = _operation_id(request, payload)
            epoch = _session_epoch(request, payload)
            current_epoch = manager.current_epoch(session_id)
            if epoch is None:
                if current_epoch != 0:
                    return JSONResponse(
                        {
                            "error": "session_epoch is required after a session reset",
                            "type": "missing_session_epoch",
                            "current_epoch": current_epoch,
                        },
                        status_code=409,
                    )
                epoch = 0
            result = await manager.step(
                session_id,
                frames,
                query,
                operation_id=operation_id,
                epoch=epoch,
            )
        except SessionOperationConflictError as exc:
            return JSONResponse(
                {"error": str(exc), "type": "operation_conflict"},
                status_code=409,
            )
        except StaleSessionEpochError as exc:
            return JSONResponse(
                {
                    "error": str(exc),
                    "type": "stale_session_epoch",
                    "current_epoch": manager.current_epoch(session_id),
                },
                status_code=409,
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc), "type": "invalid_request"}, status_code=400)
        return JSONResponse(
            _completion_response(
                config.main_model,
                result,
                session_id=session_id,
                epoch=manager.current_epoch(session_id),
                operation_id=operation_id,
            )
        )

    @app.post("/reset")
    @app.post("/v1/streaming/reset")
    async def reset(request: Request) -> JSONResponse:
        try:
            payload = await _request_payload(request, allow_empty=True)
            session_id = _session_id(request, payload)
            expected_epoch = _session_epoch(request, payload)
            if expected_epoch is None:
                return JSONResponse(
                    {
                        "error": "session_epoch is required to reset a session",
                        "type": "missing_session_epoch",
                        "current_epoch": manager.current_epoch(session_id),
                    },
                    status_code=400,
                )
            epoch, advanced = await manager.reset_with_status(session_id, expected_epoch=expected_epoch)
        except ValueError as exc:
            return JSONResponse({"error": str(exc), "type": "invalid_request"}, status_code=400)
        except StaleSessionEpochError as exc:
            return JSONResponse(
                {
                    "error": str(exc),
                    "type": "stale_session_epoch",
                    "current_epoch": manager.current_epoch(session_id),
                },
                status_code=409,
            )
        return JSONResponse(
            {
                "status": "reset",
                "session_id": session_id,
                "epoch": epoch,
                "advanced": advanced,
            }
        )

    @app.post("/v1/streaming/persona")
    async def persona(request: Request) -> JSONResponse:
        try:
            payload = await _request_payload(request, allow_empty=True)
            session_id = _session_id(request, payload)
            epoch = _session_epoch(request, payload)
            current_epoch = manager.current_epoch(session_id)
            if epoch is None and current_epoch != 0:
                return JSONResponse(
                    {
                        "error": "session_epoch is required after a session reset",
                        "type": "missing_session_epoch",
                        "current_epoch": current_epoch,
                    },
                    status_code=409,
                )
            if epoch is None:
                epoch = 0
            ok = await manager.set_persona(session_id, payload.get("persona", "default"), epoch=epoch)
        except StaleSessionEpochError as exc:
            return JSONResponse(
                {
                    "error": str(exc),
                    "type": "stale_session_epoch",
                    "current_epoch": manager.current_epoch(session_id),
                },
                status_code=409,
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc), "type": "invalid_request"}, status_code=400)
        return JSONResponse(
            {
                "status": "ok" if ok else "unknown_persona",
                "session_id": session_id,
                "epoch": manager.current_epoch(session_id),
            }
        )

    return app


def _build_config(args: argparse.Namespace) -> InteractionConfig:
    config = InteractionConfig(
        main_backend_url=args.main_backend_url,
        main_model=args.main_model,
        persona=args.persona,
        enable_memory=not args.no_memory,
        summarizer_backend_url=args.summarizer_backend_url,
        summarizer_model=args.summarizer_model,
        enable_delegation=not args.no_delegation,
        delegation_backend_url=args.delegation_backend_url,
        delegation_model=args.delegation_model,
        delegation_api_key=args.delegation_api_key,
        delegation_kind=args.delegation_kind,
        delegation_image_url=args.delegation_image_url,
        delegation_edit_url=args.delegation_edit_url,
        delegation_edit_model=args.delegation_edit_model,
        force_silence_before_query=not args.no_force_silence,
    )
    if args.chunk_frames is not None:
        config.chunk_frames = args.chunk_frames
    config.response_dedup_threshold = args.response_dedup_threshold
    config.sampling.max_tokens = args.max_tokens
    config.sampling.temperature = args.temperature
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM-Omni streaming interaction server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8070)
    parser.add_argument("--main-backend-url", default="http://127.0.0.1:8061/v1")
    parser.add_argument("--main-model", default="JoyAI-VL-Interaction-Preview")
    parser.add_argument("--persona", default="default", choices=["default", "silent", "talkative"])
    parser.add_argument("--summarizer-backend-url", default=None)
    parser.add_argument("--summarizer-model", default=None)
    parser.add_argument("--chunk-frames", type=int, default=None)
    parser.add_argument(
        "--response-dedup-threshold",
        type=float,
        default=1.0,
        help="1.0 drops only exact repeats (reference); < 1.0 also drops near-duplicate narration",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--no-memory", action="store_true", help="disable mid/long-term summarizer memory")
    parser.add_argument("--no-delegation", action="store_true", help="disable the delegation bridge")
    parser.add_argument(
        "--delegation-backend-url",
        default=None,
        help="OpenAI-compatible endpoint for the background brain; unset falls back to the stub",
    )
    parser.add_argument("--delegation-model", default=None, help="model name for the background brain")
    parser.add_argument(
        "--delegation-api-key",
        default=None,
        help="API key for the background brain endpoint (e.g. an Anthropic key for a claude-* model)",
    )
    parser.add_argument(
        "--delegation-kind",
        default="chat",
        choices=["chat", "image", "edit", "router", "stub"],
        help="chat = text/VL brain; image = text-to-image; edit = restyle the frame; "
        "router = dispatch by request; stub = canned demo/test answers. chat/image/edit/router "
        "need a backend URL — without one, delegation stays off.",
    )
    parser.add_argument("--delegation-image-url", default=None, help="router mode: text-to-image endpoint")
    parser.add_argument("--delegation-edit-url", default=None, help="router mode: image-edit endpoint")
    parser.add_argument("--delegation-edit-model", default=None, help="router mode: image-edit model name")
    parser.add_argument("--no-force-silence", action="store_true", help="run the model before any user query")
    args = parser.parse_args()

    app = create_app(_build_config(args))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
