from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from vllm_omni.experimental.fullduplex.minicpmo45.input import (
    MiniCPMO45PcmAppendBuffer,
)


@dataclass(slots=True)
class MiniCPMO45ServingSessionState:
    """Mutable serving state owned by one MiniCPM duplex session."""

    audio_buffer: MiniCPMO45PcmAppendBuffer = field(default_factory=MiniCPMO45PcmAppendBuffer)
    input_since_commit: bool = False
    speech_since_commit: bool = False
    committed_audio_payload: dict[str, object] | None = None
    committed_audio_operation_id: str | None = None
    committed_audio_reserved_bytes: int = 0
    deferred_response_create: bool = False
    deferred_precreate_response: bool = False
    data_plane_task: asyncio.Task[None] | None = None
    data_plane_restart_requested: bool = False
    continuation_owner_id: str | None = None
    continuation_units: int = 0
    pending_silence_task: asyncio.Task[bool] | None = None
    pending_silence_owner_id: str | None = None
    silence_continuation_scheduler: Callable[..., Awaitable[bool]] | None = None
    accepted_input_epoch: int = -1
    accepted_input_seq: int = 0
    accepted_input_seqs: set[int] = field(default_factory=set)

    def record_accepted_input(self, *, epoch: int, seq: int) -> None:
        if epoch != self.accepted_input_epoch:
            self.accepted_input_epoch = epoch
            self.accepted_input_seq = 0
            self.accepted_input_seqs.clear()
        self.accepted_input_seqs.add(seq)
        if seq > self.accepted_input_seq:
            self.accepted_input_seq = seq

    def accepted_input_watermark(self, *, epoch: int) -> int | None:
        if epoch != self.accepted_input_epoch or self.accepted_input_seq <= 0:
            return None
        return self.accepted_input_seq

    def mark_input_processed(self, *, epoch: int, seq: int) -> bool:
        if seq <= 0 or epoch != self.accepted_input_epoch or seq not in self.accepted_input_seqs:
            return False
        # The set represents accepted inputs still waiting for their model
        # decision. Removing the sequence both deduplicates late output and
        # bounds memory by the processing backlog rather than Session length.
        self.accepted_input_seqs.remove(seq)
        return True

    def retain_committed_audio(
        self,
        payload: dict[str, object],
        *,
        operation_id: str | None,
        reserved_bytes: int = 0,
    ) -> None:
        self.committed_audio_payload = payload
        self.committed_audio_operation_id = operation_id
        self.committed_audio_reserved_bytes += max(0, int(reserved_bytes))

    def clear_committed_audio(self) -> int:
        reserved_bytes = self.committed_audio_reserved_bytes
        self.committed_audio_payload = None
        self.committed_audio_operation_id = None
        self.committed_audio_reserved_bytes = 0
        self.deferred_response_create = False
        self.deferred_precreate_response = False
        return reserved_bytes

    def clear_continuation(self) -> None:
        self.continuation_owner_id = None
        self.continuation_units = 0
        self.pending_silence_task = None
        self.pending_silence_owner_id = None
