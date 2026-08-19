from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from vllm_omni.experimental.fullduplex.minicpmo45.input import (
    MiniCPMO45PcmAppendBuffer,
)


@dataclass(slots=True)
class MiniCPMO45InputDecision:
    outcome: str
    response_id: str | None


@dataclass(slots=True)
class MiniCPMO45AcceptedInput:
    accepted_turn_id: int
    target_turn_id: int
    generation: int
    decisions: dict[int | None, MiniCPMO45InputDecision] = field(default_factory=dict)
    processed_turns: set[int] = field(default_factory=set)


@dataclass(slots=True)
class MiniCPMO45InputLifecycle:
    """Track accepted MiniCPM input identities without widening other adapters."""

    epoch: int = -1
    accepted_watermark: int = 0
    accepted: dict[int, MiniCPMO45AcceptedInput] = field(default_factory=dict)
    accepted_by_generation: dict[int, int] = field(default_factory=dict)
    final_inputs: set[tuple[int, int, int]] = field(default_factory=set)
    generation: int = 0
    inflight: dict[object, tuple[int, bool, int]] = field(default_factory=dict)
    ambiguous: dict[object, tuple[int, bool, int]] = field(default_factory=dict)
    pending_final_generations: set[tuple[int, int]] = field(default_factory=set)
    deferred_outputs: dict[tuple[int, int], list[dict[str, object]]] = field(default_factory=dict)
    acceptance_barriers: set[tuple[int, int]] = field(default_factory=set)
    accepted_owner_segments: list[tuple[int, int, int, int]] = field(default_factory=list)
    generation_target_turns: dict[int, int] = field(default_factory=dict)
    target_turn_high_watermark: int = -1
    latest_accepted_generation: int = -1
    response_owner: tuple[str, int, int, int] | None = None

    def _reset_epoch(self, epoch: int) -> None:
        if epoch == self.epoch:
            return
        self.epoch = epoch
        self.accepted_watermark = 0
        self.accepted.clear()
        self.accepted_by_generation.clear()
        self.final_inputs.clear()
        self.generation = 0
        self.inflight = {token: value for token, value in self.inflight.items() if value[0] == epoch}
        self.ambiguous = {token: value for token, value in self.ambiguous.items() if value[0] == epoch}
        self.pending_final_generations = {pending for pending in self.pending_final_generations if pending[0] == epoch}
        self.acceptance_barriers = {key for key in self.acceptance_barriers if key[0] == epoch}
        self.deferred_outputs = {key: values for key, values in self.deferred_outputs.items() if key[0] == epoch}
        self.accepted_owner_segments.clear()
        self.generation_target_turns.clear()
        self.target_turn_high_watermark = -1
        self.latest_accepted_generation = -1
        self.response_owner = None

    def begin(
        self,
        token: object,
        *,
        epoch: int,
        final: bool,
        generation: int | None = None,
    ) -> None:
        self._reset_epoch(epoch)
        input_generation = self.generation if generation is None else generation
        self.inflight[token] = (epoch, final, input_generation)
        if final:
            self.pending_final_generations.add((epoch, input_generation))

    def seal_generation(self, *, epoch: int) -> int:
        self._reset_epoch(epoch)
        generation = self.generation
        self.generation += 1
        self.pending_final_generations.add((epoch, generation))
        return generation

    def current_generation(self, *, epoch: int) -> int:
        self._reset_epoch(epoch)
        return self.generation

    def target_turn_for_generation(
        self,
        *,
        epoch: int,
        generation: int,
        minimum_turn_id: int,
    ) -> int:
        """Keep model-turn ownership monotonic within an input generation.

        A newer committed user turn must not reuse the projector cursor of an
        older unresolved turn merely because the older model output is late.
        Appends are accepted in wire order. Existing segments provide a lower
        bound for the generation, while an advanced session or response turn
        must not be forced back onto an already-consumed projector cursor.
        """
        self._reset_epoch(epoch)
        same_generation = [
            target_turn
            for _, _, owner_generation, target_turn in self.accepted_owner_segments
            if owner_generation == generation
        ]
        reserved_target = self.generation_target_turns.get(generation)
        if reserved_target is not None:
            same_generation.append(reserved_target)
        if same_generation:
            target_turn_id = max(minimum_turn_id, max(same_generation))
            self.generation_target_turns[generation] = target_turn_id
            self.target_turn_high_watermark = max(self.target_turn_high_watermark, target_turn_id)
            return target_turn_id
        older_targets = [
            target_turn
            for owner_generation, target_turn in self.generation_target_turns.items()
            if owner_generation < generation
        ]
        if older_targets:
            minimum_turn_id = max(minimum_turn_id, max(older_targets) + 1)
        if self.target_turn_high_watermark >= 0:
            minimum_turn_id = max(minimum_turn_id, self.target_turn_high_watermark + 1)
        self.generation_target_turns[generation] = minimum_turn_id
        self.target_turn_high_watermark = max(self.target_turn_high_watermark, minimum_turn_id)
        return minimum_turn_id

    def cancel_generation(self, *, epoch: int, generation: int) -> None:
        self.pending_final_generations.discard((epoch, generation))
        self._discard_orphan_deferred(epoch)

    def cancel(self, token: object) -> None:
        pending = self.inflight.pop(token, None)
        if pending is not None and pending[1]:
            self.pending_final_generations.discard((pending[0], pending[2]))
        if pending is not None:
            self._discard_orphan_deferred(pending[0])

    def mark_ambiguous(self, token: object) -> None:
        """Keep post-dispatch state until a retry returns an authoritative ACK."""
        pending = self.inflight.pop(token, None)
        if pending is not None:
            self.ambiguous[token] = pending

    def has_ambiguous_generation(self, *, epoch: int, generation: int) -> bool:
        return any(
            pending_epoch == epoch and pending_generation == generation
            for pending_epoch, _, pending_generation in self.ambiguous.values()
        )

    def _discard_orphan_deferred(self, epoch: int) -> None:
        if any(pending_epoch == epoch for pending_epoch, _, _ in self.inflight.values()) or any(
            pending_epoch == epoch for pending_epoch, _, _ in self.ambiguous.values()
        ):
            return
        self.deferred_outputs = {
            key: outputs for key, outputs in self.deferred_outputs.items() if key[0] != epoch or key[1] in self.accepted
        }

    def accept(
        self,
        token: object,
        *,
        epoch: int,
        seq: int,
        accepted_turn_id: int,
        target_turn_id: int,
    ) -> bool:
        pending = self.inflight.pop(token, None)
        self._reset_epoch(epoch)
        if seq <= 0:
            return False
        is_final = pending is not None and pending[0] == epoch and pending[1]
        generation = pending[2] if pending is not None and pending[0] == epoch else self.generation
        self.ambiguous = {
            pending_token: value
            for pending_token, value in self.ambiguous.items()
            if not (value[0] == epoch and value[2] <= generation)
        }
        self.generation_target_turns[generation] = max(
            target_turn_id,
            self.generation_target_turns.get(generation, target_turn_id),
        )
        self.target_turn_high_watermark = max(self.target_turn_high_watermark, target_turn_id)
        self.accepted_watermark = max(self.accepted_watermark, seq)
        self.latest_accepted_generation = max(self.latest_accepted_generation, generation)
        if (
            self.accepted_owner_segments
            and self.accepted_owner_segments[-1][1] + 1 == seq
            and self.accepted_owner_segments[-1][2:] == (generation, target_turn_id)
        ):
            lower_seq, _, _, _ = self.accepted_owner_segments[-1]
            self.accepted_owner_segments[-1] = (lower_seq, seq, generation, target_turn_id)
        else:
            self.accepted_owner_segments.append((seq, seq, generation, target_turn_id))
        previous_seq = self.accepted_by_generation.get(generation)
        self.accepted.setdefault(
            seq,
            MiniCPMO45AcceptedInput(
                accepted_turn_id=accepted_turn_id,
                target_turn_id=target_turn_id,
                generation=generation,
            ),
        )
        self.accepted_by_generation[generation] = max(self.accepted_by_generation.get(generation, 0), seq)
        if is_final:
            self.pending_final_generations.discard((epoch, generation))
            self.final_inputs.add((epoch, seq, target_turn_id))
        if (
            previous_seq is not None
            and previous_seq != seq
            and not any(
                final_epoch == epoch and final_seq == previous_seq for final_epoch, final_seq, _ in self.final_inputs
            )
        ):
            self.accepted.pop(previous_seq, None)
        self.acceptance_barriers.add((epoch, seq))
        self._discard_orphan_deferred(epoch)
        return is_final

    def release(self, *, epoch: int, seq: int) -> list[dict[str, object]]:
        outputs = self.drain_deferred(epoch=epoch, seq=seq)
        self.finish_release(epoch=epoch, seq=seq)
        return outputs

    def drain_deferred(self, *, epoch: int, seq: int) -> list[dict[str, object]]:
        return self.deferred_outputs.pop((epoch, seq), [])

    def finish_release(self, *, epoch: int, seq: int) -> None:
        self.acceptance_barriers.discard((epoch, seq))
        self._discard_orphan_deferred(epoch)

    def defer_unaccepted_output(self, *, epoch: int, seq: int, output: dict[str, object]) -> bool:
        self._reset_epoch(epoch)
        if seq <= 0:
            return False
        if (epoch, seq) in self.acceptance_barriers:
            self.deferred_outputs.setdefault((epoch, seq), []).append(dict(output))
            return True
        # Stage0 identities are committed monotonically within an epoch.  A
        # later ACK therefore proves every lower identity has already been
        # accepted, even if its cross-stage output arrives out of order.
        if seq <= self.accepted_watermark:
            return False
        if any(pending_epoch == epoch for pending_epoch, _, _ in self.inflight.values()) or any(
            pending_epoch == epoch for pending_epoch, _, _ in self.ambiguous.values()
        ):
            self.deferred_outputs.setdefault((epoch, seq), []).append(dict(output))
        return True

    def promote_latest_final(
        self,
        *,
        epoch: int,
        generation: int,
        target_model_turn_id: int,
    ) -> int | None:
        self._reset_epoch(epoch)
        seq = self.accepted_by_generation.get(generation)
        if seq is None or seq <= 0:
            return None
        self.pending_final_generations.discard((epoch, generation))
        self.generation_target_turns[generation] = max(
            target_model_turn_id,
            self.generation_target_turns.get(generation, target_model_turn_id),
        )
        self.target_turn_high_watermark = max(self.target_turn_high_watermark, target_model_turn_id)
        self.final_inputs.add((epoch, seq, target_model_turn_id))
        for index, (lower_seq, upper_seq, owner_generation, _) in enumerate(self.accepted_owner_segments):
            if owner_generation == generation and lower_seq <= seq <= upper_seq:
                self.accepted_owner_segments[index] = (
                    lower_seq,
                    upper_seq,
                    owner_generation,
                    target_model_turn_id,
                )
                break
        return seq

    def _output_owner(
        self,
        *,
        epoch: int,
        seq: int,
        model_turn_id: int | None,
    ) -> tuple[int, int] | None:
        if epoch != self.epoch:
            return None
        owner = next(
            (
                (generation, target_turn_id)
                for lower_seq, upper_seq, generation, target_turn_id in self.accepted_owner_segments
                if lower_seq <= seq <= upper_seq
            ),
            None,
        )
        if owner is None:
            return None
        generation, target_turn_id = owner
        if model_turn_id is not None and model_turn_id != target_turn_id:
            return None
        return generation, target_turn_id

    def output_target_turn(
        self,
        *,
        epoch: int,
        seq: int,
        model_turn_id: int | None,
    ) -> int | None:
        owner = self._output_owner(
            epoch=epoch,
            seq=seq,
            model_turn_id=model_turn_id,
        )
        return owner[1] if owner is not None else None

    def can_claim_response(
        self,
        *,
        epoch: int,
        seq: int,
        model_turn_id: int | None,
    ) -> bool:
        owner = self._output_owner(epoch=epoch, seq=seq, model_turn_id=model_turn_id)
        if owner is None:
            return False
        generation, _ = owner
        latest_generation = max(
            [self.latest_accepted_generation]
            + [
                pending_generation
                for pending_epoch, pending_generation in self.pending_final_generations
                if pending_epoch == epoch
            ]
        )
        if generation != latest_generation:
            return False
        latest_seq = self.accepted_by_generation.get(generation)
        if latest_seq is None:
            return False
        return owner == self._output_owner(
            epoch=epoch,
            seq=latest_seq,
            model_turn_id=None,
        )

    def response_accepts_or_claims(
        self,
        *,
        response_id: str,
        epoch: int,
        seq: int,
        model_turn_id: int | None,
    ) -> bool:
        owner = self._output_owner(epoch=epoch, seq=seq, model_turn_id=model_turn_id)
        if owner is None:
            return False
        if self.response_owner is not None and self.response_owner[0] != response_id:
            self.response_owner = None
        if self.response_owner is None:
            if not self.can_claim_response(
                epoch=epoch,
                seq=seq,
                model_turn_id=model_turn_id,
            ):
                return False
            generation, target_turn_id = owner
            self.response_owner = (response_id, epoch, generation, target_turn_id)
            return True
        _, owner_epoch, owner_generation, owner_turn = self.response_owner
        generation, target_turn_id = owner
        return (owner_epoch, owner_generation, owner_turn) == (epoch, generation, target_turn_id)

    def response_origin_input_seq(self, *, response_id: str, epoch: int) -> int | None:
        if self.response_owner is None or self.response_owner[:2] != (response_id, epoch):
            return None
        _, _, generation, target_turn_id = self.response_owner
        candidates = [
            upper_seq
            for _, upper_seq, owner_generation, owner_turn in self.accepted_owner_segments
            if (owner_generation, owner_turn) == (generation, target_turn_id)
        ]
        return max(candidates) if candidates else None

    def accepted_identity(self, *, epoch: int, generation: int | None = None) -> tuple[int, int] | None:
        if epoch != self.epoch:
            return None
        seq = self.accepted_watermark if generation is None else self.accepted_by_generation.get(generation, 0)
        if seq <= 0:
            return None
        accepted = self.accepted.get(seq)
        if accepted is None:
            return None
        return seq, accepted.target_turn_id

    def is_final(self, *, epoch: int, seq: int, model_turn_id: int | None = None) -> bool:
        if epoch != self.epoch or seq <= 0:
            return False
        return (
            self._matching_final_turn(
                epoch=epoch,
                seq=seq,
                model_turn_id=model_turn_id,
            )
            is not None
        )

    def final_target_turn(self, *, epoch: int, seq: int) -> int | None:
        targets = [
            target_turn
            for final_epoch, final_seq, target_turn in self.final_inputs
            if final_epoch == epoch and final_seq == seq
        ]
        return max(targets) if targets else None

    def _matching_final_turn(
        self,
        *,
        epoch: int,
        seq: int,
        model_turn_id: int | None,
    ) -> int | None:
        targets = {
            target_turn
            for final_epoch, final_seq, target_turn in self.final_inputs
            if final_epoch == epoch and final_seq == seq
        }
        if model_turn_id is not None:
            return model_turn_id if model_turn_id in targets else None
        if len(targets) == 1:
            return next(iter(targets))
        return None

    def is_latest_final(self, *, epoch: int, seq: int, model_turn_id: int | None = None) -> bool:
        if epoch != self.epoch or seq != self.accepted_watermark:
            return False
        if any(pending_epoch == epoch for pending_epoch, _ in self.pending_final_generations) or any(
            pending_epoch == epoch and pending_final for pending_epoch, pending_final, _ in self.inflight.values()
        ):
            return False
        matching = [
            (final_seq, target_turn)
            for final_epoch, final_seq, target_turn in self.final_inputs
            if final_epoch == epoch
        ]
        candidate_turn = model_turn_id if model_turn_id is not None else self.final_target_turn(epoch=epoch, seq=seq)
        return candidate_turn is not None and bool(matching) and (seq, candidate_turn) == max(matching)

    def superseded_finals(self, *, epoch: int, before_seq: int) -> list[tuple[int, int]]:
        """Return older unresolved finals that a newer terminal decision supersedes."""
        return sorted(
            (seq, target_turn)
            for final_epoch, seq, target_turn in self.final_inputs
            if final_epoch == epoch and seq < before_seq
        )

    def resolve_final(self, *, epoch: int, seq: int, model_turn_id: int | None = None) -> None:
        owner = self._output_owner(epoch=epoch, seq=seq, model_turn_id=model_turn_id)
        self.final_inputs = {
            final
            for final in self.final_inputs
            if not (final[0] == epoch and final[1] == seq and (model_turn_id is None or final[2] == model_turn_id))
        }
        if not any(final_epoch == epoch and final_seq == seq for final_epoch, final_seq, _ in self.final_inputs):
            self.accepted_by_generation = {
                generation: accepted_seq
                for generation, accepted_seq in self.accepted_by_generation.items()
                if accepted_seq != seq
            }
            self.accepted.pop(seq, None)
        if owner is not None and not any(
            final_epoch == epoch
            and self._output_owner(
                epoch=epoch,
                seq=final_seq,
                model_turn_id=final_turn,
            )
            == owner
            for final_epoch, final_seq, final_turn in self.final_inputs
        ):
            generation, _ = owner
            self.accepted_owner_segments = [
                segment for segment in self.accepted_owner_segments if segment[2] != generation
            ]
            self.generation_target_turns.pop(generation, None)
            if self.response_owner is not None and self.response_owner[2] == generation:
                self.response_owner = None

    def remember_decision(
        self,
        *,
        epoch: int,
        seq: int,
        model_turn_id: int | None,
        outcome: str,
        response_id: str | None = None,
    ) -> None:
        if epoch != self.epoch:
            return
        accepted = self.accepted.get(seq)
        if accepted is not None:
            accepted.decisions.setdefault(
                model_turn_id,
                MiniCPMO45InputDecision(
                    outcome=outcome,
                    response_id=response_id,
                ),
            )

    def pending_processed(
        self,
        *,
        epoch: int,
        seq: int,
        model_turn_id: int,
    ) -> tuple[str, str | None] | None:
        if epoch != self.epoch:
            return None
        accepted = self.accepted.get(seq)
        if accepted is None or model_turn_id in accepted.processed_turns:
            return None
        decision = accepted.decisions.get(model_turn_id)
        if decision is None:
            decision = accepted.decisions.get(None)
        if decision is None:
            return None
        return decision.outcome, decision.response_id

    def mark_processed_emitted(self, *, epoch: int, seq: int, model_turn_id: int) -> bool:
        if epoch != self.epoch:
            return False
        accepted = self.accepted.get(seq)
        if accepted is None or model_turn_id in accepted.processed_turns:
            return False
        accepted.processed_turns.add(model_turn_id)
        return True


@dataclass(slots=True)
class MiniCPMO45ServingSessionState:
    """Mutable serving state owned by one MiniCPM duplex session."""

    audio_buffer: MiniCPMO45PcmAppendBuffer = field(default_factory=MiniCPMO45PcmAppendBuffer)
    input_since_commit: bool = False
    speech_since_commit: bool = False
    committed_audio_payload: dict[str, object] | None = None
    committed_audio_operation_id: str | None = None
    committed_audio_reserved_bytes: int = 0
    pending_final_commit_event: dict[str, object] | None = None
    pending_final_commit_generation: int | None = None
    deferred_response_create: bool = False
    deferred_precreate_response: bool = False
    data_plane_task: asyncio.Task[None] | None = None
    data_plane_restart_requested: bool = False
    continuation_owner_id: str | None = None
    continuation_units: int = 0
    pending_silence_task: asyncio.Task[bool] | None = None
    pending_silence_owner_id: str | None = None
    silence_continuation_scheduler: Callable[..., Awaitable[bool]] | None = None
    input_lifecycle: MiniCPMO45InputLifecycle = field(default_factory=MiniCPMO45InputLifecycle)

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
        self.pending_final_commit_event = None
        self.pending_final_commit_generation = None
        self.deferred_response_create = False
        self.deferred_precreate_response = False
        return reserved_bytes

    def clear_continuation(self) -> None:
        self.continuation_owner_id = None
        self.continuation_units = 0
        if isinstance(self.pending_silence_owner_id, str) and self.pending_silence_owner_id.startswith(
            "final-generation:"
        ):
            return
        self.pending_silence_task = None
        self.pending_silence_owner_id = None
