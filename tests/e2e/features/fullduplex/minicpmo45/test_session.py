from __future__ import annotations

import pytest

from vllm_omni.experimental.fullduplex.minicpmo45.session import (
    MiniCPMO45InputLifecycle,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _accept(
    lifecycle: MiniCPMO45InputLifecycle,
    *,
    seq: int,
    turn: int,
    final: bool = False,
) -> None:
    token = object()
    lifecycle.begin(token, epoch=0, final=final)
    lifecycle.accept(
        token,
        epoch=0,
        seq=seq,
        accepted_turn_id=turn,
        target_turn_id=turn,
    )
    lifecycle.release(epoch=0, seq=seq)


def test_promoted_final_does_not_reuse_a_decision_from_an_older_turn() -> None:
    lifecycle = MiniCPMO45InputLifecycle()
    _accept(lifecycle, seq=7, turn=0)
    lifecycle.remember_decision(
        epoch=0,
        seq=7,
        model_turn_id=0,
        outcome="listen",
    )

    assert lifecycle.promote_latest_final(epoch=0, generation=0, target_model_turn_id=1) == 7
    assert not lifecycle.is_final(epoch=0, seq=7, model_turn_id=0)
    assert lifecycle.is_final(epoch=0, seq=7, model_turn_id=1)
    assert lifecycle.pending_processed(epoch=0, seq=7, model_turn_id=1) is None


def test_first_decision_and_processed_identity_are_scoped_to_the_final_turn() -> None:
    lifecycle = MiniCPMO45InputLifecycle()
    _accept(lifecycle, seq=8, turn=2, final=True)
    for outcome in ("speak", "listen"):
        lifecycle.remember_decision(
            epoch=0,
            seq=8,
            model_turn_id=2,
            outcome=outcome,
            response_id="response-8",
        )

    assert lifecycle.pending_processed(epoch=0, seq=8, model_turn_id=2) == (
        "speak",
        "response-8",
    )
    assert lifecycle.mark_processed_emitted(epoch=0, seq=8, model_turn_id=2)
    assert not lifecycle.mark_processed_emitted(epoch=0, seq=8, model_turn_id=2)


def test_pending_new_final_cannot_terminalize_the_shared_request_for_an_old_final() -> None:
    lifecycle = MiniCPMO45InputLifecycle()
    _accept(lifecycle, seq=1, turn=0, final=True)
    token = object()
    lifecycle.begin(token, epoch=0, final=True, generation=1)

    assert not lifecycle.is_latest_final(epoch=0, seq=1, model_turn_id=0)

    lifecycle.cancel(token)
    assert lifecycle.is_latest_final(epoch=0, seq=1, model_turn_id=0)


def test_retrying_a_failed_final_restores_its_pending_generation_fence() -> None:
    lifecycle = MiniCPMO45InputLifecycle()
    first_generation = lifecycle.seal_generation(epoch=0)
    first_token = object()
    lifecycle.begin(first_token, epoch=0, final=True, generation=first_generation)
    lifecycle.accept(
        first_token,
        epoch=0,
        seq=1,
        accepted_turn_id=0,
        target_turn_id=0,
    )
    lifecycle.release(epoch=0, seq=1)
    retry_generation = lifecycle.seal_generation(epoch=0)
    lifecycle.cancel_generation(epoch=0, generation=retry_generation)

    retry_token = object()
    lifecycle.begin(retry_token, epoch=0, final=True, generation=retry_generation)

    assert not lifecycle.can_claim_response(epoch=0, seq=1, model_turn_id=0)
    lifecycle.cancel(retry_token)
    assert lifecycle.can_claim_response(epoch=0, seq=1, model_turn_id=0)


def test_newer_nonfinal_input_supersedes_an_older_final_for_shared_request_ownership() -> None:
    lifecycle = MiniCPMO45InputLifecycle()
    _accept(lifecycle, seq=1, turn=0, final=True)
    _accept(lifecycle, seq=2, turn=1)

    assert not lifecycle.is_latest_final(epoch=0, seq=1, model_turn_id=0)


def test_latest_owner_in_one_generation_is_the_only_response_claim_frontier() -> None:
    lifecycle = MiniCPMO45InputLifecycle()
    _accept(lifecycle, seq=1, turn=0)
    _accept(lifecycle, seq=2, turn=1)

    assert not lifecycle.can_claim_response(epoch=0, seq=1, model_turn_id=0)
    assert lifecycle.can_claim_response(epoch=0, seq=2, model_turn_id=1)


def test_late_output_for_an_older_accepted_seq_is_not_treated_as_unaccepted() -> None:
    lifecycle = MiniCPMO45InputLifecycle()
    _accept(lifecycle, seq=1, turn=0)
    _accept(lifecycle, seq=2, turn=0, final=True)

    assert 1 not in lifecycle.accepted
    assert not lifecycle.defer_unaccepted_output(
        epoch=0,
        seq=1,
        output={"input_seq": 1, "is_listen": True},
    )
    assert lifecycle.deferred_outputs == {}


def test_owner_segments_compress_one_generation_and_clear_after_final_resolution() -> None:
    lifecycle = MiniCPMO45InputLifecycle()
    _accept(lifecycle, seq=1, turn=0)
    _accept(lifecycle, seq=2, turn=0, final=True)

    assert lifecycle.accepted_owner_segments == [(1, 2, 0, 0)]
    assert lifecycle.response_accepts_or_claims(
        response_id="response-2",
        epoch=0,
        seq=2,
        model_turn_id=0,
    )

    lifecycle.resolve_final(epoch=0, seq=2, model_turn_id=0)

    assert lifecycle.accepted_owner_segments == []
    assert lifecycle.response_owner is None


def test_resolving_one_final_generation_keeps_the_next_input_owner() -> None:
    lifecycle = MiniCPMO45InputLifecycle()
    first_generation = lifecycle.seal_generation(epoch=0)
    first_token = object()
    lifecycle.begin(first_token, epoch=0, final=True, generation=first_generation)
    lifecycle.accept(
        first_token,
        epoch=0,
        seq=1,
        accepted_turn_id=0,
        target_turn_id=0,
    )
    lifecycle.release(epoch=0, seq=1)
    second_generation = lifecycle.current_generation(epoch=0)
    second_token = object()
    lifecycle.begin(second_token, epoch=0, final=False, generation=second_generation)
    lifecycle.accept(
        second_token,
        epoch=0,
        seq=2,
        accepted_turn_id=1,
        target_turn_id=1,
    )
    lifecycle.release(epoch=0, seq=2)

    lifecycle.resolve_final(epoch=0, seq=1, model_turn_id=0)

    assert lifecycle.output_target_turn(epoch=0, seq=2, model_turn_id=1) == 1


def test_new_input_generation_reserves_the_next_model_turn_until_accepted() -> None:
    lifecycle = MiniCPMO45InputLifecycle()
    first_generation = lifecycle.seal_generation(epoch=0)
    first_token = object()
    lifecycle.begin(first_token, epoch=0, final=True, generation=first_generation)
    lifecycle.accept(
        first_token,
        epoch=0,
        seq=1,
        accepted_turn_id=0,
        target_turn_id=0,
    )
    lifecycle.release(epoch=0, seq=1)
    second_generation = lifecycle.current_generation(epoch=0)

    assert (
        lifecycle.target_turn_for_generation(
            epoch=0,
            generation=second_generation,
            minimum_turn_id=0,
        )
        == 1
    )

    second_token = object()
    lifecycle.begin(second_token, epoch=0, final=False, generation=second_generation)
    lifecycle.accept(
        second_token,
        epoch=0,
        seq=2,
        accepted_turn_id=1,
        target_turn_id=1,
    )
    assert (
        lifecycle.target_turn_for_generation(
            epoch=0,
            generation=second_generation,
            minimum_turn_id=2,
        )
        == 2
    )


def test_ambiguous_append_reserves_its_turn_against_the_next_generation() -> None:
    lifecycle = MiniCPMO45InputLifecycle()
    ambiguous_generation = lifecycle.seal_generation(epoch=0)
    assert (
        lifecycle.target_turn_for_generation(
            epoch=0,
            generation=ambiguous_generation,
            minimum_turn_id=0,
        )
        == 0
    )
    token = object()
    lifecycle.begin(token, epoch=0, final=True, generation=ambiguous_generation)
    lifecycle.mark_ambiguous(token)

    assert (
        lifecycle.target_turn_for_generation(
            epoch=0,
            generation=lifecycle.current_generation(epoch=0),
            minimum_turn_id=0,
        )
        == 1
    )


def test_ambiguous_append_keeps_early_output_until_retry_ack() -> None:
    lifecycle = MiniCPMO45InputLifecycle()
    generation = lifecycle.seal_generation(epoch=0)
    first_token = object()
    lifecycle.begin(first_token, epoch=0, final=True, generation=generation)
    output = {"input_seq": 2, "is_listen": True}
    assert lifecycle.defer_unaccepted_output(epoch=0, seq=2, output=output)
    lifecycle.mark_ambiguous(first_token)

    retry_token = object()
    lifecycle.begin(retry_token, epoch=0, final=True, generation=generation)
    lifecycle.accept(
        retry_token,
        epoch=0,
        seq=2,
        accepted_turn_id=0,
        target_turn_id=0,
    )

    assert lifecycle.release(epoch=0, seq=2) == [output]
    assert lifecycle.ambiguous == {}


def test_newer_generation_ack_clears_older_ambiguous_append() -> None:
    lifecycle = MiniCPMO45InputLifecycle()
    first_generation = lifecycle.seal_generation(epoch=0)
    first_token = object()
    lifecycle.begin(first_token, epoch=0, final=True, generation=first_generation)
    lifecycle.mark_ambiguous(first_token)

    next_generation = lifecycle.seal_generation(epoch=0)
    next_token = object()
    lifecycle.begin(next_token, epoch=0, final=True, generation=next_generation)
    lifecycle.accept(
        next_token,
        epoch=0,
        seq=2,
        accepted_turn_id=1,
        target_turn_id=1,
    )

    assert lifecycle.ambiguous == {}


def test_resolved_generation_keeps_turn_high_watermark_for_queued_generation() -> None:
    lifecycle = MiniCPMO45InputLifecycle()
    first_generation = lifecycle.seal_generation(epoch=0)
    first_token = object()
    lifecycle.begin(first_token, epoch=0, final=True, generation=first_generation)
    lifecycle.accept(
        first_token,
        epoch=0,
        seq=1,
        accepted_turn_id=0,
        target_turn_id=0,
    )
    lifecycle.release(epoch=0, seq=1)
    next_generation = lifecycle.seal_generation(epoch=0)

    lifecycle.resolve_final(epoch=0, seq=1, model_turn_id=0)

    assert (
        lifecycle.target_turn_for_generation(
            epoch=0,
            generation=next_generation,
            minimum_turn_id=0,
        )
        == 1
    )


def test_output_turn_metadata_cannot_override_the_accepted_owner_turn() -> None:
    lifecycle = MiniCPMO45InputLifecycle()
    _accept(lifecycle, seq=1, turn=1, final=True)

    assert lifecycle.output_target_turn(epoch=0, seq=1, model_turn_id=1) == 1
    assert lifecycle.output_target_turn(epoch=0, seq=1, model_turn_id=0) is None


def test_response_continuation_uses_the_latest_input_in_its_owner_segment() -> None:
    lifecycle = MiniCPMO45InputLifecycle()
    _accept(lifecycle, seq=1, turn=0)
    _accept(lifecycle, seq=2, turn=0, final=True)
    assert lifecycle.response_accepts_or_claims(
        response_id="response-2",
        epoch=0,
        seq=1,
        model_turn_id=0,
    )

    assert lifecycle.response_origin_input_seq(response_id="response-2", epoch=0) == 2
