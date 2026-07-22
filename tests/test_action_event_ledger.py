import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from examples.reward_function import StepCount_mask_reward as stepcount_reward
from verl.trainer.core_algos import compute_bok_grpo_step_advantage
from verl.protocol import DataProto
from verl.utils.action_ledger import (
    ACTION_TAG_TOKEN_IDS,
    action_event_ledger_digest,
    build_native_action_event_row,
    canonical_interleaved_stop_sequences,
    classify_empty_generation,
    ledger_to_dense,
    make_action_event,
    validate_action_tag_token_ids,
    validate_action_event_row,
)
from verl.workers.reward.function import SequentialFunctionRewardManager


@pytest.fixture(autouse=True)
def _action_env(monkeypatch):
    monkeypatch.setenv("ACTION_EVENT_LEDGER_ENABLE", "1")
    monkeypatch.setenv("ACTION_EVENT_REWARD_ENABLE", "1")
    monkeypatch.setenv("V37_ACTION_PARSER_CONTRACT", "1")
    monkeypatch.setenv("V37_ACTION_LEDGER_CONTRACT", "1")
    monkeypatch.setenv("BOK_CORRECTNESS_FIRST", "1")
    monkeypatch.setenv("BOK_STEP_WEIGHT", "1")
    monkeypatch.setenv("BOK_FILTER_ALL_CORRECT", "1")
    monkeypatch.setenv("VCRL_ENABLE", "0")
    monkeypatch.delenv("V37_REWARD_FAIL_CLOSED", raising=False)


def _event(ordinal, event_type, start, end, *, closed=True, truncated=False, turn_index=None):
    return make_action_event(
        ordinal=ordinal,
        event_type=event_type,
        span_start=start,
        span_end=end,
        decision_start=start,
        decision_end=end,
        closed=closed,
        truncated=truncated,
        turn_index=ordinal if turn_index is None else turn_index,
        stop_reason="turn_cap" if event_type == "cap" else ("abort" if event_type == "abort" else event_type),
    )


def test_ledger_supports_53_points_and_cap_without_offset_guessing():
    events = [_event(i, "point", i, i + 1) for i in range(53)]
    events.append(_event(53, "cap", 53, 53, closed=False, truncated=True))

    validated = validate_action_event_row(events, response_length=53)

    assert len(validated) == 54
    assert validated[52]["type"] == "point"
    assert validated[-1]["type"] == "cap"
    assert validated[-1]["truncated"] is True


def test_cap_cannot_overlap_preceding_action_and_is_uncredited():
    events = [_event(0, "point", 0, 2), _event(1, "cap", 1, 2, closed=False, truncated=True)]
    with pytest.raises(ValueError, match="response boundary"):
        validate_action_event_row(events, response_length=2)

    marker = [_event(0, "point", 0, 2), _event(1, "cap", 2, 2, closed=False, truncated=True)]
    with pytest.raises(ValueError, match="cannot carry local action credit"):
        ledger_to_dense(marker, [1.0, -1.0], response_length=2)
    mask, dense, _ = ledger_to_dense(marker, [1.0, 0.0], response_length=2)
    assert mask == [1.0, 1.0]
    assert dense == [1.0, 1.0]


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("ordinal", 0.0, "integer"),
        ("span_start", 0.5, "integer"),
        ("closed", "true", "boolean"),
        ("turn_index", -1, "nonnegative"),
    ],
)
def test_ledger_rejects_type_coercion(field, value, match):
    event = _event(0, "point", 0, 1)
    event[field] = value
    with pytest.raises(ValueError, match=match):
        validate_action_event_row([event], response_length=1)


def test_native_token_builder_handles_split_tags_and_abort_semantics():
    # Both opening and closing tags cross generation-turn boundaries.
    tokens = [10, 11, 20, 12, 13, 30, 31, 40, 32, 33]
    events = build_native_action_event_row(
        tokens,
        point_open_ids=[10, 11],
        point_close_ids=[12, 13],
        answer_open_ids=[30, 31],
        answer_close_ids=[32, 33],
        turn_spans=[(0, 1, 0), (1, 4, 1), (4, 7, 2), (7, 10, 3)],
        termination_type=None,
        termination_reason=None,
    )
    assert [(event["type"], event["span_start"], event["span_end"]) for event in events] == [
        ("point", 0, 5),
        ("answer", 5, 10),
    ]

    empty_abort = build_native_action_event_row(
        [],
        point_open_ids=[10],
        point_close_ids=[11],
        answer_open_ids=[12],
        answer_close_ids=[13],
        turn_spans=[],
        termination_type="abort",
        termination_reason="empty_completion",
    )
    assert empty_abort[0]["type"] == "abort"
    assert empty_abort[0]["span_start"] == empty_abort[0]["span_end"] == 0


@pytest.mark.parametrize(
    "completion_present,stop_reason,finish_reason,expected",
    [
        (False, None, None, "generation_error"),
        (True, None, "eos", "immediate_eos"),
        (True, "stop", None, "immediate_eos"),
        (True, None, None, "empty_completion"),
    ],
)
def test_zero_token_generation_classification(
    completion_present, stop_reason, finish_reason, expected
):
    assert classify_empty_generation(
        completion_present=completion_present,
        stop_reason=stop_reason,
        finish_reason=finish_reason,
    ) == expected


def test_nonempty_plain_eos_row_has_one_canonical_abort_boundary():
    tokens = [99]
    envelope = build_native_action_event_row(
        tokens,
        point_open_ids=ACTION_TAG_TOKEN_IDS["point_open"],
        point_close_ids=ACTION_TAG_TOKEN_IDS["point_close"],
        answer_open_ids=ACTION_TAG_TOKEN_IDS["answer_open"],
        answer_close_ids=ACTION_TAG_TOKEN_IDS["answer_close"],
        turn_spans=[(0, 1, 0)],
        termination_type="abort",
        termination_reason="eos",
        return_envelope=True,
    )
    events = validate_action_event_row(
        envelope, 1, response_token_ids=tokens, require_token_binding=True
    )
    assert [(event["type"], event["stop_reason"]) for event in events] == [("abort", "eos")]
    assert events[0]["span_start"] == events[0]["span_end"] == 1


def test_ledger_invalid_offset_fails_instead_of_clamping():
    with pytest.raises(ValueError, match="invalid event span"):
        validate_action_event_row([_event(0, "point", 3, 7)], response_length=6)


def test_single_bad_ledger_row_fails_closed_with_metric():
    manager = SequentialFunctionRewardManager.__new__(SequentialFunctionRewardManager)
    manager.reward_fn = lambda *_args, **_kwargs: {"overall": 1.0}
    manager.config = SimpleNamespace(skip_special_tokens=False)
    manager.tokenizer = SimpleNamespace(decode=lambda *_args, **_kwargs: "response")
    manager._reward_debug_calls = 0
    manager._reward_debug_every = 100
    manager._reward_sample_debug = False
    manager._reward_sample_debug_max = 1
    manager._reward_health_debug = False
    manager._reward_health_every = 100
    ledger = np.empty(1, dtype=object)
    ledger[0] = [_event(0, "point", 0, 3)]
    data = DataProto.from_dict(
        tensors={"responses": torch.tensor([[1, 2]]), "response_mask": torch.ones((1, 2))},
        non_tensors={"ground_truth": np.asarray(["1"], dtype=object), "action_event_ledger": ledger},
    )

    reward_tensor, metrics = manager.compute_reward(data)

    assert torch.equal(reward_tensor, torch.zeros_like(reward_tensor))
    assert metrics["action_ledger_invalid"] == [1.0]


def test_single_bad_ledger_row_aborts_v37_batch(monkeypatch):
    monkeypatch.setenv("V37_REWARD_FAIL_CLOSED", "1")
    manager = SequentialFunctionRewardManager.__new__(SequentialFunctionRewardManager)
    manager.reward_fn = lambda *_args, **_kwargs: {"overall": 1.0}
    manager.config = SimpleNamespace(skip_special_tokens=False)
    manager.tokenizer = SimpleNamespace(decode=lambda *_args, **_kwargs: "response")
    manager._reward_debug_calls = 0
    manager._reward_debug_every = 100
    manager._reward_sample_debug = False
    manager._reward_sample_debug_max = 1
    manager._reward_health_debug = False
    manager._reward_health_every = 100
    ledger = np.empty(1, dtype=object)
    ledger[0] = [_event(0, "point", 0, 3)]
    data = DataProto.from_dict(
        tensors={"responses": torch.tensor([[1, 2]]), "response_mask": torch.ones((1, 2))},
        non_tensors={"ground_truth": np.asarray(["1"], dtype=object), "action_event_ledger": ledger},
    )

    with pytest.raises(RuntimeError, match="Invalid V37 action ledger"):
        manager.compute_reward(data)


def test_baseline_validates_shared_ledger_without_emitting_action_credit(monkeypatch):
    monkeypatch.setenv("ACTION_EVENT_REWARD_ENABLE", "0")
    manager = SequentialFunctionRewardManager.__new__(SequentialFunctionRewardManager)
    manager.reward_fn = lambda *_args, **kwargs: {
        "overall": 1.0, "answer": 1.0, "answer_correct": 1.0,
        "raw_success": 1.0, "trusted_trajectory": 1.0,
        "trajectory_quality": 1.0,
        "seen_action_events": float(len(kwargs["action_events"])),
    }
    manager.config = SimpleNamespace(skip_special_tokens=False)
    manager.tokenizer = SimpleNamespace(decode=lambda *_args, **_kwargs: "response")
    manager._reward_debug_calls = 0
    manager._reward_debug_every = 100
    manager._reward_sample_debug = False
    manager._reward_sample_debug_max = 1
    manager._reward_health_debug = False
    manager._reward_health_every = 100
    ledger = np.empty(1, dtype=object)
    tokens = [
        *ACTION_TAG_TOKEN_IDS["point_open"], *ACTION_TAG_TOKEN_IDS["point_close"],
        *ACTION_TAG_TOKEN_IDS["answer_open"], *ACTION_TAG_TOKEN_IDS["answer_close"],
    ]
    ledger[0] = build_native_action_event_row(
        tokens,
        point_open_ids=ACTION_TAG_TOKEN_IDS["point_open"],
        point_close_ids=ACTION_TAG_TOKEN_IDS["point_close"],
        answer_open_ids=ACTION_TAG_TOKEN_IDS["answer_open"],
        answer_close_ids=ACTION_TAG_TOKEN_IDS["answer_close"],
        turn_spans=[(0, 2, 0), (2, 4, 1)],
        termination_type=None, termination_reason=None, return_envelope=True,
    )
    data = DataProto.from_dict(
        tensors={"responses": torch.tensor([tokens]), "response_mask": torch.ones((1, 4), dtype=torch.long)},
        non_tensors={"ground_truth": np.asarray(["1"], dtype=object), "action_event_ledger": ledger},
    )

    _reward_tensor, metrics = manager.compute_reward(data)

    assert metrics["action_ledger_invalid"] == [0.0]
    assert metrics["seen_action_events"] == [2.0]
    assert "_action_events" not in metrics
    assert "_action_event_values" not in metrics


def test_native_action_credit_does_not_leak_point_signal_into_answer_tail():
    rewards = torch.zeros((2, 6))
    response_mask = torch.ones_like(rewards)
    action_spans = torch.tensor([[[0, 2], [4, 6]], [[0, 2], [4, 6]]])
    action_values = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    action_types = torch.tensor([[1, 2], [1, 2]])

    advantages, _ = compute_bok_grpo_step_advantage(
        rewards,
        response_mask,
        [0, 0],
        answer_correct_scores=torch.zeros(2),
        raw_success_scores=torch.zeros(2),
        trajectory_quality_scores=torch.zeros(2),
        action_step_span=action_spans,
        action_step_value=action_values,
        action_step_type=action_types,
    )

    assert torch.all(advantages[0, 0:2] > 0)
    assert torch.all(advantages[1, 0:2] < 0)
    assert torch.equal(advantages[:, 2:], torch.zeros((2, 4)))


def test_native_action_credit_is_stable_under_n16_shuffle():
    order = list(range(16))
    random.Random(7).shuffle(order)
    values = torch.tensor([1.0] * 8 + [-1.0] * 8)[order].unsqueeze(-1)
    types = torch.ones((16, 1), dtype=torch.long)
    spans = torch.tensor([[[0, 3]]] * 16)

    advantages, _ = compute_bok_grpo_step_advantage(
        torch.zeros((16, 3)),
        torch.ones((16, 3)),
        ["same_prompt"] * 16,
        answer_correct_scores=torch.zeros(16),
        raw_success_scores=torch.zeros(16),
        trajectory_quality_scores=torch.zeros(16),
        action_step_span=spans,
        action_step_value=values,
        action_step_type=types,
    )

    assert torch.all(advantages[values.squeeze(-1) > 0] > 0)
    assert torch.all(advantages[values.squeeze(-1) < 0] < 0)


def test_capped_prefix_keeps_point_values_and_separate_cap(monkeypatch):
    monkeypatch.setattr(
        stepcount_reward,
        "_trajectory_point_dense_reward_without_gt_points_details",
        lambda **_: (0.5, [0.5, 0.0], [1.0, 1.0], [0.0, 1.0], 3.0, 2.0),
    )
    prediction = (
        '<point>{"point_2d":[10,10],"count_number":"1"}</point>'
        '<point>{"point_2d":[10,10],"count_number":"2"}</point>'
    )
    events = [_event(0, "point", 0, 1), _event(1, "point", 1, 2), _event(2, "cap", 1, 2)]
    score = stepcount_reward.compute_score(
        prediction,
        json.dumps({"count_number": 3, "point_gts": [{}, {}, {}]}),
        action_events=events,
    )

    assert score["_action_event_values"] == [1.0, -1.0, 0.0]
    assert score["raw_success"] == 0.0


def test_extra_point_is_explicit_overshoot_not_inferred_answer(monkeypatch):
    monkeypatch.setattr(
        stepcount_reward,
        "_trajectory_point_dense_reward_without_gt_points_details",
        lambda **_: (1.0, [1.0], [1.0], [0.0], 1.0, 1.0),
    )
    prediction = (
        '<point>{"point_2d":[10,10],"count_number":"1"}</point>'
        '<point>{"point_2d":[20,20],"count_number":"2"}</point>'
        '<answer>1</answer>'
    )
    events = [_event(0, "point", 0, 1), _event(1, "point", 1, 2), _event(2, "answer", 2, 3)]
    score = stepcount_reward.compute_score(
        prediction,
        json.dumps({"count_number": 1, "point_gts": [{}]}),
        action_events=events,
        max_turns=5,
    )

    assert score["answer_exact"] == 1.0
    assert score["raw_success"] == 0.0
    assert score["_action_event_values"][0] == 1.0
    assert score["_action_event_values"][1] == -1.0
    assert score["_action_event_values"][2] == 0.0


def test_native_point_count_number_mismatch_keeps_negative_penalty(monkeypatch):
    monkeypatch.setattr(
        stepcount_reward,
        "_trajectory_point_dense_reward_without_gt_points_details",
        lambda **_: (1.0, [1.0], [1.0], [0.0], 1.0, 1.0),
    )
    prediction = (
        '<point>{"point_2d":[10,10],"count_number":"9"}</point>'
        '<answer>1</answer>'
    )
    events = [_event(0, "point", 0, 1), _event(1, "answer", 1, 2)]
    score = stepcount_reward.compute_score(
        prediction,
        json.dumps({"count_number": 1, "point_gts": [{}]}),
        action_events=events,
        max_turns=4,
    )
    assert score["point_count_number_mismatch"] == 1.0
    assert score["_action_event_values"] == [-1.0, 0.0]


def test_malformed_point_keeps_ordinal_before_valid_point(monkeypatch):
    monkeypatch.setattr(stepcount_reward, "mask_point_reward", lambda *args, **kwargs: None)
    monkeypatch.setattr(stepcount_reward, "point_reward", lambda *args, **kwargs: 1.0)
    prediction = (
        '<point>{not-json}</point>'
        '<point>{"point_2d":[20,20]}</point>'
        '<answer>2</answer>'
    )
    scores = stepcount_reward._trajectory_point_step_scores(
        prediction,
        {"point_sequence": [{"point_2d": [10, 10]}, {"point_2d": [20, 20]}]},
        100,
        100,
        None,
        None,
    )
    assert scores == [0.0, 1.0]


def test_reward_scores_all_53_native_point_events(monkeypatch):
    monkeypatch.setattr(
        stepcount_reward,
        "_trajectory_point_dense_reward_without_gt_points_details",
        lambda **_: (1.0, [1.0], [1.0], [0.0], 1.0, 1.0),
    )
    prediction = "".join(
        f'<point>{{"point_2d":[{i},{i}],"count_number":"{i + 1}"}}</point>'
        for i in range(53)
    ) + "<answer>1</answer>"
    events = [{"ordinal": i, "type": "point", "closed": True} for i in range(53)]
    events.append({"ordinal": 53, "type": "answer", "closed": True})

    score = stepcount_reward.compute_score(
        prediction,
        json.dumps({"count_number": 1, "point_gts": [{}]}),
        action_events=events,
        max_turns=100,
    )

    assert len(score["_action_event_values"]) == 54
    assert score["_action_event_values"][0] == 1.0
    assert score["_action_event_values"][1:53] == [-1.0] * 52


def test_answer_soft_min_gate_changes_native_action_ppo_signal(monkeypatch):
    monkeypatch.setenv("BOK_STEP_GATE", "answer_soft")
    monkeypatch.setenv("BOK_STEP_MIN_GATE", "0.2")
    advantages, _ = compute_bok_grpo_step_advantage(
        torch.zeros((2, 3)),
        torch.ones((2, 3)),
        ["prompt", "prompt"],
        answer_scores=torch.tensor([0.0, 1.0]),
        answer_correct_scores=torch.zeros(2),
        raw_success_scores=torch.zeros(2),
        trajectory_quality_scores=torch.zeros(2),
        action_step_span=torch.tensor([[[0, 2]], [[0, 2]]]),
        action_step_value=torch.tensor([[1.0], [-1.0]]),
        action_step_type=torch.ones((2, 1), dtype=torch.long),
    )
    assert advantages[0, 0] > 0 and advantages[1, 0] < 0
    assert advantages[0, 0].abs().item() == pytest.approx(0.2 * advantages[1, 0].abs().item())
    assert torch.equal(advantages[:, 2], torch.zeros(2))


def test_homogeneous_native_action_group_has_zero_local_advantage():
    advantages, _ = compute_bok_grpo_step_advantage(
        torch.zeros((4, 2)),
        torch.ones((4, 2)),
        [0, 0, 0, 0],
        answer_scores=torch.ones(4),
        answer_correct_scores=torch.zeros(4),
        raw_success_scores=torch.zeros(4),
        trajectory_quality_scores=torch.zeros(4),
        action_step_span=torch.tensor([[[0, 1]]] * 4),
        action_step_value=torch.ones((4, 1)),
        action_step_type=torch.ones((4, 1), dtype=torch.long),
    )
    assert torch.equal(advantages, torch.zeros_like(advantages))


def test_compact_action_storage_contract_at_representative_shape():
    batch, events, response = 2048, 54, 16384
    spans = torch.empty((batch, events, 2), dtype=torch.int64, device="meta")
    values = torch.empty((batch, events), dtype=torch.float32, device="meta")
    types = torch.empty((batch, events), dtype=torch.int64, device="meta")
    local = torch.empty((batch, response), dtype=torch.float32, device="meta")
    compact_bytes = (
        spans.numel() * spans.element_size()
        + values.numel() * values.element_size()
        + types.numel() * types.element_size()
        + local.numel() * local.element_size()
    )
    forbidden_dense_bytes = batch * events * response * 4
    assert spans.shape == (batch, events, 2)
    assert compact_bytes < 140 * 1024**2
    assert forbidden_dense_bytes == int(6.75 * 1024**3)


def test_v36_default_malformed_first_reward_is_exactly_legacy(monkeypatch):
    monkeypatch.delenv("ACTION_EVENT_LEDGER_ENABLE", raising=False)
    monkeypatch.delenv("ACTION_EVENT_REWARD_ENABLE", raising=False)
    monkeypatch.delenv("V37_ACTION_PARSER_CONTRACT", raising=False)
    monkeypatch.delenv("V37_ACTION_LEDGER_CONTRACT", raising=False)
    monkeypatch.setenv("TRAJ_ANSWER_GATE_MODE", "off")
    prediction = (
        '<point>{not-json}</point>'
        '<point>{"point_2d":[20,20],"count_number":"1"}</point>'
        '<answer>1</answer>'
    )
    score = stepcount_reward.compute_score(
        prediction,
        json.dumps({"count_number": 1, "point_gts": [{}]}),
        max_turns=2,
        answer_weight=0.47,
        point_weight=0.0,
        trajectory_format_weight=0.0,
    )
    assert score["overall"] == pytest.approx(0.47)
    assert score["turns_exceeded"] == 0.0
    assert len(stepcount_reward._parse_pred_point_slots(prediction)) == 1


def test_token_bound_ledger_rejects_forged_or_stale_response_tokens():
    tokens = [151669, 42, 151670, 151667, 1, 151668]
    envelope = build_native_action_event_row(
        tokens,
        point_open_ids=ACTION_TAG_TOKEN_IDS["point_open"],
        point_close_ids=ACTION_TAG_TOKEN_IDS["point_close"],
        answer_open_ids=ACTION_TAG_TOKEN_IDS["answer_open"],
        answer_close_ids=ACTION_TAG_TOKEN_IDS["answer_close"],
        turn_spans=[(0, 3, 0), (3, 6, 1)],
        termination_type=None,
        termination_reason=None,
        return_envelope=True,
    )
    assert len(validate_action_event_row(
        envelope, 6, response_token_ids=tokens, require_token_binding=True
    )) == 2
    forged = list(tokens)
    forged[0] = 999
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_action_event_row(envelope, 6, response_token_ids=forged, require_token_binding=True)

    reordered = dict(envelope)
    reordered["events"] = list(reversed(envelope["events"]))
    with pytest.raises(ValueError, match="event-ledger digest"):
        validate_action_event_row(reordered, 6, response_token_ids=tokens, require_token_binding=True)


def _canonical_point_answer_envelope():
    tokens = [151669, 42, 151670, 151667, 1, 151668]
    envelope = build_native_action_event_row(
        tokens,
        point_open_ids=ACTION_TAG_TOKEN_IDS["point_open"],
        point_close_ids=ACTION_TAG_TOKEN_IDS["point_close"],
        answer_open_ids=ACTION_TAG_TOKEN_IDS["answer_open"],
        answer_close_ids=ACTION_TAG_TOKEN_IDS["answer_close"],
        turn_spans=[(0, 3, 0), (3, 6, 1)],
        termination_type=None,
        termination_reason=None,
        return_envelope=True,
    )
    return tokens, envelope


@pytest.mark.parametrize(
    "field,value",
    [
        ("ordinal", 9),
        ("type", "answer"),
        ("span_start", 1),
        ("span_end", 2),
        ("decision_start", 1),
        ("decision_end", 2),
        ("closed", False),
        ("truncated", True),
        ("turn_index", 7),
        ("stop_reason", "answer"),
    ],
)
def test_event_digest_binds_every_event_field(field, value):
    tokens, envelope = _canonical_point_answer_envelope()
    forged = dict(envelope)
    forged["events"] = [dict(event) for event in envelope["events"]]
    forged["events"][0][field] = value
    with pytest.raises(ValueError, match="event-ledger digest"):
        validate_action_event_row(forged, len(tokens), response_token_ids=tokens, require_token_binding=True)


def test_schema_v1_and_missing_event_digest_are_stale():
    tokens, envelope = _canonical_point_answer_envelope()
    stale = dict(envelope)
    stale["version"] = 1
    with pytest.raises(ValueError, match="unsupported action ledger version"):
        validate_action_event_row(stale, len(tokens), response_token_ids=tokens, require_token_binding=True)
    missing = dict(envelope)
    missing.pop("event_ledger_digest")
    with pytest.raises(ValueError, match="missing or unknown fields"):
        validate_action_event_row(missing, len(tokens), response_token_ids=tokens, require_token_binding=True)
    extra = dict(envelope)
    extra["events"] = [dict(event) for event in envelope["events"]]
    extra["events"][0]["forged"] = True
    extra["event_ledger_digest"] = action_event_ledger_digest(extra["events"])
    with pytest.raises(ValueError, match="event has missing or unknown fields"):
        validate_action_event_row(extra, len(tokens), response_token_ids=tokens, require_token_binding=True)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("decision_start", 1, "decision span"),
        ("decision_end", 2, "decision span"),
        ("turn_index", 4, "turn_index"),
        ("stop_reason", "answer", "stop_reason"),
    ],
)
def test_rehashed_forged_action_metadata_still_fails_canonical_reconstruction(field, value, match):
    tokens, envelope = _canonical_point_answer_envelope()
    forged = dict(envelope)
    forged["events"] = [dict(event) for event in envelope["events"]]
    forged["events"][0][field] = value
    forged["event_ledger_digest"] = action_event_ledger_digest(forged["events"])
    with pytest.raises(ValueError, match=match):
        validate_action_event_row(forged, len(tokens), response_token_ids=tokens, require_token_binding=True)


def test_rehashed_event_count_duplicate_and_reordering_fail_token_reconstruction():
    tokens, envelope = _canonical_point_answer_envelope()

    missing = dict(envelope)
    missing["events"] = [dict(envelope["events"][1])]
    missing["events"][0]["ordinal"] = 0
    missing["events"][0]["turn_index"] = 0
    missing["event_ledger_digest"] = action_event_ledger_digest(missing["events"])
    with pytest.raises(ValueError, match="canonical native-token reconstruction"):
        validate_action_event_row(missing, len(tokens), response_token_ids=tokens, require_token_binding=True)

    duplicate = dict(envelope)
    duplicate["events"] = [dict(envelope["events"][0]), dict(envelope["events"][0]), dict(envelope["events"][1])]
    for ordinal, event in enumerate(duplicate["events"]):
        event["ordinal"] = ordinal
        event["turn_index"] = ordinal
    duplicate["event_ledger_digest"] = action_event_ledger_digest(duplicate["events"])
    with pytest.raises(ValueError, match="out-of-order|canonical native-token reconstruction"):
        validate_action_event_row(duplicate, len(tokens), response_token_ids=tokens, require_token_binding=True)

    reordered = dict(envelope)
    reordered["events"] = [dict(envelope["events"][1]), dict(envelope["events"][0])]
    for ordinal, event in enumerate(reordered["events"]):
        event["ordinal"] = ordinal
        event["turn_index"] = ordinal
    reordered["event_ledger_digest"] = action_event_ledger_digest(reordered["events"])
    with pytest.raises(ValueError, match="legal action order|out-of-order|canonical native-token reconstruction"):
        validate_action_event_row(reordered, len(tokens), response_token_ids=tokens, require_token_binding=True)


@pytest.mark.parametrize(
    "tokens",
    [
        [151669, 7],
        [7, 151670],
        [151667, 1],
        [151669, 151667, 1, 151668, 151670],
        [151667, 1, 151668, 151667, 2, 151668],
        [151667, 1, 151668, 151669, 2, 151670],
        [151669, 1, 151670, 151667, 2, 151668, 99],
    ],
)
def test_unmatched_multiple_answer_event_after_answer_and_post_answer_tokens_abort(tokens):
    envelope = build_native_action_event_row(
        tokens,
        point_open_ids=ACTION_TAG_TOKEN_IDS["point_open"],
        point_close_ids=ACTION_TAG_TOKEN_IDS["point_close"],
        answer_open_ids=ACTION_TAG_TOKEN_IDS["answer_open"],
        answer_close_ids=ACTION_TAG_TOKEN_IDS["answer_close"],
        turn_spans=[(0, len(tokens), 0)] if tokens else [],
        termination_type=None,
        termination_reason=None,
        return_envelope=True,
    )
    assert len(envelope["events"]) == 1
    assert envelope["events"][0]["type"] == "abort"
    assert envelope["events"][0]["stop_reason"] == "invalid_native_span"


def test_v37_rollout_uses_same_strict_scheduler_for_both_arms_and_preserves_v36_branch():
    assert canonical_interleaved_stop_sequences() == ("</point>", "</answer>")
    source = Path("verl/workers/rollout/vllm_rollout_spmd.py").read_text(encoding="utf-8")
    assert "normal_point_stop = canonical_action_stops" in source
    assert "if action_event_mode and not strict_action_scheduler" in source
    assert "exact_string=strict_action_scheduler" in source
    assert "if strict_action_scheduler:" in source
    assert "strict_action_scheduler\n                        and len(response_token_ids" in source
    assert "A fully closed answer is terminal" in source
    assert 'else ["</point>", "<answer>"]' in source

    early_answer = build_native_action_event_row(
        [151667, 3, 151668],
        point_open_ids=ACTION_TAG_TOKEN_IDS["point_open"],
        point_close_ids=ACTION_TAG_TOKEN_IDS["point_close"],
        answer_open_ids=ACTION_TAG_TOKEN_IDS["answer_open"],
        answer_close_ids=ACTION_TAG_TOKEN_IDS["answer_close"],
        turn_spans=[(0, 3, 0)],
        termination_type=None,
        termination_reason=None,
    )
    assert [event["type"] for event in early_answer] == ["answer"]

    packed_actions = build_native_action_event_row(
        [151669, 7, 151670, 151667, 1, 151668],
        point_open_ids=ACTION_TAG_TOKEN_IDS["point_open"],
        point_close_ids=ACTION_TAG_TOKEN_IDS["point_close"],
        answer_open_ids=ACTION_TAG_TOKEN_IDS["answer_open"],
        answer_close_ids=ACTION_TAG_TOKEN_IDS["answer_close"],
        turn_spans=[(0, 6, 0)],
        termination_type=None,
        termination_reason=None,
    )
    assert [(event["type"], event["stop_reason"]) for event in packed_actions] == [
        ("abort", "stop_semantics_violation")
    ]


def test_reward_ingress_fails_closed_on_forged_response_token():
    manager = SequentialFunctionRewardManager.__new__(SequentialFunctionRewardManager)
    manager.reward_fn = lambda *_args, **_kwargs: {"overall": 1.0, "_action_event_values": [1.0]}
    manager.config = SimpleNamespace(skip_special_tokens=False)
    manager.tokenizer = SimpleNamespace(decode=lambda *_args, **_kwargs: "<point>x</point>")
    manager._reward_debug_calls = 0
    manager._reward_debug_every = 100
    manager._reward_sample_debug = False
    manager._reward_sample_debug_max = 1
    manager._reward_health_debug = False
    manager._reward_health_every = 100
    original = [151669, 42, 151670]
    envelope = build_native_action_event_row(
        original,
        point_open_ids=ACTION_TAG_TOKEN_IDS["point_open"],
        point_close_ids=ACTION_TAG_TOKEN_IDS["point_close"],
        answer_open_ids=ACTION_TAG_TOKEN_IDS["answer_open"],
        answer_close_ids=ACTION_TAG_TOKEN_IDS["answer_close"],
        turn_spans=[(0, 3, 0)],
        termination_type=None,
        termination_reason=None,
        return_envelope=True,
    )
    ledgers = np.empty(1, dtype=object)
    ledgers[0] = envelope
    data = DataProto.from_dict(
        tensors={
            "responses": torch.tensor([[999, 42, 151670]]),
            "response_mask": torch.ones((1, 3)),
        },
        non_tensors={
            "ground_truth": np.asarray(["1"], dtype=object),
            "action_event_ledger": ledgers,
        },
    )
    rewards, metrics = manager.compute_reward(data)
    assert torch.equal(rewards, torch.zeros_like(rewards))
    assert metrics["action_ledger_invalid"] == [1.0]
    assert metrics["_answer_correct_for_routing"] == [0.0]


def test_terminal_marker_metadata_and_value_are_canonical_and_uncredited():
    tokens = [151669, 42, 151670]
    envelope = build_native_action_event_row(
        tokens,
        point_open_ids=ACTION_TAG_TOKEN_IDS["point_open"],
        point_close_ids=ACTION_TAG_TOKEN_IDS["point_close"],
        answer_open_ids=ACTION_TAG_TOKEN_IDS["answer_open"],
        answer_close_ids=ACTION_TAG_TOKEN_IDS["answer_close"],
        turn_spans=[(0, 3, 0)],
        termination_type="cap",
        termination_reason="turn_cap",
        return_envelope=True,
    )
    assert [event["type"] for event in validate_action_event_row(
        envelope, len(tokens), response_token_ids=tokens, require_token_binding=True
    )] == ["point", "cap"]

    for field, value in (
        ("span_start", 2),
        ("span_end", 2),
        ("decision_start", 2),
        ("decision_end", 2),
        ("turn_index", 8),
        ("stop_reason", "abort"),
        ("closed", True),
        ("truncated", False),
    ):
        forged = dict(envelope)
        forged["events"] = [dict(event) for event in envelope["events"]]
        forged["events"][-1][field] = value
        forged["event_ledger_digest"] = action_event_ledger_digest(forged["events"])
        with pytest.raises(ValueError):
            validate_action_event_row(
                forged, len(tokens), response_token_ids=tokens, require_token_binding=True
            )

    manager = SequentialFunctionRewardManager.__new__(SequentialFunctionRewardManager)
    manager.reward_fn = lambda *_args, **_kwargs: {
        "overall": 1.0,
        "answer_correct": 1.0,
        "_action_event_values": [1.0, 0.5],
    }
    manager.config = SimpleNamespace(skip_special_tokens=False)
    manager.tokenizer = SimpleNamespace(decode=lambda *_args, **_kwargs: "<point>x</point>")
    manager._reward_debug_calls = 0
    manager._reward_debug_every = 100
    manager._reward_sample_debug = False
    manager._reward_sample_debug_max = 1
    manager._reward_health_debug = False
    manager._reward_health_every = 100
    ledgers = np.empty(1, dtype=object)
    ledgers[0] = envelope
    data = DataProto.from_dict(
        tensors={
            "responses": torch.tensor([tokens]),
            "response_mask": torch.ones((1, len(tokens))),
        },
        non_tensors={
            "ground_truth": np.asarray(["1"], dtype=object),
            "action_event_ledger": ledgers,
        },
    )
    rewards, metrics = manager.compute_reward(data)
    assert torch.equal(rewards, torch.zeros_like(rewards))
    assert metrics["action_ledger_invalid"] == [1.0]
    assert metrics["_answer_correct_for_routing"] == [0.0]

    manager.reward_fn = lambda *_args, **_kwargs: {
        "overall": 1.0,
        "answer_correct": 1.0,
        "_action_event_values": [1.0],
    }
    rewards, metrics = manager.compute_reward(data)
    assert torch.equal(rewards, torch.zeros_like(rewards))
    assert metrics["action_ledger_invalid"] == [1.0]


def test_exact_singleton_tag_contract_and_split_encoding_fail_closed():
    fixture = {name: list(ids) for name, ids in ACTION_TAG_TOKEN_IDS.items()}
    assert validate_action_tag_token_ids(fixture, require_expected=True) == ACTION_TAG_TOKEN_IDS
    split = dict(fixture)
    split["point_open"] = [151669, 7]
    with pytest.raises(ValueError, match="singleton"):
        validate_action_tag_token_ids(split, require_expected=True)


def test_real_checkpoint_declares_expected_added_token_ids():
    tokenizer_config = Path(
        "/mnt/shared-storage-user/zhangchenhao/work/StepcountModel/model/"
        "StepCount-7B-SFT-1M-resume-from-1963/checkpoint-476/tokenizer_config.json"
    )
    if not tokenizer_config.is_file():
        pytest.skip("local checkpoint tokenizer config is unavailable")
    decoder = json.loads(tokenizer_config.read_text(encoding="utf-8"))["added_tokens_decoder"]
    assert decoder["151667"]["content"] == "<answer>"
    assert decoder["151668"]["content"] == "</answer>"
    assert decoder["151669"]["content"] == "<point>"
    assert decoder["151670"]["content"] == "</point>"


def test_real_checkpoint_tokenizer_singletons_when_transformers_available():
    transformers = pytest.importorskip("transformers")
    model_path = Path(
        "/mnt/shared-storage-user/zhangchenhao/work/StepcountModel/model/"
        "StepCount-7B-SFT-1M-resume-from-1963/checkpoint-476"
    )
    if not (model_path / "tokenizer.json").is_file():
        pytest.skip("local checkpoint tokenizer is unavailable")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True
    )
    observed = {
        "answer_open": tokenizer.encode("<answer>", add_special_tokens=False),
        "answer_close": tokenizer.encode("</answer>", add_special_tokens=False),
        "point_open": tokenizer.encode("<point>", add_special_tokens=False),
        "point_close": tokenizer.encode("</point>", add_special_tokens=False),
    }
    assert validate_action_tag_token_ids(observed, require_expected=True) == ACTION_TAG_TOKEN_IDS


@pytest.mark.parametrize(
    "reason,metric",
    [
        ("immediate_eos", "immediate_eos"),
        ("empty_completion", "empty_response"),
        ("generation_error", "generation_error"),
    ],
)
def test_empty_generation_rows_are_explicit_and_do_not_abort_reward_batch(reason, metric):
    manager = SequentialFunctionRewardManager.__new__(SequentialFunctionRewardManager)
    manager.reward_fn = stepcount_reward.compute_score
    manager.config = SimpleNamespace(skip_special_tokens=False)
    manager.tokenizer = SimpleNamespace(decode=lambda *_args, **_kwargs: "")
    manager._reward_debug_calls = 0
    manager._reward_debug_every = 100
    manager._reward_sample_debug = False
    manager._reward_sample_debug_max = 1
    manager._reward_health_debug = False
    manager._reward_health_every = 100
    ledger = np.empty(1, dtype=object)
    ledger[0] = build_native_action_event_row(
        [],
        point_open_ids=ACTION_TAG_TOKEN_IDS["point_open"],
        point_close_ids=ACTION_TAG_TOKEN_IDS["point_close"],
        answer_open_ids=ACTION_TAG_TOKEN_IDS["answer_open"],
        answer_close_ids=ACTION_TAG_TOKEN_IDS["answer_close"],
        turn_spans=[],
        termination_type="abort",
        termination_reason=reason,
        return_envelope=True,
    )
    data = DataProto.from_dict(
        tensors={"responses": torch.zeros((1, 4), dtype=torch.long), "response_mask": torch.zeros((1, 4))},
        non_tensors={
            "ground_truth": np.asarray([json.dumps({"count_number": 1, "point_gts": [{}]})], dtype=object),
            "action_event_ledger": ledger,
        },
    )
    reward_tensor, metrics = manager.compute_reward(data)
    assert torch.equal(reward_tensor, torch.zeros_like(reward_tensor))
    assert metrics[metric] == [1.0]
    assert metrics["invalid_generation"] == [1.0]
    assert metrics["action_abort_count"] == [1.0]
