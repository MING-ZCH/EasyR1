import json

import numpy as np
import pytest

from examples.reward_function import StepCount_mask_reward as reward


@pytest.fixture(autouse=True)
def _strict_reward_env(monkeypatch):
    names = (
        "ACTION_EVENT_REWARD_ENABLE",
        "STEPCOUNT_MASK_REQUIRE",
        "TRAJ_FORMAT_GRADED",
        "TRAJ_FORMAT_STRICT_KEY",
        "TRAJ_FORMAT_TYPO_CREDIT",
        "TRAJ_POINT_COUNT_NUMBER_CHECK",
        "TRAJ_RETURN_POINT_STEP_SCORES",
        "TRAJ_STRICT_ANSWER_INTEGER_PARSE",
        "V37_ACTION_PARSER_CONTRACT",
        "V37_RAW_SUCCESS_STRICT_WINNER",
        "V37_REWARD_FAIL_CLOSED",
        "V37_STRICT_POINT_PARSER_CONTRACT",
        "V37_WINNER_MODE",
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TRAJ_STRICT_ANSWER_INTEGER_PARSE", "1")
    monkeypatch.setenv("V37_RAW_SUCCESS_STRICT_WINNER", "1")
    monkeypatch.setenv("V37_STRICT_POINT_PARSER_CONTRACT", "1")
    monkeypatch.setenv("TRAJ_FORMAT_STRICT_KEY", "1")
    monkeypatch.setenv("TRAJ_FORMAT_TYPO_CREDIT", "0")
    monkeypatch.setenv("TRAJ_POINT_COUNT_NUMBER_CHECK", "required")


def _gt(count):
    return json.dumps(
        {
            "count_number": count,
            "point_gts": [
                {"point_2d": [10 + 10 * index, 10], "label": "object"}
                for index in range(count)
            ],
        }
    )


def _events(count):
    return [
        {"ordinal": index, "type": "point", "closed": True}
        for index in range(count)
    ] + [{"ordinal": count, "type": "answer", "closed": True}]


def _score(
    monkeypatch, prediction, count, *, action_mode=False, duplicates=None, hits=None, evidence=True
):
    monkeypatch.setattr(
        reward,
        "_strict_duplicate_evidence",
        lambda **_: (
            list(duplicates if duplicates is not None else [0.0] * count),
            list(hits if hits is not None else [1.0] * count),
            evidence,
        ),
    )
    if action_mode:
        monkeypatch.setenv("ACTION_EVENT_REWARD_ENABLE", "1")
        monkeypatch.setenv("V37_ACTION_PARSER_CONTRACT", "1")
    return reward.compute_score(
        prediction,
        _gt(count),
        action_events=_events(count) if action_mode else None,
    )


@pytest.mark.parametrize("action_mode", [False, True])
@pytest.mark.parametrize(
    "payload",
    [
        '{"point_2d":null,"count_number":"1"}',
        '{"point_2d":[],"count_number":"1"}',
        '{"point_2d":"12","count_number":"1"}',
        '{"point_2d":[NaN,10],"count_number":"1"}',
        '{"point_2d":[10,10],"count_number":"1 or 9"}',
    ],
)
def test_invalid_point_payload_is_ineligible_in_both_arms(monkeypatch, action_mode, payload):
    score = _score(monkeypatch, f"<point>{payload}</point><answer>1</answer>", 1, action_mode=action_mode)
    assert score["answer_correct"] == 1.0
    assert score["answer"] > 0.0
    assert score["raw_success"] == 0.0
    assert score["raw_success_point_payload_violation"] == 1.0


def test_valid_strict_structure_is_eligible_with_complete_duplicate_evidence(monkeypatch):
    prediction = '<point>{"point_2d":[10,10],"count_number":"1"}</point><answer>1</answer>'
    score = _score(monkeypatch, prediction, 1)
    assert score["answer_correct"] == 1.0
    assert score["raw_success"] == 1.0
    assert score["trusted_trajectory"] == 1.0


def test_missing_duplicate_evidence_fails_closed_but_keeps_answer_credit(monkeypatch):
    prediction = '<point>{"point_2d":[10,10],"count_number":"1"}</point><answer>1</answer>'
    score = _score(monkeypatch, prediction, 1, duplicates=[], evidence=False)
    assert score["answer_correct"] == 1.0
    assert score["answer"] > 0.0
    assert score["raw_success"] == 0.0
    assert score["raw_success_duplicate_evidence_missing"] == 1.0


def test_mask_miss_is_not_a_strict_winner_but_keeps_answer_credit(monkeypatch):
    prediction = '<point>{"point_2d":[10,10],"count_number":"1"}</point><answer>1</answer>'
    score = _score(monkeypatch, prediction, 1, hits=[0.0])
    assert score["answer_correct"] == 1.0
    assert score["answer"] > 0.0
    assert score["raw_success"] == 0.0
    assert score["raw_success_miss_violation"] == 1.0


def test_outcome_success_keeps_mask_miss_out_of_winner_but_not_trust(monkeypatch):
    monkeypatch.setenv("V37_WINNER_MODE", "outcome_success")
    prediction = '<point>{"point_2d":[10,10],"count_number":"1"}</point><answer>1</answer>'
    score = _score(monkeypatch, prediction, 1, hits=[0.0])
    assert score["answer_correct"] == 1.0
    assert score["raw_success"] == 1.0
    assert score["trusted_trajectory"] == 0.0
    assert score["raw_success_miss_violation"] == 1.0


def test_outcome_success_keeps_duplicate_out_of_winner_but_not_trust(monkeypatch):
    monkeypatch.setenv("V37_WINNER_MODE", "outcome_success")
    prediction = (
        '<point>{"point_2d":[10,10],"count_number":"1"}</point>'
        '<point>{"point_2d":[10.00001,10],"count_number":"2"}</point>'
        '<answer>2</answer>'
    )
    score = _score(monkeypatch, prediction, 2, duplicates=[0.0, 1.0], hits=[1.0, 0.0])
    assert score["answer_correct"] == 1.0
    assert score["raw_success"] == 1.0
    assert score["trusted_trajectory"] == 0.0
    assert score["raw_success_duplicate_violation"] == 1.0


@pytest.mark.parametrize(
    "prediction,violation",
    [
        (
            '<point>{"point_22d":[10,10],"count_number":"1"}</point><answer>1</answer>',
            "raw_success_point_payload_violation",
        ),
        (
            '<point>{"point_2d":[10,10],"count_number":"2"}</point><answer>1</answer>',
            "point_count_number_violation",
        ),
    ],
)
def test_outcome_success_still_rejects_point_and_count_number_format(
    monkeypatch, prediction, violation,
):
    monkeypatch.setenv("V37_WINNER_MODE", "outcome_success")
    score = _score(monkeypatch, prediction, 1)
    assert score["answer_correct"] == 1.0
    assert score["raw_success"] == 0.0
    assert score[violation] == 1.0


def test_legacy_all_hit_mode_remains_default_and_explicit(monkeypatch):
    prediction = '<point>{"point_2d":[10,10],"count_number":"1"}</point><answer>1</answer>'
    default_score = _score(monkeypatch, prediction, 1, hits=[0.0])
    monkeypatch.setenv("V37_WINNER_MODE", "legacy_all_hit")
    explicit_score = _score(monkeypatch, prediction, 1, hits=[0.0])
    assert default_score["raw_success"] == 0.0
    assert explicit_score["raw_success"] == 0.0
    assert default_score["raw_success"] == explicit_score["raw_success"]


def test_object_duplicate_with_coordinate_jitter_is_not_a_strict_winner(monkeypatch):
    prediction = (
        '<point>{"point_2d":[10,10],"count_number":"1"}</point>'
        '<point>{"point_2d":[10.00001,10],"count_number":"2"}</point>'
        '<answer>2</answer>'
    )
    score = _score(monkeypatch, prediction, 2, duplicates=[0.0, 1.0], evidence=True)
    assert score["answer_correct"] == 1.0
    assert score["raw_success"] == 0.0
    assert score["raw_success_duplicate_violation"] == 1.0


@pytest.mark.parametrize(
    "prediction",
    [
        '</point><point>{"point_2d":[10,10],"count_number":"1"}</point><answer>1</answer>',
        '<point><point>{"point_2d":[10,10],"count_number":"1"}</point></point><answer>1</answer>',
        '<answer>1</answer><point>{"point_2d":[10,10],"count_number":"1"}</point>',
    ],
)
def test_control_tags_must_form_ordered_actions(monkeypatch, prediction):
    score = _score(monkeypatch, prediction, 1)
    assert score["raw_success"] == 0.0
    assert score["raw_success_tag_order_violation"] == 1.0


def test_discrete_structure_not_near_one_format_reward_controls_winner(monkeypatch):
    monkeypatch.setattr(reward, "_trajectory_format_reward", lambda *_: 0.9999999999)
    malformed = '<point>{"point_2d":null,"count_number":"1"}</point><answer>1</answer>'
    score = _score(monkeypatch, malformed, 1)
    assert score["format"] == pytest.approx(0.9999999999)
    assert score["raw_success"] == 0.0


def test_object_identity_duplicate_evidence_uses_matched_sample_id(monkeypatch):
    class FakeHelper:
        def get_sequence_from_image_path(self, image_path):
            return "sequence", [{"sample_id": "object-1"}, {"sample_id": "object-2"}]

        def get_sequence_masks(self, sequence_id):
            mask = np.ones((1, 2, 2), dtype=bool)
            return [("object-1", 0, mask), ("object-2", 1, mask)]

        def check_point_in_sequence_masks(self, *, used_sample_ids, **kwargs):
            duplicate = "object-1" in used_sample_ids
            return {
                "in_any_mask": True,
                "in_unused_mask": not duplicate,
                "matched_sample_id": "object-1",
                "is_duplicate": duplicate,
            }

    monkeypatch.setattr(reward, "_get_mask_helper", lambda: FakeHelper())
    prediction = (
        '<point>{"point_2d":[10,10],"count_number":"1"}</point>'
        '<point>{"point_2d":[10.00001,10],"count_number":"2"}</point>'
        '<answer>2</answer>'
    )
    flags, hits, complete = reward._strict_duplicate_evidence(
        prediction, json.loads(_gt(2)) | {"point_sequence": [{}, {}]}, "image.png"
    )
    assert complete is True
    assert flags == [0.0, 1.0]
    assert hits == [1.0, 0.0]


def test_mask_route_attestation_is_independent_of_model_point_count_and_payload(monkeypatch):
    class FakeHelper:
        def get_sequence_from_image_path(self, image_path):
            return "sequence", [{"sample_id": "object-1"}, {"sample_id": "object-2"}]

        def get_sequence_masks(self, sequence_id):
            mask = np.ones((1, 2, 2), dtype=bool)
            return [("object-1", 0, mask), ("object-2", 1, mask)]

        def check_point_in_sequence_masks(self, **kwargs):
            return {
                "in_any_mask": True,
                "in_unused_mask": True,
                "matched_sample_id": "object-1",
                "is_duplicate": False,
            }

    monkeypatch.setattr(reward, "_get_mask_helper", lambda: FakeHelper())
    gt = json.loads(_gt(2)) | {"point_sequence": [{}, {}]}

    short = '<point>{"point_2d":[10,10],"count_number":"1"}</point><answer>1</answer>'
    flags, hits, complete = reward._strict_duplicate_evidence(short, gt, "image.png")
    assert (flags, hits, complete) == ([0.0], [1.0], True)

    malformed = '<point>{"point_2d":null,"count_number":"1"}</point><answer>1</answer>'
    flags, hits, complete = reward._strict_duplicate_evidence(malformed, gt, "image.png")
    assert (flags, hits, complete) == ([0.0], [0.0], True)


def test_v37_required_mask_failure_propagates(monkeypatch, tmp_path):
    monkeypatch.setenv("STEPCOUNT_MASK_REQUIRE", "1")
    monkeypatch.setenv("V37_REWARD_FAIL_CLOSED", "1")
    monkeypatch.setenv("STEPCOUNT_MASKS_METADATA", str(tmp_path / "missing.json"))
    monkeypatch.setenv("STEPCOUNT_MASKS_DIR", str(tmp_path / "missing-masks"))
    monkeypatch.setattr(reward, "_MASK_HELPER", None)
    prediction = '<point>{"point_2d":[10,10],"count_number":"1"}</point><answer>1</answer>'
    with pytest.raises(RuntimeError, match="mask resources are missing"):
        reward.compute_score(prediction, _gt(1), image_path="missing.png")


def test_v36_default_reward_failure_remains_zero_score(monkeypatch, tmp_path):
    monkeypatch.setenv("STEPCOUNT_MASK_REQUIRE", "1")
    monkeypatch.delenv("V37_REWARD_FAIL_CLOSED", raising=False)
    monkeypatch.setenv("STEPCOUNT_MASKS_METADATA", str(tmp_path / "missing.json"))
    monkeypatch.setenv("STEPCOUNT_MASKS_DIR", str(tmp_path / "missing-masks"))
    monkeypatch.setattr(reward, "_MASK_HELPER", None)
    prediction = '<point>{"point_2d":[10,10],"count_number":"1"}</point><answer>1</answer>'
    score = reward.compute_score(prediction, _gt(1), image_path="missing.png")
    assert score["overall"] == 0.0
    assert score["raw_success"] == 0.0


def test_v36_feature_off_keeps_malformed_slot_filtering(monkeypatch):
    monkeypatch.delenv("V37_RAW_SUCCESS_STRICT_WINNER", raising=False)
    monkeypatch.delenv("V37_STRICT_POINT_PARSER_CONTRACT", raising=False)
    monkeypatch.delenv("TRAJ_STRICT_ANSWER_INTEGER_PARSE", raising=False)
    prediction = (
        '<point>{"point_2d":null,"count_number":"999"}</point>'
        '<point>{"point_2d":[10,10],"count_number":"1"}</point>'
        '<answer>1</answer>'
    )
    score = reward.compute_score(prediction, _gt(1))
    assert score["answer_correct"] == 1.0
    assert len(reward._parse_pred_point_slots(prediction)) == 1
    # Legacy count-number integrity may still reject this trajectory; the
    # compatibility contract here is that the malformed slot remains filtered.
    assert "raw_success_point_payload_violation" not in score
    assert "trusted_trajectory" not in score
