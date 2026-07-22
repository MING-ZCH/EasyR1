import json

import pytest
import torch

from examples.reward_function import StepCount_mask_reward as stepcount_reward
from verl.trainer.core_algos import compute_bok_grpo_advantage, compute_bok_grpo_step_advantage


@pytest.fixture(autouse=True)
def _clean_bok_environment(monkeypatch):
    for name in (
        "BOK_ADV_NORMALIZE",
        "BOK_ALLWRONG_ANSWER_THRESHOLD",
        "BOK_ALLWRONG_CAP",
        "BOK_ALLWRONG_NEG_ONLY",
        "BOK_DAPO_AUTO_DISABLE_THRESHOLD",
        "BOK_DAPO_FILTER",
        "BOK_EASY_SCALE",
        "BOK_EASY_SCORE_THRESHOLD",
        "BOK_EASY_THRESHOLD",
        "BOK_FALLBACK_MODE",
        "BOK_FILTER_ALL_CORRECT",
        "BOK_LOW_VAR_THRESHOLD",
        "BOK_QUALITY_BONUS",
        "BOK_CORRECTNESS_FIRST",
        "BOK_CORRECTNESS_TASK_WEIGHT",
        "BOK_CORRECTNESS_QUALITY_WEIGHT",
        "BOK_CORRECTNESS_PARTIAL_SCALE",
        "BOK_CORRECTNESS_SIGN_MARGIN",
        "BOK_ALLWRONG_TERMINAL_ZERO",
        "BOK_SMART_FILTER_THRESHOLD",
        "BOK_STEP_WEIGHT",
        "BOK_TAU_ADAPTIVE",
        "BOK_WINNER_BOOST",
        "VCRL_ENABLE",
        "ACTION_EVENT_REWARD_ENABLE",
        "TRAJ_STRICT_ANSWER_INTEGER_PARSE",
        "TRAJ_FORMAT_GRADED",
        "TRAJ_FORMAT_STRICT_KEY",
        "TRAJ_FORMAT_TYPO_CREDIT",
        "TRAJ_POINT_COUNT_NUMBER_CHECK",
        "TRAJ_RETURN_POINT_STEP_SCORES",
        "STEPCOUNT_MASK_REQUIRE",
        "V37_ACTION_PARSER_CONTRACT",
        "V37_RAW_SUCCESS_STRICT_WINNER",
        "V37_STRICT_POINT_PARSER_CONTRACT",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BOK_DAPO_FILTER", "0")
    monkeypatch.setenv("VCRL_ENABLE", "0")


def _advantages(
    scores,
    indices,
    *,
    answer_scores=None,
    answer_correct_scores=None,
    raw_success_scores=None,
    trajectory_quality_scores=None,
    **kwargs,
):
    rewards = torch.tensor(scores, dtype=torch.float32).unsqueeze(-1)
    mask = torch.ones_like(rewards)
    advantages, _ = compute_bok_grpo_advantage(
        rewards,
        mask,
        indices,
        answer_scores=None if answer_scores is None else torch.tensor(answer_scores, dtype=torch.float32),
        answer_correct_scores=(
            None
            if answer_correct_scores is None
            else torch.tensor(answer_correct_scores, dtype=torch.float32)
        ),
        raw_success_scores=None if raw_success_scores is None else torch.tensor(raw_success_scores, dtype=torch.float32),
        trajectory_quality_scores=(
            None
            if trajectory_quality_scores is None
            else torch.tensor(trajectory_quality_scores, dtype=torch.float32)
        ),
        **kwargs,
    )
    return advantages.squeeze(-1)


def test_trajectory_answer_correct_is_binary_and_independent_of_shaping(monkeypatch):
    monkeypatch.setenv("TRAJ_ANSWER_GATE_MODE", "hard")
    monkeypatch.setenv("TRAJ_ANSWER_GATE_THRESHOLD", "0.4")
    monkeypatch.setattr(stepcount_reward, "_trajectory_point_dense_reward", lambda **_: 0.0)

    ground_truth = json.dumps(
        {
            "count_number": 2,
            "point_gts": [
                {"point_2d": [100, 100]},
                {"point_2d": [200, 200]},
            ],
        }
    )
    prediction = '<point>{"point_2d":[900,900]}</point><answer>2</answer>'

    train_score = stepcount_reward.compute_score(prediction, ground_truth, is_eval=False)
    eval_score = stepcount_reward.compute_score(prediction, ground_truth, is_eval=True)
    wrong_score = stepcount_reward.compute_score(
        prediction.replace("<answer>2</answer>", "<answer>3</answer>"),
        ground_truth,
        is_eval=False,
    )
    missing_score = stepcount_reward.compute_score("<answer>not-a-number</answer>", ground_truth)

    assert train_score["answer"] < 0.5
    assert train_score["answer_correct"] == 1.0
    assert eval_score["answer_correct"] == 1.0
    assert wrong_score["answer_correct"] == 0.0
    assert missing_score["answer_correct"] == 0.0
    assert "answer_correct" not in stepcount_reward.compute_score(
        "anything", "not-a-valid-ground-truth"
    )
    point_only = stepcount_reward.compute_score(
        '<point>{"point_2d":[100,100]}</point>',
        json.dumps({"point_2d": [100, 100], "label": "object"}),
    )
    assert "answer_correct" not in point_only


@pytest.mark.parametrize("answer", ["5 or 6", "5.0", "5 objects", "about 5"])
def test_v37_strict_answer_parse_rejects_ambiguous_text(monkeypatch, answer):
    monkeypatch.setenv("TRAJ_STRICT_ANSWER_INTEGER_PARSE", "1")
    score = stepcount_reward.compute_score(
        f"<answer>{answer}</answer>",
        json.dumps({"count_number": 5}),
    )
    assert score["answer_correct"] == 0.0
    assert score["answer_exact"] == 0.0
    assert score["raw_success"] == 0.0


@pytest.mark.parametrize("answer", ["5", "+5", "005"])
def test_v37_strict_answer_parse_accepts_complete_integer(monkeypatch, answer):
    monkeypatch.setenv("TRAJ_STRICT_ANSWER_INTEGER_PARSE", "1")
    score = stepcount_reward.compute_score(
        f"<answer>{answer}</answer>",
        json.dumps({"count_number": 5}),
    )
    assert score["answer_correct"] == 1.0


def test_v36_default_answer_parse_remains_permissive(monkeypatch):
    monkeypatch.delenv("TRAJ_STRICT_ANSWER_INTEGER_PARSE", raising=False)
    score = stepcount_reward.compute_score(
        "<answer>5 objects</answer>",
        json.dumps({"count_number": 5}),
    )
    assert score["answer_correct"] == 1.0


@pytest.mark.parametrize("case", ["duplicate", "format"])
def test_v37_strict_winner_rejects_structural_point_defects_but_keeps_answer_credit(
    monkeypatch, case,
):
    monkeypatch.setenv("V37_RAW_SUCCESS_STRICT_WINNER", "1")
    monkeypatch.setenv("TRAJ_RETURN_POINT_STEP_SCORES", "1")
    monkeypatch.setenv("TRAJ_FORMAT_STRICT_KEY", "1")
    monkeypatch.setenv("TRAJ_FORMAT_TYPO_CREDIT", "0")
    first_key = "point_22d" if case == "format" else "point_2d"
    second_point = [10, 10] if case == "duplicate" else [20, 20]
    prediction = (
        f'<point>{{"{first_key}":[10,10],"count_number":"1"}}</point>'
        f'<point>{{"point_2d":{second_point},"count_number":"2"}}</point>'
        '<answer>2</answer>'
    )
    ground_truth = json.dumps({
        "count_number": 2,
        "point_gts": [{"point_2d": [10, 10]}, {"point_2d": [20, 20]}],
    })
    score = stepcount_reward.compute_score(prediction, ground_truth)
    assert score["answer_correct"] == 1.0
    assert score["answer"] > 0.0
    assert score["raw_success"] == 0.0
    assert score[f"raw_success_{case}_violation"] == 1.0


def test_action_reward_requires_slot_preserving_parser_contract(monkeypatch):
    monkeypatch.setenv("ACTION_EVENT_REWARD_ENABLE", "1")
    monkeypatch.delenv("V37_ACTION_PARSER_CONTRACT", raising=False)
    with pytest.raises(RuntimeError, match="V37_ACTION_PARSER_CONTRACT=1"):
        stepcount_reward._parse_pred_point_slots(
            '<point>{not-json}</point><point>{"point_2d":[10,10]}</point>'
        )


def test_shaped_answer_below_half_is_not_routed_as_all_wrong(monkeypatch):
    monkeypatch.setenv("BOK_ALLWRONG_CAP", "0.2")
    advantages = _advantages(
        [0.9, 0.1],
        [0, 0],
        answer_scores=[0.2, 0.1],
        answer_correct_scores=[1.0, 0.0],
    )

    assert advantages[0].item() > 0.0

    # The step estimator must forward the same binary routing channel even
    # when its auxiliary process signal is disabled.
    monkeypatch.setenv("BOK_STEP_WEIGHT", "0")
    rewards = torch.tensor([[0.9], [0.1]], dtype=torch.float32)
    step_advantages, _ = compute_bok_grpo_step_advantage(
        rewards,
        torch.ones_like(rewards),
        [0, 0],
        answer_scores=torch.tensor([0.2, 0.1]),
        answer_correct_scores=torch.tensor([1.0, 0.0]),
    )
    assert step_advantages[0, 0].item() > 0.0


def test_v36_low_variance_fallback_is_unchanged_when_v37_is_off(monkeypatch):
    monkeypatch.setenv("BOK_ALLWRONG_CAP", "0")
    monkeypatch.setenv("BOK_FALLBACK_MODE", "zscore")
    advantages = _advantages(
        [0.8, 0.8, 0.1, 0.3],
        [0, 0, 1, 1],
        answer_scores=[0.8, 0.8, 0.1, 0.3],
        answer_correct_scores=[0.0, 0.0, 0.0, 1.0],
    )

    batch = torch.tensor([0.8, 0.8, 0.1, 0.3])
    expected = (torch.tensor(0.8) - batch.mean()) / (batch.std() + 1e-6)
    assert advantages[0].item() == pytest.approx(expected.item())
    assert advantages[1].item() == pytest.approx(expected.item())
    assert advantages[0].item() > 0.0


def test_low_variance_all_correct_group_obeys_filter_even_with_smart_release(monkeypatch):
    monkeypatch.setenv("BOK_CORRECTNESS_FIRST", "1")
    monkeypatch.setenv("BOK_FILTER_ALL_CORRECT", "1")
    monkeypatch.setenv("BOK_SMART_FILTER_THRESHOLD", "0.955")
    advantages = _advantages(
        [0.8, 0.8, 0.1, 0.3],
        [0, 0, 1, 1],
        answer_scores=[0.2, 0.2, 0.1, 0.3],
        answer_correct_scores=[1.0, 1.0, 0.0, 1.0],
        raw_success_scores=[1.0, 1.0, 0.0, 1.0],
    )

    assert torch.equal(advantages[:2], torch.zeros(2))


def test_non_low_variance_legacy_routing_is_unchanged(monkeypatch):
    monkeypatch.setenv("BOK_ALLWRONG_CAP", "0.25")
    monkeypatch.setenv("BOK_WINNER_BOOST", "1.5")
    monkeypatch.setenv("BOK_QUALITY_BONUS", "0.2")
    shaped_answers = [1.0, 0.0, 0.0]

    legacy = _advantages(
        [0.9, 0.4, 0.1],
        [0, 0, 0],
        answer_scores=shaped_answers,
    )
    explicit = _advantages(
        [0.9, 0.4, 0.1],
        [0, 0, 0],
        answer_scores=shaped_answers,
        answer_correct_scores=shaped_answers,
    )

    assert torch.allclose(explicit, legacy)


def test_binary_correctness_breaks_shaped_reward_tie(monkeypatch):
    monkeypatch.setenv("BOK_CORRECTNESS_FIRST", "1")
    monkeypatch.setenv("BOK_WINNER_BOOST", "1.5")
    advantages = _advantages(
        [0.4, 0.4],
        [0, 0],
        answer_scores=[0.1, 0.1],
        answer_correct_scores=[1.0, 0.0],
        raw_success_scores=[1.0, 0.0],
    )

    assert advantages[0].item() > 0.0
    assert advantages[1].item() < 0.0


def test_missing_binary_channel_falls_back_per_group(monkeypatch):
    shaped_answers = [1.0, 0.0]
    legacy = _advantages(
        [0.9, 0.1],
        [0, 0],
        answer_scores=shaped_answers,
    )
    partial = _advantages(
        [0.9, 0.1],
        [0, 0],
        answer_scores=shaped_answers,
        answer_correct_scores=[float("nan"), 1.0],
    )

    assert torch.allclose(partial, legacy)


def test_default_off_ignores_correctness_channels():
    legacy = _advantages([0.9, 0.2], [0, 0], answer_scores=[0.1, 0.9])
    supplied = _advantages(
        [0.9, 0.2],
        [0, 0],
        answer_scores=[0.1, 0.9],
        answer_correct_scores=[1.0, 0.0],
        raw_success_scores=[0.0, 1.0],
        trajectory_quality_scores=[100.0, -100.0],
    )
    assert torch.allclose(supplied, legacy)


def test_dapo_safety_does_not_overwrite_independent_filters(monkeypatch):
    monkeypatch.setenv("BOK_CORRECTNESS_FIRST", "1")
    monkeypatch.setenv("BOK_DAPO_FILTER", "1")
    monkeypatch.setenv("BOK_DAPO_AUTO_DISABLE_THRESHOLD", "0.1")
    monkeypatch.setenv("BOK_FILTER_ALL_CORRECT", "1")
    advantages = _advantages(
        [0.8, 0.8, 0.7, 0.7, 0.6, 0.6],
        [0, 0, 1, 1, 2, 2],
        answer_scores=[1.0, 1.0, 0.0, 0.0, 0.2, 0.2],
        answer_correct_scores=[1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        raw_success_scores=[1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
    )

    assert torch.equal(advantages[:2], torch.zeros(2))
    assert torch.all(advantages[2:4] <= 0.0)
    assert advantages[4].item() > 0.0
    assert advantages[5].item() < 0.0


def test_binary_channel_shape_is_checked():
    with pytest.raises(ValueError, match="answer_correct_scores"):
        _advantages(
            [0.9, 0.1],
            [0, 0],
            answer_correct_scores=[1.0],
        )


@pytest.mark.parametrize("n_correct", [1, 15])
def test_correctness_first_signs_survive_adversarial_shaping(monkeypatch, n_correct):
    monkeypatch.setenv("BOK_CORRECTNESS_FIRST", "1")
    monkeypatch.setenv("BOK_WINNER_BOOST", "3")
    raw = [1.0] * n_correct + [0.0] * (16 - n_correct)
    shaped = [100.0 if value == 0.0 else -100.0 for value in raw]
    quality = list(reversed(range(16)))

    advantages = _advantages(
        shaped,
        [0] * 16,
        answer_correct_scores=raw,
        raw_success_scores=raw,
        trajectory_quality_scores=quality,
    )

    assert torch.all(advantages[torch.tensor(raw).bool()] > 0)
    assert torch.all(advantages[~torch.tensor(raw).bool()] < 0)


@pytest.mark.parametrize("raw_value", [0.0, 1.0])
def test_correctness_first_homogeneous_terminal_advantage_is_exact_zero(monkeypatch, raw_value):
    monkeypatch.setenv("BOK_CORRECTNESS_FIRST", "1")
    advantages = _advantages(
        list(range(16)),
        [0] * 16,
        answer_correct_scores=[raw_value] * 16,
        raw_success_scores=[raw_value] * 16,
        trajectory_quality_scores=list(range(16)),
    )
    assert torch.equal(advantages, torch.zeros(16))


def test_exact_answer_partial_credit_is_learned_without_a_strict_winner(monkeypatch):
    monkeypatch.setenv("BOK_CORRECTNESS_FIRST", "1")
    monkeypatch.setenv("BOK_ALLWRONG_TERMINAL_ZERO", "1")
    advantages = _advantages(
        [0.25, 0.0, 0.0],
        [0, 0, 0],
        answer_correct_scores=[1.0, 0.0, 0.0],
        raw_success_scores=[0.0, 0.0, 0.0],
        trajectory_quality_scores=[0.1, 0.1, 0.1],
    )
    assert advantages[0].item() > 0.0
    assert advantages[1].item() < 0.0
    assert advantages[2].item() < 0.0


def test_all_answer_wrong_group_remains_terminal_zero(monkeypatch):
    monkeypatch.setenv("BOK_CORRECTNESS_FIRST", "1")
    monkeypatch.setenv("BOK_ALLWRONG_TERMINAL_ZERO", "1")
    advantages = _advantages(
        [0.9, 0.1],
        [0, 0],
        answer_correct_scores=[0.0, 0.0],
        raw_success_scores=[0.0, 0.0],
        trajectory_quality_scores=[1.0, 0.0],
    )
    assert torch.equal(advantages, torch.zeros(2))


def test_exact_invalid_loser_is_ranked_above_wrong_loser_without_crossing_zero(monkeypatch):
    monkeypatch.setenv("BOK_CORRECTNESS_FIRST", "1")
    advantages = _advantages(
        [1.0, 0.25, 0.0],
        [0, 0, 0],
        answer_correct_scores=[1.0, 1.0, 0.0],
        raw_success_scores=[1.0, 0.0, 0.0],
        trajectory_quality_scores=[1.0, 0.2, 0.2],
    )
    assert advantages[0].item() > 0.0
    assert advantages[1].item() < 0.0
    assert advantages[2].item() < advantages[1].item()


def test_allwrong_terminal_zero_is_independent_opt_in(monkeypatch):
    monkeypatch.setenv("BOK_ALLWRONG_TERMINAL_ZERO", "1")
    advantages = _advantages(
        [0.1, 0.9],
        [0, 0],
        answer_correct_scores=[0.0, 0.0],
    )
    assert torch.equal(advantages, torch.zeros(2))


def test_exact_but_capped_is_not_raw_success(monkeypatch):
    monkeypatch.setenv("ACTION_EVENT_REWARD_ENABLE", "1")
    monkeypatch.setenv("V37_ACTION_PARSER_CONTRACT", "1")
    monkeypatch.setattr(
        stepcount_reward,
        "_trajectory_point_dense_reward_without_gt_points_details",
        lambda **_: (1.0, [0.5, 0.5], [1.0, 1.0], [0.0, 0.0], 2.0, 2.0),
    )
    prediction = (
        '<point>{"point_2d":[10,10],"count_number":"1"}</point>'
        '<point>{"point_2d":[20,20],"count_number":"2"}</point>'
        '<answer>2</answer>'
    )
    events = [
        {"ordinal": 0, "type": "point", "closed": True},
        {"ordinal": 1, "type": "point", "closed": True},
        {"ordinal": 2, "type": "answer", "closed": True},
        {"ordinal": 3, "type": "cap", "closed": False},
    ]
    score = stepcount_reward.compute_score(
        prediction,
        json.dumps({"count_number": 2}),
        action_events=events,
        max_turns=8,
    )
    assert score["answer_exact"] == 1.0
    assert score["raw_success"] == 0.0
    assert score["answer"] > 0.0


def test_raw_success_requires_one_normal_closed_terminal_answer(monkeypatch):
    monkeypatch.setattr(
        stepcount_reward,
        "_trajectory_point_dense_reward_without_gt_points_details",
        lambda **_: (1.0, [0.5, 0.5], [1.0, 1.0], [0.0, 0.0], 2.0, 2.0),
    )
    points = (
        '<point>{"point_2d":[10,10],"count_number":"1"}</point>'
        '<point>{"point_2d":[20,20],"count_number":"2"}</point>'
    )
    ground_truth = json.dumps({"count_number": 2, "point_gts": [{}, {}]})
    normal = stepcount_reward.compute_score(points + "<answer>2</answer>", ground_truth)
    duplicate_terminal = stepcount_reward.compute_score(
        points + "<answer>2</answer><answer>2</answer>", ground_truth
    )
    trailing_junk = stepcount_reward.compute_score(points + "<answer>2</answer>junk", ground_truth)
    unclosed = stepcount_reward.compute_score(points + "<answer>2", ground_truth)

    assert normal["answer_exact"] == 1.0
    assert normal["raw_success"] == 1.0
    assert duplicate_terminal["answer_exact"] == 1.0
    assert duplicate_terminal["raw_success"] == 0.0
    assert trailing_junk["answer_exact"] == 1.0
    assert trailing_junk["raw_success"] == 0.0
    assert unclosed["raw_success"] == 0.0
