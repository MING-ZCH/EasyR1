import math
from types import SimpleNamespace

import pytest
import torch

from verl.trainer.core_algos import AdaptiveKLController, compute_stable_low_var_kl
from verl.trainer.ray_trainer import strict_optimizer_counter_metrics
from verl.utils import torch_functional as VF
from verl.utils.checkpoint.checkpoint_manager import reference_content_identity
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.workers.actor.dp_actor import (
    DataParallelPPOActor,
    coherent_actual_loss_metrics,
    masked_mean_to_token_sum,
    resolve_logprob_function,
    scale_full_shard_token_sum,
    validate_adaptive_actor_kl_topology,
)


def test_stable_low_var_kl_is_finite_nonnegative_and_differentiable():
    new_log_probs = torch.tensor([[-1000.0, -2.0, -0.5]], requires_grad=True)
    ref_log_probs = torch.tensor([[1000.0, -1.0, -0.5]])
    kld = compute_stable_low_var_kl(new_log_probs, ref_log_probs)
    loss = kld.mean()
    loss.backward()

    assert torch.isfinite(kld).all()
    assert torch.all(kld >= 0)
    assert torch.isfinite(new_log_probs.grad).all()


def test_stable_low_var_kl_extreme_tails_keep_restoring_gradient_sign():
    new_log_probs = torch.tensor([-1000.0, 1000.0], requires_grad=True)
    ref_log_probs = torch.zeros(2)
    compute_stable_low_var_kl(new_log_probs, ref_log_probs).sum().backward()

    assert torch.isfinite(new_log_probs.grad).all()
    assert new_log_probs.grad[0].item() < 0.0
    assert new_log_probs.grad[1].item() > 0.0
    assert torch.all(new_log_probs.grad != 0)


def test_stable_low_var_kl_is_finite_at_opposite_fp32_limits():
    limit = torch.finfo(torch.float32).max
    new_log_probs = torch.tensor([-limit, limit], dtype=torch.float32, requires_grad=True)
    ref_log_probs = torch.tensor([limit, -limit], dtype=torch.float32)
    kld = compute_stable_low_var_kl(new_log_probs, ref_log_probs)
    kld.sum().backward()

    assert torch.isfinite(kld).all()
    assert torch.isfinite(new_log_probs.grad).all()
    assert new_log_probs.grad[0] < 0
    assert new_log_probs.grad[1] > 0
    assert torch.all(new_log_probs.grad != 0)


def test_global_token_mean_kl_gradient_is_microbatch_invariant():
    ref = torch.tensor([[-2.0, -1.5, -1.0], [-0.7, -0.4, -0.2]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    full_new = torch.tensor([[-1.8, -1.6, -3.0], [-0.8, -0.5, -0.1]], requires_grad=True)
    full_loss = (compute_stable_low_var_kl(full_new, ref) * mask).sum() / mask.sum()
    full_loss.backward()
    full_grad = full_new.grad.clone()

    split_new = full_new.detach().clone().requires_grad_(True)
    total_count = mask.sum()
    for row in range(2):
        micro_loss = (
            compute_stable_low_var_kl(split_new[row : row + 1], ref[row : row + 1])
            * mask[row : row + 1]
        ).sum() / total_count
        micro_loss.backward()

    assert torch.allclose(split_new.grad, full_grad, atol=1e-7, rtol=1e-6)


def test_adaptive_kl_controller_state_roundtrip_and_validation():
    controller = AdaptiveKLController(0.08, 0.15, 10000)
    controller.update(0.3, 2)
    state = controller.state_dict()
    restored = AdaptiveKLController(0.01, 0.2, 5)
    restored.load_state_dict(state)

    assert restored.state_dict() == state
    with pytest.raises(ValueError, match="finite"):
        restored.update(float("nan"), 1)
    with pytest.raises(ValueError, match="positive"):
        AdaptiveKLController(0.1, 0.0, 10)


def test_legacy_reward_side_controller_accepts_signed_kl_but_actor_is_strict():
    legacy = AdaptiveKLController(0.1, 0.2, 100, strict_nonnegative_kl=False)
    legacy.update(-0.05, 1)
    assert legacy.kl_coef > 0

    actor = AdaptiveKLController(0.1, 0.2, 100, strict_nonnegative_kl=True)
    with pytest.raises(ValueError, match="nonnegative"):
        actor.update(-0.05, 1)


def test_reference_identity_hash_changes_with_local_content(tmp_path):
    model_dir = tmp_path / "reference"
    model_dir.mkdir()
    weights = model_dir / "weights.bin"
    weights.write_bytes(b"version-one")
    first = reference_content_identity(str(model_dir))
    weights.write_bytes(b"version-two")
    second = reference_content_identity(str(model_dir))

    assert first["kind"] == "directory-content"
    assert first["sha256"] != second["sha256"]


def test_actual_loss_metrics_are_token_weighted_across_unequal_microbatches():
    # Microbatch means 1 and 5 over 1 and 3 tokens respectively must produce
    # 4, not the misleading arithmetic mean 3.
    actual = coherent_actual_loss_metrics(
        policy_sum=1.0 * 1 + 5.0 * 3,
        entropy_sum=2.0 * 1 + 6.0 * 3,
        entropy_bonus_sum=0.2 * 1 + 0.6 * 3,
        loss_token_count=4,
        kl_sum=0.5 * 2 + 1.5 * 2,
        kl_token_count=4,
        kl_coef=0.25,
    )

    assert actual["policy_loss"] == pytest.approx(4.0)
    assert actual["entropy_loss"] == pytest.approx(5.0)
    assert actual["kl_loss"] == pytest.approx(1.0)
    assert actual["total_loss"] == pytest.approx(4.25 - 0.5)


def test_full_shard_rank_average_preserves_global_unequal_token_mean_gradient():
    rank_0_sum = torch.tensor(2.0, requires_grad=True)  # one valid token
    rank_1_sum = torch.tensor(9.0, requires_grad=True)  # three valid tokens
    global_count = torch.tensor(4.0)
    scaled_0 = scale_full_shard_token_sum(rank_0_sum, global_count, gradient_world_size=2)
    scaled_1 = scale_full_shard_token_sum(rank_1_sum, global_count, gradient_world_size=2)
    fsdp_averaged_loss = (scaled_0 + scaled_1) / 2
    fsdp_averaged_loss.backward()

    assert fsdp_averaged_loss.item() == pytest.approx(11.0 / 4.0)
    assert rank_0_sum.grad.item() == pytest.approx(1.0 / 4.0)
    assert rank_1_sum.grad.item() == pytest.approx(1.0 / 4.0)


def test_global_objective_gradient_matches_unequal_microbatches_and_ranks():
    # Two ranks each execute two microbatches, but their valid-token counts are
    # [1, 2] and [3, 1].  The helper must cancel both the caller's /2 gradient
    # accumulation and FSDP's rank-average.
    rank_coefficients = (
        (torch.tensor([1.0]), torch.tensor([2.0, 3.0])),
        (torch.tensor([4.0, 5.0, 6.0]), torch.tensor([7.0])),
    )
    global_count = torch.tensor(7.0)
    rank_parameters = [torch.tensor(0.25, requires_grad=True) for _ in range(2)]

    for parameter, microbatches in zip(rank_parameters, rank_coefficients):
        for coefficients in microbatches:
            token_losses = (parameter * coefficients).square()
            mask = torch.ones_like(token_losses)
            local_mean = VF.masked_mean(token_losses, mask)
            local_sum = masked_mean_to_token_sum(local_mean, mask.sum())
            scaled = scale_full_shard_token_sum(
                local_sum,
                global_count,
                gradient_world_size=2,
                gradient_accumulation=2,
            )
            (scaled / 2).backward()

    fsdp_averaged_gradient = sum(parameter.grad for parameter in rank_parameters) / 2
    reference_parameter = torch.tensor(0.25, requires_grad=True)
    all_coefficients = torch.cat([item for rank in rank_coefficients for item in rank])
    reference_loss = (reference_parameter * all_coefficients).square().sum() / global_count
    reference_loss.backward()

    assert fsdp_averaged_gradient.item() == pytest.approx(reference_parameter.grad.item(), rel=1e-6)


def test_adaptive_actor_kl_fails_closed_for_ulysses_and_hybrid_sharding():
    with pytest.raises(RuntimeError, match="Ulysses"):
        validate_adaptive_actor_kl_topology(ulysses_size=2, fsdp_size=-1, world_size=8)
    with pytest.raises(RuntimeError, match="hybrid-sharded"):
        validate_adaptive_actor_kl_topology(ulysses_size=1, fsdp_size=4, world_size=8)
    validate_adaptive_actor_kl_topology(ulysses_size=1, fsdp_size=8, world_size=8)


def test_torch_logprob_fallback_preserves_legacy_sign_and_opt_in_correction(monkeypatch):
    logits = torch.tensor([[3.0, 1.0, -2.0]], dtype=torch.float32)
    labels = torch.tensor([0])
    monkeypatch.setattr(VF, "FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE", False)

    legacy_result = VF.log_probs_from_logits(logits, labels)
    corrected_result = VF.log_probs_from_logits(
        logits, labels, correct_torch_fallback_logprob_sign=True
    )
    expected_logprob = torch.log_softmax(logits, dim=-1)[0, 0]

    monkeypatch.setattr(VF, "FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE", True)
    monkeypatch.setattr(
        VF,
        "log_probs_from_logits_flash_attn",
        lambda flash_logits, flash_labels: -torch.nn.functional.cross_entropy(
            flash_logits.float(), flash_labels, reduction="none"
        ),
    )
    flash_result = VF.log_probs_from_logits(logits, labels)

    assert legacy_result.item() >= 0.0
    assert math.isclose(legacy_result.item(), -expected_logprob.item(), rel_tol=1e-6)
    assert corrected_result.item() <= 0.0
    assert math.isclose(corrected_result.item(), expected_logprob.item(), rel_tol=1e-6)
    assert torch.allclose(corrected_result, flash_result)


def test_logprob_fallback_policy_is_pluggable_and_v37_can_fail_closed(monkeypatch):
    monkeypatch.setattr(VF, "FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE", False)
    assert resolve_logprob_function("legacy") is VF.log_probs_from_logits
    assert resolve_logprob_function("correct") is not VF.log_probs_from_logits
    with pytest.raises(RuntimeError, match="requires the FlashAttention"):
        resolve_logprob_function("error")
    with pytest.raises(ValueError, match="legacy, correct, or error"):
        resolve_logprob_function("unknown")

    monkeypatch.setattr(VF, "FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE", True)
    assert resolve_logprob_function("error") is VF.log_probs_from_logits


def test_optimizer_step_reports_executed_and_nonfinite_skip():
    actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
    actor.actor_module = torch.nn.Linear(2, 1)
    actor.actor_optimizer = torch.optim.SGD(actor.actor_module.parameters(), lr=0.1)
    actor.config = SimpleNamespace(max_grad_norm=1.0)
    actor._spike_enabled = False
    actor._fp16_monitor_enabled = False
    actor._nonfinite_count = 0
    actor._optimizer_step_count = 0

    actor.actor_module(torch.ones(1, 2)).sum().backward()
    actor._optimizer_step()
    assert actor._last_optimizer_step_executed is True
    assert actor._last_optimizer_skip_reason == ""

    for parameter in actor.actor_module.parameters():
        parameter.grad = torch.full_like(parameter, float("nan"))
    actor._optimizer_step()
    assert actor._last_optimizer_step_executed is False
    assert actor._last_optimizer_skip_reason == "nonfinite_grad"


def _runtime_actor(adaptive=True):
    config = SimpleNamespace(
        max_grad_norm=1.0,
        adaptive_actor_kl=adaptive,
        use_torch_compile=False,
        optim=SimpleNamespace(lr=0.1),
    )
    module = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    return DataParallelPPOActor(config, module, optimizer)


def test_actor_runtime_state_exact_resume_during_cooldown(monkeypatch):
    monkeypatch.setenv("GRAD_SPIKE_PROTECT", "1")
    actor = _runtime_actor()
    actor._grad_norm_ema = 0.75
    actor._spike_cooldown_remaining = 2
    actor._original_lrs = [0.1]
    actor.actor_optimizer.param_groups[0]["lr"] = 0.01
    actor._spike_count = 3
    actor._spike_history = [2, 5, 8]
    actor._optimizer_step_count = 9
    actor._last_optimizer_step_executed = True

    state = actor.runtime_state_dict()
    restored = _runtime_actor()
    restored.load_runtime_state_dict(state)

    assert restored.runtime_state_dict() == state
    assert restored.actor_optimizer.param_groups[0]["lr"] == 0.01
    assert restored.config.optim.lr == 0.01  # prevents legacy worker LR override


def test_actor_runtime_state_exact_resume_after_nonfinite_skip(monkeypatch):
    monkeypatch.setenv("GRAD_SPIKE_PROTECT", "1")
    actor = _runtime_actor()
    for parameter in actor.actor_module.parameters():
        parameter.grad = torch.full_like(parameter, float("nan"))
    actor._optimizer_step()
    state = actor.runtime_state_dict()

    restored = _runtime_actor()
    restored.load_runtime_state_dict(state)
    assert restored._last_optimizer_step_executed is False
    assert restored._last_optimizer_skip_reason == "nonfinite_grad"
    assert restored._nonfinite_count == 1
    assert restored._nonfinite_history == [1]
    assert restored._optimizer_step_count == 1
    assert restored._spike_cooldown_remaining > 0


def test_adaptive_off_loads_legacy_runtime_state_but_adaptive_fails_closed(monkeypatch):
    monkeypatch.setenv("GRAD_SPIKE_PROTECT", "1")
    _runtime_actor(adaptive=False).load_runtime_state_dict(None)
    with pytest.raises(RuntimeError, match="missing per-rank"):
        _runtime_actor(adaptive=True).load_runtime_state_dict(None)


def test_distributed_counter_disagreement_fails(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def _unequal(tensor, op):
        if op == torch.distributed.ReduceOp.MAX:
            tensor.add_(1)

    monkeypatch.setattr(torch.distributed, "all_reduce", _unequal)
    with pytest.raises(RuntimeError, match="disagreement"):
        DataParallelPPOActor._distributed_agreed_int("executed", 1, torch.device("cpu"))


def test_checkpoint_runtime_disagreement_fails_closed(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def _disagree(output, state):
        output[:] = [state, {**state, "optimizer_attempt_count": state["optimizer_attempt_count"] + 1}]

    monkeypatch.setattr(torch.distributed, "all_gather_object", _disagree)
    with pytest.raises(RuntimeError, match="disagrees across ranks"):
        FSDPCheckpointManager._assert_runtime_agreement({"optimizer_attempt_count": 3}, "actor runtime")


def test_trainer_counter_contract_rejects_fractional_or_unattested_values():
    good = {
        "actor/optimizer_counter_agreement": 1.0,
        "actor/optimizer_steps_attempted": 3.0,
        "actor/optimizer_steps_executed": 2.0,
        "actor/optimizer_steps_skipped": 1.0,
    }
    assert strict_optimizer_counter_metrics(good) == (3, 2, 1)
    with pytest.raises(RuntimeError, match="exact nonnegative integer"):
        strict_optimizer_counter_metrics({**good, "actor/optimizer_steps_executed": 1.5})
    with pytest.raises(RuntimeError, match="attest"):
        strict_optimizer_counter_metrics({**good, "actor/optimizer_counter_agreement": 0.0})


def test_actor_emits_rank_agreed_nonfinite_counter_for_formal_evidence():
    source = (ROOT / "verl/workers/actor/dp_actor.py").read_text(encoding="utf-8")
    assert 'self._distributed_agreed_int(\n                "nonfinite_grad_count"' in source
    assert 'metrics["actor/nonfinite_counter_agreement"] = [1.0]' in source
    trainer = (ROOT / "verl/trainer/ray_trainer.py").read_text(encoding="utf-8")
    assert 'actor_metrics.get("actor/nonfinite_counter_agreement") != 1.0' in trainer
