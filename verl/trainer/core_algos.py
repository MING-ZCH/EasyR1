# Copyright 2022 The HuggingFace Team
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..utils import torch_functional as VF


if TYPE_CHECKING:
    from .config import AlgorithmConfig


class KLController(ABC):
    kl_coef: float
    """KL coefficient."""

    @abstractmethod
    def update(self, current_kl: float, n_steps: int) -> None:
        """Update kl_coef according to current KL."""
        ...


class AdaptiveKLController(KLController):
    """Adaptive KL controller described in: https://arxiv.org/pdf/1909.08593.pdf

    Copied from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/utils.py#L54"""

    def __init__(self, init_kl_coef: float, target_kl: float, horizon: float):
        self.kl_coef = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl: float, n_steps: int) -> None:
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.kl_coef *= mult


class FixedKLController(KLController):
    """Fixed KL controller.

    Copeid from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/utils.py#L72"""

    def __init__(self, init_kl_coef: float):
        self.kl_coef = init_kl_coef

    def update(self, current_kl: float, n_steps: int) -> None:
        pass


def get_kl_controller(algorithm_config: "AlgorithmConfig") -> KLController:
    """Adapted from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L319"""
    if algorithm_config.kl_type == "fixed":
        kl_ctrl = FixedKLController(init_kl_coef=algorithm_config.kl_coef)
    elif algorithm_config.kl_type == "adaptive":
        assert algorithm_config.kl_horizon > 0, f"horizon must be larger than 0. Got {algorithm_config.kl_horizon}."
        kl_ctrl = AdaptiveKLController(
            init_kl_coef=algorithm_config.kl_coef,
            target_kl=algorithm_config.kl_target,
            horizon=algorithm_config.kl_horizon,
        )
    else:
        raise ValueError(f"Unknown kl type: {algorithm_config.kl_type}.")

    return kl_ctrl


@torch.no_grad()
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Adapted from https://github.com/huggingface/trl/blob/v0.16.0/trl/trainer/ppo_trainer.py#L513

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length). The token after eos tokens have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    lastgaelam = 0
    advantages_reversed = []
    gen_len = token_level_rewards.shape[-1]
    for t in reversed(range(gen_len)):
        nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
        delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam
        advantages_reversed.append(lastgaelam)

    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values
    advantages = VF.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@torch.no_grad()
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean, id2std = {}, {}

    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    for idx in id2score:
        assert len(id2score[idx]) > 1, "GRPO needs rollout.n > 1."
        id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
        id2std[idx] = torch.std(torch.tensor(id2score[idx]))

    for i in range(bsz):
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + eps)

    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns


# ---------------------------------------------------------------------------
#  Dr.GRPO: handles the *reward homogeneity* failure mode of standard GRPO.
#
#  When all n responses for a prompt receive (nearly) identical rewards,
#  standard per-group z-normalisation yields zero advantages -> zero gradient
#  -> no learning.
#
#  Dr.GRPO addresses this with a two-level strategy:
#    1. *Within-group* normalisation (standard GRPO) for groups that have
#       sufficient reward variance  (std > `low_var_threshold`).
#    2. *Global-baseline* normalisation for low-variance groups:
#           adv_i  =  (score_i  -  batch_mean) / (batch_std + eps)
#       This preserves a between-prompt signal: easy prompts (high reward)
#       get positive advantage; hard prompts (low reward) get negative.
#    3. Optional: clip per-group advantage magnitude to avoid outlier steps
#       dominating the gradient (controlled by `adv_clip`).
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_drgrpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor,
    eps: float = 1e-6,
    low_var_threshold: float = 1e-3, # 1e-4
    adv_clip: float = 2.5,  # 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dr.GRPO advantage -- robust to within-group reward homogeneity.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask:       (bs, response_length)
        index:               (bs,) -- prompt uid for grouping
        eps:                 numerical stability
        low_var_threshold:   groups with std < this use global baseline
        adv_clip:            if >0, clip per-sample advantage to [-clip, +clip]

    Returns:
        advantages, returns -- both (bs, response_length)
    """
    # ---- score per trajectory (response-length normalised) ----
    response_len = response_mask.sum(dim=-1).clamp(min=1.0)  # (bs,)
    scores = token_level_rewards.sum(dim=-1) / response_len  # normalise by length

    # ---- per-group statistics ----
    id2indices: dict = defaultdict(list)
    bsz = scores.shape[0]
    for i in range(bsz):
        id2indices[index[i]].append(i)

    id2mean: dict = {}
    id2std: dict = {}
    for idx, indices in id2indices.items():
        group_scores = torch.tensor([scores[j].item() for j in indices])
        id2mean[idx] = group_scores.mean()
        id2std[idx] = group_scores.std() if len(indices) > 1 else torch.tensor(0.0)

    # ---- batch-level statistics (fallback baseline) ----
    batch_mean = scores.mean()
    batch_std = scores.std()

    n_low_var = 0
    for i in range(bsz):
        idx = index[i]
        group_std = id2std[idx]
        if group_std.item() > low_var_threshold:
            # Standard within-group z-normalisation
            scores[i] = (scores[i] - id2mean[idx]) / (group_std + eps)
        else:
            # Low-variance fallback: use batch-level baseline
            n_low_var += 1
            if batch_std.item() > eps:
                scores[i] = (scores[i] - batch_mean) / (batch_std + eps)
            else:
                scores[i] = 0.0

    if adv_clip > 0:
        scores = scores.clamp(-adv_clip, adv_clip)

    # Log low-variance fraction periodically
    _ctr = getattr(compute_drgrpo_outcome_advantage, "_log_ctr", 0) + 1
    compute_drgrpo_outcome_advantage._log_ctr = _ctr
    if _ctr % 10 == 1 or n_low_var > bsz * 0.8:
        import logging
        logging.getLogger(__name__).info(
            "[DrGRPO] batch=%d  low_var_groups=%d/%d  batch_mean=%.4f batch_std=%.4f  adv_range=[%.3f, %.3f]",
            bsz, n_low_var, bsz,
            batch_mean.item(), batch_std.item(),
            scores.min().item(), scores.max().item(),
        )

    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns


# ---------------------------------------------------------------------------
#  Step-level GRPO (Process Reward GRPO / GSPO):
#  Distributes and normalises advantages *per reward-bearing token position*
#  instead of per sequence.  This is crucial for interleaved multi-turn tasks
#  where per-turn point rewards may differ even when overall outcome rewards
#  are identical.
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_grpo_step_level_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor,
    eps: float = 1e-6,
    low_var_threshold: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Step-level GRPO advantage for multi-turn process rewards.

    Expects token_level_rewards to have non-zero values at turn boundary
    positions (placed by the reward worker).  For each reward position,
    advantages are normalised independently across the group.  The
    normalised advantage is then broadcast to all tokens between this
    reward position and the next.

    Falls back to batch-global baseline when within-group variance at a
    step is zero (same Dr.GRPO strategy).

    Args:
        token_level_rewards: (bs, response_length) -- per-turn rewards at
                             turn-boundary token positions
        response_mask:       (bs, response_length)
        index:               (bs,) prompt uid for grouping

    Returns:
        advantages, returns -- both (bs, response_length)
    """
    bsz, seq_len = token_level_rewards.shape

    # Group responses by prompt
    id2indices: dict = defaultdict(list)
    for i in range(bsz):
        id2indices[index[i]].append(i)

    advantages = torch.zeros_like(token_level_rewards)

    for idx, indices in id2indices.items():
        group_rewards = token_level_rewards[indices]  # (group_size, seq_len)
        group_mask = response_mask[indices]

        # Find positions where ANY response in the group has a reward signal
        reward_positions = (group_rewards.abs() > 1e-9).any(dim=0).nonzero(as_tuple=True)[0]

        if len(reward_positions) == 0:
            # No reward signal at all -- set advantage to 0
            continue

        # For each reward position, normalise across the group
        step_advantages = torch.zeros(
            (len(indices), len(reward_positions)),
            dtype=token_level_rewards.dtype,
            device=token_level_rewards.device,
        )
        for step_idx, pos in enumerate(reward_positions):
            pos_rewards = group_rewards[:, pos]  # (group_size,)
            pos_mean = pos_rewards.mean()
            pos_std = (
                pos_rewards.std()
                if len(indices) > 1
                else torch.tensor(0.0, dtype=token_level_rewards.dtype, device=token_level_rewards.device)
            )

            if pos_std.item() > low_var_threshold:
                step_advantages[:, step_idx] = (pos_rewards - pos_mean) / (pos_std + eps)
            else:
                # Fallback: batch-level stats at this position
                batch_pos_rewards = token_level_rewards[:, pos]
                batch_pos_mean = batch_pos_rewards.mean()
                batch_pos_std = batch_pos_rewards.std()
                if batch_pos_std.item() > eps:
                    step_advantages[:, step_idx] = (pos_rewards - batch_pos_mean) / (batch_pos_std + eps)
                # else: leave as 0

        # Broadcast step advantages to all tokens between consecutive reward positions
        for step_idx, pos in enumerate(reward_positions):
            # Find the range of tokens this step advantage applies to
            # (from the previous reward position + 1 to this position)
            if step_idx == 0:
                start = 0
            else:
                start = reward_positions[step_idx - 1].item() + 1
            end = pos.item() + 1  # inclusive

            for j, global_i in enumerate(indices):
                adv_val = step_advantages[j, step_idx].item()
                mask_slice = group_mask[j, start:end]
                advantages[global_i, start:end] = adv_val * mask_slice


    # ---- Per-response centering (GSPO bias fix) ----
    # After broadcasting, the token-level advantage may have non-zero mean
    # because different segments have different lengths (e.g. answer turn
    # is much longer than point turns).  Re-center each response's
    # advantage to mean=0 so that the policy gradient is unbiased.
    for idx, indices in id2indices.items():
        group_mask_local = response_mask[indices]
        for j, global_i in enumerate(indices):
            resp_mask = group_mask_local[j]  # (seq_len,)
            resp_adv = advantages[global_i]
            n_tokens = resp_mask.sum()
            if n_tokens > 0:
                resp_mean = (resp_adv * resp_mask).sum() / n_tokens
                advantages[global_i] = (resp_adv - resp_mean) * resp_mask

    returns = advantages.clone()
    return advantages, returns


@torch.no_grad()

# ---------------------------------------------------------------------------
#  Best-of-K GRPO (BoK-GRPO):
#  Designed for tasks where pass@K >> pass@1 (e.g. counting with pass@32~95%
#  but pass@1~79%).  Instead of symmetric z-normalisation, BoK-GRPO uses
#  a reward-temperature softmax to compute soft "selection probabilities"
#  for each trajectory, then computes advantages as the deviation from
#  uniform selection.
#
#  Key idea:  We want the model to *maximise P(argmax reward)*.
#  The advantage for response i in a group of K is:
#
#      w_i  =  softmax(r / tau)_i      --  soft best-of-K weight
#      A_i  =  w_i  -  1/K             --  advantage = softmax - uniform
#
#  When tau -> 0 this is pure best-of-K (only best gets positive advantage).
#  When tau -> inf this recovers uniform weighting (no gradient signal).
#  A moderate tau (0.1-0.5) provides a smooth interpolation.
#
#  Anti-collapse measures:
#    1. Entropy bonus: if w_max > collapse_threshold, we mix w with uniform.
#    2. Advantage clipping: clip to [-clip, +clip] to avoid extreme gradients.
#    3. Low-variance fallback: if all rewards are the same, use Dr.GRPO's
#       batch-level baseline.
#
#  References:
#    - BOND (Best-of-N Distillation), Gui et al. 2024
#    - RAFT (Reward Ranked Fine-Tuning), Dong et al. 2023
#    - Dr.GRPO, Liu et al. 2025
#    - GRPO, Shao et al. 2024
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_bok_grpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor,
    eps: float = 1e-6,
    bok_tau: float = 0.3,
    bok_clip: float = 3.0,
    bok_uniform_mix: float = 0.1,
    collapse_threshold: float = 0.9,
    low_var_threshold: float = 1e-5,
    bok_tau_init: float = 0.0,
    bok_tau_final: float = 0.0,
    global_step: int = 0,
    total_steps: int = 1,
    answer_scores: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Best-of-K GRPO advantage with Dr.GRPO-enhanced fallback.

    Uses softmax-temperature weighting to concentrate gradient signal on
    the best trajectories within diverse groups, with a robust global-baseline
    fallback for homogeneous groups (Dr.GRPO strategy).

    New in v2:
      - BOK_FALLBACK_MODE: controls low-var fallback strategy
          'zscore'   : (default) batch z-normalization  (score - μ) / (σ + ε)
          'drgrpo'   : Dr.GRPO style  (score - μ), no std division
          'clip_std' : clipped std  (score - μ) / max(σ, min_std)
      - BOK_DAPO_FILTER: if '1', zero-out homogeneous groups (DAPO dynamic sampling)
      - Enhanced per-group diagnostics every 10 calls.

    Args:
        token_level_rewards: (bs, response_length) per-token rewards
        response_mask:       (bs, response_length) valid token mask
        index:               (bs,) prompt uid for grouping
        eps:                 numerical stability
        bok_tau:             softmax temperature (lower = more selective)
        bok_clip:            clip advantage magnitude
        bok_uniform_mix:     mix ratio with uniform (anti-collapse)
        collapse_threshold:  if max softmax weight > this, increase uniform mix
        low_var_threshold:   fallback to batch baseline when group std < this
        bok_tau_init / bok_tau_final / global_step / total_steps: τ annealing

    Returns:
        advantages, returns -- both (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    # Guard against non-finite reward spikes from upstream reward pipelines.
    scores = torch.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)

    # ---- τ annealing: cosine schedule from tau_init -> tau_final ----
    import os, math
    _bok_schedule = os.environ.get("BOK_TAU_SCHEDULE", "cosine")  # "linear" or "cosine"
    if bok_tau_init > 0 and bok_tau_final > 0 and total_steps > 1:
        progress = min(1.0, max(0.0, global_step / total_steps))
        if _bok_schedule == "cosine":
            bok_tau = bok_tau_final + (bok_tau_init - bok_tau_final) * 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            bok_tau = bok_tau_init + (bok_tau_final - bok_tau_init) * progress

    # ---- Configuration from environment ----
    _bok_adv_normalize = int(os.environ.get("BOK_ADV_NORMALIZE", "0")) > 0
    _lvt_env = os.environ.get("BOK_LOW_VAR_THRESHOLD", "")
    if _lvt_env:
        low_var_threshold = float(_lvt_env)
    _logit_cap = float(os.environ.get("BOK_LOGIT_CAP", "12.0"))
    _tau_adaptive = int(os.environ.get("BOK_TAU_ADAPTIVE", "1")) > 0
    _tau_min = float(os.environ.get("BOK_TAU_MIN", "0.08"))

    # NEW: Fallback mode for low-var groups
    _fallback_mode = os.environ.get("BOK_FALLBACK_MODE", "zscore")  # zscore | drgrpo | clip_std
    _dapo_filter = int(os.environ.get("BOK_DAPO_FILTER", "0")) > 0
    _min_batch_std = float(os.environ.get("BOK_MIN_BATCH_STD", "0.1"))
    # Safety: auto-disable DAPO filter when low-var ratio exceeds threshold to prevent death spiral.
    # When DAPO filter removes too many samples, remaining data is insufficient for stable training.
    _dapo_auto_disable_threshold = float(os.environ.get("BOK_DAPO_AUTO_DISABLE_THRESHOLD", "0.5"))

    # NEW V11: Conditional-Advantage routing thresholds
    # Groups with pass_rate > _easy_threshold are routed to DrGRPO advantage (stronger gradient for easy groups)
    _easy_threshold = float(os.environ.get("BOK_EASY_THRESHOLD", "0.75"))
    _easy_score_threshold = float(os.environ.get("BOK_EASY_SCORE_THRESHOLD", "0.5"))
    # P2: Filter all-correct groups (pass@K=K/K) — zero gradient for trivially easy samples
    _filter_all_correct = int(os.environ.get("BOK_FILTER_ALL_CORRECT", "1")) > 0
    # V24 Smart Filter: only filter all-correct groups whose group_mean >= threshold.
    # group_mean ≈ 0.7 + 0.3*mean(point) when format=1, so threshold 0.955 ≈ mean(point)>=0.85.
    # Groups below threshold fall through to Easy path to continue optimizing point quality.
    # Set to 0 to disable (= filter ALL all-correct groups, V23 behavior).
    _smart_filter_threshold = float(os.environ.get("BOK_SMART_FILTER_THRESHOLD", "0"))

    # AllWrong Advantage Cap: cap BoK advantages for groups where ALL trajectories have wrong answers.
    # Preserves directional signal from soft_decay but prevents overpowered +bok_clip advantage.
    _allwrong_cap = float(os.environ.get("BOK_ALLWRONG_CAP", "0"))  # 0 = disabled
    _allwrong_answer_threshold = float(os.environ.get("BOK_ALLWRONG_ANSWER_THRESHOLD", "0.5"))

    # --- V3 improvements: Winner Amplification, Easy Dampening, Quality Bonus ---
    # Winner Amplification: boost correct-trajectory advantages in BoK groups
    # boost = min(boost_max, sqrt(K / n_correct)). Set to 0 to disable.
    _winner_boost = float(os.environ.get("BOK_WINNER_BOOST", "0"))
    # Easy Gradient Dampening: scale easy_drgrpo advantages by this factor.
    # 1.0 = no change (default), 0.5 = halve easy gradient to shift budget to BoK.
    _easy_scale = float(os.environ.get("BOK_EASY_SCALE", "1.0"))
    # Quality-Ranked BoK: add bonus to correct trajectories proportional to
    # relative point-quality rank within BoK group. Set to 0 to disable.
    _quality_bonus = float(os.environ.get("BOK_QUALITY_BONUS", "0"))

    # ---- per-group statistics ----
    id2indices: dict = defaultdict(list)
    bsz = scores.shape[0]
    for i in range(bsz):
        id2indices[index[i]].append(i)

    # ---- batch-level statistics (fallback baseline) ----
    batch_mean = scores.mean()
    batch_std = scores.std()

    advantages_1d = torch.zeros_like(scores)  # (bs,)
    n_low_var = 0
    n_collapsed = 0
    n_dapo_filtered = 0
    n_easy_drgrpo = 0
    n_all_correct_filtered = 0
    n_all_correct_released = 0  # V24: all-correct groups released to Easy by smart filter
    n_allwrong_capped = 0
    n_winner_boosted = 0
    n_tau_bumped = 0
    all_group_stds = []  # for diagnostics

    for idx, indices in id2indices.items():
        K = len(indices)
        group_scores = torch.stack([scores[j] for j in indices])  # (K,)
        group_scores = torch.nan_to_num(group_scores, nan=0.0, posinf=1.0, neginf=0.0)
        group_mean = group_scores.mean()
        group_std = group_scores.std() if K > 1 else torch.tensor(0.0, device=scores.device)
        if not torch.isfinite(group_std):
            group_std = torch.tensor(0.0, device=scores.device)
        all_group_stds.append(group_std.item())

        if group_std.item() <= low_var_threshold:
            # ---- Low-variance fallback ----
            n_low_var += K

            # DAPO filtering: optionally zero-out homogeneous groups
            if _dapo_filter:
                n_dapo_filtered += K
                # Leave advantages as 0 → no gradient for this group
                continue

            # Fallback advantage computation (configurable mode)
            if _fallback_mode == "drgrpo":
                # Dr.GRPO: score - batch_mean, NO std division
                # Removes "difficulty bias" from std normalization (Dr.GRPO paper)
                for j, global_i in enumerate(indices):
                    advantages_1d[global_i] = scores[global_i] - batch_mean
            elif _fallback_mode == "clip_std":
                # Clipped std: prevents over-amplification when batch_std is tiny
                effective_std = max(batch_std.item(), _min_batch_std)
                for j, global_i in enumerate(indices):
                    advantages_1d[global_i] = (scores[global_i] - batch_mean) / effective_std
            else:
                # "zscore" (default, backward-compatible with v1)
                if batch_std.item() > eps:
                    for j, global_i in enumerate(indices):
                        advantages_1d[global_i] = (scores[global_i] - batch_mean) / (batch_std + eps)
                # else: leave as 0
            continue

        # ---- P2: All-correct group filter (V11) ----
        # If all K trajectories in a group score > threshold, skip (zero gradient)
        # This removes trivially easy samples that waste gradient budget
        if _filter_all_correct:
            # V25: use answer_scores for routing when available (more precise than mixed score)
            if answer_scores is not None:
                group_answer = torch.stack([answer_scores[j] for j in indices])
                pass_rate_full = (group_answer >= _allwrong_answer_threshold).float().mean().item()
            else:
                pass_rate_full = (group_scores > _easy_score_threshold).float().mean().item()
            if pass_rate_full >= 1.0 - 1e-6:  # all K correct
                if _smart_filter_threshold > 0 and group_mean.item() < _smart_filter_threshold:
                    # V24: all-correct but point quality below threshold -> release to Easy path
                    n_all_correct_released += K
                else:
                    n_all_correct_filtered += K
                    # advantages_1d already initialized to 0 — no action needed
                    continue

        # ---- Difficulty-Aware Advantage Routing (V11 / V25 answer-based) ----
        # Easy groups (high pass rate) -> Dr.GRPO raw centering (score - mean, NO std division)
        # Hard groups (low pass rate) -> BOK softmax (concentrate probability on rare correct trajectories)
        # V25: route by answer correctness, not mixed score (avoids point-inflated routing errors)
        # V31: Fixed z-score amplification bug — Easy path now uses true Dr.GRPO (raw centering)
        if answer_scores is not None:
            group_answer = torch.stack([answer_scores[j] for j in indices])
            pass_rate = (group_answer >= _allwrong_answer_threshold).float().mean().item()
        else:
            pass_rate = (group_scores > _easy_score_threshold).float().mean().item()
        if pass_rate > _easy_threshold:
            n_easy_drgrpo += K
            for j, global_i in enumerate(indices):
                # V31: Dr.GRPO raw centering — no std division.
                # When group_std ≈ 0 (homogeneous easy groups), z-normalization
                # amplifies noise to ±clip.  True Dr.GRPO uses only mean-centering
                # so that nearly-uniform groups get near-zero advantage (correct
                # signal: "already mastered, no extra gradient needed"), while
                # groups with genuine variance retain natural discriminability.
                adv = (scores[global_i] - group_mean).item()
                adv = max(-3.0, min(3.0, adv))  # soft clip for safety
                advantages_1d[global_i] = adv * _easy_scale  # V3: Easy Gradient Dampening
            continue

        # ---- Softmax-temperature weighting (BoK path) ----
        # Shift for numerical stability
        shifted = (group_scores - group_mean) / (group_std + eps)  # z-normalise first
        shifted = torch.nan_to_num(shifted, nan=0.0, posinf=_logit_cap, neginf=-_logit_cap)

        # Keep low-tau advantage by default, but adaptively raise effective tau
        # only for numerically dangerous groups where |shifted / tau| would explode.
        effective_tau = max(bok_tau, eps)
        if _tau_adaptive:
            max_abs_shifted = shifted.abs().max().item()
            safe_tau = max_abs_shifted / max(_logit_cap, eps)
            adjusted_tau = max(effective_tau, safe_tau, _tau_min)
            if adjusted_tau > effective_tau + 1e-12:
                n_tau_bumped += K
            effective_tau = adjusted_tau
        else:
            effective_tau = max(effective_tau, _tau_min)

        logits = shifted / effective_tau
        logits = torch.nan_to_num(logits, nan=0.0, posinf=_logit_cap, neginf=-_logit_cap)
        logits = logits.clamp(-_logit_cap, _logit_cap)

        # Softmax weights
        w = torch.softmax(logits.float(), dim=0).to(logits.dtype)  # (K,)
        if not torch.isfinite(w).all():
            # Degenerate case fallback: uniform avoids contaminating gradients with NaN.
            w = torch.ones_like(w) / K

        # Anti-collapse: if max weight is too high, mix with uniform
        w_max = w.max().item()
        uniform = torch.ones_like(w) / K
        mix_ratio = bok_uniform_mix
        if w_max > collapse_threshold:
            mix_ratio = max(mix_ratio, 0.3)
            n_collapsed += 1
        w = (1.0 - mix_ratio) * w + mix_ratio * uniform

        # Advantage = softmax weight - uniform weight, scaled by K for gradient magnitude
        for j, global_i in enumerate(indices):
            advantages_1d[global_i] = (w[j].item() - 1.0 / K) * K

        # ---- V3: Winner Amplification ----
        # Boost correct-trajectory advantages proportional to their rarity.
        # Rare correct solutions (low pass_rate) get stronger gradient.
        if _winner_boost > 0 and answer_scores is not None:
            group_answer = torch.stack([answer_scores[j] for j in indices])
            n_correct = int((group_answer >= _allwrong_answer_threshold).sum().item())
            if 0 < n_correct < K:
                boost = min(_winner_boost, math.sqrt(K / max(n_correct, 1)))
                for j, global_i in enumerate(indices):
                    if answer_scores[global_i] >= _allwrong_answer_threshold:
                        advantages_1d[global_i] = advantages_1d[global_i] * boost
                n_winner_boosted += K

        # ---- V3: Quality-Ranked BoK ----
        # Among correct trajectories, add bonus for higher point quality.
        # Encourages model to learn the best correct path, not just any correct path.
        if _quality_bonus > 0 and answer_scores is not None:
            correct_indices = [j for j in indices if answer_scores[j] >= _allwrong_answer_threshold]
            if len(correct_indices) >= 2:
                # quality ≈ point + format component (overall - 0.6*answer)
                qualities = [scores[j].item() - 0.6 for j in correct_indices]
                q_min, q_max = min(qualities), max(qualities)
                if q_max > q_min + eps:
                    for ci, j in enumerate(correct_indices):
                        rel_q = (qualities[ci] - q_min) / (q_max - q_min + eps)
                        advantages_1d[j] = advantages_1d[j] + _quality_bonus * rel_q

        # ---- AllWrong Advantage Cap ----
        # If all K trajectories have wrong answers, cap BoK advantages to ±allwrong_cap.
        # This preserves the directional signal (best wrong > worst wrong) but
        # prevents overpowered positive advantage that would push wrong answers too hard.
        if _allwrong_cap > 0:
            if answer_scores is not None:
                # Precise detection using per-sample answer scores from reward function
                group_answer_scores = torch.stack([answer_scores[j] for j in indices])
                is_allwrong = (group_answer_scores < _allwrong_answer_threshold).all().item()
            else:
                # Fallback: use overall scores (less precise but backward-compatible)
                is_allwrong = (group_scores < _allwrong_answer_threshold).all().item()
            if is_allwrong:
                n_allwrong_capped += K
                # V25: force advantages to [-cap, 0] for all-wrong groups.
                # Rationale: in all-wrong groups, the "best wrong answer" still gets
                # positive BoK advantage. This wrongly reinforces incorrect trajectories.
                # By capping at 0, we only penalize (down-weight) wrong trajectories
                # without ever rewarding the "least wrong" answer.
                _allwrong_neg_only = int(os.environ.get("BOK_ALLWRONG_NEG_ONLY", "1")) > 0
                for global_i in indices:
                    v = advantages_1d[global_i].item()
                    if _allwrong_neg_only:
                        advantages_1d[global_i] = max(-_allwrong_cap, min(0.0, v))
                    else:
                        advantages_1d[global_i] = max(-_allwrong_cap, min(_allwrong_cap, v))

        # Per-group advantage normalization (controlled by BOK_ADV_NORMALIZE, default OFF)
        if _bok_adv_normalize:
            group_adv = torch.stack([advantages_1d[i] for i in indices])
            adv_std = group_adv.std()
            if adv_std.item() > eps:
                for i in indices:
                    advantages_1d[i] = advantages_1d[i] / adv_std

    # ---- Safety: detect and mitigate death spiral from DAPO filter ----
    low_var_rate = n_low_var / max(bsz, 1)
    if _dapo_filter and low_var_rate > _dapo_auto_disable_threshold:
        # DAPO filter is removing too many samples → death spiral risk.
        # Re-compute advantages for filtered samples using fallback instead of zero.
        print(
            f"[BoK-GRPO][SAFETY] low_var_rate={low_var_rate:.1%} > {_dapo_auto_disable_threshold:.0%} "
            f"with DAPO filter ON! Auto-falling back to '{_fallback_mode}' for "
            f"{n_dapo_filtered}/{bsz} filtered samples to prevent death spiral."
        )
        # Re-run low-var groups with fallback instead of filtering
        for idx, indices in id2indices.items():
            K = len(indices)
            group_scores = torch.stack([scores[j] for j in indices])
            group_std = group_scores.std() if K > 1 else torch.tensor(0.0, device=scores.device)
            if group_std.item() <= low_var_threshold:
                # Apply fallback advantage instead of zero
                if _fallback_mode == "drgrpo":
                    for j, global_i in enumerate(indices):
                        advantages_1d[global_i] = scores[global_i] - batch_mean
                elif _fallback_mode == "clip_std":
                    effective_std = max(batch_std.item(), _min_batch_std)
                    for j, global_i in enumerate(indices):
                        advantages_1d[global_i] = (scores[global_i] - batch_mean) / effective_std
                else:  # zscore
                    if batch_std.item() > eps:
                        for j, global_i in enumerate(indices):
                            advantages_1d[global_i] = (scores[global_i] - batch_mean) / (batch_std + eps)

    # ---- VCRL: Variance-based Curriculum RL ----
    # Per-group reward variance determines learning potential:
    #   variance in [VCRL_LOW, VCRL_HIGH] → optimal zone → 2x advantage boost
    #   variance outside range → too easy or too hard → 0.5x advantage reduction
    # Disabled by default (VCRL_ENABLE=0); env-var gated for backward compatibility.
    _vcrl_enable = int(os.environ.get("VCRL_ENABLE", "0")) > 0
    if _vcrl_enable:
        _vcrl_low = float(os.environ.get("VCRL_LOW", "0.15"))
        _vcrl_high = float(os.environ.get("VCRL_HIGH", "0.55"))
        _vcrl_boost = float(os.environ.get("VCRL_BOOST", "2.0"))
        _vcrl_reduce = float(os.environ.get("VCRL_REDUCE", "0.5"))
        n_vcrl_boosted = 0
        n_vcrl_reduced = 0
        for _vcrl_idx, _vcrl_indices in id2indices.items():
            _K = len(_vcrl_indices)
            _g_scores = torch.stack([scores[_j] for _j in _vcrl_indices])
            _g_var = _g_scores.var().item() if _K > 1 else 0.0
            if not math.isfinite(_g_var):
                _g_var = 0.0
            if _vcrl_low <= _g_var <= _vcrl_high:
                for _gi in _vcrl_indices:
                    advantages_1d[_gi] = advantages_1d[_gi] * _vcrl_boost
                n_vcrl_boosted += _K
            else:
                for _gi in _vcrl_indices:
                    advantages_1d[_gi] = advantages_1d[_gi] * _vcrl_reduce
                n_vcrl_reduced += _K
        print(
            f"[VCRL] step={global_step} enable=1 "
            f"range=[{_vcrl_low},{_vcrl_high}] boost={_vcrl_boost} reduce={_vcrl_reduce} "
            f"boosted={n_vcrl_boosted}/{bsz} reduced={n_vcrl_reduced}/{bsz}"
        )

    # ---- Clip advantages ----
    # Asymmetric clip: positive ceiling raised by sqrt(winner_boost) to allow
    # Winner Amplification to take effect for rare correct trajectories.
    # Without this, the global clip at bok_clip=4.0 would negate the boost.
    if bok_clip > 0:
        if _winner_boost > 1.0:
            pos_clip = bok_clip * math.sqrt(_winner_boost)
        else:
            pos_clip = bok_clip
        advantages_1d = advantages_1d.clamp(-bok_clip, pos_clip)

    if not torch.isfinite(advantages_1d).all():
        bad_count = (~torch.isfinite(advantages_1d)).sum().item()
        print(
            f"[BoK-GRPO][SAFETY] non-finite advantages detected: {bad_count}/{bsz}. "
            "Replacing with finite values."
        )
        clip_bound = bok_clip if bok_clip > 0 else 3.0
        advantages_1d = torch.nan_to_num(advantages_1d, nan=0.0, posinf=clip_bound, neginf=-clip_bound)

    # ---- Log diagnostics every step (first ppo_epoch call per step) ----
    _last_step = getattr(compute_bok_grpo_advantage, "_last_logged_step", -1)
    _should_log = (global_step != _last_step)
    if _should_log:
        compute_bok_grpo_advantage._last_logged_step = global_step


        print(
            f"[BoK-GRPO] batch={bsz} tau={bok_tau:.3f} step={global_step}/{total_steps}  "
            f"low_var={n_low_var}/{bsz}  collapsed={n_collapsed}  lvt={low_var_threshold:.1e}  "
            f"tau_bumped={n_tau_bumped}/{bsz}  logit_cap={_logit_cap:.1f}  "
            f"adv_normalize={_bok_adv_normalize}  fallback={_fallback_mode}  easy_drgrpo={n_easy_drgrpo}/{bsz}  all_correct_filtered={n_all_correct_filtered}/{bsz}  ac_released={n_all_correct_released}/{bsz}  allwrong_capped={n_allwrong_capped}/{bsz}  winner_boosted={n_winner_boosted}/{bsz}  "
            f"adv_mean={advantages_1d.mean().item():.4f} adv_std={advantages_1d.std().item():.4f} "
            f"adv_range=[{advantages_1d.min().item():.3f}, {advantages_1d.max().item():.3f}]"
        )

        # Enhanced per-group diagnostics
        if all_group_stds:
            sorted_stds = sorted(all_group_stds)
            n_groups = len(sorted_stds)
            n_zero = sum(1 for s in sorted_stds if s < 1e-10)
            p25_idx = max(0, n_groups // 4 - 1)
            p50_idx = max(0, n_groups // 2 - 1)
            p75_idx = max(0, 3 * n_groups // 4 - 1)
            print(
                f"[BoK-GRPO-Detail] n_groups={n_groups} K_typical={bsz // max(n_groups, 1)} "
                f"group_std: min={sorted_stds[0]:.6f} p25={sorted_stds[p25_idx]:.6f} "
                f"median={sorted_stds[p50_idx]:.6f} p75={sorted_stds[p75_idx]:.6f} "
                f"max={sorted_stds[-1]:.6f} n_zero_std={n_zero}/{n_groups} "
                f"score: range=[{scores.min().item():.4f},{scores.max().item():.4f}] "
                f"batch_mean={batch_mean.item():.4f} batch_std={batch_std.item():.4f} "
                f"dapo_filter={_dapo_filter} dapo_filtered={n_dapo_filtered}/{bsz} "
                f"easy_drgrpo={n_easy_drgrpo}/{bsz} all_correct_filtered={n_all_correct_filtered}/{bsz} ac_released={n_all_correct_released}/{bsz} allwrong_capped={n_allwrong_capped}/{bsz} "
                f"smart_filter_th={_smart_filter_threshold:.3f} easy_threshold={_easy_threshold:.2f} easy_score_threshold={_easy_score_threshold:.2f} allwrong_cap={_allwrong_cap:.2f} allwrong_answer_th={_allwrong_answer_threshold:.2f} "
                f"winner_boost={_winner_boost:.1f} easy_scale={_easy_scale:.2f} quality_bonus={_quality_bonus:.1f}"
            )


    # ---- Health monitoring: detect training instability early ----
    _low_var_pct = n_low_var / max(bsz, 1)
    _n_zero_reward = int((scores == 0).sum().item())
    _zero_reward_pct = _n_zero_reward / max(bsz, 1)
    _effective_training_pct = 1.0 - (n_dapo_filtered / max(bsz, 1)) if _dapo_filter else 1.0
    _easy_drgrpo_pct = n_easy_drgrpo / max(bsz, 1)

    if _low_var_pct > 0.3 or _zero_reward_pct > 0.5:
        _severity = "CRITICAL" if _low_var_pct > 0.5 else "WARNING"
        print(
            f"[BoK-GRPO][{_severity}] step={global_step} "
            f"low_var_rate={_low_var_pct:.1%} zero_reward_rate={_zero_reward_pct:.1%} "
            f"effective_training_data={_effective_training_pct:.1%} "
            f"batch_mean={batch_mean.item():.4f} adv_std={advantages_1d.std().item():.4f} "
            f"Consider: increase kl_coef, increase tau, disable DAPO filter"
        )

    returns = advantages_1d.unsqueeze(-1) * response_mask
    return returns, returns


@torch.no_grad()
def compute_bok_grpo_step_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor,
    eps: float = 1e-6,
    bok_tau: float = 0.3,
    bok_clip: float = 3.0,
    bok_uniform_mix: float = 0.1,
    collapse_threshold: float = 0.9,
    low_var_threshold: float = 1e-5,
    bok_tau_init: float = 0.0,
    bok_tau_final: float = 0.0,
    global_step: int = 0,
    total_steps: int = 1,
    answer_scores: torch.Tensor = None,
    point_step_mask: torch.Tensor = None,
    point_step_value: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Outcome-primary BoK-GRPO with semantic point-step auxiliary credit.

    This estimator is opt-in and keeps the original BoK-GRPO outcome advantage
    as the base signal. When process rewards expose point-step token positions,
    it aligns the kth point in each rollout by semantic order within each prompt
    group, normalizes point-step rewards per semantic step, and adds a gated
    auxiliary advantage only to the corresponding point-generation span.
    """
    import os

    outcome_advantages, _ = compute_bok_grpo_advantage(
        token_level_rewards,
        response_mask,
        index,
        eps=eps,
        bok_tau=bok_tau,
        bok_clip=bok_clip,
        bok_uniform_mix=bok_uniform_mix,
        collapse_threshold=collapse_threshold,
        low_var_threshold=low_var_threshold,
        bok_tau_init=bok_tau_init,
        bok_tau_final=bok_tau_final,
        global_step=global_step,
        total_steps=total_steps,
        answer_scores=answer_scores,
    )

    bsz, seq_len = token_level_rewards.shape
    valid_lengths = response_mask.long().sum(dim=-1)
    safe_lengths = valid_lengths.clamp_min(1).to(dtype=token_level_rewards.dtype)
    outcome_1d = (outcome_advantages * response_mask).sum(dim=-1) / safe_lengths
    advantages = outcome_advantages.clone()

    step_lambda = float(os.environ.get("BOK_STEP_WEIGHT", os.environ.get("BOK_STEP_LAMBDA", "0.2")))
    if step_lambda <= 0:
        return advantages * response_mask, advantages * response_mask

    step_clip = float(os.environ.get("BOK_STEP_CLIP", "2.5"))
    total_clip = float(os.environ.get("BOK_HYBRID_CLIP", os.environ.get("BOK_STEP_TOTAL_CLIP", "4.0")))
    step_low_var_threshold = float(os.environ.get("BOK_STEP_LOW_VAR_THRESHOLD", "1e-6"))
    step_reward_eps = float(os.environ.get("BOK_STEP_REWARD_EPS", "1e-12"))
    max_semantic_steps = int(os.environ.get("BOK_STEP_MAX_SEMANTIC_STEPS", "64"))
    gate_mode = os.environ.get("BOK_STEP_GATE", os.environ.get("BOK_STEP_GATE_MODE", "answer_soft")).strip().lower()
    step_min_gate = float(os.environ.get("BOK_STEP_MIN_GATE", os.environ.get("BOK_STEP_GATE_FLOOR", "0.2")))
    outcome_weight = float(os.environ.get("BOK_STEP_OUTCOME_WEIGHT", "1.0"))
    exclude_final = os.environ.get("BOK_STEP_EXCLUDE_FINAL", "1").lower() in ("1", "true", "yes")

    has_explicit_mask = point_step_mask is not None and tuple(point_step_mask.shape) == tuple(token_level_rewards.shape)
    if has_explicit_mask:
        point_mask = (point_step_mask > 0) & (response_mask > 0)
    else:
        point_mask = (token_level_rewards.abs() > step_reward_eps) & (response_mask > 0)

    sample_positions: list[list[int]] = []
    n_point_positions = 0
    n_no_point_positions = 0
    max_observed_steps = 0
    for row_idx in range(bsz):
        answer_pos = max(int(valid_lengths[row_idx].item()) - 1, 0)
        positions = point_mask[row_idx].nonzero(as_tuple=True)[0].tolist()
        if exclude_final:
            positions = [int(pos) for pos in positions if int(pos) < answer_pos]
        else:
            positions = [int(pos) for pos in positions if int(pos) <= answer_pos]
        positions.sort()
        if max_semantic_steps > 0:
            positions = positions[:max_semantic_steps]
        if not positions:
            n_no_point_positions += 1
        n_point_positions += len(positions)
        max_observed_steps = max(max_observed_steps, len(positions))
        sample_positions.append(positions)

    if n_point_positions == 0:
        _last_step = getattr(compute_bok_grpo_step_advantage, "_last_logged_step", -1)
        if global_step != _last_step:
            compute_bok_grpo_step_advantage._last_logged_step = global_step
            print(
                f"[BoK-GRPO-Step] batch={bsz} step={global_step}/{total_steps} "
                f"no point-step positions found; fallback=outcome_only explicit_mask={has_explicit_mask}"
            )
        return advantages * response_mask, advantages * response_mask

    id2indices: dict = defaultdict(list)
    for row_idx in range(bsz):
        id2indices[index[row_idx]].append(row_idx)

    # Per-step signal source: when an explicit `point_step_value` is provided
    # (progress / stoptiming arms), use it as the step signal; otherwise fall
    # back to the placed token rewards (legacy `pointhit` behaviour).
    _use_step_value = (
        point_step_value is not None
        and tuple(point_step_value.shape) == tuple(token_level_rewards.shape)
    )
    step_value_src = point_step_value if _use_step_value else token_level_rewards

    step_adv_values = [
        torch.zeros(len(positions), dtype=token_level_rewards.dtype, device=token_level_rewards.device)
        for positions in sample_positions
    ]
    n_step_values = 0
    n_low_var_steps = 0

    for _, indices in id2indices.items():
        group_max_steps = max((len(sample_positions[row_idx]) for row_idx in indices), default=0)
        for step_idx in range(group_max_steps):
            present = [row_idx for row_idx in indices if len(sample_positions[row_idx]) > step_idx]
            if len(present) < 2:
                continue
            rewards = torch.stack([
                step_value_src[row_idx, sample_positions[row_idx][step_idx]]
                for row_idx in present
            ])
            rewards = torch.nan_to_num(rewards, nan=0.0, posinf=1.0, neginf=0.0)
            reward_std = rewards.std() if len(present) > 1 else torch.tensor(0.0, device=token_level_rewards.device)
            if torch.isfinite(reward_std) and reward_std.item() > step_low_var_threshold:
                normalized = (rewards - rewards.mean()) / (reward_std + eps)
            else:
                batch_present = [row_idx for row_idx in range(bsz) if len(sample_positions[row_idx]) > step_idx]
                batch_rewards = torch.stack([
                    step_value_src[row_idx, sample_positions[row_idx][step_idx]]
                    for row_idx in batch_present
                ]) if len(batch_present) >= 2 else rewards
                batch_rewards = torch.nan_to_num(batch_rewards, nan=0.0, posinf=1.0, neginf=0.0)
                batch_std = batch_rewards.std() if len(batch_present) > 1 else torch.tensor(0.0, device=token_level_rewards.device)
                if torch.isfinite(batch_std) and batch_std.item() > step_low_var_threshold:
                    normalized = (rewards - batch_rewards.mean()) / (batch_std + eps)
                else:
                    normalized = torch.zeros_like(rewards)
                    n_low_var_steps += 1
            if step_clip > 0:
                normalized = normalized.clamp(-step_clip, step_clip)
            for local_idx, row_idx in enumerate(present):
                step_adv_values[row_idx][step_idx] = normalized[local_idx]
                n_step_values += 1

    if gate_mode == "none":
        gates = torch.ones_like(outcome_1d)
    elif gate_mode in ("positive_outcome", "traj_positive"):
        gates = (outcome_1d > 0).to(dtype=token_level_rewards.dtype)
    elif gate_mode in ("outcome_sigmoid", "traj_soft"):
        gates = torch.sigmoid(outcome_1d)
    elif gate_mode in ("answer_hard", "hard_answer"):
        if answer_scores is not None:
            gates = (answer_scores.to(dtype=token_level_rewards.dtype) >= 0.5).to(dtype=token_level_rewards.dtype)
        else:
            gates = (outcome_1d > 0).to(dtype=token_level_rewards.dtype)
    else:
        # answer_soft / soft_count_closeness / answer use the available answer score.
        # In StepCount this is usually binary or soft-decayed; min_gate preserves
        # a small amount of point learning for answer-wrong but visually useful traces.
        if answer_scores is not None:
            gates = torch.nan_to_num(answer_scores.to(dtype=token_level_rewards.dtype), nan=0.0, posinf=1.0, neginf=0.0)
            gates = gates.clamp(0.0, 1.0)
        else:
            gates = torch.sigmoid(outcome_1d)
        if step_min_gate > 0:
            gates = step_min_gate + (1.0 - step_min_gate) * gates

    aux_components = []
    n_active_spans = 0
    # Build the aux contribution into a separate tensor so we can re-center it
    # per response (bias fix) WITHOUT touching the outcome broadcast base.
    aux_tensor = torch.zeros_like(token_level_rewards)
    rows_with_spans = []
    for row_idx, positions in enumerate(sample_positions):
        if not positions:
            continue
        previous_end = 0
        row_has_span = False
        for step_idx, pos in enumerate(positions):
            start = previous_end
            end = min(pos + 1, seq_len)
            if end <= start:
                previous_end = end
                continue
            aux = step_lambda * gates[row_idx] * step_adv_values[row_idx][step_idx]
            aux = torch.nan_to_num(aux, nan=0.0, posinf=step_clip, neginf=-step_clip)
            aux_tensor[row_idx, start:end] = aux
            aux_components.append(aux.detach())
            n_active_spans += 1
            row_has_span = True
            previous_end = end
        if row_has_span:
            rows_with_spans.append(row_idx)

    # Per-response AUX re-centering (bias fix): subtract the masked mean of the
    # aux contribution so Sum(aux) over each response == 0. This keeps the
    # trajectory-broadcast outcome advantage (base) intact while removing the
    # non-zero-mean bias that gated per-step credit would otherwise inject
    # (mirrors compute_grpo_step_level_advantage's per-response centering, but
    # applied to AUX only — NOT to base+aux, which would erase the outcome signal).
    # Default: re-center only for the new value-based arms (progress/stoptiming),
    # so the legacy `pointhit` path stays byte-identical. Explicit env overrides.
    _aux_recenter_env = os.environ.get("BOK_STEP_AUX_RECENTER", "auto").lower()
    if _aux_recenter_env == "auto":
        _aux_recenter = bool(_use_step_value)
    else:
        _aux_recenter = _aux_recenter_env in ("1", "true", "yes")
    for row_idx in rows_with_spans:
        rmask = response_mask[row_idx]
        n_tok = rmask.sum()
        if n_tok > 0:
            if _aux_recenter:
                aux_mean = (aux_tensor[row_idx] * rmask).sum() / n_tok
                aux_tensor[row_idx] = (aux_tensor[row_idx] - aux_mean) * rmask
            base = outcome_weight * outcome_1d[row_idx]
            advantages[row_idx] = (base + aux_tensor[row_idx]) * rmask

    advantages = torch.nan_to_num(advantages, nan=0.0, posinf=total_clip if total_clip > 0 else bok_clip, neginf=-(total_clip if total_clip > 0 else bok_clip))
    advantages = advantages * response_mask
    if total_clip > 0:
        advantages = advantages.clamp(-total_clip, total_clip) * response_mask

    _last_step = getattr(compute_bok_grpo_step_advantage, "_last_logged_step", -1)
    if global_step != _last_step:
        compute_bok_grpo_step_advantage._last_logged_step = global_step
        if aux_components:
            _aux_stack = torch.stack(aux_components)
            aux_log_mean = _aux_stack.mean().item()
            aux_std = _aux_stack.std().item() if _aux_stack.numel() > 1 else 0.0
        else:
            aux_log_mean = 0.0
            aux_std = 0.0
        gate_mean = gates.mean().item() if gates.numel() > 0 else 0.0
        print(
            f"[BoK-GRPO-Step] batch={bsz} step={global_step}/{total_steps} "
            f"lambda={step_lambda:.3f} gate={gate_mode} min_gate={step_min_gate:.3f} gate_mean={gate_mean:.4f} "
            f"exclude_final={exclude_final} explicit_mask={has_explicit_mask} point_positions={n_point_positions} "
            f"no_point_rows={n_no_point_positions}/{bsz} max_semantic_step={max_observed_steps} "
            f"step_values={n_step_values} low_var_steps={n_low_var_steps} active_spans={n_active_spans} "
            f"aux_mean={aux_log_mean:.4f} aux_std={aux_std:.4f} "
            f"adv_mean={advantages.mean().item():.4f} adv_std={advantages.std().item():.4f}"
        )

    returns = advantages.clone()
    return advantages, returns



def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2sum = {}
    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    for idx in id2score:
        id2sum[idx] = torch.sum(torch.tensor(id2score[idx]))

    for i in range(bsz):
        sample_num = len(id2score[index[i]])
        assert sample_num > 1, "RLOO needs rollout.n > 1."
        baseline = (id2sum[index[i]] - scores[i]) / (sample_num - 1)
        scores[i] = scores[i] - baseline

    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns


@torch.no_grad()
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    returns = torch.zeros_like(token_level_rewards)
    running_return = 0
    for t in reversed(range(token_level_rewards.shape[1])):
        running_return = token_level_rewards[:, t] + gamma * running_return
        returns[:, t] = running_return
        # Reset after EOS
        running_return = running_return * response_mask[:, t]

    advantages = VF.masked_whiten(returns, response_mask)
    return advantages, returns


@torch.no_grad()
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    scores = token_level_rewards.sum(dim=-1) - reward_baselines
    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns


def compute_rewards(
    token_level_scores: torch.Tensor,
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    kl_ratio: float,
) -> torch.Tensor:
    kl = log_probs - ref_log_probs
    return token_level_scores - kl * kl_ratio


def compute_policy_loss(
    old_log_probs: torch.Tensor,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
    clip_ratio_dual: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the policy loss.

    Adapted from https://github.com/huggingface/trl/blob/v0.15.0/trl/trainer/ppo_trainer.py#L568

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        clip_ratio_low: (float)
            The lower clip range used in PPO. See https://arxiv.org/abs/1707.06347
        clip_ratio_high: (float)
            The higher clip range used in DAPO. See https://arxiv.org/pdf/2503.14476
        clip_ratio_dual: (float)
            The dual clip range used in Dual-clip PPO. See https://arxiv.org/pdf/1912.09729

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac_higher: (float)
            a float number indicating the fraction of policy gradient loss being clipped to a higher value
        pg_clipfrac_lower: (float)
            a float number indicating the fraction of policy gradient loss being clipped to a lower value
        ppo_kl: (float)
            a float number indicating the mean KL divergence between the old policy and the new policy

    """
    import os
    negative_approx_kl = log_probs - old_log_probs  # token-level log-ratio (KL metric uses this)

    # ---- Importance-sampling granularity (GSPO / GSPO-token), env-gated ----
    # POLICY_LOSS_IS_LEVEL: "token" (default; standard GRPO/PPO token-level IS)
    #   | "sequence"       (GSPO, arXiv 2507.18071): geometric-mean sequence ratio,
    #                       sequence-level gradient (same ratio on every token).
    #   | "sequence_token" (GSPO-token): sequence-magnitude ratio but PER-TOKEN
    #                       gradient via stop-grad trick.
    # Why for long-horizon 21-turn dense: token-level IS variance grows ~L*v with
    # length; the geometric-mean sequence ratio is ~v/L -> far lower variance, and
    # it aligns the IS unit with our TRAJECTORY-level BoK reward/advantage (GSPO's
    # core "match objective unit to reward unit" principle). "sequence_token" is
    # preferred for us: it does NOT drop whole trajectories (preserves the gradient
    # BoK deliberately selected) and stays valid when advantages are PER-TURN (Arm C).
    # NOTE: pure "sequence" mode expects much smaller clip_ratio (~3e-3); tune before use.
    _is_level = os.environ.get("POLICY_LOSS_IS_LEVEL", "token").strip().lower()
    if _is_level in ("sequence", "gspo", "sequence_token", "gspo_token"):
        # fp32 reduction for numerical stability over long (21-turn, ~2k-token) seqs
        _seq_tok = response_mask.sum(dim=-1).clamp(min=1.0).float()              # (bs,)
        _seq_log_ratio = (negative_approx_kl.float() * response_mask.float()).sum(dim=-1) / _seq_tok  # (bs,)
        if _is_level in ("sequence", "gspo"):
            effective_log_ratio = _seq_log_ratio.unsqueeze(-1).expand_as(negative_approx_kl)
        else:  # sequence_token (GSPO-token): seq magnitude + per-token gradient
            effective_log_ratio = (
                _seq_log_ratio.detach().unsqueeze(-1)
                + (negative_approx_kl - negative_approx_kl.detach())
            )
    else:
        effective_log_ratio = negative_approx_kl                                  # token-level (default)

    # Clamp log-ratio before exp to avoid inf/NaN.
    # NOTE: keep KL metrics computed from the unclamped token-level log-ratio.
    # see: https://github.com/pytorch/pytorch/issues/10729
    safe_log_ratio = torch.clamp(effective_log_ratio, min=-20.0, max=20.0)
    # exp in fp32 to avoid bf16/fp16 overflow
    ratio = torch.exp(safe_log_ratio.float())
    clipped_ratio = torch.exp(
        torch.clamp(effective_log_ratio, np.log(1.0 - clip_ratio_low), np.log(1.0 + clip_ratio_high))
    )

    pg_loss = -advantages * ratio
    pg_loss2 = -advantages * clipped_ratio
    pg_loss3 = -advantages * clip_ratio_dual

    clipped_pg_loss_higher = torch.max(pg_loss, pg_loss2)  # clip if pg_loss < pg_loss2
    pg_clipfrac_higher = (pg_loss < pg_loss2).float()
    clipped_pg_loss_lower = torch.min(clipped_pg_loss_higher, pg_loss3)  # clip if pg_loss > pg_loss3 and adv < 0
    final_pg_loss = torch.where(advantages < 0, clipped_pg_loss_lower, clipped_pg_loss_higher)
    pg_clipfrac_lower = (clipped_pg_loss_higher > pg_loss3).float() * (advantages < 0).float()

    final_pg_loss = VF.masked_mean(final_pg_loss, response_mask)
    pg_clipfrac_higher = VF.masked_mean(pg_clipfrac_higher, response_mask)
    pg_clipfrac_lower = VF.masked_mean(pg_clipfrac_lower, response_mask)
    ppo_kl = VF.masked_mean(-negative_approx_kl, response_mask)
    return final_pg_loss, pg_clipfrac_higher, pg_clipfrac_lower, ppo_kl


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    action_mask: torch.Tensor,
    cliprange_value: float,
) -> Tuple[torch.Tensor, float]:
    """Compute the value loss.

    Adapted from https://github.com/huggingface/trl/blob/v0.15.0/trl/trainer/ppo_trainer.py#L556

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        action_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange_value: (float)
            The clip range for value net used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = torch.clamp(vpreds, values - cliprange_value, values + cliprange_value)
    vf_loss1 = torch.square(vpreds - returns)
    vf_loss2 = torch.square(vpredclipped - returns)
    vf_loss = 0.5 * VF.masked_mean(torch.max(vf_loss1, vf_loss2), action_mask)  # clip if vf_loss1 < vf_loss2
    vf_clipfrac = VF.masked_mean((vf_loss1 < vf_loss2).float(), action_mask)
    return vf_loss, vf_clipfrac


def compute_kl(log_probs: torch.FloatTensor, ref_log_probs: torch.FloatTensor, kl_penalty: str) -> torch.Tensor:
    """Compute KL divergence given log_probs and ref_log_probs.

    Adapted from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L1150

    Args:
        log_probs: torch.Tensor
        ref_log_probs: torch.Tensor
        kl_penalty: str

    Returns:
        kl_div: torch.Tensor

    """
    log_probs, ref_log_probs = log_probs.float(), ref_log_probs.float()
    if kl_penalty == "kl":
        return log_probs - ref_log_probs

    if kl_penalty == "abs":
        return (log_probs - ref_log_probs).abs()

    if kl_penalty == "mse":
        return 0.5 * (log_probs - ref_log_probs).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # URL http://joschu.net/blog/kl-approx.html
    if kl_penalty == "low_var_kl":
        kl = ref_log_probs - log_probs
        kld = (kl.exp() - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        return F.kl_div(ref_log_probs, log_probs, log_target=True, reduction="none").sum(-1)

    raise NotImplementedError(f"Unknown KL penalty: {kl_penalty}.")
