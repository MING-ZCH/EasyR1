# BOK-GRPO: Best-of-K Group Relative Policy Optimization  
## Interleaved Point-to-Count for VLM Object Counting

---

# Slide 1: Overview & Motivation

## The Counting Challenge
- **Task**: Visual object counting (1–50 objects) with explicit spatial grounding
- **Gap**: VLMs achieve pass@32 ≈ 95% but pass@1 ≈ 79% → rich latent capability trapped in stochastic sampling
- **Goal**: Collapse pass@32 performance into pass@1 via RL training

## Key Insight
> Human counters point to each object sequentially, shifting attention explicitly.  
> We replicate this with **interleaved multi-turn point-to-count** — the model points one object per turn, receives visual feedback (red dot on image), then continues.

## Our Contributions
1. **Interleaved Point-to-Count Paradigm** — explicit dynamic attention via multi-turn VLM interaction  
2. **BOK-GRPO Algorithm** — best-of-K advantage weighting to maximize correct-path probability  
3. **Dense Mask Point Reward** — spatial grounding reward without explicit point GT annotations  
4. **Trajectory-level RL Training** — real-time visual feedback within rollout generation

---

# Slide 2: Interleaved Point-to-Count Paradigm

## Architecture

```
Turn 1: [Original Image] + Question
  → Model: <think>...</think> <point>{"point_2d": [x,y], "label": "person", "count_number": "1"}</point>
  → Environment: Draws red dot at (x,y) on image

Turn 2: [Image + 1 red dot] + "Continue counting..."
  → Model: <think>observe red dot, find next...</think> <point>{...count_number: "2"}</point>
  → Environment: Draws another red dot

Turn 3: [Image + 2 red dots] + "Continue counting..."
  → Model: <think>all marked</think> <answer>2</answer>
  → STOP: trajectory complete
```

## Design Principles

| Feature | Design Choice | Rationale |
|---------|--------------|-----------|
| **One point per turn** | Sequential counting | Prevents duplicate/skip errors |
| **Visual feedback** | Red dots on image | Explicit attention memory |
| **Top-left to bottom-right** | Spatial ordering hint | Reduces confusion in dense scenes |
| **Max turns = 11** | Hard limit | Prevents infinite loops |
| **Stop condition** | `</answer>` tag generation | Natural language stop signal |

## Prompts

**First Turn**: "Count objects one-by-one from top-left to bottom-right by outputting `<point>` tags. Each pointed object will be marked with a red dot."

**Subsequent Turns**: "Observe the red dots (already counted). If unmarked objects remain, point the next one. If all counted, output `<answer>`."

---

# Slide 3: BOK-GRPO Algorithm — Core Idea

## Problem with Standard GRPO

Standard GRPO computes group-level z-normalized advantages:
$$A_i^{\text{GRPO}} = \frac{r_i - \mu_g}{\sigma_g}$$

**Failure mode**: When most rollouts have similar rewards (e.g., 14/16 correct), $\sigma_g \approx 0$ → gradient vanishes. The model cannot learn from its own best trajectories.

## BOK-GRPO Core Formulation

Replace z-normalization with **softmax-temperature weighting**:

$$w_i = \text{softmax}\left(\frac{r_i - \mu_g}{\sigma_g \cdot \tau}\right)_i$$

$$A_i^{\text{BoK}} = \left(w_i - \frac{1}{K}\right) \times K$$

**Interpretation**:
- $\tau \to 0$: Pure best-of-K (only highest-reward trajectory gets positive advantage)
- $\tau \to \infty$: Uniform weights (standard GRPO equivalent)
- $\tau \in [0.3, 0.7]$: Smooth interpolation — concentrates gradient on top trajectories

## Temperature Annealing (Cosine Schedule)

$$\tau(t) = \tau_{\text{final}} + (\tau_{\text{init}} - \tau_{\text{final}}) \times \frac{1 + \cos(\pi \cdot t/T)}{2}$$

- **Early training** ($\tau = 0.5$): Exploratory, distributes credit across multiple good trajectories
- **Late training** ($\tau = 0.3$): Selective, focuses on the single best trajectory per group

---

# Slide 4: BOK-GRPO — 4-Way Routing

## Adaptive Routing Per Prompt Group

For each prompt group (K = 16 rollouts), compute `pass_rate` and `group_std`, then route:

```
┌─────────────────────────────────────────────────┐
│              Per-Group Routing                   │
│                                                  │
│  group_std ≤ ε? ──YES──► ROUTE 1: LowVar        │
│       │ NO                  (batch-level DrGRPO) │
│       ▼                                         │
│  all K correct? ──YES──► ROUTE 2: AllCorrect     │
│       │ NO                  (zero gradient)      │
│       ▼                                         │
│  pass_rate > θ? ──YES──► ROUTE 3: EasyDrGRPO     │
│       │ NO                  (z-norm, clip±2.5)   │
│       ▼                                         │
│  DEFAULT ─────────────► ROUTE 4: BoK Softmax     │
│                          (softmax advantages)    │
└─────────────────────────────────────────────────┘
```

| Route | Condition | Advantage Formula | Purpose |
|-------|-----------|-------------------|---------|
| **LowVar** | $\sigma_g \leq \epsilon$ | Batch-level DrGRPO | Avoid 0/0 division |
| **AllCorrect** | All $r_i > \theta_{score}$ | $A_i = 0$ | Skip trivially solved |
| **EasyDrGRPO** | pass_rate $> \theta_{easy}$ | $\text{clip}\left(\frac{r_i - \mu_g}{\sigma_g}, \pm 2.5\right)$ | Standard RL for easy |
| **BoK Softmax** | Default | $(w_i - 1/K) \times K$ | Focus on best trajectory |

## Typical Routing Distribution (mixed data)

| Route | V12 (easy data) | V16 (mixed data) |
|-------|----------------|------------------|
| LowVar | 14.8% | 7.2% |
| AllCorrect | 30.0% | 15.3% |
| EasyDrGRPO | 29.8% | 46.8% |
| **BoK Softmax** | **25.4%** | **30.8%** |

*Routing distribution is data-driven — harder data → fewer AllCorrect, more EasyDrGRPO/BoK.*

---

# Slide 5: BOK-GRPO — Safety Mechanisms

## Numerical Stability Suite

| Mechanism | Implementation | Purpose |
|-----------|---------------|---------|
| **NaN Guard** | `nan_to_num(scores, nan=0)` | Prevent NaN propagation |
| **Logit Cap** | `clamp(logits, ±12.0)` | Prevent softmax overflow |
| **τ-Adaptive** | $\tau_{eff} = \max(\tau, \frac{\max|z|}{C_{logit}}, \tau_{min})$ | Dynamic τ floor based on z-score range |
| **Advantage Clip** | `clamp(A, ±BOK_CLIP)` | Bound final advantages |
| **Collapse Detection** | If $\max(w) > 0.9$: increase uniform mixing | Prevent single-trajectory domination |
| **Uniform Mixing** | $w = (1-\alpha) \cdot w_{softmax} + \alpha \cdot \frac{1}{K}$ | Gradient smoothing |

## KL Regularization

$$\mathcal{L}_{KL} = \beta_{KL} \cdot D_{KL}(\pi_\theta \| \pi_{ref})$$

- Prevents excessive policy drift from SFT reference
- $\beta_{KL} = 0.03$ (tuned for stability with LR = 1e-6)

## PPO-Style Clipping

$$\mathcal{L}_{clip} = -\min\left(r(\theta) A, \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon) A\right)$$

- $\epsilon_{high} = 0.28$ (asymmetric clipping — allows upward policy ratio)

---

# Slide 6: Dense Mask Point Reward

## Challenge
- Training data provides only: image + question + GT count number (N)
- No per-step GT point annotations during RL
- Need **dense per-turn reward** for each pointed location

## Solution: Mask-Based Spatial Matching

Pre-computed segmentation masks for all target objects in training images:

```
masks_metadata.json:
  sample_id → {
    image_path, mask_path,
    points_pixel: [(x,y)],     # mask center coordinates
    labels: ["person"],
    image_size: (W, H)
  }

Index structures:
  coord_index[(x,y)] → [matching masks]
  sequence_index[seq_id] → [{sample_id, turn_number, mask}]
```

## Per-Point Reward Calculation

```
For each predicted point (px, py):
  1. Find matching mask sequence via coordinate lookup
  2. Convert predicted point to pixel coordinates
  3. Check if point falls inside any GT mask:
  
     HIT (unused mask): reward = 1.0, mark mask as used
     HIT (duplicate):   reward = 0.0 (penalize re-counting)
     MISS:              reward = distance_decay(pred, nearest_mask)
```

## Distance Decay Shaping (for misses)

$$r_{miss} = \begin{cases}
1.0 & \text{if } d \leq d_{tol} \\
e^{-\alpha \cdot (d - d_{tol})} & \text{if } d_{tol} < d \leq d_{fail} \\
e^{-\alpha \cdot d_{fail}} \cdot e^{-\alpha_{pen} \cdot (d - d_{fail})} & \text{if } d > d_{fail}
\end{cases}$$

| Parameter | Value | Description |
|-----------|-------|-------------|
| $\alpha$ | 20.0 | Base decay steepness |
| $\alpha_{pen}$ | 50.0 | Penalty acceleration beyond failure threshold |
| $d_{tol}$ | 0.0 | Tolerance zone (inside mask boundary) |
| $d_{fail}$ | 0.02 | Rapid decay threshold (normalized by image size) |

---

# Slide 7: Trajectory Reward Composition

## Three-Component Weighted Reward

$$R_{overall} = w_a \cdot R_{answer} + w_p \cdot R_{point} + w_f \cdot R_{format}$$

| Component | Weight | Computation | Signal Type |
|-----------|--------|------------|-------------|
| **Answer** ($R_{answer}$) | 0.6 | Correct → 1.0; Wrong → soft decay | Terminal (sparse) |
| **Point** ($R_{point}$) | 0.3 | Mask-matching dense reward | Per-turn (dense) |
| **Format** ($R_{format}$) | 0.1 | Valid JSON + proper tags → 1.0 | Per-trajectory |

## Answer Reward (Soft Decay for Wrong Answers)

$$R_{answer} = \begin{cases}
1.0 & \text{if pred} = \text{GT} \\
\min\left(e^{-\alpha_{base} \cdot k \cdot \frac{|\text{pred} - \text{GT}|}{\max(\text{GT}, 1)}},\; 0.4\right) & \text{otherwise}
\end{cases}$$

**Asymmetric penalty**:
- Over-counting ($\text{pred} > \text{GT}$): $k = 2.0$ (hallucination — heavy penalty)
- Under-counting ($\text{pred} < \text{GT}, \text{GT} > 5$): $k = 1 + 0.5 \cdot \frac{\text{GT} - 5}{5}$ (graceful for large GT)
- Decay cap = 0.4 ensures minimum 0.48 gap between correct and wrong

## Dense Point Reward (Aggregated over trajectory)

$$R_{point} = \frac{\sum_{t=1}^{T} \mathbb{1}[\text{hit\_unused}_t]}{N_{target}} + \sum_{t: \text{miss}} \frac{r_{miss,t}}{N_{target}} - \lambda \cdot \frac{\max(0, T - N_{target})}{N_{target}}$$

- $\lambda = 0.3$ penalty for over-pointing (predicting more points than GT)
- Hit-unused contributes +1/N per correctly matched object
- Miss contributes shaped distance reward / N
- Duplicate hits contribute 0

## Consistency Check

If $\text{pred\_answer} \neq |\text{pred\_points}|$:
$$R_{answer} \mathrel{*}= (1 - \gamma_{cons}), \quad \gamma_{cons} = 0.5$$

*Penalizes inconsistency between stated count and actual number of points.*

---

# Slide 8: Training Pipeline

## End-to-End Trajectory RL Training

```
┌─────────────────────────────────────────────────────────┐
│                    RL Training Loop                      │
│                                                          │
│  1. Sample batch of images + questions                   │
│  2. Generate K=16 rollout trajectories per sample        │
│     ┌──────────────────────────────────────┐             │
│     │  For each rollout:                    │             │
│     │    Turn 1: Generate point             │             │
│     │    → Draw red dot on image            │             │
│     │    Turn 2: Generate point (with dots) │             │
│     │    → Draw another red dot             │             │
│     │    ...                                │             │
│     │    Turn T: Generate <answer>N</answer>│             │
│     └──────────────────────────────────────┘             │
│  3. Compute trajectory rewards:                          │
│     R = 0.6×answer + 0.3×point + 0.1×format             │
│  4. BOK-GRPO advantage computation:                      │
│     Route → {LowVar, AllCorrect, Easy, BoK}             │
│  5. PPO update with KL regularization                    │
│  6. Repeat                                               │
└─────────────────────────────────────────────────────────┘
```

## Key Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Base Model | Qwen2.5-VL-7B | Vision-Language Model |
| SFT Pre-training | 30k samples, 3537 steps | Supervised fine-tuning on counting data |
| Rollouts per prompt | K = 16 | Group size for advantage computation |
| Learning Rate | 1e-6 | Policy update rate |
| PPO Epochs | 2 | Updates per rollout batch |
| Clip Ratio (high) | 0.28 | Asymmetric PPO clipping |
| KL Coefficient | 0.03 | Reference policy regularization |
| τ Schedule | 0.5 → 0.3 (cosine) | BoK temperature annealing |
| Max Turns | 11 | Trajectory length limit |
| FSDP | Full Shard (4 GPUs) | Distributed training |

---

# Slide 9: Experimental Results

## Benchmark Performance (0-10 object range)

| Model | pixmo-test (529) | countbench (491) | Method |
|-------|-----------------|-----------------|--------|
| SFT Base | 75.6% | — | Supervised Only |
| V7-GRPO | 80.72% | — | Standard GRPO |
| **V12 BOK-GRPO** | **82.04%** | **79.43%** | BOK-GRPO (best) |
| V15 BOK-GRPO | 80.15% | 79.02% | Conservative LR |

## Key Findings

1. **BOK-GRPO > Standard GRPO**: V12 (82.04%) vs V7 (80.72%) = +1.32pp improvement
2. **+6.44pp over SFT base**: 75.6% → 82.04% from RL training alone
3. **Dense point reward enables stable training**: Format compliance maintained at 96.6%
4. **Zero NaN with proper eff_intensity tuning**: Stable training at 5.6e-7

## Learning Efficiency

| Metric | SFT Base | After RL (V12) |
|--------|---------|----------------|
| pass@1 | 75.6% | **82.04%** |
| pass@32 | 95.2% | — |
| **Gap** | **19.6pp** | **~13pp** |

*BOK-GRPO reduces the pass@1–pass@32 gap by ~33%.*

---

# Slide 10: Ablation — What Matters

## Critical Design Decisions

| Decision | Without | With | Impact |
|----------|---------|------|--------|
| Interleaved (multi-turn) | Single-turn pointing | Multi-turn sequential | Enables explicit attention tracking |
| Dense point reward | Answer-only (sparse) | Answer + point (dense) | Stable training, no reward hacking |
| Softmax advantage (BoK) | Z-norm (GRPO) | Softmax weighting | +1.32pp on pixmo-test |
| Mask-based matching | Distance-only | Mask hit/miss | Precise point reward |
| Consistency penalty | No penalty | 50% penalty | Reduces output inconsistency |
| Soft answer decay | Binary (0/1) | Exponential decay | Finer reward shaping |

## Temperature τ Sensitivity

| τ (fixed) | Effect |
|-----------|--------|
| 0.1 | Too aggressive — winner-take-all, high variance |
| 0.3 | Selective — focuses on top 2-3 trajectories |
| 0.5 | Balanced — distributes credit across top half |
| 1.0 | Nearly uniform — similar to standard GRPO |
| **0.5→0.3 cosine** | **Best — explores first, then specializes** |

---

# Slide 11: Future Directions

## Near-Term Goals
- **Dense scenes (GT 11-50)**: Current accuracy ~30%, target ≥50% on stepcount-500 benchmark
- **Pass@1 > 85%** on sparse scenes (GT ≤ 10)
- **Curriculum learning**: Start easy, progressively increase difficulty

## Algorithm Extensions
- **Per-turn process reward**: Assign advantage at each turn, not just trajectory-level
- **Adaptive routing thresholds**: Dynamic easy/hard boundary based on training progress
- **History-aware prompting**: Include recent pointing history in context window

## Scaling
- **Larger base models**: Qwen2.5-VL-72B with BOK-GRPO
- **Cross-domain transfer**: Apply interleaved paradigm to other spatial reasoning tasks
- **Real-time interactive counting**: Deploy with streaming visual feedback

---

# Appendix A: Algorithm Pseudocode

```python
def BOK_GRPO(rollouts, K=16, τ_init=0.5, τ_final=0.3, T_total=90):
    """Best-of-K GRPO advantage computation."""
    # 1. Compute trajectory rewards
    for group in rollouts.group_by_prompt():
        scores = [compute_reward(r) for r in group]  # R = 0.6*ans + 0.3*pt + 0.1*fmt
        
        # 2. Temperature annealing
        τ = τ_final + (τ_init - τ_final) * 0.5 * (1 + cos(π * t/T))
        
        # 3. Route decision
        σ_g = std(scores)
        pass_rate = mean(score > θ_score for score in scores)
        
        if σ_g ≤ ε:                           # LowVar
            A = score - batch_mean             # batch-level baseline
        elif all(score > θ_score):             # AllCorrect
            A = 0                              # skip trivially solved
        elif pass_rate > θ_easy:               # EasyDrGRPO
            A = clip((score - μ_g) / σ_g, ±2.5)  # standard z-norm
        else:                                  # BoK Softmax
            z = (score - μ_g) / σ_g
            τ_eff = max(τ, max(|z|)/C, τ_min)
            w = softmax(z / τ_eff)
            w = (1-α)*w + α/K                 # uniform mixing
            A = (w - 1/K) * K                 # BoK advantage
        
        # 4. Final clipping
        A = clamp(A, -BOK_CLIP, BOK_CLIP)
    
    # 5. PPO update with KL penalty
    L = -E[min(r(θ)*A, clip(r(θ), 1±ε)*A)] + β_KL * D_KL(π_θ || π_ref)
    return L
```

---

# Appendix B: Mask Point Reward Detail

```python
def compute_trajectory_point_reward(trajectory, masks_db):
    """Dense mask-based point reward for a multi-turn trajectory."""
    used_masks = set()
    total_score = 0
    
    for turn in trajectory.turns:
        pred_point = turn.parsed_point  # (x, y) normalized
        
        # Convert to pixel coordinates
        px, py = pred_point[0] * W, pred_point[1] * H
        
        # Find matching mask via spatial index
        candidates = masks_db.find_masks_near(px, py)
        
        hit = False
        for mask in candidates:
            if point_in_mask((px, py), mask.binary_mask):
                if mask.id in used_masks:
                    score = 0.0  # duplicate penalty
                else:
                    score = 1.0  # correct hit
                    used_masks.add(mask.id)
                hit = True
                break
        
        if not hit:
            # Distance-based shaping for misses
            d = min_distance_to_any_mask(px, py, candidates)
            d_norm = d / (W + H)
            if d_norm ≤ 0:
                score = 1.0
            elif d_norm ≤ 0.02:
                score = exp(-20 * d_norm)
            else:
                score = exp(-20 * 0.02) * exp(-50 * (d_norm - 0.02))
        
        total_score += score / GT_count
    
    # Over-pointing penalty
    extra = max(0, len(trajectory.points) - GT_count)
    penalty = 0.3 * extra / GT_count
    
    return clamp(total_score - penalty, 0, 1)
```
