# 11-50 Training Analysis & V2 Optimization

## 1. Task Summary

Three subtasks completed:
1. ✅ V14-V16 docs reviewed, 11-50 data + mask verified, V1 script generated
2. ✅ V1 training analyzed (Steps 0-4), compared with 0-10 paradigm, V2 script created
3. ✅ stepcount-500 packaged as parquet val dataset

## 2. Data Verification

### 11-50 Training Data (StepCountQA-RL-Traj_11_50)
- **Samples**: 9,508 (vs 0-10: 11,455)
- **Shards**: 4 parquet files
- **Schema**: `{images, problem, answer}` — matches 0-10 format ✅
- **Count Distribution**: 11(707) through 50(122), declining with count

### Mask Coverage (OLD metadata file, 428MB, 282K entries)
- **11-50 coverage**: 100% ✅
- **Training log confirms**: uses `sam_mask/.../masks_metadata.json`
- **Train mask stats**: hit_unused=55.2%, miss=33.7%, dup=11.1%

### stepcount-500 Val Dataset (NEW)
- **Source**: `eval/eval_stepcount_bench_500.json`
- **Output**: `dataset/stepcount-500/train.parquet` (141 MB, 500 rows)
- **Schema**: matches pixmo-test exactly (`images`, `problem`, `answer`)
- **Count range**: 11-50, evenly distributed (130/130/120/120 per decade)

## 3. V1 Training Analysis (Steps 0-4)

### Key Metrics

| Metric | Step 0 (val) | Step 1 | Step 2 | Step 3 | Step 4 |
|--------|-------------|--------|--------|--------|--------|
| Val answer_reward | 0.749 | — | — | — | — |
| Train answer_mean | — | 0.265 | 0.271 | 0.264 | 0.228 |
| Train point_mean | — | 0.446 | 0.438 | 0.478 | 0.407 |
| entropy_loss | — | 0.970 | 0.997 | 0.974 | 0.964 |
| kl_loss | — | 0.013 | 0.003 | 0.003 | 0.005 |
| format_fail% | — | 12.2% | 13.8% | 7.7% | 8.8% |
| turns_exceeded% | — | 12.0% | 13.7% | 7.7% | 8.8% |
| BoK easy_drgrpo | — | 144/1024 | 224/1024 | 240/1024 | 208/1024 |

### Critical Observations

1. **answer_mean ~0.26** (extremely low vs 0-10's 0.75) — SFT model struggles on 11-50
2. **Val (pixmo-test) is 0-10 range** — gives inflated 0.749 that doesn't reflect 11-50 training
3. **format_fail 8-14%** — almost entirely turns_exceeded (50 turns overflow)
4. **point_mean ~0.44** — decent but significantly lower than 0-10's ~0.94
5. **answer_mean declining** (0.265 → 0.228 by Step 4) — potential early collapse concern

## 4. V12 (0-10 best) vs V1 (11-50) Parameter Comparison

| Parameter | V12 (best) | V1 (11-50) | V2 (optimized) |
|-----------|-----------|------------|----------------|
| Data | 0_10 (11,455) | 11_50 (9,508) | 11_50 (9,508) |
| **Val dataset** | pixmo-test (0-10) | pixmo-test (0-10) ❌ | **stepcount-500 (11-50) ✅** |
| MAX_TURNS | 11 | 51 | 51 |
| **ppo_epochs** | **2** | 1 | **2** |
| LR | 1.5e-6 | 1.5e-6 | 1.5e-6 |
| max_grad_norm | 0.5 | 1.0 | 1.0 |
| **kl_coef** | **0.02** | 0.03 | **0.02** |
| BOK_CLIP | 3.0 | 4.0 | 4.0 |
| BOK_TAU_INIT | 0.7 | 0.5 | 0.5 |
| BOK_EASY_THRESHOLD | 0.75 | 0.50 | 0.50 |
| **ANSWER_WEIGHT** | 0.6 | 0.6 | **0.5** |
| **POINT_WEIGHT** | 0.3 | 0.3 | **0.4** |
| **save_freq** | **20** | 30 | **20** |
| val_freq | 15 | 15 | 15 |
| Time/step | ~40min | ~53min | ~53min (expected) |

## 5. V2 Optimization Rationale

### Change 1: Val dataset → stepcount-500
**Problem**: pixmo-test (0-10) gives val_answer=0.749 which doesn't reflect 11-50 training.
**Fix**: Use stepcount-500 (11-50 range, 500 samples) for monitoring actual 11-50 performance.

### Change 2: ppo_epochs 1→2
**Problem**: V1 used ppo_epochs=1, but V12 (best) used 2.
**Rationale**: With very low answer_mean (~0.26), the model needs more gradient steps per rollout to extract learning signal from sparse correct trajectories.

### Change 3: ANSWER_WEIGHT 0.6→0.5, POINT_WEIGHT 0.3→0.4
**Problem**: In 11-50, point accuracy is critical — each point error compounds over 11-50 steps.
**Rationale**: Dense scenes need stronger point supervision. Point error is the primary source of answer error.

### Change 4: save_freq 30→20
**Rationale**: Match V12. With 11-50's low answer_mean, we need finer checkpointing to catch early improvements.

### Change 5: kl_coef 0.03→0.02
**Rationale**: Match V12 (best). Lower KL penalty allows more exploration, important when base model has very low accuracy (0.26).

## 6. Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `dataset/eval/convert_stepcount500_to_parquet.py` | Created | Conversion script for stepcount-500 |
| `dataset/stepcount-500/train.parquet` | Created | 500 samples, 141MB, schema matches pixmo-test |
| `examples/..._v2.sh` | Created | Optimized 11-50 training script |

## 7. Next Steps

- [ ] Run V2 with `bash examples/qwen2_5_vl_7b_StepCount_11_50_grpo_interleaved_traj_v2.sh`
- [ ] Monitor stepcount-500 val metrics (expect initially lower than pixmo-test's 0.749)
- [ ] Compare V1 vs V2 at step 20+ checkpoint
- [ ] Consider curriculum: train 0-10 first → finetune 11-50 (instead of cold-start from SFT)
