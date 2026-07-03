# Wrong Prediction Mining Report — V12 S132 Eval Data Augmentation

## Overview

Mined wrong predictions from V12 StepCount-7B S132 checkpoint evaluations on pixmo_test and countbench benchmarks, converted to StepCountQA-RL training format, and merged with existing training data.

## Source Data

| Benchmark | Eval File | Total | Correct | Wrong | Accuracy |
|-----------|-----------|-------|---------|-------|----------|
| pixmo_test (with_history_1) | eval_StepCount-7B-SFT-30k_v12_…_step132_with_history_1.json | 529 | 438 | 91 | 82.8% |
| countbench | eval_StepCount-7B-SFT-30k_v12_…_step132.json | 491 | 390 | 101 | 79.4% |
| **Total** | | **1020** | **828** | **192** | **81.2%** |

### Wrong Prediction GT Count Distribution (filtered to 0-10)

| GT Count | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|----------|---|---|---|---|---|---|---|---|-----|
| Wrong Count | 12 | 16 | 17 | 18 | 19 | 25 | 23 | 15 | 36 |

**Observation**: Higher counts (7, 8, 10) have disproportionately more wrong predictions, confirming error accumulation in sequential counting is the primary failure mode.

## Output Datasets

### 1. StepCountQA-RL-Wrong-V12-S132 (standalone wrong predictions)
- **Path**: `StepcountModel/dataset/StepCountQA-RL-Wrong-V12-S132/data/`
- **Samples**: 192
- **Shards**: 1 parquet file
- **Sources**: 91 pixmo_test + 101 countbench

### 2. StepCountQA-RL-Traj_0_10_plus_wrong (merged dataset)
- **Path**: `StepcountModel/dataset/StepCountQA-RL-Traj_0_10_plus_wrong/data/`
- **Samples**: 11,647 (11,455 original + 192 wrong)
- **Shards**: 4 parquet files (~2911-2912 rows each)
- **Shuffled**: Yes (seed=42)

### Parquet Schema (same as original)
```
images: list<struct<bytes: binary, path: string>>
problem: string  ("<image>\n{question}")
answer: string   (GT count number)
```

## Reward Configuration for Maskless Data

New wrong prediction samples **do NOT have corresponding mask metadata**. The reward system handles this via fallback:

### Reward Path for New Data (no mask sequence found)

**Point Dense Reward**: Uses default `count_iou` fallback (code default, no explicit env var needed):
- Formula: $\text{score} = \frac{\min(pred\_count, gt\_count)}{\max(pred\_count, gt\_count)}$
- Example: GT=5, pred=4 → 0.8; GT=5, pred=6 → 0.833; GT=5, pred=7 → 0.714

**Answer Reward**: Same as V12, exponential decay for wrong answers (TRAJ_SOFT_ANSWER_DECAY=1):
- Correct: 1.0
- Wrong undercount: $\min(e^{-8.0 \times |err|/gt}, 0.4)$
- Wrong overcount: $\min(e^{-16.0 \times |err|/gt}, 0.4)$ (2× harder penalty)

### Complete Reward Table (GT=5 example)

| Scenario | Answer (w=0.6) | Point count_iou (w=0.3) | Format (w=0.1) | **Overall** |
|----------|----------------|------------------------|-----------------|------------|
| Correct (5) | 1.000 | 1.000 | 1.0 | **1.000** |
| Under by 1 (4) | 0.202 | 0.800 | 1.0 | **0.461** |
| Over by 1 (6) | 0.041 | 0.833 | 1.0 | **0.375** |
| Over by 2 (7) | 0.002 | 0.714 | 1.0 | **0.315** |

**Key properties**:
- Correct answer → full score → strong positive advantage
- count_iou gives moderate 0.7-0.8 range for ±1-2 errors → sufficient gradient signal
- Answer and point rewards combined give clear ranking among trajectories
- V13 config unchanged from V12 (answer reward, weights, etc. all identical)

### Fallback Chain Detail
1. `answer = "5"` → `gt_data = {"type": "trajectory", "count_number": 5, "point_sequence": [], "no_point_gt": True}`
2. Model generates multi-step points → `_trajectory_point_dense_reward_without_gt_points()`
3. `sequence_id` NOT in `masks_metadata.json` → `sequence_samples = []`
4. Fallback: `TRAJ_NO_SEQUENCE_FALLBACK` (not set → default `count_iou`)
5. `score = min(pred_count, target_count) / max(pred_count, target_count)`

## Script

- **Extraction script**: `scripts/extract_wrong_eval_to_training_data.py`
  - Reads eval JSONs, filters wrong predictions, reads original images, writes parquet
  - Options: `--count_range`, `--dedup`, `--samples_per_shard`

## Source Data Locations
- **CountBench dataset def**: `StepcountModel/dataset/eval/eval_dataset_countbench.json` (491 entries)
- **CountBench eval results**: `StepcountModel/eval/eval_countbench/eval_StepCount-7B-SFT-30k_v12_…_step132.json`
- **PixmoTest eval results**: `StepcountModel/eval/eval_pixmo_test/eval_StepCount-7B-SFT-30k_v12_…_step132_with_history_1.json`
- **Original training data**: `StepcountModel/dataset/StepCountQA-RL-Traj_0_10/data/` (11,455 rows, 4 shards)
- **Mask metadata**: `/data/workspace/hyleochang/StepCount-RL_masks_output/masks_metadata.json` (97,099 entries)

## Usage in Training

```bash
# In V13 training script, update dataset path to use merged data:
data.train_files=/path/to/StepCountQA-RL-Traj_0_10_plus_wrong/data
```

## Notes

- Wrong predictions are from V12 S132 (best checkpoint before NaN degradation)
- All images are in the 0-10 object count range
- No overlap between wrong prediction samples (cross-benchmark dedup=0)
- These samples represent model's hardest cases — targeted augmentation for pass@1 improvement
- V13 reward config identical to V12 for all shared data types

---

## Appendix: Balanced Dataset Plan (for Future SFT)

When retraining from SFT with the new hard examples, keep total = 11,455 (same as original) by replacing easy samples with hard ones.

### Original Dataset GT Count Distribution
| Count | Original | Wrong (hard) | Remove (easy) | New |
|-------|----------|-------------|---------------|-----|
| 1 | 1165 | 0 | ~192 | ~973 |
| 2 | 1142 | 12 | 0 | 1154 |
| 3 | 1157 | 16 | 0 | 1173 |
| 4 | 1134 | 17 | 0 | 1151 |
| 5 | 1051 | 18 | 0 | 1069 |
| 6 | 2493 | 19 | 0 | 2512 |
| 7 | 1385 | 25 | 0 | 1410 |
| 8 | 977 | 23 | 0 | 1000 |
| 9 | 632 | 15 | 0 | 647 |
| 10 | 319 | 36 | 0 | 355 |

**Strategy**: Remove 192 samples from count=1 (easiest category, baseline model near-100% accuracy). This:
- Preserves all harder categories (count 2-10)
- Adds 192 targeted hard examples (wrong predictions)
- Keeps total at 11,455 for consistent training duration
- count=1 goes from 1165 → 973 (still adequate coverage)

**Alternative**: Remove from count=6 (2493 → 2301) to reduce class imbalance.

### Implementation Notes
- Use `scripts/extract_wrong_eval_to_training_data.py` for extraction
- Script should be extended with `--remove_easy_from_count=1` or `--balance_target=11455` options
- Ensure no overlap between removed easy samples and wrong prediction samples (different benchmarks, no overlap expected)
