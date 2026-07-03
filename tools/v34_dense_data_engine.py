#!/usr/bin/env python3
"""V34 Phase2: Dense data engine — LocateAnything-3B box→point pseudo-mask pipeline.

Produces RL-ready trajectories with pseudo point_sequence from LA-3B detections.
Does NOT modify existing datasets in-place; writes new parquet/json shards.

Usage:
  python3 tools/v34_dense_data_engine.py \\
    --images_dir /path/to/dense/images \\
    --la_model /apdcephfs_hldy2/.../LocateAnything-3B \\
    --output_dir /apdcephfs_hldy2/.../StepCountQA-RL-Traj_11_50_LA_pseudo \\
    --count_min 11 --count_max 50 \\
    --dry_run

Requires GPU pod with LocateAnything-3B installed (see eval/scripts/eval_supplement_v3.sh).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List


def box_center_to_point(box: List[float]) -> List[float]:
    """Convert [x1,y1,x2,y2] normalized box to center point."""
    x1, y1, x2, y2 = box[:4]
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


def build_pseudo_trajectory(
    image_path: str,
    question: str,
    boxes: List[List[float]],
    labels: List[str],
) -> Dict[str, Any]:
    points = [box_center_to_point(b) for b in boxes]
    seq = [
        {"point_2d": p, "label": labels[i] if i < len(labels) else "object"}
        for i, p in enumerate(points)
    ]
    return {
        "image": image_path,
        "question": question,
        "answer": str(len(points)),
        "type": "trajectory",
        "point_sequence": seq,
        "pseudo_label_source": "LocateAnything-3B",
        "bbox_source": boxes,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_manifest", required=True, help="JSON list of {image_path, question}")
    ap.add_argument("--la_model", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--count_min", type=int, default=11)
    ap.add_argument("--count_max", type=int, default=50)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    manifest = json.load(open(args.images_manifest, encoding="utf-8"))
    os.makedirs(args.output_dir, exist_ok=True)

    out_rows: List[Dict] = []
    for row in manifest:
        n = row.get("expected_count")
        if n is not None and (n < args.count_min or n > args.count_max):
            continue
        if args.dry_run:
            out_rows.append(
                {
                    "image": row.get("image_path"),
                    "status": "dry_run_skip_la_inference",
                }
            )
            continue
        # LA-3B inference hook — implement on pod using eval_supplement_v3 LA runner
        raise NotImplementedError(
            "Run on GPU pod: integrate LocateAnything-3B detect() from eval/scripts/eval_supplement_v3.sh"
        )

    out_path = os.path.join(args.output_dir, "pseudo_manifest.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "la_model": args.la_model,
                "count_range": [args.count_min, args.count_max],
                "n_rows": len(out_rows),
                "rows": out_rows[:100],
                "next_step": "Convert to HF parquet + register pseudo masks via StepCount-RL_Masks pipeline",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Wrote {out_path} ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
