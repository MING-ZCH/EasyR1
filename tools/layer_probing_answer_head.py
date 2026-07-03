#!/usr/bin/env python3
"""Stub: layer-wise probing for SFT vs RL answer-head bottleneck (Phase0-7 optional).

Usage:
  python3 tools/layer_probing_answer_head.py \\
    --sft_ckpt /path/sft --rl_ckpt /path/rl --probe_layers -1,-2,-4
"""

from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft_ckpt", required=True)
    ap.add_argument("--rl_ckpt", required=True)
    ap.add_argument("--probe_layers", default="-1")
    ap.add_argument("--output", default="docs/v34_reports/layer_probing_results.json")
    args = ap.parse_args()
    plan = {
        "status": "stub",
        "sft_ckpt": args.sft_ckpt,
        "rl_ckpt": args.rl_ckpt,
        "probe_layers": args.probe_layers,
        "procedure": [
            "Extract hidden states at layers L for pixmo 0-10 samples",
            "Train linear probe: hidden -> count answer",
            "Compare SFT vs RL probe accuracy; if encoder layers equal but last layer drops, output-layer bottleneck",
        ],
        "reference": "Related_Paper_Survey_Counting_Attention_20260603.md",
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
