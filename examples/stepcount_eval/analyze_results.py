#!/usr/bin/env python3
"""严格验证 V36 两个 checkpoint 的四套结果并生成 paired 对比报告。"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from bundle_common import (
    MODEL_ORDER,
    PROTOCOL_TAG,
    PROTOCOL_VERSION,
    SUITE_ORDER,
    SUITES,
    ids_sha256,
    parse_key_value,
    sha256_file,
    validate_dataset_rows,
)


def exact_mcnemar_p(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(first_only, second_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def load_datasets(mapping: dict[str, str]) -> dict[str, dict]:
    loaded = {}
    for suite in SUITE_ORDER:
        path = Path(mapping[suite]).expanduser().resolve()
        with path.open("r", encoding="utf-8") as handle:
            rows, ids, answers = validate_dataset_rows(json.load(handle), suite)
        spec = SUITES[suite]
        if len(rows) != spec.expected_count:
            raise ValueError(f"{suite}: dataset 样本数 {len(rows)} != {spec.expected_count}")
        loaded[suite] = {
            "path": path,
            "rows": rows,
            "ids": ids,
            "id_set": set(ids),
            "by_id": {str(row["id"]): row for row in rows},
            "sha256": sha256_file(path),
            "ids_sha256": ids_sha256(ids),
            "answers": answers,
        }
    return loaded


def require_equal(actual, expected, context: str) -> None:
    if actual != expected:
        raise ValueError(f"{context}: {actual!r} != {expected!r}")


def require_bool(mapping: dict, key: str, expected: bool, context: str) -> None:
    value = mapping.get(key)
    if type(value) is not bool:
        raise TypeError(f"{context}.{key}: 必须是 bool，实际 {type(value).__name__}")
    require_equal(value, expected, f"{context}.{key}")


def require_int(mapping: dict, key: str, expected: int, context: str) -> None:
    value = mapping.get(key)
    if type(value) is not int:
        raise TypeError(f"{context}.{key}: 必须是 int，实际 {type(value).__name__}")
    require_equal(value, expected, f"{context}.{key}")


def explicit_integer_from_response(response: str) -> str:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    value = match.group(1).strip()
    return value if re.fullmatch(r"[+-]?\d+", value) else ""


def has_closed_answer(response: str) -> bool:
    return re.search(r"<answer>.*?</answer>", response, re.IGNORECASE | re.DOTALL) is not None


def has_parseable_point(response: str) -> bool:
    for match in re.findall(r"<point>\s*(\{[^}]+\})\s*</point>", response, re.IGNORECASE):
        try:
            raw = json.loads(match)
            point = {str(key).strip('"').strip("'").replace("\\", ""): value for key, value in raw.items()}
            coords = point.get("point_2d")
            if not isinstance(coords, list) or len(coords) != 2:
                continue
            x, y = float(coords[0]), float(coords[1])
            int(point.get("count_number", 0))
            if math.isfinite(x) and math.isfinite(y):
                return True
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return False


def validate_result_set(*, step: str, suite: str, result_dir: Path, dataset: dict) -> dict:
    result_path = result_dir / "results.json"
    manifest_path = result_dir / "run_manifest.json"
    with result_path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(rows, list):
        raise TypeError(f"{result_path}: 顶层必须是 list")
    spec = SUITES[suite]
    require_equal(len(rows), spec.expected_count, f"step{step}/{suite} sample count")
    result_ids = [str(row.get("id")) for row in rows]
    require_equal(result_ids, dataset["ids"], f"step{step}/{suite} dataset ID/order")

    manifest_expected = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_tag": PROTOCOL_TAG,
        "suite": suite,
        "dataset_id": suite,
        "dataset_sha256": dataset["sha256"],
        "sample_ids_sha256": dataset["ids_sha256"],
        "expected_samples": spec.expected_count,
        "actual_samples": spec.expected_count,
        "model_label": f"step{step}",
        "dtype": "bfloat16",
        "do_sample": False,
        "num_beams": 1,
        "num_beam_groups": 1,
        "num_return_sequences": 1,
        "history_mode": 0,
        "adaptive_max_rounds": True,
        "adaptive_max_rounds_extra": 3,
        "task_cap": spec.task_cap,
        "require_explicit_answer": True,
        "allow_point_count_fallback": False,
        "stop_after_first_complete_tag": True,
        "stop_on_no_progress": False,
        "keep_eval_images": False,
    }
    for key, expected in manifest_expected.items():
        require_equal(manifest.get(key), expected, f"step{step}/{suite} manifest.{key}")
    if not manifest.get("model_identity"):
        raise ValueError(f"step{step}/{suite}: manifest 缺少 model_identity")
    if not manifest.get("eval_code_sha256"):
        raise ValueError(f"step{step}/{suite}: manifest 缺少 eval_code_sha256")
    expected_eval_code_sha256 = sha256_file(Path(__file__).with_name("eval_stepcount.py"))
    require_equal(
        manifest["eval_code_sha256"],
        expected_eval_code_sha256,
        f"step{step}/{suite} evaluator code hash",
    )
    for key, expected in {
        "do_sample": False,
        "adaptive_max_rounds": True,
        "require_explicit_answer": True,
        "allow_point_count_fallback": False,
        "stop_after_first_complete_tag": True,
        "stop_on_no_progress": False,
        "keep_eval_images": False,
    }.items():
        require_bool(manifest, key, expected, f"step{step}/{suite} manifest")
    for key, expected in {
        "history_mode": 0,
        "num_beams": 1,
        "num_beam_groups": 1,
        "num_return_sequences": 1,
        "adaptive_max_rounds_extra": 3,
        "task_cap": spec.task_cap,
    }.items():
        require_int(manifest, key, expected, f"step{step}/{suite} manifest")

    row_model_identities = set()
    row_code_hashes = set()
    fingerprints = set()
    correct_map = {}
    for row in rows:
        sample_id = str(row["id"])
        gt = int(str(dataset["by_id"][sample_id]["answer"]).strip())
        expected_turns = min(spec.task_cap, gt + 3)
        context = f"step{step}/{suite}/{sample_id}"
        require_equal(int(str(row.get("correct_answer")).strip()), gt, context + " GT")
        require_equal(row.get("effective_max_rounds"), expected_turns, context + " GT+3")
        require_equal(row.get("adaptive_max_rounds_extra"), 3, context + " adaptive extra")
        require_equal(row.get("eval_protocol_version"), PROTOCOL_VERSION, context + " protocol")
        require_equal(row.get("eval_protocol"), "oracle_min_global_gt_plus_3", context + " protocol detail")
        require_equal(str(row.get("model_dtype", "")).lower(), "bfloat16", context + " dtype")
        require_equal(row.get("dataset_id"), suite, context + " dataset_id")
        require_equal(row.get("model_label"), f"step{step}", context + " model_label")
        require_int(row, "history_mode", 0, context)
        require_int(row, "num_beams", 1, context)
        require_int(row, "num_beam_groups", 1, context)
        require_int(row, "num_return_sequences", 1, context)
        require_bool(row, "do_sample", False, context)
        require_bool(row, "require_explicit_answer", True, context)
        require_bool(row, "allow_point_count_fallback", False, context)
        require_bool(row, "used_point_count_fallback", False, context)
        require_bool(row, "stop_after_first_complete_tag", True, context)
        require_bool(row, "stop_on_no_progress", False, context)
        require_bool(row, "keep_eval_images", False, context)
        require_equal(row.get("task_cap"), spec.task_cap, context + " task cap")
        require_equal(row.get("dataset_sha256"), dataset["sha256"], context + " dataset hash")
        if not isinstance(row.get("num_rounds"), int) or not (1 <= row["num_rounds"] <= expected_turns):
            raise ValueError(f"{context}: num_rounds 越界 {row.get('num_rounds')}")
        output = row.get("output")
        if not isinstance(output, list):
            raise TypeError(f"{context}: output 必须是 list")
        if any(not isinstance(message, dict) for message in output):
            raise TypeError(f"{context}: output 中每条 message 必须是 object")
        model_messages = [message for message in output if message.get("role") == "model"]
        require_equal(len(model_messages), row["num_rounds"], context + " model message count")
        response_events = []
        closed_answer_events = []
        for message in model_messages:
            content = message.get("content")
            if not isinstance(content, str):
                raise TypeError(f"{context}: model content 必须是 string")
            closing = re.search(r"</(point|answer)\s*>", content, re.IGNORECASE)
            if closing and closing.end() != len(content):
                raise ValueError(f"{context}: 首个完整 closing tag 后仍有输出")
            response_events.append(closing.group(1).lower() if closing else None)
            closed_answer_events.append(has_closed_answer(content))
        require_equal(
            row.get("last_response_event"),
            response_events[-1],
            context + " last response event",
        )
        if row.get("image_paths") != []:
            raise ValueError(f"{context}: keep_eval_images=false 但 image_paths 未清空")
        predicted = str(row.get("predicted_answer", ""))
        for key in ("explicit_answer_detected", "is_correct"):
            if type(row.get(key)) is not bool:
                raise TypeError(f"{context}.{key}: 必须是 bool")
        parsed_explicit = explicit_integer_from_response(model_messages[-1]["content"])
        explicit = bool(parsed_explicit)
        if any(closed_answer_events[:-1]):
            raise ValueError(f"{context}: 较早轮已有闭合 answer 却仍继续生成")
        final_closed_answer = closed_answer_events[-1]
        require_equal(row["explicit_answer_detected"], explicit, context + " explicit answer parse")
        require_equal(predicted, parsed_explicit, context + " predicted answer parse")
        if predicted and not explicit:
            raise ValueError(f"{context}: 非空 predicted_answer 没有显式闭合 answer")
        if row.get("is_correct") is True and not explicit:
            raise ValueError(f"{context}: 正确样本没有显式闭合 answer")
        if final_closed_answer and row.get("last_response_event") != "answer":
            raise ValueError(f"{context}: closed answer 但最后 tag event 不是 answer")
        termination_reason = row.get("termination_reason")
        if termination_reason not in {"explicit_answer", "max_rounds", "annotation_failed"}:
            raise ValueError(f"{context}: 非法 termination_reason={termination_reason!r}")
        if final_closed_answer:
            require_equal(termination_reason, "explicit_answer", context + " answer termination")
        elif termination_reason == "explicit_answer":
            raise ValueError(f"{context}: 无闭合 answer 却标记 explicit_answer 终止")
        if termination_reason == "max_rounds":
            require_equal(row["num_rounds"], expected_turns, context + " max-round termination")
        if termination_reason == "annotation_failed":
            require_equal(row.get("last_response_event"), "point", context + " annotation event")
            if not has_parseable_point(model_messages[-1]["content"]):
                raise ValueError(f"{context}: annotation_failed 但最后一轮没有可解析 point")
        expected_correct = explicit and predicted.lstrip("+-").isdigit() and int(predicted) == gt
        require_equal(row.get("is_correct"), expected_correct, context + " correctness")
        row_model_identities.add(row.get("model_identity"))
        row_code_hashes.add(row.get("eval_code_sha256"))
        fingerprints.add(row.get("config_fingerprint"))
        correct_map[sample_id] = bool(row["is_correct"])

    require_equal(row_model_identities, {manifest["model_identity"]}, f"step{step}/{suite} model identity")
    require_equal(row_code_hashes, {manifest["eval_code_sha256"]}, f"step{step}/{suite} code hash")
    if len(fingerprints) != 1 or None in fingerprints:
        raise ValueError(f"step{step}/{suite}: config fingerprint 不一致")
    require_equal(
        next(iter(fingerprints)),
        manifest.get("config_fingerprint"),
        f"step{step}/{suite} manifest/row config fingerprint",
    )
    correct = sum(correct_map.values())
    return {
        "samples": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "model_identity": manifest["model_identity"],
        "eval_code_sha256": manifest["eval_code_sha256"],
        "dataset_sha256": dataset["sha256"],
        "sample_ids_sha256": dataset["ids_sha256"],
        "correct_map": correct_map,
    }


def compare(validated: dict) -> dict:
    suites = {}
    for suite in SUITE_ORDER:
        first = validated["77"][suite]["correct_map"]
        second = validated["60"][suite]["correct_map"]
        first_only = sum(first[key] and not second[key] for key in first)
        second_only = sum(second[key] and not first[key] for key in first)
        both = sum(first[key] and second[key] for key in first)
        neither = len(first) - first_only - second_only - both
        suites[suite] = {
            "step77_only": first_only,
            "step60_only": second_only,
            "both_correct": both,
            "both_wrong": neither,
            "exact_mcnemar_p": exact_mcnemar_p(first_only, second_only),
        }
    scores = {}
    for step in MODEL_ORDER:
        dense = validated[step]["stepcount-500"]["accuracy"]
        broad = sum(validated[step][suite]["accuracy"] for suite in ("pixmo-test", "countqa", "bias")) / 3
        scores[f"step{step}"] = {
            "dense": dense,
            "broad_macro": broad,
            "composite_50_50": 0.5 * dense + 0.5 * broad,
        }
    primary = max(scores, key=lambda label: scores[label]["composite_50_50"])
    return {"paired": suites, "selection_scores": scores, "primary_by_composite": primary}


def json_ready(validated: dict) -> dict:
    return {
        f"step{step}": {
            suite: {key: value for key, value in item.items() if key != "correct_map"}
            for suite, item in suites.items()
        }
        for step, suites in validated.items()
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# V36 checkpoint 严格评测对比",
        "",
        f"- 协议：`{PROTOCOL_VERSION}`（BF16、greedy、`min(task cap, GT+3)`、显式闭合 answer、无 fallback）",
        "- 顺序：`step77 -> step60`；suite 为 `pixmo-test -> stepcount-500 -> countqa -> bias`。",
        "- 本报告只接受样本数、dataset hash/ID、模型身份、代码 hash 与协议元数据全部通过的结果。",
        "",
        "| suite | step77 | step60 | 差值 77-60 | paired flips 77-only/60-only | exact McNemar p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    validated = report["validated"]
    for suite in SUITE_ORDER:
        left = validated["step77"][suite]
        right = validated["step60"][suite]
        paired = report["comparison"]["paired"][suite]
        delta = left["accuracy"] - right["accuracy"]
        lines.append(
            f"| {suite} | {left['correct']}/{left['samples']} ({left['accuracy']:.2%}) | "
            f"{right['correct']}/{right['samples']} ({right['accuracy']:.2%}) | {delta:+.2%} | "
            f"{paired['step77_only']}/{paired['step60_only']} | {paired['exact_mcnemar_p']:.6f} |"
        )
    lines.extend(
        ["", "## 预设 50/50 选模口径", "", "| checkpoint | dense | broad macro | composite |", "|---|---:|---:|---:|"]
    )
    for label in ("step77", "step60"):
        score = report["comparison"]["selection_scores"][label]
        lines.append(
            f"| {label} | {score['dense']:.2%} | {score['broad_macro']:.2%} | {score['composite_50_50']:.2%} |"
        )
    lines.extend(
        [
            "",
            f"按预设 composite 的主 checkpoint：`{report['comparison']['primary_by_composite']}`。",
            "",
            "注意：小幅差值不自动代表稳定总体提升；bias 的重复模板分布也不能外推为普遍鲁棒性。",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--dataset", action="append", default=[], required=True, metavar="SUITE=JSON")
    parser.add_argument("--json-report", type=Path)
    parser.add_argument("--markdown-report", type=Path)
    args = parser.parse_args(argv)
    mapping = parse_key_value(args.dataset, what="--dataset")
    if set(mapping) != set(SUITE_ORDER):
        raise ValueError(f"--dataset 必须恰好覆盖 {list(SUITE_ORDER)}")
    datasets = load_datasets(mapping)
    validated = {step: {} for step in MODEL_ORDER}
    for step in MODEL_ORDER:
        for suite in SUITE_ORDER:
            validated[step][suite] = validate_result_set(
                step=step,
                suite=suite,
                result_dir=args.results_root / f"step{step}" / suite,
                dataset=datasets[suite],
            )
        identities = {validated[step][suite]["model_identity"] for suite in SUITE_ORDER}
        if len(identities) != 1:
            raise ValueError(f"step{step}: 四套结果的 model identity 不一致: {identities}")
        code_hashes = {validated[step][suite]["eval_code_sha256"] for suite in SUITE_ORDER}
        if len(code_hashes) != 1:
            raise ValueError(f"step{step}: 四套结果的 evaluator code hash 不一致: {code_hashes}")
    step_identities = {step: validated[step][SUITE_ORDER[0]]["model_identity"] for step in MODEL_ORDER}
    if len(set(step_identities.values())) != len(MODEL_ORDER):
        raise ValueError(f"step77 与 step60 的 model identity 意外相同: {step_identities}")

    report = {
        "protocol": PROTOCOL_VERSION,
        "validated": json_ready(validated),
        "comparison": compare(validated),
    }
    json_path = args.json_report or args.results_root / "v36_checkpoint_comparison.json"
    markdown_path = args.markdown_report or args.results_root / "v36_checkpoint_comparison.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"验证通过：{json_path}")
    print(f"对比报告：{markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
