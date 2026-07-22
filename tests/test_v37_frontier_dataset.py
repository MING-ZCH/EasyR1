import hashlib
import json
from pathlib import Path

import pytest

from tools import build_v37_frontier_dataset as builder

H = {
    "model": "1" * 64,
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def row(
    pid, seed, cid, success=False, quality=.8, dup=0, cap=False,
    source_binding=None, model_hash=None,
):
    generation_config = {"seed": seed, "max_new_tokens": 512}
    sampling_config = {"temperature": 0.7, "top_p": 1.0}
    response = f"{pid}:{seed}:{cid}"
    result = {
        "schema_version": 1,
        "prompt_id": pid,
        "seed": seed,
        "declared_seed": seed,
        "candidate_id": cid,
        "answer_exact": success,
        "raw_success": success,
        "trajectory_quality": quality,
        "process": {"unique_hit": 4, "duplicate": dup, "miss": 1, "cap": cap},
        "termination": "cap" if cap else "answer",
        "model_checkpoint_sha256": model_hash or H["model"],
        "generation_config": generation_config,
        "sampling_config": sampling_config,
        "response": response,
        "generation_config_sha256": canonical_digest(generation_config),
        "sampling_config_sha256": canonical_digest(sampling_config),
        "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
    }
    if source_binding is not None:
        result.update(source_binding)
    return result


def write_audit(path, seed, prompts, count=32, winners=3, bindings=None, model_hash=None):
    with path.open("w") as handle:
        for pid in prompts:
            for index in range(count):
                quality = .4 if index % 2 == 0 else .8
                handle.write(json.dumps(row(
                    pid, seed, f"{pid}-{seed}-{index}", index < winners, quality,
                    source_binding=bindings.get(pid) if bindings else None,
                    model_hash=model_hash,
                )) + "\n")


def test_32x2_completeness_and_unique(tmp_path):
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    write_audit(first, 1, ["p"])
    write_audit(second, 2, ["p"])
    seeds, groups, _ = builder.validate_audits([first, second], {"p"}, 32, require_provenance=True)
    assert seeds == ("1", "2") and len(groups["p"]["1"]) == 32
    write_audit(second, 2, ["p"], 31)
    with pytest.raises(builder.ValidationError):
        builder.validate_audits([first, second], {"p"}, 32, require_provenance=True)


def test_duplicate_json_keys_and_unhashed_raw_config_are_rejected(tmp_path):
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text('{"schema_version":1,"schema_version":1}\n')
    with pytest.raises(builder.ValidationError, match="duplicate JSON key"):
        builder.load_audit(duplicate)

    item = row("p", 1, "c")
    item["generation_config"]["seed"] = 2
    path = tmp_path / "bad-config.jsonl"
    path.write_text(json.dumps(item) + "\n")
    with pytest.raises(builder.ValidationError, match="embedded config seed"):
        builder.load_audit(path, require_provenance=True)


@pytest.mark.parametrize(
    "winners,expected",
    [(0, "process-hard"), (2, None), (3, "outcome-frontier"), (20, "outcome-frontier"), (21, None)],
)
def test_winner_boundaries_and_zero_winner_process_hard(tmp_path, winners, expected):
    paths = [tmp_path / f"{seed}.jsonl" for seed in (1, 2)]
    for path, seed in zip(paths, (1, 2)):
        write_audit(path, seed, ["p"], winners=winners)
    _, groups, _ = builder.validate_audits(paths, {"p"})
    assert builder.classify_prompt(groups["p"]) == expected


def test_process_hard_requires_quality_variation(tmp_path):
    paths = [tmp_path / f"{seed}.jsonl" for seed in (1, 2)]
    for path, seed in zip(paths, (1, 2)):
        write_audit(path, seed, ["p"], winners=0)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for item in rows:
            item["trajectory_quality"] = .8
        path.write_text("\n".join(json.dumps(item) for item in rows) + "\n")
    _, groups, _ = builder.validate_audits(paths, {"p"})
    assert builder.classify_prompt(groups["p"]) is None


def test_copied_seed_audit_is_rejected(tmp_path):
    first = tmp_path / "one.jsonl"
    second = tmp_path / "two.jsonl"
    write_audit(first, 1, ["p"], winners=0)
    copied = []
    for line in first.read_text().splitlines():
        item = json.loads(line)
        item["seed"] = item["declared_seed"] = 2
        item["candidate_id"] = item["candidate_id"].replace("-1-", "-2-")
        item["generation_config"]["seed"] = 2
        item["generation_config_sha256"] = canonical_digest(item["generation_config"])
        copied.append(json.dumps(item))
    second.write_text("\n".join(copied) + "\n")
    with pytest.raises(builder.ValidationError, match="identical response content"):
        builder.validate_audits([first, second], {"p"}, require_provenance=True)


def test_generation_configs_may_differ_only_by_seed(tmp_path):
    paths = [tmp_path / f"{seed}.jsonl" for seed in (1, 2)]
    for path, seed in zip(paths, (1, 2)):
        write_audit(path, seed, ["p"])
    rows = [json.loads(line) for line in paths[1].read_text().splitlines()]
    for item in rows:
        item["generation_config"]["max_new_tokens"] = 999
        item["generation_config_sha256"] = canonical_digest(item["generation_config"])
    paths[1].write_text("\n".join(json.dumps(item) for item in rows) + "\n")
    with pytest.raises(builder.ValidationError, match="outside the declared seed field"):
        builder.validate_audits(paths, {"p"}, require_provenance=True)


def test_formal_audit_is_bound_to_source_row_images_prompt_and_live_model(tmp_path):
    binding = {
        "source_row_sha256": "2" * 64,
        "source_image_sha256": ["3" * 64],
        "rendered_prompt_sha256": hashlib.sha256(b"<image>\nCount.").hexdigest(),
        "rendered_prompt": "<image>\nCount.",
    }
    paths = [tmp_path / f"{seed}.jsonl" for seed in (1, 2)]
    for path, seed in zip(paths, (1, 2)):
        write_audit(
            path, seed, ["p"], bindings={"p": binding}, model_hash="1" * 64,
        )
    builder.validate_audits(
        paths, {"p"}, require_provenance=True, source_bindings={"p": binding},
        expected_model_checkpoint_sha256="1" * 64,
    )

    rows = [json.loads(line) for line in paths[1].read_text().splitlines()]
    rows[0]["source_image_sha256"] = ["4" * 64]
    paths[1].write_text("\n".join(json.dumps(item) for item in rows) + "\n")
    with pytest.raises(builder.ValidationError, match="audit/source row"):
        builder.validate_audits(
            paths, {"p"}, require_provenance=True, source_bindings={"p": binding},
            expected_model_checkpoint_sha256="1" * 64,
        )

    for path, seed in zip(paths, (1, 2)):
        write_audit(
            path, seed, ["p"], bindings={"p": binding}, model_hash="1" * 64,
        )
    with pytest.raises(builder.ValidationError, match="live checkpoint tree"):
        builder.validate_audits(
            paths, {"p"}, require_provenance=True, source_bindings={"p": binding},
            expected_model_checkpoint_sha256="9" * 64,
        )


def test_default_bucket_quotas_determinism_and_canary_completeness():
    assert builder.largest_remainder(100, builder.RATIOS) == [40, 10, 20, 20, 10]
    eligible = {}
    for label in builder.LABELS:
        eligible[label + "|outcome-frontier"] = [f"{label}-o-{index}" for index in range(100)]
        eligible[label + "|process-hard"] = [f"{label}-p-{index}" for index in range(100)]
    first, stats = builder.select_ids(eligible, 100, selection_seed=9)
    second, _ = builder.select_ids(eligible, 100, selection_seed=9)
    assert first == second and [stats[label]["quota"] for label in builder.LABELS] == [40, 10, 20, 20, 10]
    assert sum(item["outcome-frontier"] for item in stats.values()) in (75, 76)
    with pytest.raises(builder.ValidationError, match="insufficient quota"):
        builder.select_ids({}, 10, formal=False, require_complete=True)


def test_formal_custom_ratio_and_m_rejected_before_read(tmp_path):
    args = builder.parser().parse_args([
        "--source", str(tmp_path / "missing.parquet"), "--audit", str(tmp_path / "a"),
        "--audit", str(tmp_path / "b"), "--output-dir", str(tmp_path / "out"),
        "--run-class", "formal", "--sample-count", "10", "--bucket-ratios", ".2,.2,.2,.2,.2",
        "--expected-source-sha256", "a" * 64, "--expected-audit-sha256", "b" * 64,
        "--expected-audit-sha256", "c" * 64,
    ])
    args.bucket_ratios = (.2,) * 5
    with pytest.raises(builder.ValidationError, match="locked"):
        builder.build(args)


def test_dangling_output_symlink_is_rejected_before_read(tmp_path):
    output = tmp_path / "out"
    output.symlink_to(tmp_path / "missing-target", target_is_directory=True)
    args = builder.parser().parse_args([
        "--source", str(tmp_path / "missing.parquet"),
        "--audit", str(tmp_path / "a"),
        "--audit", str(tmp_path / "b"),
        "--output-dir", str(output),
        "--run-class", "debug",
        "--sample-count", "1",
    ])
    args.bucket_ratios = builder.RATIOS
    with pytest.raises(builder.ValidationError, match="already exists"):
        builder.build(args)


def test_source_schema_preserved_and_complete_artifacts_published(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    prompts, answers = [], []
    for bucket_index, (low, _) in enumerate(builder.BUCKETS):
        for index in range(20):
            prompts.append(f"p{bucket_index}-{index}")
            answers.append(low)
    source = tmp_path / "source.parquet"
    for prompt_id in prompts:
        (tmp_path / f"{prompt_id}.png").write_bytes(f"image:{prompt_id}".encode())
    table = pa.table({
        "prompt_id": prompts,
        "sample_id": [f"s-{item}" for item in prompts],
        "prompt": ["<image>\nCount the objects." for _ in prompts],
        "answer": answers,
        "images": [[f"{item}.png"] for item in prompts],
        "keep": list(range(100)),
    })
    pq.write_table(table, source)
    bindings = builder.build_source_bindings(table, source)
    mining_checkpoint = tmp_path / "mining-checkpoint"
    mining_checkpoint.mkdir()
    (mining_checkpoint / "model.bin").write_bytes(b"mining-model")
    mining_hash = builder.path_sha256(mining_checkpoint)
    audits = []
    for seed in (1, 2):
        path = tmp_path / f"audit-{seed}.jsonl"
        write_audit(
            path, seed, prompts, winners=3,
            bindings=bindings, model_hash=mining_hash,
        )
        rows = []
        for line in path.read_text().splitlines():
            item = json.loads(line)
            if int(item["prompt_id"].split("-")[1]) >= 15:
                item["answer_exact"] = item["raw_success"] = False
            rows.append(json.dumps(item))
        path.write_text("\n".join(rows) + "\n")
        audits.append(path)
    output = tmp_path / "out"
    args = builder.parser().parse_args([
        "--source", str(source), "--audit", str(audits[0]), "--audit", str(audits[1]),
        "--output-dir", str(output), "--run-class", "formal", "--sample-count", "20",
        "--expected-source-sha256", digest(source),
        "--expected-audit-sha256", digest(audits[0]),
        "--expected-audit-sha256", digest(audits[1]),
        "--mining-checkpoint", str(mining_checkpoint),
        "--expected-mining-checkpoint-sha256", mining_hash,
    ])
    args.bucket_ratios = builder.RATIOS
    manifest = builder.build(args)
    result = pq.read_table(output / "frontier_rl.parquet")
    assert result.schema == table.schema and "response" not in result.column_names
    assert (output / "_COMPLETE.json").is_file()
    assert manifest["selected_sample_count"] == 20
    assert manifest["artifact_sha256"]["frontier_rl.parquet"] == digest(output / "frontier_rl.parquet")
    assert manifest["schema_version"] == 3
    assert manifest["mining_generation_contract"]["source_binding_sha256"]


def test_source_must_be_directly_rlhf_dataset_consumable(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    source = tmp_path / "source.parquet"
    pq.write_table(pa.table({"prompt_id": ["p"], "sample_id": ["s"], "answer": [2]}), source)
    args = builder.parser().parse_args([
        "--source", str(source), "--audit", str(tmp_path / "a"), "--audit", str(tmp_path / "b"),
        "--output-dir", str(tmp_path / "out"), "--run-class", "debug", "--sample-count", "1",
    ])
    args.bucket_ratios = builder.RATIOS
    with pytest.raises(builder.ValidationError, match="RLHFDataset-consumable"):
        builder.build(args)


def test_source_with_candidate_trajectory_columns_is_rejected_by_builder(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    source = tmp_path / "source.parquet"
    pq.write_table(pa.table({
        "prompt_id": ["p"], "sample_id": ["s"],
        "prompt": ["<image>\nCount."], "answer": [2], "images": [["image.png"]],
        "response": ["fabricated candidate trajectory"],
    }), source)
    args = builder.parser().parse_args([
        "--source", str(source), "--audit", str(tmp_path / "a"),
        "--audit", str(tmp_path / "b"), "--output-dir", str(tmp_path / "out"),
        "--run-class", "debug", "--sample-count", "1",
    ])
    args.bucket_ratios = builder.RATIOS
    with pytest.raises(builder.ValidationError, match="candidate response/audit columns"):
        builder.build(args)
