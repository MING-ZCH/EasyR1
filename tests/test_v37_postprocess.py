import hashlib
import json
import os
from pathlib import Path

import pytest

from tools import v37_postprocess as postprocess
from tools import v37_eval_producer as eval_producer


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(
    tmp_path: Path,
    *,
    with_evidence: bool = True,
    pipeline_producer: Path | None = None,
) -> tuple[Path, Path, list[Path]]:
    root = tmp_path / "ab"
    root.mkdir()
    plan_path = root / "ab_plan.json"
    cells = []
    cell_dirs = []
    for seed in (11, 22):
        for arm in ("baseline", "progress"):
            cell_id = f"seed{seed}-{arm}"
            cell_dir = root / cell_id
            cell_dir.mkdir()
            cell_dirs.append(cell_dir)
            cells.append({
                "cell_id": cell_id,
                "seed": seed,
                "arm": arm,
                "target_dir": str(cell_dir),
            })
    evaluator = tmp_path / "formal-evaluator.py"
    evaluator.write_text(
        """#!/usr/bin/env python3
import json
import sys
from pathlib import Path

if len(sys.argv) != 6:
    raise SystemExit(2)
Path(sys.argv[4]).write_text(json.dumps({"producer_invocation_nonce": sys.argv[5]}))
""",
        encoding="utf-8",
    )
    os.chmod(evaluator, 0o750)
    recipe_path = tmp_path / "formal-eval-recipe.json"
    recipe_steps = {
        name: {
            "arms": eval_producer.STEP_ARMS[name],
            "program": {"path": str(evaluator), "sha256": postprocess._file_sha256(evaluator)},
            "argv": [
                "{manifest}", "{checkpoint}", "{eval_model}", "{output}",
                "{invocation_nonce}",
            ],
            "output": eval_producer.STEP_OUTPUTS[name],
        }
        for name in eval_producer.STEP_NAMES
    }
    _write(recipe_path, {
        "schema_version": 1,
        "recipe_type": eval_producer.RECIPE_TYPE,
        "environment": {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            **eval_producer.REQUIRED_ENVIRONMENT,
        },
        "steps": recipe_steps,
    })
    validation = tmp_path / "heldout.parquet"
    validation.write_bytes(b"paired heldout fixture")
    validation_snapshot = postprocess.gate.content_snapshot(validation, parquet_only=True)
    universe_rows = []
    for index, bucket in enumerate(postprocess.gate.BUCKETS):
        ground_truth = postprocess.gate.BUCKET_RANGES[bucket][0]
        projection = {
            "sample_id": f"s-{index}", "prompt_id": f"p-{index}",
            "bucket": bucket, "ground_truth": ground_truth,
        }
        universe_rows.append({
            **projection,
            "source_row_sha256": hashlib.sha256(
                postprocess._canonical_bytes(projection)
            ).hexdigest(),
        })
    universe_path = tmp_path / "paired-sample-universe.json"
    _write(universe_path, {
        "schema_version": postprocess.gate.PAIRED_UNIVERSE_SCHEMA_VERSION,
        "artifact_type": postprocess.gate.PAIRED_UNIVERSE_ARTIFACT_TYPE,
        "data_path": str(validation.resolve()),
        "data_sha256": validation_snapshot["sha256"],
        "sample_count": len(universe_rows),
        "bucket_counts": {bucket: 1 for bucket in postprocess.gate.BUCKETS},
        "ordered_rows_sha256": hashlib.sha256(
            postprocess.gate.canonical(universe_rows).encode("utf-8")
        ).hexdigest(),
        "rows": universe_rows,
    })
    bound_producer = pipeline_producer or Path(eval_producer.__file__).resolve()
    plan = {
        "schema_version": 2,
        "ab_run_id": "v37-ab-" + "1" * 32,
        "plan_path": str(plan_path),
        "seeds": [11, 22],
        "execution": {
            "run_class": "formal",
            "data_mode": "frontier_rl",
            "optimizer_steps": 12,
            "nnodes": 1,
            "gpus_per_node": 8,
            "world_size": 8,
            "scheduling": "sequential",
        },
        "preregistered": {
            "schema_version": 1,
            "complete": True,
            "missing": [],
            "input_snapshots": {
                "model": {
                    "root": "/synthetic/checkpoint-476", "file_count": 1,
                    "total_bytes": 1, "sha256": postprocess.gate.FORMAL_INITIAL_MODEL_SHA256,
                    "files": [{"path": "model.bin", "size": 1, "sha256": "1" * 64}],
                }, "training_data": {},
                "validation_data": [{"suite": "heldout", "dataset": validation_snapshot}],
                "mask_metadata": {}, "mask_tree": {}, "frontier_manifest": {},
                "coverage_report": {}, "filtered_manifest": {},
            },
            "implementation_sha256": {},
            "image_path_remap": None,
            "eval_contract": {
                "paired_metrics": postprocess.gate.FROZEN_PAIRED_METRICS,
                "benchmark_dataset_sha256": postprocess.gate.FORMAL_BENCHMARK_DATASET_SHA256,
                "benchmark_ordered_ids_sha256": postprocess.gate.FORMAL_BENCHMARK_ORDERED_IDS_SHA256,
                "benchmark_thresholds": {
                    "pixmo_correct_min": postprocess.gate.PIXMO_CORRECT_MIN,
                    "pixmo_total": postprocess.gate.PIXMO_TOTAL,
                    "stepcount_correct_min": postprocess.gate.STEPCOUNT_CORRECT_MIN,
                    "stepcount_total": postprocess.gate.STEPCOUNT_TOTAL,
                },
                "pipeline": {
                    "producer": {
                        "path": str(bound_producer),
                        "sha256": postprocess._file_sha256(bound_producer),
                    },
                    "recipe": {
                        "path": str(recipe_path),
                        "sha256": postprocess._file_sha256(recipe_path),
                    },
                },
                "paired_sample_universe": {
                    "path": str(universe_path.resolve()),
                    "sha256": postprocess._file_sha256(universe_path),
                },
            },
        },
        "cells": cells,
    }
    _write(plan_path, plan)
    plan_hash = postprocess._file_sha256(plan_path)
    manifest_bindings = []
    for cell, cell_dir in zip(cells, cell_dirs, strict=True):
        checkpoint = cell_dir / "global_step_12"
        checkpoint.mkdir()
        (checkpoint / "state.bin").write_bytes(cell["cell_id"].encode())
        eval_model = checkpoint / "actor" / "huggingface"
        eval_model.mkdir(parents=True)
        (eval_model / "model.safetensors").write_bytes(cell["cell_id"].encode())
        manifest = {
            "arm": cell["arm"],
            "seed": cell["seed"],
            "run_class": "formal",
            "data_mode": "frontier_rl",
            "ab_preregistration": {
                "schema_version": 1,
                "ab_run_id": plan["ab_run_id"],
                "plan_path": str(plan_path),
                "plan_sha256": plan_hash,
                "cell_id": cell["cell_id"],
            },
            "final_checkpoint_path": str(checkpoint),
            "final_checkpoint_sha256": postprocess._tree_sha256(checkpoint),
        }
        manifest_path = cell_dir / "v37_run_manifest.json"
        _write(manifest_path, manifest)
        manifest_bindings.append({
            "path": str(manifest_path),
            "sha256": postprocess._file_sha256(manifest_path),
        })
        _write(cell_dir / postprocess.TRAINING_EVIDENCE_NAME, {"cell": cell["cell_id"]})
        if with_evidence:
            evidence_dir = cell_dir / "v37_postprocess"
            evidence_dir.mkdir()
            _write(evidence_dir / postprocess.PAIRED_EVIDENCE_NAME, {"paired": cell["cell_id"]})
            _write(evidence_dir / postprocess.PRODUCER_RECEIPT_NAME, {"receipt": cell["cell_id"]})
            if cell["arm"] == "progress":
                _write(evidence_dir / postprocess.PIXMO_EVIDENCE_NAME, {"suite": "pixmo"})
                _write(
                    evidence_dir / postprocess.STEPCOUNT_EVIDENCE_NAME,
                    {"suite": "stepcount"},
                )
    index_path = root / "ab_evidence_index.json"
    _write(index_path, {
        "plan": {"path": str(plan_path), "sha256": plan_hash},
        "manifests": manifest_bindings,
    })
    return index_path, root, cell_dirs


def test_assemble_discovers_four_cells_and_publishes_atomically(tmp_path, monkeypatch) -> None:
    index_path, root, _ = _fixture(tmp_path)
    monkeypatch.setattr(
        postprocess.gate,
        "evaluate",
        lambda spec: {
            "schema_version": 2,
            "decision": "GO",
            "hard_failure": False,
            "computed_paired_metrics": {},
            "errors": [],
        },
    )
    monkeypatch.setattr(postprocess, "_run_producer", lambda *args: None)
    output = root / "formal_evidence"
    report = postprocess.assemble(
        ab_index=index_path,
        output_dir=output,
        producer=Path(postprocess.eval_producer.__file__).resolve(),
        recipe=tmp_path / "formal-eval-recipe.json",
    )
    assert report["decision"] == "GO"
    spec = json.loads((output / "v37_final_gate_spec.json").read_text())
    assert [(run["seed"], run["arm"]) for run in spec["runs"]] == [
        (11, "baseline"), (11, "progress"), (22, "baseline"), (22, "progress")
    ]
    assert all("pixmo_eval" not in run for run in spec["runs"] if run["arm"] == "baseline")
    assert all("pixmo_eval" in run for run in spec["runs"] if run["arm"] == "progress")
    final_index = json.loads((output / "v37_final_evidence_index.json").read_text())
    assert final_index["decision"] == "GO"
    assert final_index["index_type"] == postprocess.INDEX_TYPE
    for name in ("gate_spec", "gate_result"):
        binding = final_index[name]
        published = Path(binding["path"])
        assert published.parent == output
        assert published.is_file()
        assert postprocess._file_sha256(published) == binding["sha256"]


def test_real_preregistered_producer_receipts_pass_postprocess_validation(tmp_path) -> None:
    index_path, _, cell_dirs = _fixture(tmp_path, with_evidence=False)
    _, _, cells = postprocess._validate_ab_index(index_path)
    producer = Path(eval_producer.__file__).resolve()
    recipe = tmp_path / "formal-eval-recipe.json"
    postprocess._run_producer(
        producer, postprocess._file_sha256(producer),
        recipe, postprocess._file_sha256(recipe), cells,
    )

    for cell_dir in cell_dirs:
        evidence_dir = cell_dir / "v37_postprocess"
        receipt = json.loads((evidence_dir / postprocess.PRODUCER_RECEIPT_NAME).read_text())
        assert len(receipt["invocation_request_sha256"]) == 64
        assert receipt["invocation_nonce"] == json.loads(
            (evidence_dir / postprocess.PAIRED_EVIDENCE_NAME).read_text()
        )["producer_invocation_nonce"]


def test_postprocess_executes_frozen_producer_after_original_path_is_replaced(
    tmp_path, monkeypatch,
) -> None:
    index_path, _, cell_dirs = _fixture(tmp_path, with_evidence=False)
    _, _, cells = postprocess._validate_ab_index(index_path)
    source = Path(eval_producer.__file__).resolve()
    frozen_producer = tmp_path / "frozen-producer.py"
    frozen_producer.write_bytes(source.read_bytes())
    os.chmod(frozen_producer, 0o750)
    producer_sha256 = postprocess._file_sha256(frozen_producer)
    recipe = tmp_path / "formal-eval-recipe.json"
    run = postprocess.subprocess.run
    replaced = False

    def replace_then_run(*args, **kwargs):
        nonlocal replaced
        if not replaced:
            frozen_producer.write_text(
                "#!/usr/bin/env python3\nraise SystemExit(91)\n", encoding="utf-8",
            )
            os.chmod(frozen_producer, 0o750)
            replaced = True
        return run(*args, **kwargs)

    monkeypatch.setattr(postprocess.subprocess, "run", replace_then_run)
    postprocess._run_producer(
        frozen_producer, producer_sha256,
        recipe, postprocess._file_sha256(recipe), cells,
    )

    assert replaced is True
    assert all((cell / "v37_postprocess" / "producer_receipt.json").is_file()
               for cell in cell_dirs)


def test_missing_fixed_evidence_fails_before_gate(tmp_path, monkeypatch) -> None:
    index_path, root, cell_dirs = _fixture(tmp_path)
    (cell_dirs[0] / "v37_postprocess" / postprocess.PAIRED_EVIDENCE_NAME).unlink()
    called = False

    def _unexpected(_):
        nonlocal called
        called = True
        return {"decision": "GO"}

    monkeypatch.setattr(postprocess.gate, "evaluate", _unexpected)
    monkeypatch.setattr(postprocess, "_run_producer", lambda *args: None)
    with pytest.raises(postprocess.PostprocessError, match="paired evidence does not exist"):
        postprocess.assemble(
            ab_index=index_path,
            output_dir=root / "formal_evidence",
            producer=Path(postprocess.eval_producer.__file__).resolve(),
            recipe=tmp_path / "formal-eval-recipe.json",
        )
    assert called is False


def test_existing_output_is_never_overwritten(tmp_path) -> None:
    index_path, root, _ = _fixture(tmp_path)
    output = root / "formal_evidence"
    output.mkdir()
    with pytest.raises(postprocess.PostprocessError, match="must not exist"):
        postprocess.assemble(ab_index=index_path, output_dir=output)


def test_formal_postprocess_cannot_consume_preexisting_evidence_without_producer(tmp_path) -> None:
    index_path, root, _ = _fixture(tmp_path)
    with pytest.raises(postprocess.PostprocessError, match="must execute"):
        postprocess.assemble(ab_index=index_path, output_dir=root / "formal_evidence")


def test_selected_pipeline_must_match_preregistration(tmp_path) -> None:
    index_path, root, _ = _fixture(tmp_path)
    other = tmp_path / "other-producer"
    other.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    os.chmod(other, 0o750)
    with pytest.raises(postprocess.PostprocessError, match="differs from A/B preregistration"):
        postprocess.assemble(
            ab_index=index_path,
            output_dir=root / "formal_evidence",
            producer=other,
            recipe=tmp_path / "formal-eval-recipe.json",
        )


def test_producer_checkpoint_mutation_is_rejected(tmp_path) -> None:
    producer = tmp_path / "producer.py"
    producer.write_text(
        """#!/usr/bin/env python3
import argparse, json
from pathlib import Path
p=argparse.ArgumentParser()
for name in ('cell-id','arm','seed','manifest','checkpoint','output-dir','invocation-nonce'):
    p.add_argument('--'+name, required=True)
p.add_argument('--recipe')
a=p.parse_args()
out=Path(a.output_dir)
(out/'paired_eval.json').write_text('{}')
if a.arm == 'progress':
    (out/'pixmo_benchmark_evidence.json').write_text('{}')
    (out/'stepcount_benchmark_evidence.json').write_text('{}')
(Path(a.checkpoint)/'tampered.bin').write_bytes(b'tampered')
""",
        encoding="utf-8",
    )
    os.chmod(producer, 0o750)
    index_path, root, cell_dirs = _fixture(tmp_path, with_evidence=False)
    _, _, cells = postprocess._validate_ab_index(index_path)
    with pytest.raises(postprocess.PostprocessError, match="changed checkpoint"):
        postprocess._run_producer(
            producer, postprocess._file_sha256(producer),
            tmp_path / "formal-eval-recipe.json",
            postprocess._file_sha256(tmp_path / "formal-eval-recipe.json"), cells,
        )
    assert not (root / "formal_evidence").exists()
    assert (cell_dirs[0] / "global_step_12" / "tampered.bin").exists()
    assert all(not (cell_dir / "v37_postprocess").exists() for cell_dir in cell_dirs)


@pytest.mark.parametrize("mode", ["failure", "extra", "symlink"])
def test_producer_failure_never_publishes_partial_cell_outputs(tmp_path, mode) -> None:
    producer = tmp_path / "producer.py"
    producer.write_text(
        """#!/usr/bin/env python3
import argparse, json, os, sys
from pathlib import Path
p=argparse.ArgumentParser()
for name in ('cell-id','arm','seed','manifest','checkpoint','output-dir','invocation-nonce'):
    p.add_argument('--'+name, required=True)
p.add_argument('--recipe')
a=p.parse_args()
out=Path(a.output_dir)
(out/'paired_eval.json').write_text('{}')
if a.arm == 'progress':
    (out/'pixmo_benchmark_evidence.json').write_text('{}')
    (out/'stepcount_benchmark_evidence.json').write_text('{}')
mode='__MODE__'
if mode == 'failure' and a.cell_id.endswith('progress'):
    raise SystemExit(7)
if mode == 'extra':
    (out/'unexpected.json').write_text('{}')
if mode == 'symlink':
    (out/'paired_eval.json').unlink()
    os.symlink('/dev/null', out/'paired_eval.json')
""".replace("__MODE__", mode),
        encoding="utf-8",
    )
    os.chmod(producer, 0o750)
    index_path, root, cell_dirs = _fixture(tmp_path, with_evidence=False)
    _, _, cells = postprocess._validate_ab_index(index_path)
    with pytest.raises(postprocess.PostprocessError):
        postprocess._run_producer(
            producer, postprocess._file_sha256(producer),
            tmp_path / "formal-eval-recipe.json",
            postprocess._file_sha256(tmp_path / "formal-eval-recipe.json"), cells,
        )
    assert all(not (cell_dir / "v37_postprocess").exists() for cell_dir in cell_dirs)
    assert not any(
        staging
        for cell_dir in cell_dirs
        for staging in cell_dir.glob(".v37_postprocess.*")
    )
