import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from tools import v37_benchmark_evidence as evidence


REAL_LEGACY_PIXMO = Path(
    "/mnt/shared-storage-user/zhangchenhao/work/StepcountModel/eval/eval_pixmo_test/"
    "eval_StepCount-7B-v36-focused10k-step60-bf16-oracle-gtplus3-strict-v3.json"
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@dataclass
class EvidenceCase:
    root: Path
    suite: str
    total: int
    checkpoint_tree: Path
    eval_model: Path
    dataset: Path
    image: Path
    image_manifest: Path
    raw_results: Path
    run_manifest: Path
    evaluator: Path
    validator: Path
    prompt: Path
    requirements: Path

    def rows(self) -> list[dict[str, object]]:
        return json.loads(self.raw_results.read_text(encoding="utf-8"))

    def write_rows(self, rows: list[dict[str, object]]) -> None:
        _write_json(self.raw_results, rows)
        self.refresh_run_manifest()

    def refresh_run_manifest(self) -> None:
        manifest = {
            "schema_version": 2,
            "manifest_type": evidence.RUN_MANIFEST_TYPE,
            "suite": self.suite,
            "artifacts": {
                "checkpoint_tree": evidence.artifact_binding(self.checkpoint_tree, kind="tree"),
                "eval_model": evidence.artifact_binding(self.eval_model, kind="tree"),
                "dataset": evidence.artifact_binding(self.dataset, kind="file"),
                "external_image_manifest": evidence.artifact_binding(
                    self.image_manifest, kind="file"
                ),
                "raw_results": evidence.artifact_binding(self.raw_results, kind="file"),
            },
            "implementations": {
                "evaluator": evidence.artifact_binding(self.evaluator, kind="file"),
                "validator": evidence.artifact_binding(self.validator, kind="file"),
                "prompt": evidence.artifact_binding(self.prompt, kind="file"),
                "requirements": evidence.artifact_binding(self.requirements, kind="file"),
            },
            "protocol": evidence.frozen_protocol(self.suite),
        }
        _write_json(self.run_manifest, manifest)

    def build(self, output: Path | None = None) -> dict[str, object]:
        return evidence.build_descriptor(
            suite=self.suite,
            checkpoint_tree=self.checkpoint_tree,
            eval_model=self.eval_model,
            dataset=self.dataset,
            external_image_manifest=self.image_manifest,
            raw_results=self.raw_results,
            eval_run_manifest=self.run_manifest,
            evaluator=self.evaluator,
            validator=self.validator,
            prompt=self.prompt,
            requirements=self.requirements,
            output=output,
        )

    def cli_args(self, output: Path) -> list[str]:
        return [
            "build",
            "--suite",
            self.suite,
            "--checkpoint-tree",
            str(self.checkpoint_tree),
            "--eval-model",
            str(self.eval_model),
            "--dataset",
            str(self.dataset),
            "--external-image-manifest",
            str(self.image_manifest),
            "--raw-results",
            str(self.raw_results),
            "--eval-run-manifest",
            str(self.run_manifest),
            "--evaluator",
            str(self.evaluator),
            "--validator",
            str(self.validator),
            "--prompt",
            str(self.prompt),
            "--requirements",
            str(self.requirements),
            "--output",
            str(output),
        ]


def _make_case(root: Path, suite: str = "pixmo-test") -> EvidenceCase:
    root.mkdir()
    total = evidence.SUITE_SPECS[suite]["total"]
    task_cap = evidence.SUITE_SPECS[suite]["task_cap"]
    checkpoint_tree = root / "checkpoint"
    checkpoint_tree.mkdir()
    (checkpoint_tree / "state.bin").write_bytes(b"checkpoint-state")
    eval_model = root / "eval-model"
    eval_model.mkdir()
    (eval_model / "model.safetensors").write_bytes(b"model-weights")
    image = root / "source.img"
    image.write_bytes(b"source-image-bytes")
    evaluator = root / "evaluator.py"
    evaluator.write_text("def evaluate():\n    return None\n", encoding="utf-8")
    validator = root / "validator.py"
    validator.write_text("def validate():\n    return None\n", encoding="utf-8")
    prompt = root / "prompt.txt"
    prompt.write_text("BOUND SYSTEM PROMPT\n", encoding="utf-8")
    requirements = root / "requirements.txt"
    requirements.write_text("transformers==0\n", encoding="utf-8")

    dataset = root / "dataset.json"
    dataset_rows = [
        {
            "id": f"sample-{index:04d}",
            "question": f"How many objects are in image {index}?",
            "answer": index % 11 + 1,
            "image_path": str(image),
        }
        for index in range(total)
    ]
    _write_json(dataset, dataset_rows)
    dataset_hash = evidence.sha256_file(dataset)
    image_hash = evidence.sha256_file(image)
    checkpoint_hash = evidence.tree_sha256(checkpoint_tree)
    model_hash = evidence.tree_sha256(eval_model)

    image_manifest = root / "images.json"
    _write_json(
        image_manifest,
        {
            "schema_version": 2,
            "suite": suite,
            "images": [
                {"id": row["id"], "path": str(image), "sha256": image_hash}
                for row in dataset_rows
            ],
        },
    )

    raw_results = root / "raw-results.json"
    result_rows: list[dict[str, object]] = []
    for row in dataset_rows:
        answer = int(row["answer"])
        response = f"<answer>{answer}</answer>"
        result_rows.append(
            {
                "id": row["id"],
                "question": row["question"],
                "correct_answer": answer,
                "predicted_answer": str(answer),
                "explicit_answer_detected": True,
                "is_correct": True,
                "effective_max_rounds": min(task_cap, answer + 3),
                "num_rounds": 1,
                "termination_reason": "explicit_answer",
                "last_response_event": "answer",
                "used_point_count_fallback": False,
                "dataset_row_sha256": evidence.dataset_row_sha256(row),
                "source_image_sha256": image_hash,
                "dataset_sha256": dataset_hash,
                "checkpoint_sha256": checkpoint_hash,
                "eval_model_sha256": model_hash,
                "model_path": str(eval_model),
                "model_dtype": "bf16",
                "do_sample": False,
                "num_beams": 1,
                "num_return_sequences": 1,
                "adaptive_max_rounds": True,
                "adaptive_max_rounds_extra": 3,
                "max_rounds_config": task_cap,
                "require_explicit_answer": True,
                "point_count_fallback_enabled": False,
                "stop_after_first_complete_tag": True,
                "output": [
                    {"role": "system", "content": "BOUND SYSTEM PROMPT"},
                    {
                        "role": "human",
                        "content": row["question"],
                        "image": str(image),
                    },
                    {"role": "model", "content": response, "round": 1},
                ],
                "final_response": response,
            }
        )
    _write_json(raw_results, result_rows)
    run_manifest = root / "run-manifest.json"
    case = EvidenceCase(
        root=root,
        suite=suite,
        total=total,
        checkpoint_tree=checkpoint_tree,
        eval_model=eval_model,
        dataset=dataset,
        image=image,
        image_manifest=image_manifest,
        raw_results=raw_results,
        run_manifest=run_manifest,
        evaluator=evaluator,
        validator=validator,
        prompt=prompt,
        requirements=requirements,
    )
    case.refresh_run_manifest()
    return case


@pytest.fixture
def case(tmp_path: Path) -> EvidenceCase:
    return _make_case(tmp_path / "case")


@pytest.mark.parametrize(
    ("suite", "total", "task_cap"),
    [("pixmo-test", 529, 13), ("stepcount-500", 500, 53)],
)
def test_build_and_verify_exact_suite_totals(
    tmp_path: Path, suite: str, total: int, task_cap: int
) -> None:
    item = _make_case(tmp_path / suite, suite)
    descriptor_path = item.root / "descriptor.json"
    descriptor = item.build(descriptor_path)
    assert descriptor["schema_version"] == 2
    assert descriptor["protocol"]["round_limit"]["task_cap"] == task_cap
    assert descriptor["summary"] == {"total": total, "correct": total, "incorrect": 0}
    assert len(descriptor["samples"]) == total
    assert evidence.verify_descriptor(descriptor_path)["summary"] == descriptor["summary"]
    for reference in descriptor["artifacts"].values():
        assert Path(reference["path"]).is_absolute()
        assert len(reference["sha256"]) == 64


def test_tree_hash_matches_v37_gate_contract(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "b").write_bytes(b"B")
    (tree / "a").write_bytes(b"A")
    entries = [
        {"path": "a", "size": 1, "sha256": hashlib.sha256(b"A").hexdigest()},
        {"path": "b", "size": 1, "sha256": hashlib.sha256(b"B").hexdigest()},
    ]
    expected = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert evidence.tree_sha256(tree) == expected


def test_final_response_cache_is_optional(case: EvidenceCase) -> None:
    rows = case.rows()
    for row in rows:
        row.pop("final_response")
    case.write_rows(rows)
    descriptor = case.build()
    assert descriptor["summary"] == {"total": 529, "correct": 529, "incorrect": 0}


@pytest.mark.parametrize(
    "field",
    [
        "num_beams",
        "num_return_sequences",
        "adaptive_max_rounds",
        "require_explicit_answer",
        "stop_after_first_complete_tag",
    ],
)
def test_missing_per_row_protocol_evidence_is_rejected(
    case: EvidenceCase, field: str
) -> None:
    rows = case.rows()
    rows[0].pop(field)
    case.write_rows(rows)
    with pytest.raises(evidence.EvidenceError, match=f"missing protocol evidence field {field}"):
        case.build()


@pytest.mark.parametrize(
    "field",
    [
        "dataset_row_sha256",
        "source_image_sha256",
        "checkpoint_sha256",
        "eval_model_sha256",
    ],
)
def test_missing_per_row_provenance_is_rejected(case: EvidenceCase, field: str) -> None:
    rows = case.rows()
    rows[0].pop(field)
    case.write_rows(rows)
    with pytest.raises(evidence.EvidenceError, match=f"missing provenance field {field}"):
        case.build()


def test_forged_row_correctness_is_rejected(case: EvidenceCase) -> None:
    rows = case.rows()
    wrong = int(rows[0]["correct_answer"]) + 1
    response = f"<answer>{wrong}</answer>"
    rows[0]["output"][-1]["content"] = response
    rows[0]["final_response"] = response
    rows[0]["predicted_answer"] = str(wrong)
    rows[0]["is_correct"] = True
    case.write_rows(rows)
    with pytest.raises(evidence.EvidenceError, match="is_correct is forged or stale"):
        case.build()


def test_forged_descriptor_summary_is_rejected(case: EvidenceCase) -> None:
    descriptor = case.build()
    descriptor["summary"]["correct"] -= 1
    descriptor["summary"]["incorrect"] += 1
    with pytest.raises(evidence.EvidenceError, match="summary cache"):
        evidence.verify_descriptor(descriptor)


def test_missing_result_row_is_rejected(case: EvidenceCase) -> None:
    rows = case.rows()
    rows.pop()
    case.write_rows(rows)
    with pytest.raises(evidence.EvidenceError, match="exactly 529"):
        case.build()


def test_duplicate_result_id_is_rejected(case: EvidenceCase) -> None:
    rows = case.rows()
    rows[1] = copy.deepcopy(rows[0])
    case.write_rows(rows)
    with pytest.raises(evidence.EvidenceError, match="duplicate raw result ID"):
        case.build()


def test_changed_result_ground_truth_is_rejected(case: EvidenceCase) -> None:
    rows = case.rows()
    rows[0]["correct_answer"] = int(rows[0]["correct_answer"]) + 1
    case.write_rows(rows)
    with pytest.raises(evidence.EvidenceError, match="canonical dataset"):
        case.build()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("do_sample", True, "do_sample must be false"),
        ("model_dtype", "fp16", "model_dtype must be BF16"),
        ("num_beams", 2, "num_beams must be 1"),
        ("num_return_sequences", 2, "num_return_sequences must be 1"),
        ("adaptive_max_rounds", False, "adaptive_max_rounds must be true"),
        ("adaptive_max_rounds_extra", 1, "adaptive_max_rounds_extra must be 3"),
        ("max_rounds_config", 12, "max_rounds_config must be 13"),
        ("require_explicit_answer", False, "require_explicit_answer must be true"),
        ("point_count_fallback_enabled", True, "point_count_fallback_enabled must be false"),
        ("stop_after_first_complete_tag", False, "stop_after_first_complete_tag must be true"),
    ],
)
def test_non_frozen_protocol_is_rejected(
    case: EvidenceCase, field: str, value: object, message: str
) -> None:
    rows = case.rows()
    rows[0][field] = value
    case.write_rows(rows)
    with pytest.raises(evidence.EvidenceError, match=message):
        case.build()


def test_used_point_count_fallback_is_rejected(case: EvidenceCase) -> None:
    rows = case.rows()
    rows[0]["used_point_count_fallback"] = True
    case.write_rows(rows)
    with pytest.raises(evidence.EvidenceError, match="forbidden point-count fallback"):
        case.build()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        ("tail", "content after the first complete tag"),
        ("final", "final_response differs"),
        ("rounds", "num_rounds differs from transcript"),
        ("termination", "termination_reason mismatch"),
    ],
)
def test_transcript_and_termination_caches_are_recomputed(
    case: EvidenceCase, mutate: str, message: str
) -> None:
    rows = case.rows()
    if mutate == "tail":
        response = str(rows[0]["final_response"]) + " forged tail"
        rows[0]["output"][-1]["content"] = response
        rows[0]["final_response"] = response
    elif mutate == "final":
        rows[0]["final_response"] = "<answer>999</answer>"
    elif mutate == "rounds":
        rows[0]["num_rounds"] = 2
    else:
        rows[0]["termination_reason"] = "max_rounds"
    case.write_rows(rows)
    with pytest.raises(evidence.EvidenceError, match=message):
        case.build()


@pytest.mark.parametrize(
    "target",
    [
        "dataset",
        "image",
        "image_manifest",
        "raw_results",
        "run_manifest",
        "evaluator",
        "validator",
        "prompt",
        "requirements",
        "checkpoint_tree",
        "eval_model",
    ],
)
def test_bound_artifact_drift_is_rejected(case: EvidenceCase, target: str) -> None:
    descriptor = case.build()
    path = getattr(case, target)
    if path.is_dir():
        (path / "drift.bin").write_bytes(b"drift")
    else:
        path.write_bytes(path.read_bytes() + b"\nDRIFT")
    with pytest.raises(evidence.EvidenceError, match="drift|changed|hash mismatch"):
        evidence.verify_descriptor(descriptor)


def test_dataset_row_hash_must_match_canonical_row(case: EvidenceCase) -> None:
    rows = case.rows()
    rows[0]["dataset_row_sha256"] = "a" * 64
    case.write_rows(rows)
    with pytest.raises(evidence.EvidenceError, match="dataset_row_sha256 provenance mismatch"):
        case.build()


def test_source_image_hash_is_checked_against_manifest(case: EvidenceCase) -> None:
    rows = case.rows()
    rows[0]["source_image_sha256"] = "a" * 64
    case.write_rows(rows)
    with pytest.raises(evidence.EvidenceError, match="source_image_sha256 provenance mismatch"):
        case.build()


def test_existing_portable_image_snapshot_contract_is_supported(case: EvidenceCase) -> None:
    image_record = {"resolved_path": str(case.image), "sha256": evidence.sha256_file(case.image)}
    groups = {
        "train": [],
        "validation": [],
        "forbidden": [image_record],
    }
    snapshot: dict[str, object] = {
        "schema_version": 1,
        "snapshot_contract": evidence.PORTABLE_IMAGE_SNAPSHOT_CONTRACT,
        "embedded_bytes_contract": evidence.PORTABLE_EMBEDDED_BYTES_CONTRACT,
    }
    group_hashes = {}
    for split, files in groups.items():
        aggregate = evidence.canonical_json_sha256(files)
        snapshot[split] = {
            "file_count": len(files),
            "files": files,
            "aggregate_sha256": aggregate,
        }
        group_hashes[split] = aggregate
    snapshot["aggregate_sha256"] = evidence.canonical_json_sha256(group_hashes)
    _write_json(case.image_manifest, snapshot)
    case.refresh_run_manifest()
    descriptor = case.build()
    assert descriptor["summary"]["total"] == 529


def test_strict_json_rejects_duplicate_keys(case: EvidenceCase) -> None:
    text = case.raw_results.read_text(encoding="utf-8")
    text = text.replace('"id": "sample-0000",', '"id": "sample-0000", "id": "sample-0000",', 1)
    case.raw_results.write_text(text, encoding="utf-8")
    case.refresh_run_manifest()
    with pytest.raises(evidence.EvidenceError, match="duplicate JSON key"):
        case.build()


def test_strict_json_rejects_nan(case: EvidenceCase) -> None:
    text = case.raw_results.read_text(encoding="utf-8")
    text = text.replace('"is_correct": true', '"is_correct": NaN', 1)
    case.raw_results.write_text(text, encoding="utf-8")
    case.refresh_run_manifest()
    with pytest.raises(evidence.EvidenceError, match="non-finite JSON constant"):
        case.build()


def test_symlink_ancestor_is_rejected(case: EvidenceCase) -> None:
    linked_root = case.root.parent / "linked"
    linked_root.symlink_to(case.root, target_is_directory=True)
    with pytest.raises(evidence.EvidenceError, match="symlink component"):
        evidence.sha256_file(linked_root / case.dataset.name)


def test_schema_v1_formal_descriptor_is_rejected() -> None:
    with pytest.raises(evidence.EvidenceError, match="schema v1|schema_version"):
        evidence.verify_descriptor(
            {
                "schema_version": 1,
                "descriptor_type": evidence.DESCRIPTOR_TYPE,
                "suite": "pixmo-test",
                "artifacts": {},
                "implementations": {},
                "protocol": {},
                "samples": [],
                "summary": {},
            }
        )


def test_cli_build_and_verify(case: EvidenceCase, capsys: pytest.CaptureFixture[str]) -> None:
    descriptor = case.root / "descriptor.json"
    assert evidence.main(case.cli_args(descriptor)) == 0
    build_output = json.loads(capsys.readouterr().out)
    assert build_output["summary"]["total"] == 529
    assert evidence.main(["verify", str(descriptor)]) == 0
    verify_output = json.loads(capsys.readouterr().out)
    assert verify_output["valid"] is True


def test_cli_does_not_invent_missing_greedy_evidence(
    case: EvidenceCase, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = case.rows()
    rows[0].pop("do_sample")
    case.write_rows(rows)
    descriptor = case.root / "descriptor.json"
    assert evidence.main(case.cli_args(descriptor)) == 2
    assert "missing protocol evidence field do_sample" in capsys.readouterr().err
    assert not descriptor.exists()


@pytest.mark.skipif(not REAL_LEGACY_PIXMO.is_file(), reason="local legacy artifact is unavailable")
def test_real_legacy_artifact_is_readable_but_not_promotable() -> None:
    with pytest.raises(evidence.EvidenceError, match="missing protocol evidence field do_sample"):
        evidence.validate_raw_results_protocol(REAL_LEGACY_PIXMO, "pixmo-test")
