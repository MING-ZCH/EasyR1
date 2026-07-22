import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from tools import v37_continuation as continuation


def _checkpoint_hash(checkpoint: Path) -> str:
    content = (checkpoint / "state.bin").read_bytes()
    entries = [{
        "path": "state.bin",
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }]
    return hashlib.sha256(json.dumps(
        entries, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def _evidence_record(step: int) -> dict:
    return {
        "global_step": step,
        "optimizer_microsteps_attempted": 1,
        "optimizer_microsteps_executed": 1,
        "optimizer_microsteps_skipped": 0,
        "nonfinite_count_cumulative": 0,
        "oom_count_cumulative": 0,
        "sample_count": 2,
        "progress_degraded_count": 0,
        "cap_turn_exceeded_count": 0,
        "early_stop_count": 0,
        "selector_kl_contribution": 0.0,
        "action_rows_expected": 2,
        "action_rows_mapped": 2,
        "mask_rows_expected": 2,
        "mask_rows_scored": 2,
        "frontier_group_count": 1,
        "frontier_mixed_group_count": 1,
        "outcome_allwrong_group_count": 0,
        "correctness_sign_error_count": 0,
    }


def _source(tmp_path: Path) -> tuple[Path, Path, str, str]:
    checkpoint = tmp_path / "global_step_4"
    checkpoint.mkdir()
    (checkpoint / "state.bin").write_bytes(b"state")
    checkpoint_hash = _checkpoint_hash(checkpoint)
    frontier = tmp_path / "frontier-manifest.json"
    frontier.write_text('{"frontier":"frozen"}\n', encoding="utf-8")
    frontier_hash = hashlib.sha256(frontier.read_bytes()).hexdigest()
    config = {"world_size": 8, "pilot_steps": 4, "micro_update": 4, "actor_lr": "1e-6"}
    source_path = tmp_path / "source.json"
    source = {
        "manifest_version": 3,
        "data_mode": "frontier_rl",
        "run_class": "canary",
        "promotable_candidate": False,
        "arm": "progress",
        "seed": 22,
        **continuation.live_git_identity(),
        "dataset_manifest": str(frontier),
        "dataset_manifest_sha256": frontier_hash,
        "dataset_sha256": "b" * 64,
        "metadata_sha256": "c" * 64,
        "mask_tree_path": "/masks",
        "mask_tree_sha256": "d" * 64,
        "paired_eval_data_sha256": "e" * 64,
        "metadata_coverage_report_sha256": "f" * 64,
        "filtered_manifest_sha256": "1" * 64,
        "input_snapshots": {
            "model": {"sha256": "2" * 64},
            "training_data": {"sha256": "6" * 64},
        },
        "runtime_environment": {"python": "3.10"},
        "implementation_sha256": {"trainer": "3" * 64},
        "config_file_sha256": "4" * 64,
        "source_config_sha256": "5" * 64,
        "mechanisms": {"adaptive_actor_kl": "true"},
        "mechanism_config": {"schema_version": 2},
        "mining_contract": {"candidates_per_seed": 32},
        "preflight_summary": {"sample_count": 8},
        "audited_environment": {
            "ACTOR_LR": "1e-6",
            "NCCL_DEBUG": "WARN",
            "BOK_TOTAL_STEPS": "4",
            "V36_MAX_STEPS": "4",
            "V37_EVIDENCE_START_STEP": "0",
            "V37_PILOT_STEPS": "4",
            "V37_RUN_MANIFEST": str(source_path),
        },
        "execution_environment_sha256": "6" * 64,
        "ab_preregistration": None,
        "resume_evidence": None,
        "config": config,
        "continuation_config_sha256": continuation.continuation_config_sha256(config),
        "final_checkpoint_path": str(checkpoint),
        "final_checkpoint_sha256": checkpoint_hash,
    }
    for field in continuation.EXACT_BINDING_FIELDS:
        source.setdefault(field, None)
    source["config_sha256"] = continuation._manifest_config_sha256(source)
    source["evidence_run_id"] = continuation._manifest_evidence_run_id(source)
    source_path.write_text(json.dumps(source), encoding="utf-8")
    evidence_path = tmp_path / "training-evidence.json"
    recorder = continuation.training_evidence.TrainingEvidenceRecorder(
        evidence_path, source_path, expected_steps=4
    )
    for step in range(1, 5):
        recorder.record_step(_evidence_record(step))
    recorder.finalize(
        completed=True,
        checkpoint_id="progress-seed22-global_step_4",
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_hash,
        kl_recoverable=True,
    )
    source["training_evidence_path"] = str(evidence_path)
    source["training_evidence_sha256"] = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    source_path.write_text(json.dumps(source), encoding="utf-8")
    return source_path, checkpoint, checkpoint_hash, frontier_hash


def _seal(tmp_path: Path, source: Path, checkpoint: Path, checkpoint_hash: str) -> Path:
    path = tmp_path / "seal.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "purpose": "controlled_continuation",
        "checkpoint_path": str(checkpoint),
        "checkpoint_tree_sha256": checkpoint_hash,
        "world_size": 8,
        "source_run_manifest_path": str(source),
        "source_run_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }), encoding="utf-8")
    return path


def _continued_manifest(source_path: Path) -> dict:
    current = json.loads(source_path.read_text(encoding="utf-8"))
    current["config"]["pilot_steps"] = 8
    current["continuation_config_sha256"] = continuation.continuation_config_sha256(
        current["config"]
    )
    current["audited_environment"].update({
        "V37_PILOT_STEPS": "8",
        "V36_MAX_STEPS": "8",
        "BOK_TOTAL_STEPS": "8",
        "V37_EVIDENCE_START_STEP": "4",
        "V37_RUN_MANIFEST": str(source_path.with_name("continued-manifest.json")),
        "V37_CONTINUATION_MODE": "1",
        "V37_RESUME_MODE": "controlled_continuation",
    })
    current["resume_evidence"] = None
    return current


def test_seal_binds_checkpoint_arm_seed_frontier_and_config(tmp_path):
    source, checkpoint, checkpoint_hash, frontier_hash = _source(tmp_path)
    seal = _seal(tmp_path, source, checkpoint, checkpoint_hash)
    assert continuation.verify_seal(
        seal,
        checkpoint,
        checkpoint_hash,
        world_size=8,
        run_class="canary",
        arm="progress",
        seed=22,
        frontier_manifest_sha256=frontier_hash,
    ) == source
    for field, value, message in (
        ("arm", "baseline", "arm/seed"),
        ("seed", 11, "arm/seed"),
        ("frontier_manifest_sha256", "9" * 64, "frontier"),
    ):
        kwargs = {
            "world_size": 8,
            "run_class": "canary",
            "arm": "progress",
            "seed": 22,
            "frontier_manifest_sha256": frontier_hash,
        }
        kwargs[field] = value
        with pytest.raises(continuation.ContinuationError, match=message):
            continuation.verify_seal(seal, checkpoint, checkpoint_hash, **kwargs)


def test_create_seal_atomically_publishes_complete_source(tmp_path):
    source, checkpoint, checkpoint_hash, frontier_hash = _source(tmp_path)
    output = tmp_path / "created-seal.json"
    seal, seal_hash = continuation.create_seal(source, checkpoint, output)
    assert seal == output
    assert seal_hash == hashlib.sha256(output.read_bytes()).hexdigest()
    assert continuation.verify_seal(
        output,
        checkpoint,
        checkpoint_hash,
        world_size=8,
        run_class="canary",
        arm="progress",
        seed=22,
        frontier_manifest_sha256=frontier_hash,
    ) == source
    with pytest.raises(continuation.ContinuationError, match="already exists"):
        continuation.create_seal(source, checkpoint, output)


def test_complete_manifest_allows_only_larger_pilot_steps(tmp_path):
    source_path, _, _, _ = _source(tmp_path)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    current = json.loads(json.dumps(source))
    current["config"]["pilot_steps"] = 8
    current["continuation_config_sha256"] = continuation.continuation_config_sha256(current["config"])
    current["audited_environment"]["V37_PILOT_STEPS"] = "8"
    current["audited_environment"]["V36_MAX_STEPS"] = "8"
    current["audited_environment"]["BOK_TOTAL_STEPS"] = "8"
    current["audited_environment"]["V37_EVIDENCE_START_STEP"] = "4"
    current["audited_environment"]["V37_RUN_MANIFEST"] = "/current/run/manifest.json"
    continuation.verify_current_manifest(source, current)

    changed = json.loads(json.dumps(current))
    changed["config"]["micro_update"] = 8
    changed["continuation_config_sha256"] = continuation.continuation_config_sha256(changed["config"])
    with pytest.raises(continuation.ContinuationError, match="numerical config"):
        continuation.verify_current_manifest(source, changed)

    changed = json.loads(json.dumps(current))
    changed["implementation_sha256"]["trainer"] = "9" * 64
    with pytest.raises(continuation.ContinuationError, match="implementation_sha256"):
        continuation.verify_current_manifest(source, changed)

    changed = json.loads(json.dumps(current))
    changed["git_diff_sha256"] = "9" * 64
    with pytest.raises(continuation.ContinuationError, match="git_diff_sha256"):
        continuation.verify_current_manifest(source, changed)

    changed = json.loads(json.dumps(current))
    changed["implementation_sha256"]["process_prompt"] = "9" * 64
    with pytest.raises(continuation.ContinuationError, match="implementation_sha256"):
        continuation.verify_current_manifest(source, changed)

    changed = json.loads(json.dumps(current))
    changed["audited_environment"]["NCCL_DEBUG"] = "INFO"
    with pytest.raises(continuation.ContinuationError, match="NCCL_DEBUG"):
        continuation.verify_current_manifest(source, changed)


def test_rejects_relative_or_symlink_evidence(tmp_path):
    source, checkpoint, checkpoint_hash, frontier_hash = _source(tmp_path)
    seal = _seal(tmp_path, source, checkpoint, checkpoint_hash)
    link = tmp_path / "seal-link.json"
    link.symlink_to(seal)
    with pytest.raises(continuation.ContinuationError, match="symlink"):
        continuation.verify_seal(
            link,
            checkpoint,
            checkpoint_hash,
            world_size=8,
            run_class="canary",
            arm="progress",
            seed=22,
            frontier_manifest_sha256=frontier_hash,
        )


def test_late_validation_rechecks_source_completion_and_returns_bound_lineage(tmp_path):
    source, checkpoint, checkpoint_hash, frontier_hash = _source(tmp_path)
    seal = _seal(tmp_path, source, checkpoint, checkpoint_hash)
    current = _continued_manifest(source)
    lineage = continuation.validate_continuation(
        seal,
        checkpoint,
        current,
        expected_seal_sha256=hashlib.sha256(seal.read_bytes()).hexdigest(),
        expected_checkpoint_sha256=checkpoint_hash,
        expected_source_manifest_path=source,
        expected_frontier_manifest_sha256=frontier_hash,
    )
    assert lineage["contract"] == "v37_controlled_continuation_v2"
    assert lineage["source_global_step"] == 4
    assert lineage["target_global_step"] == 8
    assert lineage["source_evidence_run_id"] == json.loads(
        source.read_text(encoding="utf-8")
    )["evidence_run_id"]
    assert len(lineage["binding_sha256"]) == 64


def test_late_validation_rejects_current_live_git_identity_drift(tmp_path):
    source, checkpoint, checkpoint_hash, frontier_hash = _source(tmp_path)
    seal = _seal(tmp_path, source, checkpoint, checkpoint_hash)
    current = _continued_manifest(source)
    current["git_commit"] = "f" * 40
    with pytest.raises(continuation.ContinuationError, match="live Git identity"):
        continuation.validate_continuation(
            seal,
            checkpoint,
            current,
            expected_seal_sha256=hashlib.sha256(seal.read_bytes()).hexdigest(),
            expected_checkpoint_sha256=checkpoint_hash,
            expected_source_manifest_path=source,
            expected_frontier_manifest_sha256=frontier_hash,
        )


def test_live_git_identity_ignores_repository_override_environment(tmp_path, monkeypatch):
    fake = tmp_path / "fake-repo"
    subprocess.run(["git", "init", "-q", str(fake)], check=True)
    monkeypatch.setenv("GIT_DIR", str(fake / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(fake))
    identity = continuation.live_git_identity()
    expected = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=Path(continuation.__file__).resolve().parents[1],
        check=True, capture_output=True, env={
            key: value for key, value in os.environ.items() if not key.startswith("GIT_")
        },
    ).stdout.decode("ascii").strip()
    assert identity["git_commit"] == expected


def test_late_validation_rejects_source_or_training_evidence_replacement(tmp_path):
    source, checkpoint, checkpoint_hash, frontier_hash = _source(tmp_path)
    seal = _seal(tmp_path, source, checkpoint, checkpoint_hash)
    current = _continued_manifest(source)
    expected = {
        "expected_seal_sha256": hashlib.sha256(seal.read_bytes()).hexdigest(),
        "expected_checkpoint_sha256": checkpoint_hash,
        "expected_source_manifest_path": source,
        "expected_frontier_manifest_sha256": frontier_hash,
    }

    original_source = source.read_bytes()
    source.write_bytes(original_source + b"\n")
    with pytest.raises(continuation.ContinuationError, match="source manifest changed"):
        continuation.validate_continuation(seal, checkpoint, current, **expected)
    source.write_bytes(original_source)

    source_payload = json.loads(source.read_text(encoding="utf-8"))
    evidence_path = Path(source_payload["training_evidence_path"])
    evidence_path.write_bytes(evidence_path.read_bytes() + b"\n")
    with pytest.raises(continuation.ContinuationError, match="training evidence hash"):
        continuation.validate_continuation(seal, checkpoint, current, **expected)


def test_late_validation_rejects_bool_seed_and_equal_target_step(tmp_path):
    source, checkpoint, checkpoint_hash, frontier_hash = _source(tmp_path)
    seal = _seal(tmp_path, source, checkpoint, checkpoint_hash)
    kwargs = {
        "expected_seal_sha256": hashlib.sha256(seal.read_bytes()).hexdigest(),
        "expected_checkpoint_sha256": checkpoint_hash,
        "expected_source_manifest_path": source,
        "expected_frontier_manifest_sha256": frontier_hash,
    }
    current = _continued_manifest(source)
    current["seed"] = True
    with pytest.raises(continuation.ContinuationError, match="seed"):
        continuation.validate_continuation(seal, checkpoint, current, **kwargs)

    current = _continued_manifest(source)
    current["config"]["pilot_steps"] = 4
    current["continuation_config_sha256"] = continuation.continuation_config_sha256(
        current["config"]
    )
    current["audited_environment"].update({
        "V37_PILOT_STEPS": "4", "V36_MAX_STEPS": "4", "BOK_TOTAL_STEPS": "4",
    })
    with pytest.raises(continuation.ContinuationError, match="must extend"):
        continuation.validate_continuation(seal, checkpoint, current, **kwargs)
