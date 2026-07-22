import json
import importlib.util
from pathlib import Path

import pytest

_SOURCE = Path(__file__).resolve().parents[1] / "verl/utils/v37_training_evidence.py"
_SPEC = importlib.util.spec_from_file_location("v37_training_evidence_test", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
evidence = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(evidence)


def _manifest(run: Path) -> Path:
    path = run / "v37_run_manifest.json"
    path.write_text(json.dumps({
        "evidence_run_id": "evidence-run-1",
        "execution_environment_sha256": "1" * 64,
    }), encoding="utf-8")
    return path


def _record(step: int, *, nonfinite: int = 0) -> dict:
    return {
        "global_step": step,
        "optimizer_microsteps_attempted": 2,
        "optimizer_microsteps_executed": 2,
        "optimizer_microsteps_skipped": 0,
        "nonfinite_count_cumulative": nonfinite,
        "oom_count_cumulative": 0,
        "sample_count": 8,
        "progress_degraded_count": 0,
        "cap_turn_exceeded_count": 1,
        "early_stop_count": 1,
        "selector_kl_contribution": 0.0,
        "action_rows_expected": 8,
        "action_rows_mapped": 8,
        "mask_rows_expected": 8,
        "mask_rows_scored": 8,
        "frontier_group_count": 4,
        "frontier_mixed_group_count": 3,
        "outcome_allwrong_group_count": 1,
        "correctness_sign_error_count": 0,
    }


def test_reward_counts_are_row_exact_and_fail_closed():
    metrics = {
        "progress_degraded": [0, 1, 0],
        "turns_exceeded": [1, 0, 0],
        "early_stop": [0, 0, 1],
        "mask_evidence_complete": [1, 1, 1],
    }
    assert evidence.reward_counts(metrics, 3) == {
        "sample_count": 3,
        "progress_degraded_count": 1,
        "cap_turn_exceeded_count": 1,
        "early_stop_count": 1,
        "mask_rows_expected": 3,
        "mask_rows_scored": 3,
    }
    for field in ("turns_exceeded", "early_stop", "mask_evidence_complete"):
        broken = dict(metrics)
        broken.pop(field)
        with pytest.raises(evidence.TrainingEvidenceError):
            evidence.reward_counts(broken, 3)
    with pytest.raises(evidence.TrainingEvidenceError, match="contain 3 rows"):
        evidence.reward_counts(metrics | {"progress_degraded": [0]}, 3)
    with pytest.raises(evidence.TrainingEvidenceError, match="binary"):
        evidence.reward_counts(metrics | {"early_stop": [0, 0.5, 1]}, 3)


def test_optimizer_integrity_gate_rejects_before_checkpoint_publication():
    evidence.require_clean_optimizer_step(2, 2, 0, 0)
    with pytest.raises(evidence.TrainingEvidenceError, match="partially executed"):
        evidence.require_clean_optimizer_step(2, 1, 1, 0)
    with pytest.raises(evidence.TrainingEvidenceError, match="non-finite"):
        evidence.require_clean_optimizer_step(2, 2, 0, 1)

    trainer_source = (
        Path(__file__).resolve().parents[1] / "verl/trainer/ray_trainer.py"
    ).read_text(encoding="utf-8")
    integrity_gate = trainer_source.index("require_clean_optimizer_step(", 1000)
    evidence_record = trainer_source.index("v37_evidence.record_step({", integrity_gate)
    step_complete = trainer_source.index("self._current_step_complete = True", evidence_record)
    checkpoint_save = trainer_source.index("self._save_checkpoint()", step_complete)
    assert integrity_gate < evidence_record < step_complete < checkpoint_save
    assert "if sealed_checkpoint and not self._current_step_complete:" in trainer_source


def test_v37_resume_checkpoint_binding_accepts_only_exact_sealed_tree(tmp_path):
    checkpoint = tmp_path / "global_step_4"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"weights")
    expected = evidence.tree_sha256(checkpoint)
    environment = {
        "V37_RUN_CLASS": "canary",
        "V37_RESUME_MODE": "controlled_continuation",
        "V37_EXPECTED_RESUME_CHECKPOINT_PATH": str(checkpoint),
        "V37_EXPECTED_RESUME_CHECKPOINT_SHA256": expected,
    }
    assert evidence.validate_resume_checkpoint_binding(checkpoint, environment) == expected

    substitute = tmp_path / "global_step_5"
    substitute.mkdir()
    (substitute / "weights.bin").write_bytes(b"weights")
    with pytest.raises(evidence.TrainingEvidenceError, match="path differs"):
        evidence.validate_resume_checkpoint_binding(substitute, environment)

    (checkpoint / "weights.bin").write_bytes(b"mutated")
    with pytest.raises(evidence.TrainingEvidenceError, match="changed before Trainer load"):
        evidence.validate_resume_checkpoint_binding(checkpoint, environment)


@pytest.mark.parametrize("mode", [None, "clean_start", "typo"])
def test_v37_checkpoint_load_rejects_missing_or_noncontinuation_mode(tmp_path, mode):
    checkpoint = tmp_path / "global_step_4"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"weights")
    environment = {"V37_RUN_CLASS": "debug"}
    if mode is not None:
        environment["V37_RESUME_MODE"] = mode
    with pytest.raises(evidence.TrainingEvidenceError, match="controlled_continuation"):
        evidence.validate_resume_checkpoint_binding(checkpoint, environment)


def test_v37_checkpoint_load_rejects_bad_hash_but_v36_is_unchanged(tmp_path):
    checkpoint = tmp_path / "global_step_4"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"weights")
    environment = {
        "V37_RUN_CLASS": "debug",
        "V37_RESUME_MODE": "controlled_continuation",
        "V37_EXPECTED_RESUME_CHECKPOINT_PATH": str(checkpoint),
        "V37_EXPECTED_RESUME_CHECKPOINT_SHA256": "not-a-sha",
    }
    with pytest.raises(evidence.TrainingEvidenceError, match="lowercase full SHA256"):
        evidence.validate_resume_checkpoint_binding(checkpoint, environment)
    with pytest.raises(evidence.TrainingEvidenceError, match="explicit valid V37_RUN_CLASS"):
        evidence.validate_resume_checkpoint_binding(
            checkpoint,
            {
                "V37_RESUME_MODE": "controlled_continuation",
                "V37_EXPECTED_RESUME_CHECKPOINT_PATH": str(checkpoint),
                "V37_EXPECTED_RESUME_CHECKPOINT_SHA256": evidence.tree_sha256(checkpoint),
            },
        )
    assert evidence.validate_resume_checkpoint_binding(checkpoint, {}) is None
    assert evidence.validate_resume_checkpoint_binding(
        checkpoint, {"V37_REQUIRE_EXACT_RAY_GPUS": "1"},
    ) is None


def test_v37_checkpoint_binding_survives_lost_environment_via_trainer_config(tmp_path):
    checkpoint = tmp_path / "global_step_4"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"weights")
    binding = {
        "V37_RUN_CLASS": "canary",
        "V37_RESUME_MODE": "controlled_continuation",
        "V37_EXPECTED_RESUME_CHECKPOINT_PATH": str(checkpoint),
        "V37_EXPECTED_RESUME_CHECKPOINT_SHA256": evidence.tree_sha256(checkpoint),
    }
    assert evidence.validate_resume_checkpoint_binding(
        checkpoint, {}, config_binding=binding,
    ) == binding["V37_EXPECTED_RESUME_CHECKPOINT_SHA256"]
    with pytest.raises(evidence.TrainingEvidenceError, match="differs from the V37 runtime"):
        evidence.validate_resume_checkpoint_binding(
            checkpoint,
            {"V37_RUN_CLASS": "debug"},
            config_binding=binding,
        )


def test_recorder_finalizes_atomic_recomputable_artifact(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    manifest = _manifest(run)
    checkpoint = run / "global_step_2"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"weights")
    recorder = evidence.TrainingEvidenceRecorder(
        run / "v37_training_evidence.json", manifest, 2,
    )
    recorder.record_step(_record(1))
    recorder.record_step(_record(2))
    payload = recorder.finalize(
        completed=True,
        checkpoint_id="progress-seed11-global_step_2",
        checkpoint_path=checkpoint,
        checkpoint_sha256=evidence.tree_sha256(checkpoint),
        kl_recoverable=True,
    )
    assert payload["optimizer_steps"] == 2
    assert payload["optimizer_microsteps_executed"] == 4
    assert payload["oom_count"] == 0
    assert payload["event_mapping_coverage"] == 1.0
    assert payload["runtime_mask_coverage"] == 1.0
    assert payload["frontier_mixed_group_ratio"] == 0.75
    assert payload["outcome_frontier_allwrong_ratio"] == 0.25
    assert evidence.verify_file(run / "v37_training_evidence.json") == payload


def test_validation_oom_telemetry_updates_latest_step_and_is_recomputed(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    recorder = evidence.TrainingEvidenceRecorder(
        run / "evidence.json", _manifest(run), 2,
    )
    recorder.record_step(_record(1))
    recorder.update_latest_oom_count(2)
    second = _record(2)
    second["oom_count_cumulative"] = 2
    recorder.record_step(second)
    recorder.update_latest_oom_count(3)
    checkpoint = run / "global_step_2"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"weights")
    payload = recorder.finalize(
        completed=True,
        checkpoint_id="progress-seed11-global_step_2",
        checkpoint_path=checkpoint,
        checkpoint_sha256=evidence.tree_sha256(checkpoint),
        kl_recoverable=True,
    )
    assert payload["oom_count"] == 3
    assert evidence.verify_payload(payload) == payload

    trainer_source = (
        Path(__file__).resolve().parents[1] / "verl/trainer/ray_trainer.py"
    ).read_text(encoding="utf-8")
    assert '"oom_count_cumulative": self._v37_oom_count_cumulative' in trainer_source
    assert "v37_evidence.update_latest_oom_count(" in trainer_source
    validation_source = trainer_source[
        trainer_source.index("def _validate("):
        trainer_source.index("def _check_gpu_memory_usage")
    ]
    high_memory_fallback = validation_source.index("if exceeds_threshold:")
    per_sample_fallback = validation_source.index(
        "self._validate_per_sample(", high_memory_fallback,
    )
    assert "self._v37_oom_count_cumulative += 1" in validation_source[
        high_memory_fallback:per_sample_fallback
    ]
    finalizer = trainer_source[trainer_source.index("def _finalize_training("):]
    terminal_validation = finalizer.index("val_metrics = self._validate()")
    terminal_oom_update = finalizer.index(
        "v37_evidence.update_latest_oom_count(", terminal_validation,
    )
    terminal_log = finalizer.index("self.logger.log(data=val_metrics", terminal_validation)
    assert terminal_validation < terminal_oom_update < terminal_log


def test_recorder_rejects_overwrite_and_noncontiguous_steps(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    manifest = _manifest(run)
    output = run / "v37_training_evidence.json"
    output.write_text("{}", encoding="utf-8")
    with pytest.raises(evidence.TrainingEvidenceError, match="already exists"):
        evidence.TrainingEvidenceRecorder(output, manifest, 2)
    output.unlink()
    recorder = evidence.TrainingEvidenceRecorder(output, manifest, 2)
    with pytest.raises(evidence.TrainingEvidenceError, match="contiguous"):
        recorder.record_step(_record(2))


def test_controlled_continuation_records_only_new_absolute_steps(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    manifest = _manifest(run)
    checkpoint = run / "global_step_5"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"weights")
    recorder = evidence.TrainingEvidenceRecorder(
        run / "evidence.json", manifest, expected_steps=5, start_global_step=3,
    )
    recorder.record_step(_record(4))
    recorder.record_step(_record(5))
    payload = recorder.finalize(
        completed=True,
        checkpoint_id="progress-seed11-global_step_5",
        checkpoint_path=checkpoint,
        checkpoint_sha256=evidence.tree_sha256(checkpoint),
        kl_recoverable=True,
    )
    assert payload["start_global_step"] == 3
    assert payload["final_global_step"] == 5
    assert payload["expected_optimizer_steps"] == 2
    assert payload["optimizer_steps"] == 2
    assert evidence.verify_payload(payload) == payload


def test_checkpoint_path_is_bound_to_absolute_final_step(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    manifest = _manifest(run)
    checkpoint = run / "global_step_2"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"weights")
    recorder = evidence.TrainingEvidenceRecorder(run / "evidence.json", manifest, 2)
    recorder.record_step(_record(1))
    recorder.record_step(_record(2))
    payload = recorder.finalize(
        completed=True,
        checkpoint_id="baseline-seed11-global_step_2",
        checkpoint_path=checkpoint,
        checkpoint_sha256=evidence.tree_sha256(checkpoint),
        kl_recoverable=True,
    )
    payload["final_checkpoint_path"] = str(run / "global_step_1")
    with pytest.raises(evidence.TrainingEvidenceError, match="final_global_step"):
        evidence.verify_payload(payload)


def test_trainer_nonfinite_counter_is_rank_agreed_before_formal_evidence():
    root = Path(__file__).resolve().parents[1]
    actor_source = (root / "verl/workers/actor/dp_actor.py").read_text(encoding="utf-8")
    trainer_source = (root / "verl/trainer/ray_trainer.py").read_text(encoding="utf-8")
    assert '"nonfinite_grad_count", self._nonfinite_count, metric_device' in actor_source
    assert 'metrics["actor/nonfinite_counter_agreement"] = [1.0]' in actor_source
    assert 'actor_metrics.get("actor/nonfinite_counter_agreement") != 1.0' in trainer_source


def test_formal_hf_merge_happens_inside_checkpoint_staging_before_manifest_publication():
    root = Path(__file__).resolve().parents[1]
    trainer_source = (root / "verl/trainer/ray_trainer.py").read_text(encoding="utf-8")
    merge_call = trainer_source.index("subprocess.run(\n                    merge_command")
    manifest_write = trainer_source.index("write_checkpoint_manifest(staging_path")
    publish = trainer_source.index("publish_staged_checkpoint(staging_path")
    assert merge_call < manifest_write < publish
    assert 'required_paths.append("actor/huggingface")' in trainer_source
    assert (
        "finalize_hf = finalize_hf_requested and self.global_step == self.training_steps"
        in trainer_source
    )
    assert "self.global_step != self.training_steps" not in trainer_source[
        trainer_source.index("finalize_hf_requested"):merge_call
    ]
    assert "produced no regular safetensors weights" in trainer_source
    assert '"--host-memory-preflight"' in trainer_source
    assert '"--host-memory-safety-factor"' in trainer_source


def test_v36_checkpoint_feature_off_keeps_legacy_save_path():
    trainer_source = (
        Path(__file__).resolve().parents[1] / "verl/trainer/ray_trainer.py"
    ).read_text(encoding="utf-8")
    predicate = trainer_source.index("sealed_checkpoint = (")
    legacy = trainer_source.index("if not sealed_checkpoint:", predicate)
    sealed_guard = trainer_source.index(
        "if sealed_checkpoint and not self._current_step_complete:", legacy
    )
    staging = trainer_source.index("staging_path =", sealed_guard)
    assert legacy < sealed_guard < staging
    assert 'os.environ.get("V37_TRAINING_EVIDENCE_REQUIRED", "0") == "1"' in trainer_source
    assert 'os.environ.get("V37_FINALIZE_HF_CHECKPOINT", "0") == "1"' in trainer_source


def test_payload_rejects_tamper_and_nonmonotonic_counter(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    manifest = _manifest(run)
    checkpoint = run / "global_step_2"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"weights")
    recorder = evidence.TrainingEvidenceRecorder(run / "evidence.json", manifest, 2)
    recorder.record_step(_record(1, nonfinite=1))
    recorder.record_step(_record(2, nonfinite=1))
    payload = recorder.finalize(
        completed=True,
        checkpoint_id="baseline-seed11-global_step_2",
        checkpoint_path=checkpoint,
        checkpoint_sha256=evidence.tree_sha256(checkpoint),
        kl_recoverable=True,
    )
    tampered = json.loads(json.dumps(payload))
    tampered["step_records"][1]["nonfinite_count_cumulative"] = 0
    tampered["records_sha256"] = evidence._sha(tampered["step_records"])
    with pytest.raises(evidence.TrainingEvidenceError, match="monotonic"):
        evidence.verify_payload(tampered)
    tampered = json.loads(json.dumps(payload))
    tampered["step_records"][0]["oom_count_cumulative"] = 2
    tampered["step_records"][1]["oom_count_cumulative"] = 1
    tampered["records_sha256"] = evidence._sha(tampered["step_records"])
    with pytest.raises(evidence.TrainingEvidenceError, match="oom_count.*monotonic"):
        evidence.verify_payload(tampered)
    tampered = json.loads(json.dumps(payload))
    tampered["event_mapping_coverage"] = 0.5
    with pytest.raises(evidence.TrainingEvidenceError, match="summary cache mismatch"):
        evidence.verify_payload(tampered)


def test_feature_off_is_noop_and_required_is_fail_closed(monkeypatch):
    monkeypatch.delenv("V37_TRAINING_EVIDENCE_PATH", raising=False)
    monkeypatch.delenv("V37_TRAINING_EVIDENCE_REQUIRED", raising=False)
    assert evidence.TrainingEvidenceRecorder.from_environment() is None
    monkeypatch.setenv("V37_TRAINING_EVIDENCE_REQUIRED", "1")
    with pytest.raises(evidence.TrainingEvidenceError, match="is required"):
        evidence.TrainingEvidenceRecorder.from_environment()
