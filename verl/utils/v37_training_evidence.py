"""Fail-closed, trainer-owned V37 optimizer/reward evidence."""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 2
ARTIFACT_TYPE = "v37_training_evidence"
STEP_KEYS = {
    "global_step",
    "optimizer_microsteps_attempted",
    "optimizer_microsteps_executed",
    "optimizer_microsteps_skipped",
    "nonfinite_count_cumulative",
    "oom_count_cumulative",
    "sample_count",
    "progress_degraded_count",
    "cap_turn_exceeded_count",
    "early_stop_count",
    "selector_kl_contribution",
    "action_rows_expected",
    "action_rows_mapped",
    "mask_rows_expected",
    "mask_rows_scored",
    "frontier_group_count",
    "frontier_mixed_group_count",
    "outcome_allwrong_group_count",
    "correctness_sign_error_count",
}
SUMMARY_KEYS = {
    "optimizer_steps",
    "optimizer_microsteps_attempted",
    "optimizer_microsteps_executed",
    "skipped_updates",
    "nonfinite_count",
    "oom_count",
    "progress_degraded",
    "cap_turn_exceeded",
    "early_stop_count",
    "selector_kl_contribution",
    "event_mapping_coverage",
    "runtime_mask_coverage",
    "kl_recoverable",
    "frontier_mixed_group_ratio",
    "outcome_frontier_allwrong_ratio",
    "correctness_sign_error_count",
}
TOP_KEYS = {
    "schema_version",
    "artifact_type",
    "evidence_run_id",
    "execution_environment_sha256",
    "completed",
    "start_global_step",
    "final_global_step",
    "expected_optimizer_steps",
    "final_checkpoint_id",
    "final_checkpoint_path",
    "final_checkpoint_sha256",
    "step_records",
    "records_sha256",
    *SUMMARY_KEYS,
}


class TrainingEvidenceError(ValueError):
    """The structured V37 training evidence contract was violated."""


def require_clean_optimizer_step(
    attempted: int,
    executed: int,
    skipped: int,
    nonfinite_cumulative: int,
) -> None:
    """Reject an incomplete V37 update before it can become checkpoint-visible."""
    attempted = _integer(attempted, "optimizer_microsteps_attempted", minimum=1)
    executed = _integer(executed, "optimizer_microsteps_executed")
    skipped = _integer(skipped, "optimizer_microsteps_skipped")
    nonfinite_cumulative = _integer(nonfinite_cumulative, "nonfinite_count_cumulative")
    if executed + skipped != attempted:
        raise TrainingEvidenceError("optimizer microstep counters are inconsistent")
    if skipped or executed != attempted:
        raise TrainingEvidenceError(
            "formal V37 optimizer update was skipped or only partially executed"
        )
    if nonfinite_cumulative:
        raise TrainingEvidenceError(
            "formal V37 observed a non-finite gradient before checkpoint publication"
        )


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TrainingEvidenceError("training evidence is not finite canonical JSON") from exc


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TrainingEvidenceError(f"{label} must be an integer >= {minimum}")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingEvidenceError(f"{label} must be finite numeric")
    result = float(value)
    if not math.isfinite(result):
        raise TrainingEvidenceError(f"{label} must be finite numeric")
    return result


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise TrainingEvidenceError(f"{label} must be a lowercase full SHA256")
    return value


def _absolute_nonsymlink(path: os.PathLike[str] | str, label: str, *, missing_leaf: bool = False) -> Path:
    absolute = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    current = Path(absolute.anchor)
    for index, part in enumerate(absolute.parts[1:]):
        current /= part
        try:
            metadata = os.lstat(current)
        except (FileNotFoundError, NotADirectoryError) as exc:
            if missing_leaf and index == len(absolute.parts[1:]) - 1:
                return absolute
            raise TrainingEvidenceError(f"{label} does not exist: {current}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise TrainingEvidenceError(f"{label} contains a symlink component: {current}")
    return absolute


def tree_sha256(path: os.PathLike[str] | str) -> str:
    """Hash a canonical relative-path/size/file-digest manifest."""
    root = _absolute_nonsymlink(path, "checkpoint tree")
    if not root.is_dir():
        raise TrainingEvidenceError(f"checkpoint tree is not a directory: {root}")
    for walk_root, directories, filenames in os.walk(root, followlinks=False):
        for name in directories + filenames:
            candidate = Path(walk_root) / name
            if candidate.is_symlink():
                raise TrainingEvidenceError(f"checkpoint tree contains a symlink: {candidate}")
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise TrainingEvidenceError(f"checkpoint tree is empty: {root}")
    entries = []
    for item in files:
        if item.is_symlink():
            raise TrainingEvidenceError(f"checkpoint tree contains a symlink: {item}")
        digest = hashlib.sha256()
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        entries.append({
            "path": item.relative_to(root).as_posix(),
            "size": item.stat().st_size,
            "sha256": digest.hexdigest(),
        })
    return hashlib.sha256(_canonical_bytes(entries)).hexdigest()


def validate_resume_checkpoint_binding(
    checkpoint_path: os.PathLike[str] | str,
    environment: Mapping[str, str] | None = None,
    config_binding: Mapping[str, str | None] | None = None,
) -> str | None:
    """Bind a V37 checkpoint load to the path/tree hash validated by its seal."""
    values = os.environ if environment is None else environment
    config_values = {} if config_binding is None else dict(config_binding)
    binding_keys = (
        "V37_RUN_CLASS",
        "V37_RESUME_MODE",
        "V37_EXPECTED_RESUME_CHECKPOINT_PATH",
        "V37_EXPECTED_RESUME_CHECKPOINT_SHA256",
    )
    resolved: dict[str, str] = {}
    for key in binding_keys:
        environment_value = values.get(key)
        config_value = config_values.get(key)
        if config_value is not None:
            if not isinstance(config_value, str):
                raise TrainingEvidenceError(f"configured {key} must be a string")
            if environment_value is not None and environment_value != config_value:
                raise TrainingEvidenceError(f"configured {key} differs from the V37 runtime environment")
            resolved[key] = config_value
        elif environment_value is not None:
            resolved[key] = environment_value

    run_class = resolved.get("V37_RUN_CLASS")
    if run_class not in {"debug", "canary", "formal"}:
        identity_markers = {
            "V37_RUN_CLASS",
            "V37_ARM",
            "V37_SEED",
            "V37_RUN_PURPOSE",
            "V37_CONTINUATION_MODE",
            "V37_RESUME_MODE",
            "V37_EXPECTED_RESUME_CHECKPOINT_PATH",
            "V37_EXPECTED_RESUME_CHECKPOINT_SHA256",
            "V37_RUN_MANIFEST",
            "V37_TRAINING_EVIDENCE_REQUIRED",
        }
        has_marker = any(key in values for key in identity_markers) or any(
            value is not None for value in config_values.values()
        )
        if has_marker:
            raise TrainingEvidenceError(
                "V37 checkpoint load markers require an explicit valid V37_RUN_CLASS"
            )
        return None
    if run_class == "formal":
        raise TrainingEvidenceError("formal V37 must load from a clean start, not a checkpoint")
    if resolved.get("V37_RESUME_MODE") != "controlled_continuation":
        raise TrainingEvidenceError(
            "a V37 checkpoint load requires V37_RESUME_MODE=controlled_continuation"
        )
    expected_path = resolved.get("V37_EXPECTED_RESUME_CHECKPOINT_PATH")
    if not isinstance(expected_path, str) or not os.path.isabs(expected_path):
        raise TrainingEvidenceError("controlled continuation expected checkpoint path must be absolute")
    checkpoint = _absolute_nonsymlink(checkpoint_path, "controlled continuation checkpoint")
    expected_checkpoint = _absolute_nonsymlink(
        expected_path, "controlled continuation expected checkpoint",
    )
    if checkpoint != expected_checkpoint:
        raise TrainingEvidenceError(
            "controlled continuation checkpoint path differs from the sealed runtime binding"
        )
    expected_sha256 = _sha256(
        resolved.get("V37_EXPECTED_RESUME_CHECKPOINT_SHA256"),
        "controlled continuation expected checkpoint SHA256",
    )
    observed_sha256 = tree_sha256(checkpoint)
    if observed_sha256 != expected_sha256:
        raise TrainingEvidenceError(
            "controlled continuation checkpoint changed before Trainer load"
        )
    return observed_sha256


def _strict_json_file(path: Path, label: str) -> dict[str, Any]:
    path = _absolute_nonsymlink(path, label)
    if not path.is_file():
        raise TrainingEvidenceError(f"{label} must be a regular file: {path}")

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise TrainingEvidenceError(f"{label} contains duplicate key {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda item: (_ for _ in ()).throw(
                TrainingEvidenceError(f"{label} contains non-finite constant {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainingEvidenceError(f"{label} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise TrainingEvidenceError(f"{label} must be a JSON object")
    _canonical_bytes(value)
    return value


def reward_counts(metrics: Mapping[str, Any], sample_count: int) -> dict[str, int]:
    """Extract exact row-level counters before ``reduce_metrics`` destroys lists."""
    count = _integer(sample_count, "sample_count", minimum=1)

    def binary_sum(name: str, *, required: bool = False) -> int:
        values = metrics.get(name)
        if values is None:
            if required:
                raise TrainingEvidenceError(f"reward metric {name} is required")
            return 0
        if not isinstance(values, (list, tuple)):
            raise TrainingEvidenceError(f"reward metric {name} must be row-aligned")
        if len(values) != count:
            raise TrainingEvidenceError(f"reward metric {name} must contain {count} rows")
        total = 0
        for value in values:
            numeric = _finite(value, f"reward metric {name}")
            if numeric not in (0.0, 1.0):
                raise TrainingEvidenceError(f"reward metric {name} must be binary")
            total += int(numeric)
        return total

    mask_values = metrics.get("mask_evidence_complete")
    if not isinstance(mask_values, (list, tuple)) or len(mask_values) != count:
        raise TrainingEvidenceError("mask_evidence_complete must be present for every V37 reward row")
    mask_scored = binary_sum("mask_evidence_complete", required=True)
    return {
        "sample_count": count,
        "progress_degraded_count": binary_sum("progress_degraded"),
        "cap_turn_exceeded_count": binary_sum("turns_exceeded", required=True),
        "early_stop_count": binary_sum("early_stop", required=True),
        "mask_rows_expected": count,
        "mask_rows_scored": mask_scored,
    }


def _validate_step(record: Any, expected_step: int) -> dict[str, Any]:
    if not isinstance(record, dict) or set(record) != STEP_KEYS:
        raise TrainingEvidenceError(f"step {expected_step} schema mismatch")
    result = dict(record)
    if _integer(result["global_step"], "global_step", minimum=1) != expected_step:
        raise TrainingEvidenceError("training evidence steps must be contiguous from 1")
    integer_fields = STEP_KEYS - {"global_step", "selector_kl_contribution"}
    for field in integer_fields:
        _integer(result[field], field)
    attempted = result["optimizer_microsteps_attempted"]
    executed = result["optimizer_microsteps_executed"]
    skipped = result["optimizer_microsteps_skipped"]
    if attempted < 1 or executed + skipped != attempted:
        raise TrainingEvidenceError("optimizer microstep counters are inconsistent")
    if result["action_rows_mapped"] > result["action_rows_expected"]:
        raise TrainingEvidenceError("mapped action rows exceed expected rows")
    if result["action_rows_expected"] not in (0, result["sample_count"]):
        raise TrainingEvidenceError("expected action rows must be zero or the complete sample batch")
    if result["mask_rows_scored"] > result["mask_rows_expected"]:
        raise TrainingEvidenceError("scored mask rows exceed expected rows")
    if result["mask_rows_expected"] != result["sample_count"]:
        raise TrainingEvidenceError("expected mask rows must equal the complete sample batch")
    groups = result["frontier_group_count"]
    if result["frontier_mixed_group_count"] > groups or result["outcome_allwrong_group_count"] > groups:
        raise TrainingEvidenceError("frontier group counters are inconsistent")
    _finite(result["selector_kl_contribution"], "selector_kl_contribution")
    return result


def _summarize(records: Sequence[Mapping[str, Any]], kl_recoverable: bool) -> dict[str, Any]:
    if not records:
        raise TrainingEvidenceError("training evidence contains no optimizer steps")
    attempted = sum(int(row["optimizer_microsteps_attempted"]) for row in records)
    executed = sum(int(row["optimizer_microsteps_executed"]) for row in records)
    skipped = sum(int(row["optimizer_microsteps_skipped"]) for row in records)
    action_expected = sum(int(row["action_rows_expected"]) for row in records)
    action_mapped = sum(int(row["action_rows_mapped"]) for row in records)
    mask_expected = sum(int(row["mask_rows_expected"]) for row in records)
    mask_scored = sum(int(row["mask_rows_scored"]) for row in records)
    groups = sum(int(row["frontier_group_count"]) for row in records)
    mixed = sum(int(row["frontier_mixed_group_count"]) for row in records)
    allwrong = sum(int(row["outcome_allwrong_group_count"]) for row in records)
    selector = max(abs(float(row["selector_kl_contribution"])) for row in records)
    return {
        "optimizer_steps": sum(
            int(row["optimizer_microsteps_executed"] > 0 and row["optimizer_microsteps_skipped"] == 0)
            for row in records
        ),
        "optimizer_microsteps_attempted": attempted,
        "optimizer_microsteps_executed": executed,
        "skipped_updates": skipped,
        "nonfinite_count": max(int(row["nonfinite_count_cumulative"]) for row in records),
        "oom_count": max(int(row["oom_count_cumulative"]) for row in records),
        "progress_degraded": sum(int(row["progress_degraded_count"]) for row in records),
        "cap_turn_exceeded": sum(int(row["cap_turn_exceeded_count"]) for row in records),
        "early_stop_count": sum(int(row["early_stop_count"]) for row in records),
        "selector_kl_contribution": selector,
        "event_mapping_coverage": 1.0 if action_expected == 0 else action_mapped / action_expected,
        "runtime_mask_coverage": 1.0 if mask_expected == 0 else mask_scored / mask_expected,
        "kl_recoverable": 1.0 if kl_recoverable else 0.0,
        "frontier_mixed_group_ratio": 0.0 if groups == 0 else mixed / groups,
        "outcome_frontier_allwrong_ratio": 1.0 if groups == 0 else allwrong / groups,
        "correctness_sign_error_count": sum(
            int(row["correctness_sign_error_count"]) for row in records
        ),
    }


class TrainingEvidenceRecorder:
    def __init__(
        self,
        output_path: Path,
        manifest_path: Path,
        expected_steps: int,
        start_global_step: int = 0,
    ):
        self.output_path = _absolute_nonsymlink(
            output_path, "training evidence output", missing_leaf=True
        )
        self.manifest_path = _absolute_nonsymlink(manifest_path, "V37 run manifest")
        if self.output_path.parent != self.manifest_path.parent:
            raise TrainingEvidenceError("training evidence must stay inside the run directory")
        if self.output_path.exists() or self.output_path.is_symlink():
            raise TrainingEvidenceError(f"training evidence output already exists: {self.output_path}")
        self.expected_steps = _integer(expected_steps, "expected optimizer steps", minimum=1)
        self.start_global_step = _integer(start_global_step, "start global step")
        if self.start_global_step >= self.expected_steps:
            raise TrainingEvidenceError(
                "training evidence final target must exceed its start global step"
            )
        self.expected_new_steps = self.expected_steps - self.start_global_step
        manifest = _strict_json_file(self.manifest_path, "V37 run manifest")
        evidence_run_id = manifest.get("evidence_run_id")
        if not isinstance(evidence_run_id, str) or not evidence_run_id:
            raise TrainingEvidenceError("V37 run manifest lacks evidence_run_id")
        self.evidence_run_id = evidence_run_id
        self.execution_environment_sha256 = _sha256(
            manifest.get("execution_environment_sha256"), "execution_environment_sha256"
        )
        self.records: list[dict[str, Any]] = []

    @classmethod
    def from_environment(cls) -> "TrainingEvidenceRecorder | None":
        output = os.environ.get("V37_TRAINING_EVIDENCE_PATH")
        required = os.environ.get("V37_TRAINING_EVIDENCE_REQUIRED", "0") == "1"
        if not output:
            if required:
                raise TrainingEvidenceError("V37_TRAINING_EVIDENCE_PATH is required")
            return None
        manifest = os.environ.get("V37_RUN_MANIFEST")
        expected = os.environ.get("V37_PILOT_STEPS")
        start = os.environ.get("V37_EVIDENCE_START_STEP", "0")
        if (
            not manifest
            or not expected
            or not expected.isdigit()
            or not start.isdigit()
        ):
            raise TrainingEvidenceError("V37 training evidence environment is incomplete")
        return cls(Path(output), Path(manifest), int(expected), int(start))

    def record_step(self, record: Mapping[str, Any]) -> None:
        expected_step = self.start_global_step + len(self.records) + 1
        if len(self.records) >= self.expected_new_steps:
            raise TrainingEvidenceError("training emitted more evidence steps than preregistered")
        self.records.append(_validate_step(dict(record), expected_step))

    def update_latest_oom_count(self, cumulative: int) -> None:
        """Attach validation OOM telemetry that was observed after step recording."""
        if not self.records:
            raise TrainingEvidenceError("cannot attach OOM telemetry before the first step")
        updated = dict(self.records[-1])
        updated["oom_count_cumulative"] = _integer(cumulative, "oom_count_cumulative")
        previous = self.records[-2]["oom_count_cumulative"] if len(self.records) > 1 else 0
        if updated["oom_count_cumulative"] < previous:
            raise TrainingEvidenceError("oom_count must be cumulative and monotonic")
        self.records[-1] = _validate_step(updated, updated["global_step"])

    def finalize(
        self,
        *,
        completed: bool,
        checkpoint_id: str,
        checkpoint_path: Path,
        checkpoint_sha256: str,
        kl_recoverable: bool,
    ) -> dict[str, Any]:
        if type(completed) is not bool or not completed:
            raise TrainingEvidenceError("formal training did not complete all preregistered steps")
        if len(self.records) != self.expected_new_steps:
            raise TrainingEvidenceError(
                f"expected {self.expected_new_steps} new training records, got {len(self.records)}"
            )
        checkpoint_path = _absolute_nonsymlink(checkpoint_path, "final checkpoint")
        if not isinstance(checkpoint_id, str) or not checkpoint_id:
            raise TrainingEvidenceError("final checkpoint ID is missing")
        checkpoint_hash = _sha256(checkpoint_sha256, "final checkpoint SHA256")
        summary = _summarize(self.records, kl_recoverable)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": ARTIFACT_TYPE,
            "evidence_run_id": self.evidence_run_id,
            "execution_environment_sha256": self.execution_environment_sha256,
            "completed": True,
            "start_global_step": self.start_global_step,
            "final_global_step": self.expected_steps,
            "expected_optimizer_steps": self.expected_new_steps,
            "final_checkpoint_id": checkpoint_id,
            "final_checkpoint_path": str(checkpoint_path),
            "final_checkpoint_sha256": checkpoint_hash,
            "step_records": self.records,
            "records_sha256": _sha(self.records),
            **summary,
        }
        verify_payload(payload)
        parent = _absolute_nonsymlink(self.output_path.parent, "training evidence parent")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.output_path.name}.", suffix=".tmp", dir=parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(json.dumps(
                    payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
                ).encode("utf-8") + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            if self.output_path.exists() or self.output_path.is_symlink():
                raise TrainingEvidenceError("training evidence output appeared concurrently")
            os.link(temporary, self.output_path)
            directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)
        return payload


def verify_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != TOP_KEYS:
        actual = set(payload) if isinstance(payload, dict) else set()
        raise TrainingEvidenceError(
            f"training evidence schema mismatch; missing={sorted(TOP_KEYS - actual)}, "
            f"extra={sorted(actual - TOP_KEYS)}"
        )
    if payload["schema_version"] != SCHEMA_VERSION or payload["artifact_type"] != ARTIFACT_TYPE:
        raise TrainingEvidenceError("training evidence version/type mismatch")
    if payload["completed"] is not True:
        raise TrainingEvidenceError("training evidence is not complete")
    if not isinstance(payload["evidence_run_id"], str) or not payload["evidence_run_id"]:
        raise TrainingEvidenceError("training evidence_run_id is missing")
    _sha256(payload["execution_environment_sha256"], "execution_environment_sha256")
    start_step = _integer(payload["start_global_step"], "start_global_step")
    final_step = _integer(payload["final_global_step"], "final_global_step", minimum=1)
    expected_steps = _integer(payload["expected_optimizer_steps"], "expected_optimizer_steps", minimum=1)
    if final_step - start_step != expected_steps:
        raise TrainingEvidenceError("training evidence start/final/update count mismatch")
    records = payload["step_records"]
    if not isinstance(records, list) or len(records) != expected_steps:
        raise TrainingEvidenceError("training step record count mismatch")
    validated = [
        _validate_step(record, start_step + index + 1)
        for index, record in enumerate(records)
    ]
    if payload["records_sha256"] != _sha(validated):
        raise TrainingEvidenceError("training step record hash mismatch")
    nonfinite = [int(record["nonfinite_count_cumulative"]) for record in validated]
    if any(current < previous for previous, current in zip(nonfinite, nonfinite[1:])):
        raise TrainingEvidenceError("nonfinite_grad_count must be cumulative and monotonic")
    oom = [int(record["oom_count_cumulative"]) for record in validated]
    if any(current < previous for previous, current in zip(oom, oom[1:])):
        raise TrainingEvidenceError("oom_count must be cumulative and monotonic")
    kl_recoverable = _finite(payload["kl_recoverable"], "kl_recoverable")
    if kl_recoverable not in (0.0, 1.0):
        raise TrainingEvidenceError("kl_recoverable must be binary")
    recomputed = _summarize(validated, bool(kl_recoverable))
    for key, expected in recomputed.items():
        actual = payload[key]
        if isinstance(expected, float):
            if _finite(actual, key) != expected:
                raise TrainingEvidenceError(f"training summary cache mismatch: {key}")
        elif _integer(actual, key) != expected:
            raise TrainingEvidenceError(f"training summary cache mismatch: {key}")
    if payload["optimizer_steps"] != expected_steps:
        raise TrainingEvidenceError("not every trainer step completed an optimizer update")
    if not isinstance(payload["final_checkpoint_id"], str) or not payload["final_checkpoint_id"]:
        raise TrainingEvidenceError("final checkpoint ID is missing")
    checkpoint_path = payload["final_checkpoint_path"]
    if not isinstance(checkpoint_path, str) or not Path(checkpoint_path).is_absolute():
        raise TrainingEvidenceError("final checkpoint path must be absolute")
    if Path(checkpoint_path).name != f"global_step_{final_step}":
        raise TrainingEvidenceError("final checkpoint path does not match final_global_step")
    _sha256(payload["final_checkpoint_sha256"], "final_checkpoint_sha256")
    return payload


def verify_file(path: os.PathLike[str] | str) -> dict[str, Any]:
    return verify_payload(_strict_json_file(Path(path), "V37 training evidence"))
