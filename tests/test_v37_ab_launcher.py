import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "examples" / "rl_launch" / "run_v37_strict_ab.sh"


FAKE_WRAPPER = r'''#!/usr/bin/env python3
import json
import os
import sys
import time
from pathlib import Path

root = Path(os.environ["V37_AB_PLAN_PATH"]).parent
active = root / ".active-child"
overlap = root / "overlap"
try:
    active.mkdir()
except FileExistsError:
    overlap.write_text("overlap", encoding="utf-8")
    raise SystemExit(91)

keys = (
    "V37_AB_RUN_ID", "V37_AB_PLAN_PATH", "V37_AB_PLAN_SHA256", "V37_AB_CELL_ID",
    "V37_ARM", "V37_SEED", "V31_SAVE_CHECKPOINT_PATH", "V37_RUN_CLASS",
    "V37_DATA_MODE", "V37_RUN_PURPOSE", "V37_CONTINUATION_MODE", "V37_CONTRACT_ONLY",
    "V37_PILOT_STEPS", "V31_NNODES", "V31_N_GPUS_PER_NODE", "HOST_NUM", "HOST_GPU_NUM",
    "INDEX", "BASH_ENV", "ENV", "V37_GATE_ONLY", "V37_PREFLIGHT_ONLY", "V37_DRY_RUN",
    "V37_VALIDATE_DOWNSTREAM", "V37_RESUME_CHECKPOINT", "V37_RESUME_SEAL",
    "V37_EXPECTED_RESUME_CHECKPOINT_PATH", "V37_EXPECTED_RESUME_CHECKPOINT_SHA256",
    "V36_LOAD_CHECKPOINT_PATH", "V32_LOAD_CHECKPOINT_PATH", "V32_DRY_RUN",
    "V37_ALLOW_FOCUSED10K_PILOT", "V37_ALLOW_BENCHMARK_DEV", "V31_EXPERIMENT_NAME",
    "V37_RUN_TIMESTAMP", "TRAINER_VAL_ONLY", "V37_AB_ROOT", "V37_AB_SEEDS",
    "V37_AB_SINGLE_RUN_WRAPPER",
    "WANDB_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN",
    "GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_CEILING_DIRECTORIES", "GIT_DISCOVERY_ACROSS_FILESYSTEM",
    "GIT_EXTERNAL_DIFF", "GIT_DIFF_OPTS",
)
record = {
    "argv": sys.argv,
    "environment": {key: os.environ.get(key) for key in keys},
    "bash_functions": sorted(key for key in os.environ if key.startswith("BASH_FUNC_")),
}
with (root / "invocations.jsonl").open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(record, allow_nan=False) + "\n")

mode = os.environ.get("FAKE_MODE", "normal")
cell = os.environ["V37_AB_CELL_ID"]
time.sleep(0.02)
try:
    plan_path = Path(os.environ["V37_AB_PLAN_PATH"])
    if cell == "seed11-baseline":
        if mode == "plan_tamper":
            plan_path.write_text("{}\n", encoding="utf-8")
        elif mode == "plan_symlink":
            backup = root / ".original-plan"
            backup.write_bytes(plan_path.read_bytes())
            plan_path.unlink()
            plan_path.symlink_to(backup)
        elif mode == "plan_nonregular":
            plan_path.unlink()
            plan_path.mkdir()
        elif mode == "precreate_future_cell":
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            Path(plan["cells"][1]["target_dir"]).mkdir()
    if mode == "plan_tamper_after_last" and cell == "seed22-progress":
        plan_path.write_text("{}\n", encoding="utf-8")
    if mode == "fail" and cell.endswith("-progress"):
        (root / "ab_evidence_index.json").write_text("untrusted child artifact", encoding="utf-8")
        raise SystemExit(7)
    if os.environ["V37_CONTRACT_ONLY"] == "1" or (mode == "missing" and cell.endswith("-progress")):
        raise SystemExit(0)
    target = Path(os.environ["V31_SAVE_CHECKPOINT_PATH"])
    target.mkdir()
    audited = {
        key: os.environ[key]
        for key in (
            "V37_AB_RUN_ID", "V37_AB_PLAN_PATH", "V37_AB_PLAN_SHA256", "V37_AB_CELL_ID",
            "V37_ARM", "V37_SEED", "V31_SAVE_CHECKPOINT_PATH", "V37_RUN_CLASS",
            "V37_DATA_MODE", "V37_PILOT_STEPS", "V31_NNODES", "V31_N_GPUS_PER_NODE",
            "V37_CONTINUATION_MODE",
        )
    }
    if mode == "tampered" and cell.endswith("-progress"):
        audited["V37_AB_PLAN_SHA256"] = "0" * 64
    evidence_id = "duplicate" if mode == "duplicate" else f"evidence-{cell}"
    manifest = {
        "manifest_version": 3,
        "run_class": "formal",
        "data_mode": "frontier_rl",
        "arm": os.environ["V37_ARM"],
        "seed": int(os.environ["V37_SEED"]),
        "config": {"world_size": 8},
        "audited_environment": audited,
        "ab_preregistration": {
            "schema_version": 1,
            "ab_run_id": os.environ["V37_AB_RUN_ID"],
            "plan_path": os.environ["V37_AB_PLAN_PATH"],
            "plan_sha256": os.environ["V37_AB_PLAN_SHA256"],
            "cell_id": os.environ["V37_AB_CELL_ID"],
        },
        "evidence_run_id": evidence_id,
    }
    manifest_path = target / "v37_run_manifest.json"
    if mode == "duplicate_json" and cell.endswith("-progress"):
        manifest_path.write_text('{"manifest_version":3,"manifest_version":3}', encoding="utf-8")
    else:
        manifest_path.write_text(json.dumps(manifest, allow_nan=False) + "\n", encoding="utf-8")
finally:
    active.rmdir()
'''


def make_wrapper(tmp_path: Path) -> Path:
    wrapper = tmp_path / "fake wrapper with spaces.py"
    wrapper.write_text(FAKE_WRAPPER, encoding="utf-8")
    wrapper.chmod(0o755)
    return wrapper


def clean_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("V37_AB_") or key.startswith("BASH_FUNC_") or key in {"BASH_ENV", "ENV"}:
            env.pop(key)
    return env


def run_ab(root: Path | None, wrapper: Path | None = None, **updates: str) -> subprocess.CompletedProcess[str]:
    env = clean_env()
    if root is not None:
        env["V37_AB_ROOT"] = str(root)
    if wrapper is not None:
        env["V37_AB_SINGLE_RUN_WRAPPER"] = str(wrapper)
    env.update({key: str(value) for key, value in updates.items()})
    return subprocess.run(
        ["bash", str(LAUNCHER)], cwd=REPO, env=env, text=True,
        capture_output=True, check=False,
    )


@pytest.mark.parametrize("value", ["true", "yes", "2", "-1"])
def test_postprocess_flag_requires_exact_boolean(tmp_path, value):
    result = run_ab(
        tmp_path / f"ab-invalid-postprocess-{value}",
        make_wrapper(tmp_path),
        V37_AB_POSTPROCESS=value,
    )
    assert result.returncode != 0
    assert "V37_AB_POSTPROCESS must be exactly 0 or 1" in result.stderr


def load_invocations(root: Path) -> list[dict]:
    path = root / "invocations.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("seed_text", "expected"),
    [(None, [11, 22]), ("0,37", [0, 37])],
)
def test_plan_exact_schema_seed_major_order_and_seed_override(tmp_path, seed_text, expected):
    root = tmp_path / f"plan {expected[0]} {expected[1]}"
    updates = {"V37_AB_PLAN_ONLY": "1"}
    if seed_text is not None:
        updates["V37_AB_SEEDS"] = seed_text
    result = run_ab(root, Path("/definitely/not/a/wrapper"), **updates)
    assert result.returncode == 0, result.stderr
    plan_path = root / "ab_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert set(plan) == {
        "schema_version", "ab_run_id", "plan_path", "seeds", "execution",
        "preregistered", "cells",
    }
    assert plan["schema_version"] == 2
    assert plan["plan_path"] == str(plan_path.resolve())
    assert plan["seeds"] == expected
    assert plan["execution"] == {
        "run_class": "formal", "data_mode": "frontier_rl", "optimizer_steps": 12,
        "nnodes": 1, "gpus_per_node": 8, "world_size": 8, "scheduling": "sequential",
    }
    assert plan["preregistered"]["complete"] is False
    assert plan["preregistered"]["input_snapshots"] is None
    pipeline = plan["preregistered"]["eval_contract"]["pipeline"]
    assert pipeline["producer"]["path"] == str((REPO / "tools/v37_eval_producer.py").resolve())
    assert pipeline["producer"]["sha256"] == sha256(REPO / "tools/v37_eval_producer.py")
    assert pipeline["recipe"] is None
    assert [(cell["seed"], cell["arm"]) for cell in plan["cells"]] == [
        (expected[0], "baseline"), (expected[0], "progress"),
        (expected[1], "baseline"), (expected[1], "progress"),
    ]
    assert all(set(cell) == {"cell_id", "seed", "arm", "target_dir"} for cell in plan["cells"])
    assert len({cell["cell_id"] for cell in plan["cells"]}) == 4
    assert len({cell["target_dir"] for cell in plan["cells"]}) == 4
    assert all(Path(cell["target_dir"]).is_absolute() for cell in plan["cells"])


def test_generated_run_ids_are_unique(tmp_path):
    roots = [tmp_path / "one", tmp_path / "two"]
    for root in roots:
        assert run_ab(root, V37_AB_PLAN_ONLY="1").returncode == 0
    ids = [json.loads((root / "ab_plan.json").read_text())["ab_run_id"] for root in roots]
    assert ids[0] != ids[1]


def test_initial_model_authority_hash_cannot_be_overridden(tmp_path):
    root = tmp_path / "wrong-model-authority"
    result = run_ab(
        root, V37_AB_PLAN_ONLY="1",
        V37_EXPECTED_INITIAL_MODEL_SHA256="f" * 64,
    )
    assert result.returncode != 0
    assert "cannot override checkpoint-476 identity" in result.stderr
    assert not root.exists()


@pytest.mark.parametrize("seeds", ["", "11", "11,22,33", "11,", ",22", "-1,22", "1.0,2", "11,11", "11, 22"])
def test_invalid_seed_forms_fail_before_root_claim(tmp_path, seeds):
    root = tmp_path / "bad-seeds"
    result = run_ab(root, V37_AB_PLAN_ONLY="1", V37_AB_SEEDS=seeds)
    # Empty means an explicit empty override and is invalid, not the default.
    if seeds == "":
        assert result.returncode != 0
    else:
        assert result.returncode != 0
    assert not root.exists()


@pytest.mark.parametrize(
    "updates",
    [
        {"V37_AB_PLAN_ONLY": "true"},
        {"V37_AB_CONTRACT_ONLY": "yes"},
        {"V37_AB_PLAN_ONLY": "1", "V37_AB_CONTRACT_ONLY": "1"},
    ],
)
def test_invalid_flags_fail_before_root_claim(tmp_path, updates):
    root = tmp_path / "bad-flags"
    result = run_ab(root, **updates)
    assert result.returncode != 0
    assert not root.exists()


def test_required_existing_and_symlink_roots_are_rejected(tmp_path):
    missing = run_ab(None, V37_AB_PLAN_ONLY="1")
    assert missing.returncode != 0
    assert "V37_AB_ROOT is required" in missing.stderr

    existing = tmp_path / "existing"
    existing.mkdir()
    assert run_ab(existing, V37_AB_PLAN_ONLY="1").returncode != 0

    symlink = tmp_path / "root-link"
    symlink.symlink_to(tmp_path / "nowhere")
    assert run_ab(symlink, V37_AB_PLAN_ONLY="1").returncode != 0


def test_root_inside_git_worktree_is_rejected_before_publication(tmp_path):
    root = REPO / f".v37-ab-inside-{tmp_path.name}"
    assert not root.exists()
    result = run_ab(root, V37_AB_PLAN_ONLY="1")
    assert result.returncode != 0
    assert "outside the Git worktree" in result.stderr
    assert not root.exists()


def test_invalid_provided_recipe_fails_before_first_child(tmp_path):
    root = tmp_path / "invalid-recipe"
    recipe = tmp_path / "invalid-recipe.json"
    recipe.write_text("{}\n", encoding="utf-8")
    result = run_ab(
        root,
        make_wrapper(tmp_path),
        V37_AB_EVAL_RECIPE=str(recipe),
    )
    assert result.returncode != 0
    assert "invalid formal eval recipe" in result.stderr
    assert not (root / "invocations.jsonl").exists()
    assert not (root / "ab_plan.json").exists()


def test_plan_only_never_resolves_or_invokes_wrapper(tmp_path):
    root = tmp_path / "plan-only"
    marker = tmp_path / "must-not-exist"
    result = run_ab(
        root, Path("/missing/wrapper with spaces"), V37_AB_PLAN_ONLY="1",
        FAKE_MARKER=str(marker),
    )
    assert result.returncode == 0, result.stderr
    assert (root / "ab_plan.json").is_file()
    assert not (root / "invocations.jsonl").exists()
    assert not (root / "ab_evidence_index.json").exists()
    assert not marker.exists()


def test_plan_only_accepts_python_launcher_path_with_spaces(tmp_path):
    python_launcher = tmp_path / "python launcher with spaces"
    python_launcher.symlink_to(Path(sys.executable))
    root = tmp_path / "plan-only-python-spaces"
    result = run_ab(root, V37_AB_PLAN_ONLY="1", V37_AB_PYTHON=str(python_launcher))
    assert result.returncode == 0, result.stderr
    assert (root / "ab_plan.json").is_file()


def test_contract_only_is_serial_quoted_and_sanitized(tmp_path):
    wrapper = make_wrapper(tmp_path)
    root = tmp_path / "contract root with spaces"
    poisoned = {
        "V37_GATE_ONLY": "1", "V37_PREFLIGHT_ONLY": "1", "V37_DRY_RUN": "1",
        "V37_VALIDATE_DOWNSTREAM": "1", "V37_RESUME_CHECKPOINT": "/poison/resume",
        "V37_RESUME_SEAL": "/poison/seal", "V36_LOAD_CHECKPOINT_PATH": "/poison/v36",
        "V37_EXPECTED_RESUME_CHECKPOINT_PATH": "/poison/expected",
        "V37_EXPECTED_RESUME_CHECKPOINT_SHA256": "a" * 64,
        "V32_LOAD_CHECKPOINT_PATH": "/poison/v32", "V32_DRY_RUN": "1",
        "V37_ALLOW_FOCUSED10K_PILOT": "1", "V37_ALLOW_BENCHMARK_DEV": "1",
        "V31_EXPERIMENT_NAME": "poison", "V37_RUN_TIMESTAMP": "poison",
        "TRAINER_VAL_ONLY": "true",
        "WANDB_API_KEY": "poison", "HF_TOKEN": "poison", "HUGGING_FACE_HUB_TOKEN": "poison",
        "GIT_DIR": "/poison/git-dir", "GIT_WORK_TREE": "/poison/work-tree",
        "GIT_COMMON_DIR": "/poison/common-dir", "GIT_INDEX_FILE": "/poison/index",
        "GIT_OBJECT_DIRECTORY": "/poison/objects",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES": "/poison/alternate-objects",
        "GIT_CEILING_DIRECTORIES": "/", "GIT_DISCOVERY_ACROSS_FILESYSTEM": "1",
        "GIT_EXTERNAL_DIFF": "/poison/diff", "GIT_DIFF_OPTS": "--stat",
    }
    result = run_ab(root, wrapper, V37_AB_CONTRACT_ONLY="1", **poisoned)
    assert result.returncode == 0, result.stderr
    calls = load_invocations(root)
    assert len(calls) == 4
    assert [(call["environment"]["V37_SEED"], call["environment"]["V37_ARM"]) for call in calls] == [
        ("11", "baseline"), ("11", "progress"), ("22", "baseline"), ("22", "progress"),
    ]
    for call in calls:
        env = call["environment"]
        assert call["argv"] == [str(wrapper)]
        assert env["V37_RUN_CLASS"] == "formal"
        assert env["V37_DATA_MODE"] == "frontier_rl"
        assert env["V37_RUN_PURPOSE"] == "formal_ab"
        assert env["V37_CONTRACT_ONLY"] == "1"
        assert (env["V37_PILOT_STEPS"], env["V31_NNODES"], env["V31_N_GPUS_PER_NODE"]) == ("12", "1", "8")
        assert Path(env["V31_SAVE_CHECKPOINT_PATH"]).parent == root
        for key in poisoned:
            assert env[key] is None
        assert env["BASH_ENV"] is None and env["ENV"] is None
        assert env["V37_AB_ROOT"] is None and env["V37_AB_SEEDS"] is None
        assert env["V37_AB_SINGLE_RUN_WRAPPER"] is None
        assert call["bash_functions"] == []
    assert not (root / "overlap").exists()
    assert not (root / "ab_evidence_index.json").exists()
    assert not list(root.glob("*/v37_run_manifest.json"))


def test_exported_shell_function_is_rejected_before_root_claim(tmp_path):
    root = tmp_path / "exported-function"
    env = clean_env()
    env.update({
        "V37_AB_ROOT": str(root), "V37_AB_PLAN_ONLY": "1",
        "BASH_FUNC_injected%%": "() { echo injected; }",
    })
    result = subprocess.run(
        ["bash", str(LAUNCHER)], cwd=REPO, env=env, text=True,
        capture_output=True, check=False,
    )
    assert result.returncode != 0
    assert "BASH_FUNC" in result.stderr
    assert not root.exists()


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("fail", "child failed"),
        ("missing", "missing run manifest"),
        ("tampered", "linkage mismatch"),
        ("duplicate", "duplicate manifest evidence_run_id"),
        ("duplicate_json", "duplicate JSON key"),
    ],
)
def test_child_and_manifest_failures_never_publish_index(tmp_path, mode, message):
    wrapper = make_wrapper(tmp_path)
    root = tmp_path / mode
    result = run_ab(root, wrapper, FAKE_MODE=mode)
    assert result.returncode != 0
    assert message in result.stderr
    assert not (root / "ab_evidence_index.json").exists()


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("plan_tamper", "ab_plan.json hash changed"),
        ("plan_symlink", "ab_plan.json must be a non-symlink regular file"),
        ("plan_nonregular", "ab_plan.json must be a non-symlink regular file"),
    ],
)
def test_plan_tampering_stops_before_next_child(tmp_path, mode, message):
    wrapper = make_wrapper(tmp_path)
    root = tmp_path / mode
    result = run_ab(root, wrapper, FAKE_MODE=mode)
    assert result.returncode != 0
    assert message in result.stderr
    assert len(load_invocations(root)) == 1
    assert not (root / "ab_evidence_index.json").exists()


def test_precreated_future_cell_dir_is_rejected_before_that_child(tmp_path):
    wrapper = make_wrapper(tmp_path)
    root = tmp_path / "precreated-future-cell"
    result = run_ab(root, wrapper, FAKE_MODE="precreate_future_cell")
    assert result.returncode != 0
    assert "cell target directory must not already exist or be a symlink" in result.stderr
    assert len(load_invocations(root)) == 1
    assert not (root / "ab_evidence_index.json").exists()


def test_contract_only_revalidates_plan_after_all_children(tmp_path):
    wrapper = make_wrapper(tmp_path)
    root = tmp_path / "contract-final-plan-check"
    result = run_ab(
        root, wrapper, V37_AB_CONTRACT_ONLY="1", FAKE_MODE="plan_tamper_after_last",
    )
    assert result.returncode != 0
    assert "ab_plan.json hash changed" in result.stderr
    assert len(load_invocations(root)) == 4
    assert not (root / "ab_evidence_index.json").exists()


def test_normal_mode_publishes_only_hashed_plan_and_manifests(tmp_path):
    wrapper = make_wrapper(tmp_path)
    root = tmp_path / "normal evidence"
    result = run_ab(root, wrapper)
    assert result.returncode == 0, result.stderr
    plan_path = root / "ab_plan.json"
    index_path = root / "ab_evidence_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert set(index) == {"plan", "manifests"}
    assert set(index["plan"]) == {"path", "sha256"}
    assert index["plan"] == {"path": str(plan_path.resolve()), "sha256": sha256(plan_path)}
    assert len(index["manifests"]) == 4
    for record in index["manifests"]:
        assert set(record) == {"path", "sha256"}
        manifest_path = Path(record["path"])
        assert manifest_path.is_absolute()
        assert record["sha256"] == sha256(manifest_path)
    lowered = json.dumps(index).lower()
    assert "eval" not in lowered
    assert "gate" not in lowered
    assert not (root / "overlap").exists()
