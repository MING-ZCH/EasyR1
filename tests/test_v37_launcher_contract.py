import json
import hashlib
import os
import subprocess
from pathlib import Path

import pytest
from tools import preflight_v37_training as preflight_module
from tools import v37_continuation as continuation


REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "examples" / "v37_strict_winner_step_rl_pilot.sh"
ROLLOUT_SOURCE = REPO / "verl" / "workers" / "rollout" / "vllm_rollout_spmd.py"
ROLLOUT_CONFIG_SOURCE = REPO / "verl" / "workers" / "rollout" / "config.py"
TRAINER_SOURCE = REPO / "verl" / "trainer" / "ray_trainer.py"
V32_SOURCE = REPO / "examples" / "v32_sparse_0_10_stable_drfix.sh"
PYARROW_PYTHON = os.environ.get("V37_TEST_PYARROW_PYTHON", "/root/miniconda3/bin/python")


def run_launcher(tmp_path: Path, **overrides: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("V37_") or key in {
            "ACTION_EVENT_LEDGER_ENABLE",
            "ACTION_EVENT_REWARD_ENABLE",
            "ADAPTIVE_ACTOR_KL",
            "ADV_ESTIMATOR",
            "BOK_ALLWRONG_TERMINAL_ZERO",
            "BOK_CORRECTNESS_FIRST",
            "BOK_CORRECTNESS_PARTIAL_SCALE",
            "BOK_CORRECTNESS_QUALITY_WEIGHT",
            "BOK_CORRECTNESS_TASK_WEIGHT",
            "BOK_STEP_WEIGHT",
            "BOK_WINNER_BOOST",
            "KL_HORIZON",
            "KL_HORIZON_UNIT",
            "PROCESS_REWARD_ENABLE",
            "TRAJ_STRICT_ANSWER_INTEGER_PARSE",
            "USE_KL_LOSS",
        }:
            env.pop(key)
    env.update(
        V37_RUN_CLASS="debug",
        V37_DATA_MODE="frontier_rl",
        V37_ARM="baseline",
        V37_CONTRACT_ONLY="1",
        TMPDIR=str(tmp_path),
    )
    env.update({key: str(value) for key, value in overrides.items()})
    return subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_v37_process_prompt_hash_is_checked_at_worker_read_time():
    launcher = LAUNCHER.read_text(encoding="utf-8")
    rollout = ROLLOUT_SOURCE.read_text(encoding="utf-8")
    rollout_config = ROLLOUT_CONFIG_SOURCE.read_text(encoding="utf-8")
    v32 = V32_SOURCE.read_text(encoding="utf-8")
    assert "export INTERLEAVED_PROCESS_PROMPT_SHA256=" in launcher
    assert '"process_prompt": digest(os.environ["_V37_PROCESS_PROMPT_FILE"])' in launcher
    assert "interleaved_process_prompt_sha256: Optional[str] = None" in rollout_config
    assert "self.config.interleaved_process_prompt_sha256" in rollout
    assert "hashlib.sha256(raw).hexdigest()" in rollout
    assert 'f"{field_name} SHA256 mismatch:' in rollout
    assert "worker.rollout.interleaved_process_prompt_sha256=" in v32


def test_controlled_continuation_rechecks_sealed_checkpoint_at_trainer_load():
    launcher = LAUNCHER.read_text(encoding="utf-8")
    trainer = TRAINER_SOURCE.read_text(encoding="utf-8")
    v32 = V32_SOURCE.read_text(encoding="utf-8")
    assert 'export V37_EXPECTED_RESUME_CHECKPOINT_PATH="${_V37_RESUME_CHECKPOINT}"' in launcher
    assert 'export V37_EXPECTED_RESUME_CHECKPOINT_SHA256="${_V37_RESUME_CHECKPOINT_SHA256}"' in launcher
    load_function = trainer[trainer.index("    def _load_checkpoint(self)"):]
    runtime_check = load_function.index("validate_resume_checkpoint_binding(")
    first_deserialize = min(
        load_function.index("torch.load("),
        load_function.index("self.actor_rollout_wg.load_checkpoint("),
    )
    assert runtime_check < first_deserialize
    for field in (
        "v37_run_class",
        "v37_resume_mode",
        "v37_expected_resume_checkpoint_path",
        "v37_expected_resume_checkpoint_sha256",
    ):
        assert f"trainer.{field}=" in v32
        assert f"self.config.trainer.{field}" in load_function


def test_formal_training_requires_clean_git_before_run_directory_claim():
    launcher = LAUNCHER.read_text(encoding="utf-8")
    clean_check = launcher.index("formal training requires a clean committed Git worktree")
    claim = launcher.index("# Claim the run directory only after every fail-closed preflight")
    assert clean_check < claim
    assert '[[ "${V37_RUN_CLASS}" == "formal" ]]' in launcher[:claim]
    assert '! _v37_is_true "${V37_DRY_RUN:-0}"' in launcher[:claim]
    identity_recheck = launcher.index("formal Git worktree changed after preflight")
    manifest_write = launcher.index('export V37_RUN_MANIFEST=')
    assert clean_check < identity_recheck < manifest_write


def test_launcher_git_identity_uses_a_sanitized_minimal_environment():
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert '_v37_git() {' in launcher
    assert 'env -i PATH="${PATH}" HOME="${HOME:-/nonexistent}" LC_ALL=C' in launcher
    assert '_v37_git status --porcelain=v1 -uall' in launcher
    assert '_v37_git diff --no-ext-diff --no-textconv --binary HEAD' in launcher


def test_single_launcher_rejects_initial_model_hash_override(tmp_path):
    result = run_launcher(
        tmp_path, V37_EXPECTED_INITIAL_MODEL_SHA256="f" * 64,
    )
    assert result.returncode != 0
    assert "cannot override checkpoint-476 identity" in result.stderr


def formal_manifest(tmp_path: Path, **updates: object) -> Path:
    quotas = [40, 10, 20, 20, 10]
    labels = ["2-10", "11-20", "21-30", "31-40", "41-50"]
    publication = tmp_path / "frontier-publication"
    publication.mkdir(exist_ok=True)
    mining_checkpoint = tmp_path / "inputs" / "mining-checkpoint"
    mining_checkpoint.mkdir(parents=True, exist_ok=True)
    (mining_checkpoint / "model.bin").write_bytes(b"mining-checkpoint")
    mining_entries = [{
        "path": "model.bin",
        "size": len(b"mining-checkpoint"),
        "sha256": hashlib.sha256(b"mining-checkpoint").hexdigest(),
    }]
    mining_hash = hashlib.sha256(json.dumps(
        mining_entries, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    value = {
        "schema_version": 3,
        "data_mode": "frontier_rl",
        "run_class": "formal",
        "publication": "atomic_directory_v1",
        "source_sha256": "1" * 64,
        "audit_sha256": {"seed-a.jsonl": "2" * 64, "seed-b.jsonl": "3" * 64},
        "seeds": [11, 22],
        "candidates_per_seed": 32,
        "requested_sample_count": 100,
        "selected_sample_count": 100,
        "bucket_ratios": dict(zip(labels, [0.4, 0.1, 0.2, 0.2, 0.1])),
        "selection": {
            label: {
                "quota": quota,
                "outcome-frontier": int(quota * .75) + (1 if quota * .75 % 1 >= quota * .25 % 1 and quota % 4 else 0),
                "process-hard": quota - (int(quota * .75) + (1 if quota * .75 % 1 >= quota * .25 % 1 and quota % 4 else 0)),
            }
            for label, quota in zip(labels, quotas)
        },
        "outcome_ratio": .75,
        "process_ratio": .25,
        "classifier_thresholds": {
            "winner_min": 3, "winner_max": 20, "process_min_quality": .55,
            "process_min_quality_range": .10, "process_max_duplicate_rate": .25,
            "process_max_cap_rate": .25,
        },
        "mining_generation_contract": {
            "model_checkpoint_path": str(mining_checkpoint.resolve()),
            "model_checkpoint_sha256": mining_hash,
            "generation_config_by_seed": {
                "11": {"seed": 11, "max_new_tokens": 512},
                "22": {"seed": 22, "max_new_tokens": 512},
            },
            "sampling_config": {"temperature": .7, "top_p": 1.0},
            "seed_difference_allowlist": ["seed"],
            "source_binding_contract": "source_row_image_rendered_prompt_v1",
            "source_binding_sha256": "4" * 64,
        },
        "candidate_fields_injected": False,
        "published_artifacts": ["frontier_rl.parquet", "selected_ids.json", "selection_manifest.json"],
    }
    value.update(updates)
    frontier = publication / "frontier_rl.parquet"
    selected_ids = publication / "selected_ids.json"
    frontier.write_bytes(b"contract fixture; preflight validates real parquet")
    selected_ids.write_text(json.dumps([f"p-{index}" for index in range(100)]), encoding="utf-8")
    value["artifact_sha256"] = {
        frontier.name: hashlib.sha256(frontier.read_bytes()).hexdigest(),
        selected_ids.name: hashlib.sha256(selected_ids.read_bytes()).hexdigest(),
    }
    path = publication / "selection_manifest.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    seal_publication(path)
    return path


def seal_publication(manifest: Path) -> None:
    names = ("frontier_rl.parquet", "selected_ids.json", "selection_manifest.json")
    hashes = {
        name: hashlib.sha256((manifest.parent / name).read_bytes()).hexdigest()
        for name in names
    }
    complete = {
        "schema_version": 1,
        "publication": "atomic_directory_v1",
        "published_artifacts": list(names),
        "artifact_sha256": hashes,
    }
    (manifest.parent / "_COMPLETE.json").write_text(json.dumps(complete), encoding="utf-8")


@pytest.mark.parametrize(
    ("arm", "expected"),
    [
        (
            "baseline",
            {
                "adv_estimator": "bok_grpo",
                "action_event_ledger_enable": "1",
                "action_event_reward_enable": "0",
                "legacy_process_reward_enable": "0",
                "step_weight": "0",
            },
        ),
        (
            "progress",
            {
                "adv_estimator": "bok_grpo_step",
                "action_event_ledger_enable": "1",
                "action_event_reward_enable": "1",
                "legacy_process_reward_enable": "0",
                "step_weight": "0.1",
            },
        ),
    ],
)
def test_arm_switch_expansion(tmp_path, arm, expected):
    result = run_launcher(tmp_path, V37_ARM=arm)
    assert result.returncode == 0, result.stderr
    expanded = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    assert {key: expanded[key] for key in expected} == expected
    assert expanded["correctness_first"] == "1"
    assert expanded["allwrong_terminal_zero"] == "1"
    assert expanded["correctness_task_weight"] == "0.25"
    assert expanded["correctness_quality_weight"] == "0.1"
    assert expanded["correctness_partial_scale"] == "0.25"
    assert expanded["reward_fail_closed"] == "1"
    assert expanded["action_parser_contract"] == "1"
    assert expanded["action_ledger_contract"] == "1"
    assert expanded["winner_boost"] == "0"
    assert expanded["adaptive_actor_kl"] == "true"
    # Core rejects legacy use_kl_loss together with the adaptive actor controller.
    assert expanded["use_kl_loss"] == "false"
    assert expanded["cp_size"] == "1"
    assert expanded["fallback_logprob_sign_opt_in"] == "0"
    assert expanded["torch_logprob_fallback_mode"] == "error"
    assert expanded["strict_answer_integer_parse"] == "1"
    assert expanded["strict_raw_success_winner"] == "1"
    assert expanded["winner_mode"] == "outcome_success"
    assert expanded["kl_horizon_unit"] == "executed_optimizer_updates"
    assert expanded["kl"].endswith("/50")
    assert expanded["strict_point_parser_contract"] == "1"
    assert expanded["resume_mode"] == "clean_start"


def test_v37_strict_answer_parse_cannot_be_disabled(tmp_path):
    result = run_launcher(tmp_path, TRAJ_STRICT_ANSWER_INTEGER_PARSE="0")
    assert result.returncode != 0
    assert "TRAJ_STRICT_ANSWER_INTEGER_PARSE=1" in result.stderr

    result = run_launcher(tmp_path, V37_RAW_SUCCESS_STRICT_WINNER="0")
    assert result.returncode != 0
    assert "V37_RAW_SUCCESS_STRICT_WINNER=1" in result.stderr

    result = run_launcher(tmp_path, V37_WINNER_MODE="legacy_all_hit")
    assert result.returncode != 0
    assert "V37_WINNER_MODE=outcome_success" in result.stderr

    result = run_launcher(tmp_path, KL_HORIZON="10000")
    assert result.returncode != 0
    assert "locked to 50 executed optimizer updates" in result.stderr

    result = run_launcher(tmp_path, V37_STRICT_POINT_PARSER_CONTRACT="0")
    assert result.returncode != 0
    assert "V37_STRICT_POINT_PARSER_CONTRACT=1" in result.stderr

    result = run_launcher(tmp_path, V37_REWARD_FAIL_CLOSED="0")
    assert result.returncode != 0
    assert "V37_REWARD_FAIL_CLOSED=1" in result.stderr


def test_image_path_remap_is_canonicalized_and_invalid_input_is_rejected(tmp_path):
    result = run_launcher(
        tmp_path,
        STEPCOUNT_IMAGE_PATH_REMAP_JSON=' { "/old/images" : "/new/images" } ',
    )
    assert result.returncode == 0, result.stderr
    expanded = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    assert expanded["image_path_remap"] == '{"/old/images":"/new/images"}'

    invalid = run_launcher(tmp_path, STEPCOUNT_IMAGE_PATH_REMAP_JSON='{"relative":"/new"}')
    assert invalid.returncode != 0
    assert "invalid STEPCOUNT_IMAGE_PATH_REMAP_JSON" in invalid.stderr


def test_manifest_audits_execution_environment_without_secrets():
    source = LAUNCHER.read_text(encoding="utf-8")
    for prefix in ("CUBLAS_", "CUDA_", "NCCL_", "PYTORCH_", "RAY_", "TORCH_", "VLLM_"):
        assert f'"{prefix}"' in source
    for key in ("LD_LIBRARY_PATH", "LD_PRELOAD", "PATH", "PYTHONPATH"):
        assert f'"{key}"' in source
    assert '"HF_"' not in source
    assert "runtime_config_sha256" in source
    assert "runtime_environment" in source


def test_launcher_rejects_shell_startup_injection(tmp_path):
    bash_env = tmp_path / "bash_env.sh"
    bash_env.write_text("export V37_ARM=progress\n", encoding="utf-8")
    result = run_launcher(tmp_path, BASH_ENV=str(bash_env))
    assert result.returncode != 0
    assert "BASH_ENV/ENV are forbidden" in result.stderr


def test_strict_winner_rft_fails_closed(tmp_path):
    result = run_launcher(tmp_path, V37_DATA_MODE="strict_winner_rft")
    assert result.returncode != 0
    assert "incompatible with this online RL entrypoint" in result.stderr


def test_adaptive_kl_context_parallel_gt_one_fails_closed(tmp_path):
    result = run_launcher(tmp_path, V37_CP_SIZE="2")
    assert result.returncode != 0
    assert "V37_CP_SIZE must remain 1" in result.stderr


def test_canary_requires_hashed_frontier_manifest(tmp_path):
    missing = run_launcher(tmp_path, V37_RUN_CLASS="canary")
    assert missing.returncode != 0
    assert "V37_FRONTIER_MANIFEST" in missing.stderr

    manifest = formal_manifest(tmp_path, run_class="canary")
    passed = run_launcher(tmp_path, V37_RUN_CLASS="canary", V37_FRONTIER_MANIFEST=str(manifest))
    assert passed.returncode == 0, passed.stderr
    assert "promotable_candidate=0" in passed.stdout


def test_formal_manifest_quota_and_hash_contract(tmp_path):
    manifest = formal_manifest(tmp_path)
    passed = run_launcher(tmp_path, V37_RUN_CLASS="formal", V37_FRONTIER_MANIFEST=str(manifest))
    assert passed.returncode == 0, passed.stderr
    assert "promotable_candidate=1" in passed.stdout

    bad_hash = formal_manifest(tmp_path, source_sha256="unknown")
    failed_hash = run_launcher(tmp_path, V37_RUN_CLASS="formal", V37_FRONTIER_MANIFEST=str(bad_hash))
    assert failed_hash.returncode != 0
    assert "source_sha256 is missing or unknown" in failed_hash.stderr

    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["source_sha256"] = "1" * 64
    value["bucket_ratios"]["2-10"] = 0.25
    manifest.write_text(json.dumps(value), encoding="utf-8")
    seal_publication(manifest)
    failed_quota = run_launcher(tmp_path, V37_RUN_CLASS="formal", V37_FRONTIER_MANIFEST=str(manifest))
    assert failed_quota.returncode != 0
    assert "40/10/20/20/10" in failed_quota.stderr


@pytest.mark.parametrize("entry_kind", ["dangling-file", "dangling-directory", "directory"])
def test_frontier_publication_rejects_unenumerated_nonregular_entries(tmp_path, entry_kind):
    manifest = formal_manifest(tmp_path, run_class="canary")
    extra = manifest.parent / "unsealed-extra"
    if entry_kind == "directory":
        extra.mkdir()
    else:
        extra.symlink_to(
            manifest.parent / "missing-target",
            target_is_directory=entry_kind == "dangling-directory",
        )
    result = run_launcher(
        tmp_path, V37_RUN_CLASS="canary", V37_FRONTIER_MANIFEST=str(manifest)
    )
    assert result.returncode != 0
    assert "unenumerated artifacts" in result.stderr
    errors = preflight_module._validate_complete_publication(
        manifest.parent / "frontier_rl.parquet"
    )
    assert any("unenumerated artifacts" in error for error in errors)


def test_formal_rejects_debug_fallback_switches(tmp_path):
    manifest = formal_manifest(tmp_path)
    result = run_launcher(
        tmp_path,
        V37_RUN_CLASS="formal",
        V37_FRONTIER_MANIFEST=str(manifest),
        V37_ALLOW_FOCUSED10K_PILOT="1",
    )
    assert result.returncode != 0
    assert "cannot enable the focused10k fallback" in result.stderr


@pytest.mark.parametrize("bad_text", ['{"schema_version":1,"schema_version":1}', '{"schema_version":NaN}'])
def test_complete_marker_strict_json_even_for_canary(tmp_path, bad_text):
    manifest = formal_manifest(tmp_path, run_class="canary")
    (manifest.parent / "_COMPLETE.json").write_text(bad_text, encoding="utf-8")
    result = run_launcher(tmp_path, V37_RUN_CLASS="canary", V37_FRONTIER_MANIFEST=str(manifest))
    assert result.returncode != 0
    assert "atomic publication seal" in result.stderr


def test_controlled_resume_requires_sealed_same_world_size_and_is_nonpromotable(tmp_path):
    frontier_manifest = formal_manifest(tmp_path, run_class="debug")
    frontier_manifest_hash = hashlib.sha256(frontier_manifest.read_bytes()).hexdigest()
    checkpoint = tmp_path / "global_step_1"
    checkpoint.mkdir()
    (checkpoint / "state.bin").write_bytes(b"sealed-state")
    entries = [{
        "path": "state.bin",
        "size": len(b"sealed-state"),
        "sha256": hashlib.sha256(b"sealed-state").hexdigest(),
    }]
    checkpoint_hash = hashlib.sha256(json.dumps(
        entries, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    source = tmp_path / "source-manifest.json"
    source_config = {"world_size": 8, "pilot_steps": 1}
    source.write_text(json.dumps({
        "manifest_version": 3, "data_mode": "frontier_rl", "run_class": "debug",
        "arm": "baseline", "seed": 42, "config": source_config,
        "continuation_config_sha256": continuation.continuation_config_sha256(source_config),
        "dataset_manifest_sha256": frontier_manifest_hash,
        "final_checkpoint_path": str(checkpoint.resolve()), "final_checkpoint_sha256": checkpoint_hash,
    }))
    seal = tmp_path / "resume-seal.json"
    seal.write_text(json.dumps({
        "schema_version": 1, "purpose": "controlled_continuation",
        "checkpoint_path": str(checkpoint.resolve()), "checkpoint_tree_sha256": checkpoint_hash,
        "world_size": 8, "source_run_manifest_path": str(source),
        "source_run_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }))
    result = run_launcher(
        tmp_path, V37_CONTINUATION_MODE="1", V37_RUN_PURPOSE="controlled_continuation",
        V37_RESUME_CHECKPOINT=str(checkpoint), V37_RESUME_SEAL=str(seal),
        V37_FRONTIER_MANIFEST=str(frontier_manifest),
    )
    assert result.returncode == 0, result.stderr
    assert "resume_mode=controlled_continuation" in result.stdout
    assert "promotable_candidate=0" in result.stdout

    wrong_seed = run_launcher(
        tmp_path, V37_CONTINUATION_MODE="1", V37_RUN_PURPOSE="controlled_continuation",
        V37_RESUME_CHECKPOINT=str(checkpoint), V37_RESUME_SEAL=str(seal),
        V37_FRONTIER_MANIFEST=str(frontier_manifest), V37_SEED="43",
    )
    assert wrong_seed.returncode != 0
    assert "arm/seed mismatch" in wrong_seed.stderr

    wrong_arm = run_launcher(
        tmp_path, V37_CONTINUATION_MODE="1", V37_RUN_PURPOSE="controlled_continuation",
        V37_RESUME_CHECKPOINT=str(checkpoint), V37_RESUME_SEAL=str(seal),
        V37_FRONTIER_MANIFEST=str(frontier_manifest), V37_ARM="progress",
    )
    assert wrong_arm.returncode != 0
    assert "arm/seed mismatch" in wrong_arm.stderr

    alternate_root = tmp_path / "alternate"
    alternate_root.mkdir()
    alternate_frontier = formal_manifest(alternate_root, run_class="debug")
    wrong_frontier = run_launcher(
        tmp_path, V37_CONTINUATION_MODE="1", V37_RUN_PURPOSE="controlled_continuation",
        V37_RESUME_CHECKPOINT=str(checkpoint), V37_RESUME_SEAL=str(seal),
        V37_FRONTIER_MANIFEST=str(alternate_frontier),
    )
    assert wrong_frontier.returncode != 0
    assert "frontier manifest mismatch" in wrong_frontier.stderr

    formal = run_launcher(
        tmp_path, V37_RUN_CLASS="formal", V37_CONTINUATION_MODE="1",
        V37_RUN_PURPOSE="controlled_continuation", V37_RESUME_CHECKPOINT=str(checkpoint),
        V37_RESUME_SEAL=str(seal),
    )
    assert formal.returncode != 0
    assert "forbidden for formal A/B" in formal.stderr


def test_controlled_resume_runs_both_identity_gates_before_publication():
    source = LAUNCHER.read_text(encoding="utf-8")
    early = source.index('"${REPO_DIR}/tools/v37_continuation.py" verify-seal')
    contract_only = source.index("# Contract-only mode", early)
    complete = source.index("continuation_module.validate_continuation", contract_only)
    manifest_publish = source.index(
        "temporary.write_text(json.dumps(manifest", complete
    )
    assert early < contract_only < complete < manifest_publish


def test_formal_locks_ray_release_wait_and_hf_merge_memory_preflight(tmp_path):
    frontier = formal_manifest(tmp_path)
    result = run_launcher(
        tmp_path, V37_RUN_CLASS="formal", V37_FRONTIER_MANIFEST=str(frontier)
    )
    assert result.returncode == 0, result.stderr
    assert "ray_gpu_wait_timeout_seconds=900" in result.stdout
    assert "hf_merge_host_memory=error/4.0" in result.stdout

    bad_policy = run_launcher(
        tmp_path, V37_RUN_CLASS="formal", V37_FRONTIER_MANIFEST=str(frontier),
        V37_HF_MERGE_HOST_MEMORY_PREFLIGHT="off",
    )
    assert bad_policy.returncode != 0
    assert "locked to error" in bad_policy.stderr

    bad_factor = run_launcher(
        tmp_path, V37_RUN_CLASS="formal", V37_FRONTIER_MANIFEST=str(frontier),
        V37_HF_MERGE_HOST_MEMORY_SAFETY_FACTOR="2.0",
    )
    assert bad_factor.returncode != 0
    assert "locked to 4.0" in bad_factor.stderr


def test_gate_only_dispatches_to_final_gate_without_training(tmp_path):
    spec = tmp_path / "gate.json"
    spec.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    result = run_launcher(
        tmp_path,
        V37_GATE_ONLY="1",
        V37_GATE_INPUT=str(spec),
        V37_CONTRACT_ONLY="0",
    )
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["decision"] == "NO-GO"
    assert payload["hard_failure"] is True


def test_real_builder_launcher_preflight_e2e(tmp_path):
    probe = subprocess.run(
        [PYARROW_PYTHON, "-c", "import pyarrow, PIL"], capture_output=True, text=True, check=False
    )
    if probe.returncode:
        pytest.skip("a Python with pyarrow and Pillow is unavailable")
    driver = r'''
import hashlib, json, random, subprocess, sys
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

root, repo = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(repo))
from tools import build_v37_frontier_dataset as frontier_builder
assets, masks = root / "assets", root / "masks"
assets.mkdir(); masks.mkdir()
prompts, answers, images = [], [], []
for bucket, answer in enumerate((2, 11, 21, 31, 41)):
    for index in range(20):
        pid = f"p-{bucket}-{index}"
        image = assets / f"train-{bucket}-{index}.png"
        pixels = random.Random(1000 + bucket * 100 + index).randbytes(64)
        Image.frombytes("L", (8, 8), pixels).save(image)
        prompts.append(pid); answers.append(answer); images.append(str(image))
source = root / "source.parquet"
source_table = pa.table({
    "prompt_id": prompts, "sample_id": [f"s-{p}" for p in prompts],
    "prompt": ["<image>\nCount the objects." for _ in prompts],
    "answer": answers, "images": [[item] for item in images],
})
pq.write_table(source_table, source)
source_bindings = frontier_builder.build_source_bindings(source_table, source)
mining_checkpoint = root / "mining-checkpoint"
mining_checkpoint.mkdir()
(mining_checkpoint / "model.bin").write_bytes(b"mining-checkpoint")
mining_checkpoint_sha256 = frontier_builder.path_sha256(mining_checkpoint)

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

audits = []
for seed in (11, 22):
    path = root / f"audit-{seed}.jsonl"; audits.append(path)
    with path.open("w") as handle:
        for pid in prompts:
            process_hard = int(pid.rsplit("-", 1)[1]) >= 15
            for index in range(32):
                success = not process_hard and index < 3
                response = f"response:{pid}:{seed}:{index}"
                generation = {"seed": seed, "max_new_tokens": 512}
                sampling = {"temperature": 0.7, "top_p": 1.0}
                row = {"schema_version": 1, "prompt_id": pid, "seed": seed, "declared_seed": seed,
                    "candidate_id": f"candidate:{pid}:{seed}:{index}", "answer_exact": success,
                    "raw_success": success, "trajectory_quality": .4 if index % 2 == 0 else .8,
                    "process": {"unique_hit": 4, "duplicate": 0, "miss": 1, "cap": False},
                    "termination": "answer", "model_checkpoint_sha256": mining_checkpoint_sha256,
                    "generation_config": generation, "sampling_config": sampling, "response": response,
                    "generation_config_sha256": canonical_hash(generation),
                    "sampling_config_sha256": canonical_hash(sampling),
                    "response_sha256": hashlib.sha256(response.encode()).hexdigest()}
                row.update(source_bindings[pid])
                handle.write(json.dumps(row) + "\n")

digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
output = root / "built"
command = [sys.executable, str(repo / "tools/build_v37_frontier_dataset.py"), "--source", str(source),
    "--audit", str(audits[0]), "--audit", str(audits[1]), "--output-dir", str(output),
    "--run-class", "formal", "--sample-count", "20", "--expected-source-sha256", digest(source),
    "--expected-audit-sha256", digest(audits[0]), "--expected-audit-sha256", digest(audits[1]),
    "--mining-checkpoint", str(mining_checkpoint),
    "--expected-mining-checkpoint-sha256", mining_checkpoint_sha256]
subprocess.run(command, check=True)

metadata_rows = []
for image in [Path(item) for item in images]:
    mask = masks / f"{image.stem}.png"; Image.new("L", (8, 8), 255).save(mask)
    metadata_rows.append({"image": image.name, "mask": mask.name})
val_rows = []
for bucket, answer in enumerate((2, 11, 21, 31, 41)):
    image = assets / f"val-{bucket}.png"
    Image.frombytes("L", (8, 8), random.Random(5000 + bucket).randbytes(64)).save(image)
    mask = masks / f"{image.stem}.png"; Image.new("L", (8, 8), 255).save(mask)
    metadata_rows.append({"image": image.name, "mask": mask.name})
    val_rows.append((f"val-s-{bucket}", f"val-p-{bucket}", answer, str(image)))
metadata = root / "metadata.json"; metadata.write_text(json.dumps(metadata_rows))
val = root / "val.parquet"
pq.write_table(pa.table({"sample_id": [r[0] for r in val_rows], "prompt_id": [r[1] for r in val_rows],
    "answer": [r[2] for r in val_rows], "image_path": [r[3] for r in val_rows]}), val)
forbidden_image = assets / "forbidden.png"
Image.frombytes("L", (8, 8), random.Random(9000).randbytes(64)).save(forbidden_image)
forbidden = root / "forbidden.parquet"
pq.write_table(pa.table({"sample_id": ["forbidden-s"], "prompt_id": ["forbidden-p"], "answer": [7], "image_path": [str(forbidden_image)]}), forbidden)

def tree_hash(path):
    path = Path(path); files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    if path.is_file(): return hashlib.sha256(path.read_bytes()).hexdigest()
    entries = [{"path": item.relative_to(path).as_posix(), "size": item.stat().st_size,
        "sha256": hashlib.sha256(item.read_bytes()).hexdigest()} for item in files]
    return hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

selected = pq.read_table(output / "frontier_rl.parquet").to_pylist()
labels = ((2, 10, "2-10"), (11, 20, "11-20"), (21, 30, "21-30"), (31, 40, "31-40"), (41, 50, "41-50"))
observed = []
buckets = {label: 0 for _, _, label in labels}
for item in selected:
    label = next(label for low, high, label in labels if low <= item["answer"] <= high)
    buckets[label] += 1
    observed.append({"sample_id": item["sample_id"], "prompt_id": item["prompt_id"], "bucket": label})
observed.sort(key=lambda item: (item["sample_id"], item["prompt_id"]))
ids_payload = [[item["sample_id"], item["prompt_id"]] for item in observed]
loader_config = {"prompt_key": "prompt", "answer_key": "answer", "image_key": "images",
    "max_prompt_length": 12000, "truncation": "error", "max_pixels": 12845056,
    "min_pixels": 262144, "filter_overlong_prompts": True, "filter_overlong_num_proc": 64,
    "format_prompt": None, "system_prompt": str((repo / "examples/format_prompt/StepCount_interleaved_system_prompt.txt").resolve())}
filtered = root / "filtered.json"
filtered.write_text(json.dumps({"schema_version": 1, "dataset_sha256": tree_hash(output / "frontier_rl.parquet"),
    "loader_contract": "verl.utils.dataset.RLHFDataset:v1", "loader_config": loader_config,
    "sample_count": len(observed), "answer_buckets": buckets,
    "observed_ids_sha256": hashlib.sha256(json.dumps(ids_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    "observed_rows_sha256": hashlib.sha256(json.dumps(observed, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}))
coverage = root / "coverage.json"
coverage.write_text(json.dumps({"dataset_sha256": tree_hash(output / "frontier_rl.parquet"),
    "metadata_sha256": tree_hash(metadata), "mask_tree_sha256": tree_hash(masks), "coverage": 1.0,
    "sample_count": 20, "mapped_sample_count": 20, "missing_sample_count": 0,
    "ambiguous_sample_count": 0, "dimension_mismatch_count": 0, "unreadable_mask_count": 0,
    "mapping_key_normalization": "nfkc_basename_casefold_v1"}))
print(json.dumps({"manifest": str(output / "selection_manifest.json"), "data": str(output / "frontier_rl.parquet"),
    "metadata": str(metadata), "masks": str(masks), "val": str(val), "forbidden": str(forbidden),
    "filtered": str(filtered), "coverage": str(coverage), "assets": str(assets)}))
'''
    prepared = subprocess.run(
        [PYARROW_PYTHON, "-c", driver, str(tmp_path), str(REPO)],
        text=True, capture_output=True, check=True,
    )
    paths = json.loads(prepared.stdout.splitlines()[-1])
    result = run_launcher(
        tmp_path,
        V37_RUN_CLASS="formal",
        V37_PREFLIGHT_ONLY="1",
        V37_CONTRACT_ONLY="0",
        V37_PREFLIGHT_PYTHON=PYARROW_PYTHON,
        V37_FRONTIER_MANIFEST=paths["manifest"],
        V37_FRONTIER_DATA=paths["data"],
        V37_STEPCOUNT_MASKS_METADATA=paths["metadata"],
        V37_STEPCOUNT_MASKS_DIR=paths["masks"],
        STEPCOUNT_V37_VAL_DATA=paths["val"],
        V37_FORBIDDEN_DATA=paths["forbidden"],
        V37_FILTERED_MANIFEST=paths["filtered"],
        V37_METADATA_COVERAGE_REPORT=paths["coverage"],
        V37_IMAGE_ROOTS=paths["assets"],
    )
    assert result.returncode == 0, result.stderr
    assert "[V37-preflight] PASS" in result.stdout
