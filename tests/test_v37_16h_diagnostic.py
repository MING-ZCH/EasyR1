import json
import hashlib
import os
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "examples/rl_launch/run_v37_16h_diagnostic.sh"
PILOT = REPO / "examples/v37_strict_winner_step_rl_pilot.sh"


def executable(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)
    return path


def mock_environment(tmp_path: Path, wrapper_body: str) -> dict[str, str]:
    val = tmp_path / "heldout"
    val.mkdir()
    parquet = val / "diagnostic_heldout.parquet"
    parquet.write_bytes(b"mock")
    benchmark_env_names = (
        "STEPCOUNT_PIXMO_CANONICAL_JSON",
        "STEPCOUNT_STEPCOUNT500_CANONICAL_JSON",
        "STEPCOUNT_COUNTQA_DATA",
        "STEPCOUNT_BIAS_DATA",
        "STEPCOUNT_DENSE_CANONICAL_JSON",
        "STEPCOUNT_EXTREME_CANONICAL_JSON",
    )
    benchmark_paths = []
    for index, name in enumerate(benchmark_env_names):
        path = tmp_path / f"benchmark-{index}.json"
        path.write_text("[]\n", encoding="utf-8")
        benchmark_paths.append(path.resolve())
    source_paths = []
    source_hashes = {}
    for index in range(2):
        source = tmp_path / f"source-{index}"
        data = source / "data"
        data.mkdir(parents=True)
        parquet_source = data / "train.parquet"
        parquet_source.write_bytes(f"source-{index}".encode())
        source_paths.append(source.resolve())
        source_hashes[str(parquet_source.resolve())] = hashlib.sha256(parquet_source.read_bytes()).hexdigest()
    focused_dir = tmp_path / "focused"
    focused_dir.mkdir()
    focused_manifest = focused_dir / "selection_manifest.json"
    focused_manifest.write_text('{"fixture": true}\n', encoding="utf-8")
    selected = [
        {"image_exact_keys": [f"sha256:{index:064x}"]}
        for index in range(500)
    ]
    quotas = {"2-10": 200, "11-20": 50, "21-30": 100, "31-40": 100, "41-50": 50}
    (val / "selection_manifest.json").write_text(json.dumps({
        "schema_version": 2,
        "dataset_kind": "v37_nonbenchmark_diagnostic_heldout",
        "promotable": False,
        "benchmark_overlap_selected": 0,
        "selected_rows": 500,
        "quotas": quotas,
        "selected_distribution": quotas,
        "parquet_batch_size": 8,
        "benchmark_paths": [str(path) for path in benchmark_paths],
        "benchmark_input_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in benchmark_paths
        },
        "benchmark_exact_keys_sha256": "a" * 64,
        "source_paths": [str(path) for path in source_paths],
        "source_parquet_sha256": source_hashes,
        "focused_selection_manifest": str(focused_manifest.resolve()),
        "focused_selection_manifest_sha256": hashlib.sha256(focused_manifest.read_bytes()).hexdigest(),
        "focused_exact_key_count": 10_000,
        "focused_exact_duplicate_count": 0,
        "focused_exact_keys_sha256": "b" * 64,
        "focused_sequence_key_count": 10_000,
        "focused_sequence_keys_sha256": "c" * 64,
        "artifact_sha256": {
            "diagnostic_heldout.parquet": hashlib.sha256(parquet.read_bytes()).hexdigest()
        },
        "selected_indices": selected,
    }), encoding="utf-8")
    nvidia = executable(
        tmp_path / "nvidia-smi",
        """#!/usr/bin/env bash
if [[ "$*" == *"--query-gpu="* ]]; then
  for i in {0..7}; do echo "$i, NVIDIA H200, 141000, 0, Disabled"; done
fi
""",
    )
    ray = executable(tmp_path / "ray", "#!/usr/bin/env bash\nexit 0\n")
    wrapper = executable(tmp_path / "wrapper", "#!/usr/bin/env bash\n" + wrapper_body)
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("V37_") or key in {"BASH_ENV", "ENV", "WANDB_API_KEY"}:
            env.pop(key, None)
    env.update({
        "V37_DIAGNOSTIC_ROOT": str(tmp_path / "run"),
        "STEPCOUNT_V37_DIAGNOSTIC_VAL_DATA": str(val),
        "V37_DIAGNOSTIC_NVIDIA_SMI": str(nvidia),
        "V37_DIAGNOSTIC_RAY_BIN": str(ray),
        "V37_DIAGNOSTIC_SINGLE_RUN_WRAPPER": str(wrapper),
        "STEPCOUNT_REPLAY_DATA": str(source_paths[0]),
        "STEPCOUNT_DENSE_11_50_MASKCOMPLETE_DATA": str(source_paths[1]),
        "STEPCOUNT_DENSE_11_30_FOCUSED10K_DATA": str(focused_dir),
    })
    env.update({name: str(path) for name, path in zip(benchmark_env_names, benchmark_paths)})
    return env


def test_diagnostic_mock_completes_fixed_ab_ba_sequence(tmp_path):
    env = mock_environment(
        tmp_path,
        'mkdir -p "$V31_SAVE_CHECKPOINT_PATH/global_step_3"\n'
        'echo "actor/nonfinite_grad_count: 0"\n'
        'echo "mock arm=$V37_ARM seed=$V37_SEED"\n',
    )
    result = subprocess.run(
        ["bash", str(LAUNCHER)], cwd=REPO, env=env,
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    root = Path(env["V37_DIAGNOSTIC_ROOT"])
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "completed"
    assert status["promotable"] is False
    assert [(item["seed"], item["arm"]) for item in status["cells"]] == [
        (11, "baseline"), (11, "progress"), (22, "progress"), (22, "baseline"),
    ]
    assert all(item["state"] == "completed" for item in status["cells"])
    assert (root / "command_manifest.json").is_file()
    assert (root / "中文总结.md").is_file()
    assert sorted(path.name for path in (root / "logs").glob("*.log")) == [
        "seed11-baseline.log", "seed11-progress.log",
        "seed22-baseline.log", "seed22-progress.log",
    ]


def test_diagnostic_rejects_changed_benchmark_before_claiming_run_dir(tmp_path):
    env = mock_environment(tmp_path, 'exit 99\n')
    Path(env["STEPCOUNT_PIXMO_CANONICAL_JSON"]).write_text('[{"changed": true}]\n', encoding="utf-8")
    result = subprocess.run(
        ["bash", str(LAUNCHER)], cwd=REPO, env=env,
        text=True, capture_output=True, check=False,
    )
    assert result.returncode != 0
    assert "canonical benchmark content changed" in result.stderr
    assert not Path(env["V37_DIAGNOSTIC_ROOT"]).exists()


def test_diagnostic_stops_on_positive_nonfinite_counter(tmp_path):
    env = mock_environment(
        tmp_path,
        'mkdir -p "$V31_SAVE_CHECKPOINT_PATH/global_step_3"\n'
        'echo "actor/nonfinite_grad_count: 1"\n',
    )
    result = subprocess.run(
        ["bash", str(LAUNCHER)], cwd=REPO, env=env,
        text=True, capture_output=True, check=False,
    )
    assert result.returncode != 0
    status = json.loads(
        (Path(env["V37_DIAGNOSTIC_ROOT"]) / "status.json").read_text(encoding="utf-8")
    )
    assert status["cells"][0]["reason"] == "nonfinite_detected"


def test_diagnostic_sanitizes_ambient_control_and_ray_environment(tmp_path):
    env = mock_environment(
        tmp_path,
        '[[ -z "${RAY_ADDRESS:-}" && -z "${V37_DRY_RUN:-}" ]]\n'
        '[[ -z "${ACTOR_LR:-}" && -z "${ROLLOUT_N:-}" && -z "${GRAD_SPIKE_THRESHOLD:-}" ]]\n'
        '[[ "$CUDA_VISIBLE_DEVICES" == "0,1,2,3,4,5,6,7" ]]\n'
        '[[ "$V37_STEP_WEIGHT/$V37_STEP_GATE/$V37_STEP_MIN_GATE" == "0.1/answer_soft/0.2" ]]\n'
        'mkdir -p "$V31_SAVE_CHECKPOINT_PATH/global_step_3"\n',
    )
    env.update({
        "RAY_ADDRESS": "ray://wrong-cluster:10001",
        "V37_DRY_RUN": "1",
        "V37_STEP_WEIGHT": "9",
        "CUDA_VISIBLE_DEVICES": "0",
        "ACTOR_LR": "9",
        "ROLLOUT_N": "999",
        "GRAD_SPIKE_THRESHOLD": "999",
    })
    result = subprocess.run(
        ["bash", str(LAUNCHER)], cwd=REPO, env=env,
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_diagnostic_reaps_residual_fifo_writer_without_hanging(tmp_path):
    env = mock_environment(
        tmp_path,
        '(sleep 300) &\n'
        'mkdir -p "$V31_SAVE_CHECKPOINT_PATH/global_step_3"\n',
    )
    result = subprocess.run(
        ["bash", str(LAUNCHER)], cwd=REPO, env=env,
        text=True, capture_output=True, check=False, timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_diagnostic_mock_stops_on_gradspike_skip(tmp_path):
    env = mock_environment(
        tmp_path,
        'mkdir -p "$V31_SAVE_CHECKPOINT_PATH/global_step_3"\n'
        'echo "GradSpike protection requested skip"\n',
    )
    result = subprocess.run(
        ["bash", str(LAUNCHER)], cwd=REPO, env=env,
        text=True, capture_output=True, check=False,
    )
    assert result.returncode != 0
    status = json.loads(
        (Path(env["V37_DIAGNOSTIC_ROOT"]) / "status.json").read_text(encoding="utf-8")
    )
    assert status["state"] == "stopped"
    assert status["cells"][0]["reason"] == "gradspike_skip_detected"
    assert [item["state"] for item in status["cells"][1:]] == ["pending"] * 3


def test_pilot_resources_and_outside_range_flag_are_fail_closed(tmp_path):
    common = {
        **os.environ,
        "V37_RUN_CLASS": "debug", "V37_DATA_MODE": "frontier_rl",
        "V37_ARM": "baseline", "V37_CONTRACT_ONLY": "1",
    }
    common.pop("BASH_ENV", None); common.pop("ENV", None)
    micro8 = subprocess.run(
        ["bash", str(PILOT)], cwd=REPO,
        env={**common, "V37_MICRO_BATCH_UPDATE": "8"},
        text=True, capture_output=True, check=False,
    )
    assert micro8.returncode != 0
    assert "micro8 already OOMed" in micro8.stderr
    formal_flag = subprocess.run(
        ["bash", str(PILOT)], cwd=REPO,
        env={**common, "V37_RUN_CLASS": "formal",
             "V37_ALLOW_INDEPENDENT_VAL_OUTSIDE_TRAIN_RANGE": "1"},
        text=True, capture_output=True, check=False,
    )
    assert formal_flag.returncode != 0
    assert "debug-only" in formal_flag.stderr
