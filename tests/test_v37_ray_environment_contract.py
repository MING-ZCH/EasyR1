import importlib.util
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "verl" / "trainer" / "main.py"
RAY_ENV_SOURCE = Path(__file__).resolve().parents[1] / "verl" / "utils" / "ray_environment.py"
_SPEC = importlib.util.spec_from_file_location("v37_ray_environment_test", RAY_ENV_SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
RAY_WORKER_ENV_PREFIXES = _MODULE.RAY_WORKER_ENV_PREFIXES
RAY_WORKER_ENV_KEYS = _MODULE.RAY_WORKER_ENV_KEYS
collect_ray_worker_environment = _MODULE.collect_ray_worker_environment
validate_v37_remote_environment = _MODULE.validate_v37_remote_environment
FORMAL_INITIAL_MODEL_SHA256 = _MODULE._V37_FORMAL_INITIAL_MODEL_SHA256
BASE_SOURCE = Path(__file__).resolve().parents[1] / "verl" / "single_controller" / "ray" / "base.py"
FSDP_SOURCE = Path(__file__).resolve().parents[1] / "verl" / "workers" / "fsdp_workers.py"
REWARD_SOURCE = Path(__file__).resolve().parents[1] / "verl" / "workers" / "reward" / "function.py"
V32_SOURCE = Path(__file__).resolve().parents[1] / "examples" / "v32_sparse_0_10_stable_drfix.sh"
V37_SOURCE = Path(__file__).resolve().parents[1] / "examples" / "v37_strict_winner_step_rl_pilot.sh"


def test_v37_action_contract_is_forwarded_and_checked_in_remote_runner():
    source = SOURCE.read_text(encoding="utf-8")
    assert "ACTION_" not in RAY_WORKER_ENV_PREFIXES and "V37_" in RAY_WORKER_ENV_PREFIXES
    assert "ACTION_EVENT_REWARD_ENABLE" in RAY_WORKER_ENV_KEYS
    assert "collect_ray_worker_environment()" in source
    assert "_validate_v37_remote_environment()" in source

    shared = {
        "BOK_CORRECTNESS_FIRST": "1",
        "BOK_ALLWRONG_TERMINAL_ZERO": "1",
        "BOK_CORRECTNESS_TASK_WEIGHT": "0.25",
        "BOK_CORRECTNESS_QUALITY_WEIGHT": "0.1",
        "BOK_CORRECTNESS_PARTIAL_SCALE": "0.25",
        "STEPCOUNT_MASK_REQUIRE": "1",
        "TRAJ_STRICT_ANSWER_INTEGER_PARSE": "1",
        "V37_RAW_SUCCESS_STRICT_WINNER": "1",
        "V37_WINNER_MODE": "outcome_success",
        "V37_REWARD_FAIL_CLOSED": "1",
        "V37_STRICT_POINT_PARSER_CONTRACT": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "WANDB_MODE": "offline",
        "V37_TRAINING_EVIDENCE_REQUIRED": "1",
        "V37_EVIDENCE_START_STEP": "0",
        "V37_FINALIZE_HF_CHECKPOINT": "1",
        "V37_EXPECTED_INITIAL_MODEL_SHA256": FORMAL_INITIAL_MODEL_SHA256,
        "V37_RESUME_MODE": "clean_start",
    }
    baseline = shared | {
        "V37_RUN_CLASS": "formal",
        "V37_ARM": "baseline",
        "V37_SEED": "11",
        "V37_AB_CELL_ID": "seed11-baseline",
        "ACTION_EVENT_REWARD_ENABLE": "0",
    }
    progress = shared | {
        "V37_RUN_CLASS": "formal",
        "V37_ARM": "progress",
        "V37_SEED": "11",
        "V37_AB_CELL_ID": "seed11-progress",
        "ACTION_EVENT_REWARD_ENABLE": "1",
        "V37_ACTION_PARSER_CONTRACT": "1",
        "V37_ACTION_LEDGER_CONTRACT": "1",
    }
    with patch.dict(os.environ, baseline, clear=True):
        validate_v37_remote_environment("test")
        forwarded = collect_ray_worker_environment()
        assert forwarded["ACTION_EVENT_REWARD_ENABLE"] == "0"
        assert "V37_ACTION_LEDGER_CONTRACT" not in forwarded
    with patch.dict(os.environ, progress, clear=True):
        validate_v37_remote_environment("test")
        forwarded = collect_ray_worker_environment()
        assert forwarded["ACTION_EVENT_REWARD_ENABLE"] == "1"
        assert forwarded["V37_ACTION_PARSER_CONTRACT"] == "1"
        assert forwarded["V37_ACTION_LEDGER_CONTRACT"] == "1"


def test_v37_remote_runner_rejects_cross_arm_environment_leakage():
    shared = {
        "BOK_CORRECTNESS_FIRST": "1",
        "BOK_ALLWRONG_TERMINAL_ZERO": "1",
        "BOK_CORRECTNESS_TASK_WEIGHT": "0.25",
        "BOK_CORRECTNESS_QUALITY_WEIGHT": "0.1",
        "BOK_CORRECTNESS_PARTIAL_SCALE": "0.25",
        "STEPCOUNT_MASK_REQUIRE": "1",
        "TRAJ_STRICT_ANSWER_INTEGER_PARSE": "1",
        "V37_RAW_SUCCESS_STRICT_WINNER": "1",
        "V37_WINNER_MODE": "outcome_success",
        "V37_REWARD_FAIL_CLOSED": "1",
        "V37_STRICT_POINT_PARSER_CONTRACT": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "WANDB_MODE": "offline",
        "V37_TRAINING_EVIDENCE_REQUIRED": "1",
        "V37_EVIDENCE_START_STEP": "0",
        "V37_FINALIZE_HF_CHECKPOINT": "1",
        "V37_EXPECTED_INITIAL_MODEL_SHA256": FORMAL_INITIAL_MODEL_SHA256,
        "V37_RESUME_MODE": "clean_start",
    }
    missing_progress = shared | {
        "V37_RUN_CLASS": "formal",
        "V37_ARM": "progress",
        "V37_SEED": "22",
        "V37_AB_CELL_ID": "seed22-progress",
        "ACTION_EVENT_REWARD_ENABLE": "0",
    }
    with patch.dict(os.environ, missing_progress, clear=True):
        with pytest.raises(RuntimeError, match="ACTION_EVENT_REWARD_ENABLE=1"):
            validate_v37_remote_environment("test")

    leaked_baseline = shared | {
        "V37_RUN_CLASS": "formal",
        "V37_ARM": "baseline",
        "V37_SEED": "22",
        "V37_AB_CELL_ID": "seed22-baseline",
        "ACTION_EVENT_REWARD_ENABLE": "0",
        "V37_ACTION_PARSER_CONTRACT": "1",
    }
    with patch.dict(os.environ, leaked_baseline, clear=True):
        with pytest.raises(RuntimeError, match="must not inherit progress"):
            validate_v37_remote_environment("test")

    missing_strict_parser = dict(leaked_baseline)
    missing_strict_parser.pop("V37_ACTION_PARSER_CONTRACT")
    missing_strict_parser.pop("V37_STRICT_POINT_PARSER_CONTRACT")
    with patch.dict(os.environ, missing_strict_parser, clear=True):
        with pytest.raises(RuntimeError, match="V37_STRICT_POINT_PARSER_CONTRACT=1"):
            validate_v37_remote_environment("test")


def test_v37_controlled_continuation_binding_is_forwarded_and_fail_closed():
    base = {
        "V37_RUN_CLASS": "debug",
        "V37_ARM": "baseline",
        "V37_SEED": "11",
        "ACTION_EVENT_REWARD_ENABLE": "0",
        "BOK_CORRECTNESS_FIRST": "1",
        "BOK_ALLWRONG_TERMINAL_ZERO": "1",
        "BOK_CORRECTNESS_TASK_WEIGHT": "0.25",
        "BOK_CORRECTNESS_QUALITY_WEIGHT": "0.1",
        "BOK_CORRECTNESS_PARTIAL_SCALE": "0.25",
        "STEPCOUNT_MASK_REQUIRE": "1",
        "TRAJ_STRICT_ANSWER_INTEGER_PARSE": "1",
        "V37_RAW_SUCCESS_STRICT_WINNER": "1",
        "V37_WINNER_MODE": "outcome_success",
        "V37_REWARD_FAIL_CLOSED": "1",
        "V37_STRICT_POINT_PARSER_CONTRACT": "1",
        "V37_EVIDENCE_START_STEP": "4",
        "V37_RESUME_MODE": "controlled_continuation",
        "V37_EXPECTED_RESUME_CHECKPOINT_PATH": "/checkpoint/global_step_4",
        "V37_EXPECTED_RESUME_CHECKPOINT_SHA256": "a" * 64,
    }
    with patch.dict(os.environ, base, clear=True):
        validate_v37_remote_environment("test")
        forwarded = collect_ray_worker_environment()
        assert forwarded["V37_EXPECTED_RESUME_CHECKPOINT_PATH"] == "/checkpoint/global_step_4"
        assert forwarded["V37_EXPECTED_RESUME_CHECKPOINT_SHA256"] == "a" * 64
    with patch.dict(
        os.environ,
        base | {"V37_EXPECTED_RESUME_CHECKPOINT_SHA256": "not-a-sha"},
        clear=True,
    ):
        with pytest.raises(RuntimeError, match="lowercase checkpoint SHA256"):
            validate_v37_remote_environment("test")
    with patch.dict(
        os.environ,
        base | {"V37_RESUME_MODE": "clean_start"},
        clear=True,
    ):
        with pytest.raises(RuntimeError, match="must not inherit"):
            validate_v37_remote_environment("test")
    for mode in (None, "typo"):
        invalid = dict(base)
        if mode is None:
            invalid.pop("V37_RESUME_MODE")
        else:
            invalid["V37_RESUME_MODE"] = mode
        with patch.dict(os.environ, invalid, clear=True):
            with pytest.raises(RuntimeError, match="explicit valid V37_RESUME_MODE"):
                validate_v37_remote_environment("test")

    lost_class = dict(base)
    lost_class.pop("V37_RUN_CLASS")
    with patch.dict(os.environ, lost_class, clear=True):
        with pytest.raises(RuntimeError, match="no explicit valid V37_RUN_CLASS"):
            validate_v37_remote_environment("test")
        with pytest.raises(RuntimeError, match="explicit valid V37_RUN_CLASS"):
            collect_ray_worker_environment()


def test_same_environment_collector_reaches_runner_fsdp_and_reward():
    assert "collect_ray_worker_environment" in SOURCE.read_text(encoding="utf-8")
    assert "collect_ray_worker_environment" in BASE_SOURCE.read_text(encoding="utf-8")
    assert 'validate_v37_remote_environment("FSDP worker")' in FSDP_SOURCE.read_text(encoding="utf-8")
    assert 'validate_v37_remote_environment("reward worker")' in REWARD_SOURCE.read_text(encoding="utf-8")


def test_formal_offline_flags_and_sensitive_values_fail_closed():
    base = {
        "V37_RUN_CLASS": "formal",
        "V37_ARM": "baseline",
        "V37_SEED": "11",
        "V37_AB_CELL_ID": "seed11-baseline",
        "ACTION_EVENT_REWARD_ENABLE": "0",
        "BOK_CORRECTNESS_FIRST": "1",
        "BOK_ALLWRONG_TERMINAL_ZERO": "1",
        "BOK_CORRECTNESS_TASK_WEIGHT": "0.25",
        "BOK_CORRECTNESS_QUALITY_WEIGHT": "0.1",
        "BOK_CORRECTNESS_PARTIAL_SCALE": "0.25",
        "STEPCOUNT_MASK_REQUIRE": "1",
        "TRAJ_STRICT_ANSWER_INTEGER_PARSE": "1",
        "V37_RAW_SUCCESS_STRICT_WINNER": "1",
        "V37_WINNER_MODE": "outcome_success",
        "V37_REWARD_FAIL_CLOSED": "1",
        "V37_STRICT_POINT_PARSER_CONTRACT": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "WANDB_MODE": "offline",
        "V37_TRAINING_EVIDENCE_REQUIRED": "1",
        "V37_EVIDENCE_START_STEP": "0",
        "V37_FINALIZE_HF_CHECKPOINT": "1",
        "V37_EXPECTED_INITIAL_MODEL_SHA256": FORMAL_INITIAL_MODEL_SHA256,
        "V37_RESUME_MODE": "clean_start",
    }
    with patch.dict(os.environ, base, clear=True):
        validate_v37_remote_environment("test")
    with patch.dict(
        os.environ,
        base | {"V37_EXPECTED_INITIAL_MODEL_SHA256": "f" * 64},
        clear=True,
    ):
        with pytest.raises(RuntimeError, match="canonical V37_EXPECTED_INITIAL_MODEL_SHA256"):
            validate_v37_remote_environment("test")
    with patch.dict(os.environ, base | {"HF_HUB_OFFLINE": "0"}, clear=True):
        with pytest.raises(RuntimeError, match="HF_HUB_OFFLINE=1"):
            validate_v37_remote_environment("test")
    with patch.dict(os.environ, base | {"V37_API_TOKEN": "secret"}, clear=True):
        with pytest.raises(RuntimeError, match="sensitive environment"):
            collect_ray_worker_environment()
    with patch.dict(os.environ, base | {"V37_ACCESS_KEY_ID": "secret"}, clear=True):
        with pytest.raises(RuntimeError, match="sensitive environment"):
            collect_ray_worker_environment()
    with patch.dict(
        os.environ,
        base | {"V37_RUNTIME_JSON": '{"headers":{"Authorization":"hidden"}}'},
        clear=True,
    ):
        with pytest.raises(RuntimeError, match="sensitive environment"):
            collect_ray_worker_environment()


def test_v36_online_wandb_api_key_forwarding_remains_compatible():
    with patch.dict(os.environ, {"WANDB_MODE": "online", "WANDB_API_KEY": "legacy"}, clear=True):
        forwarded = collect_ray_worker_environment()
    assert forwarded["WANDB_API_KEY"] == "legacy"


def test_v36_worker_environment_keeps_historical_override_precedence():
    forwarded = collect_ray_worker_environment({
        "TOKENIZERS_PARALLELISM": "false",
        "VLLM_LOGGING_LEVEL": "DEBUG",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "0",
        "OMP_NUM_THREADS": "3",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:64",
    })
    assert forwarded["TOKENIZERS_PARALLELISM"] == "true"
    assert forwarded["VLLM_LOGGING_LEVEL"] == "WARN"
    assert forwarded["TORCH_NCCL_AVOID_RECORD_STREAMS"] == "1"
    assert forwarded["OMP_NUM_THREADS"] == "3"
    assert forwarded["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:64"


def test_v36_exact_ray_resource_flag_is_not_a_v37_identity_marker():
    source = {"V37_REQUIRE_EXACT_RAY_GPUS": "1"}
    forwarded = collect_ray_worker_environment(source)
    assert forwarded["V37_REQUIRE_EXACT_RAY_GPUS"] == "1"
    with patch.dict(os.environ, source, clear=True):
        assert validate_v37_remote_environment("test") is None


def test_v36_rejects_stray_action_event_semantics():
    source = {"ACTION_EVENT_REWARD_ENABLE": "1"}
    with pytest.raises(RuntimeError, match="explicit valid V37_RUN_CLASS"):
        collect_ray_worker_environment(source)
    with patch.dict(os.environ, source, clear=True):
        with pytest.raises(RuntimeError, match="no explicit valid V37_RUN_CLASS"):
            validate_v37_remote_environment("test")


@pytest.mark.parametrize(
    "key",
    [
        "V37_RAW_SUCCESS_STRICT_WINNER",
        "V37_ACTION_PARSER_CONTRACT",
        "V37_ACTION_LEDGER_CONTRACT",
        "V37_REWARD_FAIL_CLOSED",
        "V37_EFFECTIVE_ENVIRONMENT_PATH",
    ],
)
def test_v36_rejects_stray_v37_semantic_markers(key):
    source = {key: "1"}
    with pytest.raises(RuntimeError, match="explicit valid V37_RUN_CLASS"):
        collect_ray_worker_environment(source)
    with patch.dict(os.environ, source, clear=True):
        with pytest.raises(RuntimeError, match="no explicit valid V37_RUN_CLASS"):
            validate_v37_remote_environment("test")


def test_formal_requires_exact_ray_gpu_count_and_offline_runtime_probe():
    v32 = V32_SOURCE.read_text(encoding="utf-8")
    wrapper = V37_SOURCE.read_text(encoding="utf-8")
    assert v32.count("V37_REQUIRE_EXACT_RAY_GPUS") >= 2
    assert "formal requires exactly ${EXPECTED_GPUS} Ray GPUs" in v32
    assert "USED_GPUS" in v32 and "FREE_GPUS" in v32
    assert "AVAILABLE_GPUS_INT" not in v32
    assert "requires all ${EXPECTED_GPUS} Ray GPUs to be free" in v32
    assert "export RAY_GPU_WAIT_TIMEOUT_SECONDS=${RAY_GPU_WAIT_TIMEOUT_SECONDS:-300}" in v32
    assert "export RAY_STATUS_TIMEOUT_SECONDS=${RAY_STATUS_TIMEOUT_SECONDS:-10}" in v32
    assert 'timeout --signal=KILL "${STATUS_TIMEOUT}" "${RAY_CMD[@]}" status' in v32
    assert 'timeout --signal=KILL "${RAY_START_TIMEOUT_SECONDS}"' in v32
    assert "RAY_PLACEMENT_GROUP_TIMEOUT_SECONDS" in BASE_SOURCE.read_text(encoding="utf-8")
    assert "ray.get(ready_refs, timeout=timeout_seconds)" in BASE_SOURCE.read_text(encoding="utf-8")
    assert 'WAIT_INTERVAL=15' in v32
    assert 'sleep "${SLEEP_FOR}"' in v32
    assert "WAITED=$((WAITED + SLEEP_FOR))" in v32
    busy_check = v32.index('! "${USED_GPUS:-}" =~ ^0+([.]0+)?$') if '! "${USED_GPUS:-}" =~ ^0+([.]0+)?$' in v32 else -1
    assert busy_check == -1
    assert "timed out after ${MAX_WAIT}s with used=${USED_GPUS:-unknown}" in v32
    for assignment in (
        "export V37_REQUIRE_EXACT_RAY_GPUS=1",
        "export RAY_GPU_WAIT_TIMEOUT_SECONDS=900",
        "export RAY_STATUS_TIMEOUT_SECONDS=10",
        "export RAY_START_TIMEOUT_SECONDS=60",
        "export RAY_PLACEMENT_GROUP_TIMEOUT_SECONDS=900",
        "export HF_DATASETS_OFFLINE=1",
        "export HF_HUB_OFFLINE=1",
        "export TRANSFORMERS_OFFLINE=1",
        "export WANDB_MODE=offline",
    ):
        assert assignment in wrapper
    lock_check = '"${RAY_GPU_WAIT_TIMEOUT_SECONDS}" != "900"'
    assert lock_check in wrapper
    assert wrapper.index(lock_check) < wrapper.index("export RAY_GPU_WAIT_TIMEOUT_SECONDS=900")
    assert wrapper.index("export RAY_GPU_WAIT_TIMEOUT_SECONDS=900") < wrapper.index(
        "# Contract-only mode"
    )
    assert 'torch.cuda.device_count() != 8' in wrapper
    assert 'resolve_logprob_function("error")' in wrapper


def test_contract_only_executes_the_exact_1x8_topology_contract(tmp_path):
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("V37_") or key in {
            "BASH_ENV", "ENV", "HOST_NUM", "HOST_GPU_NUM", "V31_NNODES",
            "V31_N_GPUS_PER_NODE", "INDEX",
        }:
            env.pop(key)
    env.update(
        V37_RUN_CLASS="debug",
        V37_DATA_MODE="frontier_rl",
        V37_ARM="baseline",
        V37_CONTRACT_ONLY="1",
        TMPDIR=str(tmp_path),
    )
    result = subprocess.run(
        ["bash", str(V37_SOURCE)], cwd=V37_SOURCE.parents[1], env=env,
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "topology=1x8" in result.stdout
    assert "ray_exact=1" in result.stdout

    env["HOST_NUM"] = "2"
    result = subprocess.run(
        ["bash", str(V37_SOURCE)], cwd=V37_SOURCE.parents[1], env=env,
        text=True, capture_output=True, check=False,
    )
    assert result.returncode != 0
    assert "HOST_NUM must be 1" in result.stderr


@pytest.mark.parametrize("script", [V32_SOURCE, V37_SOURCE])
def test_ray_launcher_bash_syntax(script):
    result = subprocess.run(
        ["bash", "-n", str(script)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def _run_v32_with_mocked_ray(
    tmp_path: Path,
    *,
    busy_status_calls: int,
    exact: bool,
    timeout_seconds: int,
    status_mode: str = "normal",
    total_gpus: str = "8.0",
    status_timeout_seconds: int = 1,
    start_timeout_seconds: int = 1,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    mock_bin = tmp_path / "mock-bin"
    mock_bin.mkdir()
    status_count = tmp_path / "ray-status-count"
    sleep_log = tmp_path / "sleep.log"
    _write_executable(
        mock_bin / "ray",
        """#!/bin/bash
set -euo pipefail
if [[ "${1:-}" != "status" ]]; then
  [[ "${MOCK_RAY_STATUS_MODE}" != "start_hang" ]] || /bin/sleep 60
  exit 0
fi
count=0
[[ ! -f "${MOCK_RAY_STATUS_COUNT}" ]] || read -r count < "${MOCK_RAY_STATUS_COUNT}"
count=$((count + 1))
printf '%s\n' "${count}" > "${MOCK_RAY_STATUS_COUNT}"
case "${MOCK_RAY_STATUS_MODE}" in
  fail) exit 7 ;;
  hang) /bin/sleep 60 ;;
  start_hang) exit 7 ;;
  normal) ;;
  *) exit 9 ;;
esac
used=0.0
(( count > MOCK_RAY_BUSY_STATUS_CALLS )) || used=1.0
printf '%s\n' "${used}/${MOCK_RAY_TOTAL_GPUS} GPU"
""",
    )
    _write_executable(
        mock_bin / "sleep",
        """#!/bin/bash
set -euo pipefail
printf '%s\n' "${1:-}" >> "${MOCK_SLEEP_LOG}"
""",
    )
    _write_executable(mock_bin / "python3", "#!/bin/bash\nexit 0\n")
    _write_executable(mock_bin / "nohup", "#!/bin/bash\nexit 0\n")

    fixtures = {}
    for name in ("model", "train", "masks", "val"):
        fixtures[name] = tmp_path / name
        fixtures[name].mkdir()
    metadata = tmp_path / "masks_metadata.json"
    metadata.write_text("{}\n", encoding="utf-8")

    env = os.environ.copy()
    for key in list(env):
        if key.startswith("V37_") or key in {"BASH_ENV", "ENV", "V32_DRY_RUN"}:
            env.pop(key)
    env.update(
        PATH=f"{mock_bin}:{env['PATH']}",
        MOCK_RAY_STATUS_COUNT=str(status_count),
        MOCK_RAY_BUSY_STATUS_CALLS=str(busy_status_calls),
        MOCK_RAY_STATUS_MODE=status_mode,
        MOCK_RAY_TOTAL_GPUS=total_gpus,
        MOCK_SLEEP_LOG=str(sleep_log),
        RAY_GPU_WAIT_TIMEOUT_SECONDS=str(timeout_seconds),
        RAY_STATUS_TIMEOUT_SECONDS=str(status_timeout_seconds),
        RAY_START_TIMEOUT_SECONDS=str(start_timeout_seconds),
        RAY_PLACEMENT_GROUP_TIMEOUT_SECONDS="900",
        HOST_NUM="1",
        HOST_GPU_NUM="8",
        INDEX="0",
        STEPCOUNT_HARDWARE_PROFILE="h200",
        EASYR1_LOG_ROOT=str(tmp_path / "logs"),
        EASYR1_SAVE_ROOT=str(tmp_path / "save"),
        RAY_TMPDIR=str(tmp_path / "ray"),
        MODEL_PATH=str(fixtures["model"]),
        STEPCOUNT_TRAIN_DATA=str(fixtures["train"]),
        STEPCOUNT_MASKS_METADATA=str(metadata),
        STEPCOUNT_MASKS_DIR=str(fixtures["masks"]),
        STEPCOUNT_VAL_DATA=str(fixtures["val"]),
        V31_SAVE_CHECKPOINT_PATH=str(tmp_path / "save" / "run"),
        V30_FAILFAST_ENABLE="0",
        WANDB_MODE="offline",
    )
    if exact:
        env["V37_REQUIRE_EXACT_RAY_GPUS"] = "1"

    result = subprocess.run(
        ["bash", str(V32_SOURCE)],
        cwd=V32_SOURCE.parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    sleeps = sleep_log.read_text(encoding="utf-8").splitlines() if sleep_log.exists() else []
    return result, sleeps


def test_exact_ray_gpu_mode_waits_for_serial_predecessor_release(tmp_path):
    result, sleeps = _run_v32_with_mocked_ray(
        tmp_path,
        busy_status_calls=2,
        exact=True,
        timeout_seconds=30,
    )
    assert result.returncode == 0, result.stderr
    assert sleeps == ["15"]
    assert "Waiting... (used=1.0" in result.stdout
    assert "Ray GPUs ready: used=0.0" in result.stdout


def test_exact_ray_gpu_mode_errors_only_after_release_timeout(tmp_path):
    result, sleeps = _run_v32_with_mocked_ray(
        tmp_path,
        busy_status_calls=100,
        exact=True,
        timeout_seconds=15,
    )
    assert result.returncode != 0
    assert sleeps == ["15"]
    assert "timed out after 15s with used=1.0" in result.stderr


def test_nonexact_ray_gpu_mode_keeps_legacy_no_wait_behavior(tmp_path):
    result, sleeps = _run_v32_with_mocked_ray(
        tmp_path,
        busy_status_calls=100,
        exact=False,
        timeout_seconds=15,
    )
    assert result.returncode == 0, result.stderr
    assert sleeps == []
    assert "Ray GPUs ready: used=1.0" in result.stdout


def test_exact_ray_gpu_mode_retries_nonzero_status_until_outer_timeout(tmp_path):
    result, sleeps = _run_v32_with_mocked_ray(
        tmp_path,
        busy_status_calls=0,
        exact=True,
        timeout_seconds=15,
        status_mode="fail",
    )
    assert result.returncode != 0
    # One existing head-start delay plus one bounded release-wait interval.
    assert sleeps == ["15", "15"]
    assert "got 0 after 15s" in result.stderr


def test_exact_ray_gpu_mode_bounds_a_hung_status_probe(tmp_path):
    result, sleeps = _run_v32_with_mocked_ray(
        tmp_path,
        busy_status_calls=0,
        exact=True,
        timeout_seconds=1,
        status_mode="hang",
        status_timeout_seconds=1,
    )
    assert result.returncode != 0
    # The only sleep is the existing delay after restarting an unhealthy head.
    assert sleeps == ["15"]
    assert "after 1s" in result.stderr


def test_exact_ray_gpu_mode_bounds_a_hung_head_start(tmp_path):
    result, sleeps = _run_v32_with_mocked_ray(
        tmp_path,
        busy_status_calls=0,
        exact=True,
        timeout_seconds=15,
        status_mode="start_hang",
        start_timeout_seconds=1,
    )
    assert result.returncode != 0
    assert sleeps == []
    assert "Ray head failed to start within 1s" in result.stderr


def test_exact_ray_gpu_mode_rejects_fractional_total(tmp_path):
    result, sleeps = _run_v32_with_mocked_ray(
        tmp_path,
        busy_status_calls=0,
        exact=True,
        timeout_seconds=15,
        total_gpus="8.5",
    )
    assert result.returncode != 0
    assert sleeps == []
    assert "requires an integral Ray GPU total, got 8.5" in result.stderr


def test_formal_v37_rejects_ray_wait_timeout_override_before_asset_checks(tmp_path):
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("V37_") or key in {"BASH_ENV", "ENV"}:
            env.pop(key)
    env.update(
        V37_RUN_CLASS="formal",
        V37_DATA_MODE="frontier_rl",
        V37_ARM="baseline",
        V37_CONTRACT_ONLY="1",
        RAY_GPU_WAIT_TIMEOUT_SECONDS="899",
        TMPDIR=str(tmp_path),
    )
    result = subprocess.run(
        ["bash", str(V37_SOURCE)],
        cwd=V37_SOURCE.parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "formal RAY_GPU_WAIT_TIMEOUT_SECONDS is locked to 900" in result.stderr


def test_formal_v37_rejects_ray_status_timeout_override_before_asset_checks(tmp_path):
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("V37_") or key in {"BASH_ENV", "ENV"}:
            env.pop(key)
    env.update(
        V37_RUN_CLASS="formal",
        V37_DATA_MODE="frontier_rl",
        V37_ARM="baseline",
        V37_CONTRACT_ONLY="1",
        RAY_STATUS_TIMEOUT_SECONDS="11",
        TMPDIR=str(tmp_path),
    )
    result = subprocess.run(
        ["bash", str(V37_SOURCE)],
        cwd=V37_SOURCE.parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "formal RAY_STATUS_TIMEOUT_SECONDS is locked to 10" in result.stderr


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("RAY_START_TIMEOUT_SECONDS", "61", "formal RAY_START_TIMEOUT_SECONDS is locked to 60"),
        (
            "RAY_PLACEMENT_GROUP_TIMEOUT_SECONDS",
            "901",
            "formal RAY_PLACEMENT_GROUP_TIMEOUT_SECONDS is locked to 900",
        ),
    ],
)
def test_formal_v37_rejects_other_ray_timeout_overrides_before_asset_checks(
    tmp_path, name, value, message
):
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("V37_") or key in {"BASH_ENV", "ENV"}:
            env.pop(key)
    env.update(
        V37_RUN_CLASS="formal",
        V37_DATA_MODE="frontier_rl",
        V37_ARM="baseline",
        V37_CONTRACT_ONLY="1",
        TMPDIR=str(tmp_path),
    )
    env[name] = value
    result = subprocess.run(
        ["bash", str(V37_SOURCE)], cwd=V37_SOURCE.parents[1], env=env,
        text=True, capture_output=True, check=False,
    )
    assert result.returncode != 0
    assert message in result.stderr
