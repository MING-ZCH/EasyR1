"""Shared, fail-closed Ray environment forwarding for EasyR1 workers."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping


RAY_WORKER_ENV_PREFIXES = (
    "STEPCOUNT_", "TRAJ_", "EASYR1_", "INTERLEAVED_", "BOK_",
    "PROCESS_REWARD_", "POLICY_LOSS_", "GRAD_SPIKE_", "GRAD_NONFINITE_",
    "VCRL_", "V37_",
)

RAY_WORKER_ENV_KEYS = {
    "DISABLE_ADDMM_CUDA_LT",
    "ACTION_EVENT_REWARD_ENABLE",
    "GLOO_SOCKET_IFNAME",
    "HF_DATASETS_OFFLINE",
    "HF_HUB_DISABLE_TELEMETRY",
    "HF_HUB_OFFLINE",
    "LD_LIBRARY_PATH",
    "MKL_NUM_THREADS",
    "NCCL_DEBUG",
    "NCCL_IB_DISABLE",
    "NCCL_IB_DISABLE_ECE",
    "NCCL_IB_GID_INDEX",
    "NCCL_IB_HCA",
    "NCCL_IB_QPS_PER_CONNECTION",
    "NCCL_IB_SL",
    "NCCL_IB_TC",
    "NCCL_IB_TIMEOUT",
    "NCCL_NET_GDR_LEVEL",
    "NCCL_P2P_DISABLE",
    "NCCL_PXN_DISABLE",
    "NCCL_SOCKET_IFNAME",
    "NCCL_TIMEOUT",
    "NVIDIA_TF32_OVERRIDE",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "PYTHONHASHSEED",
    "PYTHONUNBUFFERED",
    "PYTORCH_CUDA_ALLOC_CONF",
    "REWARD_NUM_WORKERS",
    "TOKENIZERS_PARALLELISM",
    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
    "TORCH_DISTRIBUTED_TIMEOUT",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    "TORCH_NCCL_AVOID_RECORD_STREAMS",
    "TORCH_NUM_THREADS",
    "TP_SOCKET_IFNAME",
    "TRANSFORMERS_OFFLINE",
    "VLLM_ATTENTION_BACKEND",
    "VLLM_LOGGING_LEVEL",
    "VLLM_USE_V1",
    "WANDB_DIR",
    "WANDB_API_KEY",
    "WANDB_MODE",
}

_SENSITIVE_FRAGMENTS = (
    "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "API_KEY", "ACCESS_KEY",
    "PRIVATE_KEY", "AUTH", "BEARER", "COOKIE", "SESSION",
)
_V36_LEGACY_SECRET_FORWARD_KEYS = {"WANDB_API_KEY"}
_V37_RUN_CLASSES = {"debug", "canary", "formal"}
_V37_FORMAL_INITIAL_MODEL_SHA256 = "f9e6b1e8031bdbc509d34249745cdcf75af85c320d6b88918444b1abb4f580a3"
_V37_V36_SAFE_KEYS = {
    # This shared V32 resource-wait switch predates the V37 semantic contract
    # and remains intentionally usable by V36 launchers.
    "V37_REQUIRE_EXACT_RAY_GPUS",
}
_V37_SEMANTIC_KEYS = {"ACTION_EVENT_REWARD_ENABLE"}


def _has_v37_marker(values: Mapping[str, str]) -> bool:
    return any(
        (key.startswith("V37_") and key not in _V37_V36_SAFE_KEYS)
        or key in _V37_SEMANTIC_KEYS
        for key in values
    )


def _is_sensitive(key: str) -> bool:
    upper = key.upper()
    return (
        any(fragment in upper for fragment in _SENSITIVE_FRAGMENTS)
        or upper.endswith("_TOKEN")
        or "_TOKEN_" in upper
    )


def _contains_sensitive_json(value: str) -> bool:
    try:
        payload = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return False

    def visit(item) -> bool:
        if isinstance(item, dict):
            return any(_is_sensitive(str(key)) or visit(child) for key, child in item.items())
        if isinstance(item, list):
            return any(visit(child) for child in item)
        return False

    return visit(payload)


def collect_ray_worker_environment(
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Collect one shared env contract for Runner, reward, and FSDP actors."""
    values = os.environ if source is None else source
    run_class = values.get("V37_RUN_CLASS")
    v37 = run_class in _V37_RUN_CLASSES
    if not v37 and _has_v37_marker(values):
        raise RuntimeError("V37 environment markers require an explicit valid V37_RUN_CLASS.")
    selected: dict[str, str] = {}
    for key, value in values.items():
        if key in RAY_WORKER_ENV_KEYS or key.startswith(RAY_WORKER_ENV_PREFIXES):
            sensitive = _is_sensitive(key) or (key.endswith("_JSON") and _contains_sensitive_json(value))
            if sensitive:
                if v37:
                    raise RuntimeError(f"V37 refuses to forward sensitive environment variable {key}.")
                if key not in _V36_LEGACY_SECRET_FORWARD_KEYS:
                    continue
            selected[key] = value

    defaults = {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": values.get("NCCL_DEBUG", "WARN"),
        "VLLM_LOGGING_LEVEL": "WARN",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "PYTORCH_CUDA_ALLOC_CONF": values.get(
            "PYTORCH_CUDA_ALLOC_CONF",
            "expandable_segments:True,max_split_size_mb:512,roundup_power2_divisions:16",
        ),
        "PYTHONUNBUFFERED": values.get("PYTHONUNBUFFERED", "1"),
        "PYTHONHASHSEED": values.get("PYTHONHASHSEED", "0"),
        "NCCL_TIMEOUT": values.get("NCCL_TIMEOUT", "1800"),
        "TORCH_DISTRIBUTED_TIMEOUT": values.get("TORCH_DISTRIBUTED_TIMEOUT", "1800"),
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": values.get(
            "TORCH_NCCL_ASYNC_ERROR_HANDLING",
            values.get("NCCL_ASYNC_ERROR_HANDLING", "1"),
        ),
        "OMP_NUM_THREADS": values.get("OMP_NUM_THREADS", "8"),
        "MKL_NUM_THREADS": values.get("MKL_NUM_THREADS", "8"),
        "NUMEXPR_NUM_THREADS": values.get("NUMEXPR_NUM_THREADS", "8"),
        "TORCH_NUM_THREADS": values.get("TORCH_NUM_THREADS", "8"),
        "LD_LIBRARY_PATH": values.get("LD_LIBRARY_PATH", ""),
    }
    for key, value in defaults.items():
        if v37:
            selected.setdefault(key, value)
        else:
            # Match the historical Runner runtime_env contract: a few safety
            # keys are forced, while the other values are inherited/defaulted.
            selected[key] = value
    return selected


def validate_v37_remote_environment(role: str) -> dict[str, str] | None:
    """Verify V37 semantics in every remote process, not only the Ray Runner."""
    run_class = os.environ.get("V37_RUN_CLASS")
    if run_class not in _V37_RUN_CLASSES:
        if _has_v37_marker(os.environ):
            raise RuntimeError(f"V37 {role} has markers but no explicit valid V37_RUN_CLASS.")
        return None
    arm = os.environ.get("V37_ARM")
    seed = os.environ.get("V37_SEED")
    if arm not in {"baseline", "progress"} or not seed or not seed.isdigit():
        raise RuntimeError(f"V37 {role} is missing the arm/seed environment contract.")
    required = {
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
        "ACTION_EVENT_REWARD_ENABLE": "1" if arm == "progress" else "0",
    }
    if run_class == "formal":
        required.update({
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_MODE": "offline",
            "V37_TRAINING_EVIDENCE_REQUIRED": "1",
            "V37_EVIDENCE_START_STEP": "0",
            "V37_FINALIZE_HF_CHECKPOINT": "1",
        })
        expected_model_sha256 = os.environ.get("V37_EXPECTED_INITIAL_MODEL_SHA256")
        if expected_model_sha256 != _V37_FORMAL_INITIAL_MODEL_SHA256:
            raise RuntimeError(
                f"V37 formal {role} requires the canonical V37_EXPECTED_INITIAL_MODEL_SHA256."
            )
    start_step = os.environ.get("V37_EVIDENCE_START_STEP")
    if start_step is None or not start_step.isdigit():
        raise RuntimeError(f"V37 {role} requires a nonnegative V37_EVIDENCE_START_STEP.")
    for key, expected in required.items():
        if os.environ.get(key) != expected:
            raise RuntimeError(f"V37 {role} requires {key}={expected}.")
    progress_contracts = ("V37_ACTION_PARSER_CONTRACT", "V37_ACTION_LEDGER_CONTRACT")
    if arm == "progress":
        for key in progress_contracts:
            if os.environ.get(key) != "1":
                raise RuntimeError(f"V37 progress {role} requires {key}=1.")
    elif any(key in os.environ for key in progress_contracts):
        raise RuntimeError(f"V37 baseline {role} must not inherit progress action contracts.")
    if run_class == "formal" and os.environ.get("V37_AB_CELL_ID") != f"seed{seed}-{arm}":
        raise RuntimeError(f"V37 formal {role} A/B cell identity mismatch.")
    resume_mode = os.environ.get("V37_RESUME_MODE")
    expected_resume_path = os.environ.get("V37_EXPECTED_RESUME_CHECKPOINT_PATH")
    expected_resume_sha256 = os.environ.get("V37_EXPECTED_RESUME_CHECKPOINT_SHA256")
    if resume_mode not in {"clean_start", "controlled_continuation"}:
        raise RuntimeError(f"V37 {role} requires an explicit valid V37_RESUME_MODE.")
    if run_class == "formal" and resume_mode != "clean_start":
        raise RuntimeError(f"V37 formal {role} requires a clean start.")
    if resume_mode == "controlled_continuation":
        if not expected_resume_path or not os.path.isabs(expected_resume_path):
            raise RuntimeError(
                f"V37 controlled continuation {role} requires an absolute expected checkpoint path."
            )
        if (
            expected_resume_sha256 is None
            or len(expected_resume_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_resume_sha256)
        ):
            raise RuntimeError(
                f"V37 controlled continuation {role} requires a lowercase checkpoint SHA256."
            )
    elif resume_mode == "clean_start" and (
        expected_resume_path is not None or expected_resume_sha256 is not None
    ):
        raise RuntimeError(f"V37 clean-start {role} must not inherit a resume checkpoint binding.")
    collect_ray_worker_environment()
    return {key: os.environ[key] for key in sorted(required)}
