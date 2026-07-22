import copy
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tools import v37_gate as gate
from tools import preflight_v37_training as preflight_module
from tools import v37_benchmark_evidence as benchmark_evidence
training_evidence = gate.training_evidence

SHA = "a" * 64
PRODUCTION_BENCHMARK_DATASET_SHA256 = copy.deepcopy(gate.FORMAL_BENCHMARK_DATASET_SHA256)
PRODUCTION_BENCHMARK_ORDERED_IDS_SHA256 = copy.deepcopy(
    gate.FORMAL_BENCHMARK_ORDERED_IDS_SHA256
)
_SYNTHETIC_PAIRED_SOURCE_ROWS = {}
_REAL_LIVE_GIT_IDENTITY = gate._live_git_identity


@pytest.fixture(autouse=True)
def _isolate_synthetic_formal_benchmark_identities(monkeypatch):
    _SYNTHETIC_PAIRED_SOURCE_ROWS.clear()
    repo = Path(gate.__file__).resolve().parents[1]
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True,
    ).stdout.decode("ascii").strip()
    monkeypatch.setattr(gate, "_live_git_identity", lambda _repo: {
        "git_commit": git_commit,
        "git_status_sha256": gate.EMPTY_SHA256,
        "git_diff_sha256": gate.EMPTY_SHA256,
    })
    monkeypatch.setattr(gate, "FORMAL_BENCHMARK_DATASET_SHA256", {})
    monkeypatch.setattr(gate, "FORMAL_BENCHMARK_ORDERED_IDS_SHA256", {})
    monkeypatch.setattr(
        gate,
        "_recompute_paired_source_rows",
        lambda path: copy.deepcopy(_SYNTHETIC_PAIRED_SOURCE_ROWS[str(Path(path).resolve())]),
    )
    yield
    _SYNTHETIC_PAIRED_SOURCE_ROWS.clear()


def config_sha(config, environment):
    payload = json.dumps(
        {"selected": config, "environment": environment},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def test_tree_hash_has_unambiguous_file_boundaries(tmp_path):
    one = tmp_path / "one"
    two = tmp_path / "two"
    one.mkdir()
    two.mkdir()
    (one / "a").write_bytes(b"b\0c")
    (two / "a").write_bytes(b"")
    (two / "b").write_bytes(b"c")
    assert gate.tree_sha256(one) != gate.tree_sha256(two)


def test_live_git_identity_ignores_repository_override_environment(tmp_path, monkeypatch):
    fake = tmp_path / "fake-repo"
    subprocess.run(["git", "init", "-q", str(fake)], check=True)
    monkeypatch.setenv("GIT_DIR", str(fake / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(fake))
    repo = Path(gate.__file__).resolve().parents[1]
    identity = _REAL_LIVE_GIT_IDENTITY(repo)
    expected_environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    expected = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
        capture_output=True, env=expected_environment,
    ).stdout.decode("ascii").strip()
    assert identity["git_commit"] == expected


def test_stable_artifact_snapshot_rejects_path_replacement(tmp_path, monkeypatch):
    artifact_path = tmp_path / "evidence.json"
    replacement = tmp_path / "replacement.json"
    artifact_path.write_text('{"value":1}\n', encoding="utf-8")
    replacement.write_text('{"value":2}\n', encoding="utf-8")
    original_read = gate.os.read
    replaced = False

    def replacing_read(descriptor, size):
        nonlocal replaced
        chunk = original_read(descriptor, size)
        if chunk and not replaced:
            replacement.replace(artifact_path)
            replaced = True
        return chunk

    monkeypatch.setattr(gate.os, "read", replacing_read)
    with pytest.raises(gate.GateError, match="path changed while being read"):
        gate._read_stable_regular_file(artifact_path)


def test_stable_artifact_snapshot_rejects_same_inode_post_read_mutation(tmp_path, monkeypatch):
    artifact_path = tmp_path / "evidence.json"
    artifact_path.write_text('{"value":1}\n', encoding="utf-8")
    original_inode = artifact_path.stat().st_ino
    original_open = gate._open_nonsymlink_file
    calls = 0

    def mutating_open(path):
        nonlocal calls
        calls += 1
        if calls == 2:
            artifact_path.write_text('{"value":200}\n', encoding="utf-8")
            assert artifact_path.stat().st_ino == original_inode
        return original_open(path)

    monkeypatch.setattr(gate, "_open_nonsymlink_file", mutating_open)
    with pytest.raises(gate.GateError, match="path changed while being read"):
        gate._read_stable_regular_file(artifact_path)


def test_stable_artifact_snapshot_rejects_parent_symlink_substitution(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    alternate = tmp_path / "alternate"
    parent.mkdir()
    alternate.mkdir()
    artifact_path = parent / "evidence.json"
    artifact_path.write_text('{"source":"original"}\n', encoding="utf-8")
    (alternate / artifact_path.name).write_text(
        '{"source":"alternate"}\n', encoding="utf-8",
    )
    saved_parent = tmp_path / "saved-parent"
    original_open = gate.os.open
    switched = False

    def redirecting_open(path, flags, *args, **kwargs):
        nonlocal switched
        if gate.os.fspath(path) == parent.name and not switched:
            parent.rename(saved_parent)
            parent.symlink_to(alternate, target_is_directory=True)
            switched = True
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(gate.os, "open", redirecting_open)
    with pytest.raises(gate.GateError, match="non-symlink artifact path"):
        gate._read_stable_regular_file(artifact_path)


def test_artifact_hash_and_json_decode_share_one_byte_snapshot(tmp_path, monkeypatch):
    artifact_path = tmp_path / "evidence.json"
    raw = b'{"value":1}\n'
    artifact_path.write_bytes(raw)
    run = {
        "eval": str(artifact_path.resolve()),
        "eval_sha256": hashlib.sha256(raw).hexdigest(),
    }
    calls = 0
    original_reader = gate._read_stable_regular_file

    def counted_reader(path):
        nonlocal calls
        calls += 1
        return original_reader(path)

    monkeypatch.setattr(gate, "_read_stable_regular_file", counted_reader)
    assert gate.artifact(run, "eval", require_path=True) == {"value": 1}
    assert calls == 1


def test_hashed_row_loader_rejects_before_decoding_wrong_snapshot(tmp_path, monkeypatch):
    artifact_path = tmp_path / "evidence.json"
    artifact_path.write_text('{"value":1}\n', encoding="utf-8")
    decoded = False

    def decoding_spy(*args, **kwargs):
        nonlocal decoded
        decoded = True
        return {}, False

    monkeypatch.setattr(gate, "_read_stable_regular_file", lambda _path: b'{"value":2}\n')
    monkeypatch.setattr(gate, "_decode_rows", decoding_spy)
    with pytest.raises(gate.GateError, match="hash mismatch"):
        gate._load_hashed_rows(
            artifact_path,
            hashlib.sha256(b'{"value":1}\n').hexdigest(),
            "evidence",
        )
    assert decoded is False


def test_hashed_row_loader_reads_once_for_hash_and_decode(tmp_path, monkeypatch):
    artifact_path = tmp_path / "evidence.json"
    raw = b'{"value":1}\n'
    artifact_path.write_bytes(raw)
    calls = 0
    original_reader = gate._read_stable_regular_file

    def counted_reader(path):
        nonlocal calls
        calls += 1
        return original_reader(path)

    monkeypatch.setattr(gate, "_read_stable_regular_file", counted_reader)
    payload, is_jsonl = gate._load_hashed_rows(
        artifact_path, hashlib.sha256(raw).hexdigest(), "evidence",
    )
    assert payload == {"value": 1}
    assert is_jsonl is False
    assert calls == 1


def metric_rules():
    return copy.deepcopy(gate.FROZEN_PAIRED_METRICS)


def mechanism(arm):
    progress = arm == "progress"
    return {
        "schema_version": 2,
        "arm": arm,
        "estimator": "bok_grpo_step" if progress else "bok_grpo",
        "native_action_enabled": progress,
        "native_action_parser_contract": "native_action_parser_v1" if progress else "disabled",
        "native_action_ledger_contract": "native_action_ledger_v2" if progress else "disabled",
        "legacy_process_reward_enabled": False,
        "step_signal": "native_action_event" if progress else None,
        "step_weight": .1 if progress else 0.0,
        "answer_gate": {"mode": "answer_soft" if progress else None, "minimum": .2 if progress else None},
        "correctness_first": True,
        "allwrong_terminal_zero": True,
        "correctness_secondary": {
            "task_weight": .25,
            "quality_weight": .1,
            "partial_scale": .25,
            "all_answer_wrong_terminal_zero": True,
            "exact_answer_partial_enabled": True,
        },
        "reward_fail_closed": True,
        "adaptive_actor_kl": {
            "enabled": True, "type": "adaptive", "penalty": "low_var_kl",
            "init_beta": .08, "target": .15, "horizon": 10000, "selector_includes_kl": False,
        },
        "legacy_kl": {"use_kl_loss": False, "reward_kl_enabled": False},
        "cp_size": 1,
        "fallback_logprob_sign_opt_in": False,
        "torch_logprob_fallback_mode": "error",
        "strict_answer_integer_parse": True,
        "strict_raw_success_winner": True,
        "strict_point_parser_contract": "strict_point_slots_v2",
    }


def mining_contract():
    checkpoint = Path(sys.executable).resolve()
    return {
        "seeds": [11, 22],
        "candidates_per_seed": 32,
        "classifier_thresholds": copy.deepcopy(gate.FORMAL_CLASSIFIER_THRESHOLDS),
        "outcome_ratio": .75,
        "process_ratio": .25,
        "bucket_ratios": copy.deepcopy(gate.FORMAL_BUCKET_RATIOS),
        "model_checkpoint_path": str(checkpoint),
        "model_checkpoint_sha256": gate.tree_sha256(checkpoint),
        "source_binding_contract": "source_row_image_rendered_prompt_v1",
        "source_binding_sha256": "e" * 64,
        "generation_config_by_seed": {
            "11": {"seed": 11, "max_new_tokens": 512},
            "22": {"seed": 22, "max_new_tokens": 512},
        },
        "sampling_config": {"temperature": .7, "top_p": 1.0},
        "seed_difference_allowlist": ["seed"],
    }


def samples(arm):
    rows = []
    for bucket_index, bucket in enumerate(gate.BUCKETS):
        for index in range(2):
            baseline = {
                "answer_exact": .4,
                "format_compliance": .7,
                "unique_valid_hit": .4,
                "duplicate": .4,
                "cap_turn_exceeded": .3,
                "early_stop": .3,
                "progress_degraded": .2,
                "kl_recoverability": .6,
            }
            if arm == "progress":
                baseline = {
                    name: value + (.1 if name in gate.HIGHER_BETTER else -.1)
                    for name, value in baseline.items()
                }
            rows.append({
                "sample_id": f"s-{bucket_index}-{index}",
                "prompt_id": f"p-{bucket_index}-{index}",
                "bucket": bucket,
                "metrics": baseline,
            })
    return rows


def formal_samples(arm):
    rows = []
    progress = arm == "progress"
    for bucket_index, bucket in enumerate(gate.BUCKETS):
        for index in range(2):
            ground_truth = gate.BUCKET_RANGES[bucket][0]
            if progress:
                events = [
                    {"point_index": number, "matched_target_id": f"target-{number}"}
                    for number in range(1, ground_truth + 1)
                ]
                answer = ground_truth
                termination = "answer"
            else:
                events = [
                    {"point_index": number, "matched_target_id": "target-1"}
                    for number in range(1, ground_truth + 4)
                ]
                answer = None
                termination = "cap"
            transcript = [
                f'<think>inspect</think><point>{{"point_2d":[{event["point_index"]},2],"label":"object","count_number":{event["point_index"]}}}</point>'
                for event in events
            ]
            for event, turn in zip(events, transcript):
                raw_payload = gate.POINT_BLOCK_CAPTURE_RE.findall(turn)[0]
                event["point_payload_sha256"] = hashlib.sha256(raw_payload.encode()).hexdigest()
            if progress:
                transcript.append(f"<think>done</think><answer>{answer}</answer>")
            row = {
                "sample_id": f"s-{bucket_index}-{index}",
                "prompt_id": f"p-{bucket_index}-{index}",
                "predicted_answer": answer,
                "transcript": transcript,
                "point_events": events,
                "termination_reason": termination,
                "num_rounds": len(transcript),
                "configured_max_turns": 53,
                "effective_max_turns": ground_truth + 3,
                "source_row_sha256": hashlib.sha256(
                    gate.eval_producer._canonical_bytes({
                        "sample_id": f"s-{bucket_index}-{index}",
                        "prompt_id": f"p-{bucket_index}-{index}",
                        "bucket": bucket,
                        "ground_truth": ground_truth,
                    })
                ).hexdigest(),
                "metrics": {
                    "answer_exact": float(progress),
                    "format_compliance": 1.0,
                    "unique_valid_hit": (
                        1.0 if progress else 1.0 / float(ground_truth)
                    ),
                    "duplicate": (
                        0.0 if progress else (len(events) - 1) / float(len(events))
                    ),
                    "cap_turn_exceeded": float(not progress),
                    "early_stop": 0.0,
                },
            }
            gate._recompute_formal_sample(
                row, f"fixture-{arm}-{bucket_index}-{index}",
                ground_truth=ground_truth, bucket=bucket,
            )
            rows.append(row)
    return rows


def frozen_sample_hash(rows):
    indexed = {(row["sample_id"], row["prompt_id"]): row for row in rows}
    return gate._sample_set_sha256(indexed)


def valid_spec():
    runs = []
    for seed in (11, 22):
        for arm in ("baseline", "progress"):
            manifest = {
                "manifest_version": 3,
                "arm": arm,
                "seed": seed,
                "run_class": "canary",
                "data_mode": "frontier_rl",
                "promotable_candidate": False,
                "run_purpose": "canary_diagnostic",
                "resume_mode": "clean_start",
                "ab_preregistration": None,
                "mechanism_config": mechanism(arm),
                "mining_contract": mining_contract(),
                "config": {
                    "lr": 2e-7, "arm": arm, "estimator": mechanism(arm)["estimator"],
                    "pilot_steps": 4,
                    "filter_overlong_num_proc": 64,
                    "process_reward": "1" if arm == "progress" else "0",
                    "step_signal": "native_action_event" if arm == "progress" else None,
                    "step_weight": "0.1" if arm == "progress" else "0",
                    "answer_gate": {
                        "mode": "answer_soft" if arm == "progress" else None,
                        "minimum": "0.2" if arm == "progress" else None,
                    },
                },
                "input_hashes": {"train": SHA},
                "dataset_sha256": SHA,
                "mask_tree_sha256": SHA,
                "metadata_sha256": SHA,
                "input_snapshots": {"model": {"sha256": "1" * 64}},
                "paired_eval_data_sha256": "c" * 64,
            }
            environment = {
                "ACTOR_LR": "2e-7", "ADV_ESTIMATOR": mechanism(arm)["estimator"],
                "ACTION_EVENT_REWARD_ENABLE": "1" if arm == "progress" else "0",
                "CONFIG_PATH": f"/fixture/{arm}-{seed}/v37_config.yaml",
                "PYTHONHASHSEED": str(seed), "V31_DATA_SEED": str(seed),
                "V31_EXPERIMENT_NAME": f"fixture-{arm}-{seed}",
                "V31_ROLLOUT_SEED": str(seed),
                "V31_SAVE_CHECKPOINT_PATH": f"/fixture/{arm}-{seed}",
                "V37_ARM": arm, "V37_RUN_MANIFEST": f"/fixture/{arm}-{seed}/manifest.json",
                "V37_SEED": str(seed),
                "V37_PILOT_STEPS": "4", "V36_MAX_STEPS": "4", "BOK_TOTAL_STEPS": "4",
                "BOK_STEP_WEIGHT": "0.1" if arm == "progress" else "0",
                "BOK_CORRECTNESS_TASK_WEIGHT": "0.25",
                "BOK_CORRECTNESS_QUALITY_WEIGHT": "0.1",
                "BOK_CORRECTNESS_PARTIAL_SCALE": "0.25",
                "PROCESS_REWARD_ENABLE": "0", "STEPCOUNT_RL_MODE": mechanism(arm)["estimator"],
                "STEPCOUNT_MASK_REQUIRE": "1",
                "TRAJ_STRICT_ANSWER_INTEGER_PARSE": "1",
                "V37_RAW_SUCCESS_STRICT_WINNER": "1", "V37_STRICT_POINT_PARSER_CONTRACT": "1",
                "V37_REWARD_FAIL_CLOSED": "1",
            }
            if arm == "progress":
                environment.update({
                    "BOK_STEP_SIGNAL": "native_action_event", "BOK_STEP_GATE": "answer_soft",
                    "BOK_STEP_MIN_GATE": "0.2", "V37_ACTION_PARSER_CONTRACT": "1",
                    "V37_ACTION_LEDGER_CONTRACT": "1",
                })
            runtime_environment = {
                "python_executable": str(Path(sys.executable).resolve()),
                "python_executable_sha256": gate.sha256(Path(sys.executable).resolve()),
                "python_version": "fixture-python",
                "packages": {
                    name: "fixture-version"
                    for name in ("flash-attn", "numpy", "pyarrow", "ray", "tensordict", "torch", "transformers", "vllm")
                },
            }
            manifest["runtime_environment"] = runtime_environment
            manifest["audited_environment"] = environment
            manifest["continuation_config_sha256"] = gate.continuation.continuation_config_sha256(
                manifest["config"]
            )
            manifest["config_sha256"] = config_sha(manifest["config"], environment)
            evidence_run_id = hashlib.sha256(json.dumps({
                "arm": arm, "seed": seed,
                "run_directory": str(Path(environment["V37_RUN_MANIFEST"]).resolve().parent),
                "manifest_path": str(Path(environment["V37_RUN_MANIFEST"]).resolve()),
                "config_sha256": manifest["config_sha256"],
                "runtime_config_sha256": manifest.get("config_file_sha256"),
                "source_config_sha256": manifest.get("source_config_sha256"),
                "runtime_environment": runtime_environment,
                "execution_environment_sha256": manifest.get("execution_environment_sha256"),
                "ab_preregistration": manifest.get("ab_preregistration"),
                "resume_evidence": manifest.get("resume_evidence"),
                "git_commit": manifest.get("git_commit"),
                "git_status_sha256": manifest.get("git_status_sha256"),
                "git_diff_sha256": manifest.get("git_diff_sha256"),
                "implementation_sha256": manifest.get("implementation_sha256"),
                "initial_model_sha256": manifest["input_snapshots"]["model"]["sha256"],
                "dataset_sha256": manifest["dataset_sha256"],
            }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            manifest["evidence_run_id"] = evidence_run_id
            log = {
                "optimizer_steps": 4,
                "skipped_updates": 0,
                "nonfinite_count": 0,
                "oom_count": 0,
                "progress_degraded": 0,
                "correctness_sign_error_count": 0,
                "cap_turn_exceeded": 0,
                "early_stop_count": 0,
                "selector_kl_contribution": 0,
                "event_mapping_coverage": 1,
                "runtime_mask_coverage": 1,
                "kl_recoverable": 1,
                "frontier_mixed_group_ratio": .8,
                "outcome_frontier_allwrong_ratio": .2,
                "evidence_run_id": evidence_run_id,
            }
            sample_rows = samples(arm)
            evaluation = {
                "protocol_sha256": "b" * 64,
                "data_sha256": "c" * 64,
                "eval_config_sha256": "d" * 64,
                "samples": sample_rows,
                "sample_set_sha256": frozen_sample_hash(sample_rows),
                "evidence_run_id": evidence_run_id,
            }
            runs.append({"arm": arm, "seed": seed, "manifest": manifest, "log": log, "eval": evaluation})
    return {
        "schema_version": 2,
        "stage": "canary",
        "run_class": "canary",
        "preregistered": {
            "seeds": [11, 22],
            "optimizer_steps": 4,
            "confidence_level": .95,
            "ci_method": "paired_normal_95",
            "frontier_mixed_min": .7,
            "outcome_allwrong_max": .25,
            "paired_metrics": metric_rules(),
        },
        "runs": runs,
    }


def _write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _make_benchmark_base(root: Path, suite: str):
    total = benchmark_evidence.SUITE_SPECS[suite]["total"]
    image = root / "source.img"
    if not image.exists():
        image.write_bytes(b"formal-benchmark-image")
    dataset_rows = [
        {
            "id": f"{suite}-{index:04d}",
            "question": f"How many objects are in benchmark image {index}?",
            "answer": index % 11 + 1,
            "image_path": str(image.resolve()),
        }
        for index in range(total)
    ]
    dataset = root / f"{suite}-dataset.json"
    _write_json(dataset, dataset_rows)
    suite_key = "pixmo" if suite == "pixmo-test" else "stepcount"
    gate.FORMAL_BENCHMARK_DATASET_SHA256[suite_key] = benchmark_evidence.sha256_file(dataset)
    gate.FORMAL_BENCHMARK_ORDERED_IDS_SHA256[suite_key] = hashlib.sha256(
        gate.canonical([row["id"] for row in dataset_rows]).encode("utf-8")
    ).hexdigest()
    image_hash = benchmark_evidence.sha256_file(image)
    image_manifest = root / f"{suite}-images.json"
    _write_json(image_manifest, {
        "schema_version": 2,
        "suite": suite,
        "images": [
            {"id": row["id"], "path": str(image.resolve()), "sha256": image_hash}
            for row in dataset_rows
        ],
    })
    return {
        "suite": suite,
        "rows": dataset_rows,
        "dataset": dataset,
        "image": image,
        "image_hash": image_hash,
        "image_manifest": image_manifest,
    }


def _materialize_benchmark_evidence(
    tmp_path: Path,
    base,
    checkpoint: Path,
    run_index: int,
    correct_count: int,
    shared_implementations,
) -> Path:
    suite = base["suite"]
    task_cap = benchmark_evidence.SUITE_SPECS[suite]["task_cap"]
    eval_model = checkpoint / "actor" / "huggingface"
    assert eval_model.is_dir()
    dataset_hash = benchmark_evidence.sha256_file(base["dataset"])
    checkpoint_hash = benchmark_evidence.tree_sha256(checkpoint)
    model_hash = benchmark_evidence.tree_sha256(eval_model)
    result_rows = []
    for index, dataset_row in enumerate(base["rows"]):
        answer = int(dataset_row["answer"])
        predicted = answer if index < correct_count else answer + 100
        response = f"<answer>{predicted}</answer>"
        result_rows.append({
            "id": dataset_row["id"],
            "question": dataset_row["question"],
            "correct_answer": answer,
            "predicted_answer": str(predicted),
            "explicit_answer_detected": True,
            "is_correct": predicted == answer,
            "effective_max_rounds": min(task_cap, answer + 3),
            "num_rounds": 1,
            "termination_reason": "explicit_answer",
            "last_response_event": "answer",
            "used_point_count_fallback": False,
            "dataset_row_sha256": benchmark_evidence.dataset_row_sha256(dataset_row),
            "source_image_sha256": base["image_hash"],
            "dataset_sha256": dataset_hash,
            "checkpoint_sha256": checkpoint_hash,
            "eval_model_sha256": model_hash,
            "model_path": str(eval_model.resolve()),
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
                    "content": dataset_row["question"],
                    "image": str(base["image"].resolve()),
                },
                {"role": "model", "content": response, "round": 1},
            ],
            "final_response": response,
        })
    raw_results = tmp_path / f"raw-{run_index}-{suite}.json"
    _write_json(raw_results, result_rows)
    run_manifest = tmp_path / f"eval-run-{run_index}-{suite}.json"
    run_manifest_payload = {
        "schema_version": 2,
        "manifest_type": benchmark_evidence.RUN_MANIFEST_TYPE,
        "suite": suite,
        "artifacts": {
            "checkpoint_tree": benchmark_evidence.artifact_binding(checkpoint, kind="tree"),
            "eval_model": benchmark_evidence.artifact_binding(eval_model, kind="tree"),
            "dataset": benchmark_evidence.artifact_binding(base["dataset"], kind="file"),
            "external_image_manifest": benchmark_evidence.artifact_binding(
                base["image_manifest"], kind="file"
            ),
            "raw_results": benchmark_evidence.artifact_binding(raw_results, kind="file"),
        },
        "implementations": {
            key: benchmark_evidence.artifact_binding(path, kind="file")
            for key, path in shared_implementations.items()
        },
        "protocol": benchmark_evidence.frozen_protocol(suite),
    }
    _write_json(run_manifest, run_manifest_payload)
    descriptor = tmp_path / f"descriptor-{run_index}-{suite}.json"
    benchmark_evidence.build_descriptor(
        suite=suite,
        checkpoint_tree=checkpoint,
        eval_model=eval_model,
        dataset=base["dataset"],
        external_image_manifest=base["image_manifest"],
        raw_results=raw_results,
        eval_run_manifest=run_manifest,
        evaluator=shared_implementations["evaluator"],
        validator=shared_implementations["validator"],
        prompt=shared_implementations["prompt"],
        requirements=shared_implementations["requirements"],
        output=descriptor,
    )
    return descriptor


def materialize_formal(tmp_path: Path, value):
    tmp_path.mkdir(parents=True, exist_ok=True)
    value["stage"] = "final"
    value["run_class"] = "formal"
    value["preregistered"]["optimizer_steps"] = gate.FORMAL_OPTIMIZER_STEPS
    ab_run_id = "v37-ab-" + "1" * 32
    ab_plan_path = tmp_path / "ab_plan.json"
    ab_cells = []
    for seed in (11, 22):
        for arm in ("baseline", "progress"):
            cell_id = f"seed{seed}-{arm}"
            ab_cells.append({
                "cell_id": cell_id,
                "seed": seed,
                "arm": arm,
                "target_dir": str((tmp_path / cell_id).absolute()),
            })
    ab_plan = {
        "schema_version": 2,
        "ab_run_id": ab_run_id,
        "plan_path": str(ab_plan_path.absolute()),
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
        "cells": ab_cells,
    }
    benchmark_root = tmp_path / "benchmark-shared"
    benchmark_root.mkdir()
    shared_implementations = {
        "evaluator": benchmark_root / "evaluator.py",
        "validator": benchmark_root / "validator.py",
        "prompt": benchmark_root / "prompt.txt",
        "requirements": benchmark_root / "requirements.txt",
    }
    shared_implementations["evaluator"].write_text("def evaluate():\n    return None\n")
    shared_implementations["validator"].write_text("def validate():\n    return None\n")
    shared_implementations["prompt"].write_text("BOUND SYSTEM PROMPT\n")
    shared_implementations["requirements"].write_text("transformers==0\n")
    benchmark_bases = {
        suite: _make_benchmark_base(benchmark_root, suite)
        for suite in ("pixmo-test", "stepcount-500")
    }
    model = tmp_path / "initial-model"
    model.mkdir()
    (model / "model.bin").write_bytes(b"frozen-initial-model")
    training = tmp_path / "frontier_rl.parquet"
    training.write_bytes(b"formal training fixture")
    validation = tmp_path / "heldout.parquet"
    validation.write_bytes(b"formal validation fixture")
    forbidden = tmp_path / "forbidden.parquet"
    forbidden.write_bytes(b"formal forbidden fixture")
    metadata = tmp_path / "metadata.json"
    metadata.write_text("[]")
    frontier_manifest = tmp_path / "selection_manifest.json"
    frontier_manifest.write_text('{"fixture":"frontier"}')
    coverage_report = tmp_path / "coverage.json"
    coverage_report.write_text('{"fixture":"coverage"}')
    filtered_manifest = tmp_path / "filtered.json"
    filtered_manifest.write_text('{"fixture":"filtered"}')
    masks = tmp_path / "masks"
    masks.mkdir()
    (masks / "mask.bin").write_bytes(b"formal-mask")
    model_snapshot = gate.content_snapshot(model)
    # Formal unit fixtures use a tiny synthetic model; production code keeps a
    # literal checkpoint-476 identity, asserted separately below.
    gate.FORMAL_INITIAL_MODEL_SHA256 = model_snapshot["sha256"]
    training_snapshot = gate.content_snapshot(training, parquet_only=True)
    validation_snapshot = gate.content_snapshot(validation, parquet_only=True)
    universe_rows = [
        {
            "sample_id": row["sample_id"],
            "prompt_id": row["prompt_id"],
            "bucket": gate.BUCKETS[bucket_index],
            "ground_truth": gate.BUCKET_RANGES[gate.BUCKETS[bucket_index]][0],
            "source_row_sha256": row["source_row_sha256"],
        }
        for bucket_index in range(len(gate.BUCKETS))
        for row in formal_samples("baseline")[bucket_index * 2:(bucket_index + 1) * 2]
    ]
    universe_rows.sort(key=lambda row: (row["sample_id"], row["prompt_id"]))
    _SYNTHETIC_PAIRED_SOURCE_ROWS[str(validation.resolve())] = copy.deepcopy(universe_rows)
    paired_universe = tmp_path / "paired-sample-universe.json"
    _write_json(paired_universe, {
        "schema_version": gate.PAIRED_UNIVERSE_SCHEMA_VERSION,
        "artifact_type": gate.PAIRED_UNIVERSE_ARTIFACT_TYPE,
        "data_path": str(validation.resolve()),
        "data_sha256": validation_snapshot["sha256"],
        "sample_count": len(universe_rows),
        "bucket_counts": {bucket: 2 for bucket in gate.BUCKETS},
        "ordered_rows_sha256": hashlib.sha256(
            gate.canonical(universe_rows).encode("utf-8")
        ).hexdigest(),
        "rows": universe_rows,
    })
    metadata_hash = gate.sha256(metadata)
    mask_hash = gate.tree_sha256(masks)
    implementation = {
        name: gate.sha256(Path(gate.__file__).resolve().parents[1] / relative)
        for name, relative in gate.IMPLEMENTATION_PATHS.items()
    }
    repo = Path(gate.__file__).resolve().parents[1]
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True,
    ).stdout.decode("ascii").strip()
    git_status_sha256 = gate.EMPTY_SHA256
    git_diff_sha256 = gate.EMPTY_SHA256
    formal_eval_program = tmp_path / "formal-eval-program"
    formal_eval_program.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    formal_eval_program.chmod(0o750)
    formal_eval_recipe = tmp_path / "formal-eval-recipe.json"
    _write_json(formal_eval_recipe, {
        "schema_version": 1,
        "recipe_type": gate.eval_producer.RECIPE_TYPE,
        "environment": {
            "PATH": "/usr/bin:/bin",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            **gate.eval_producer.REQUIRED_ENVIRONMENT,
        },
        "steps": {
            name: {
                "arms": gate.eval_producer.STEP_ARMS[name],
                "program": {
                    "path": str(formal_eval_program.resolve()),
                    "sha256": gate.sha256(formal_eval_program),
                },
                "argv": [
                    "{manifest}", "{checkpoint}", "{eval_model}", "{output}",
                    "{invocation_nonce}",
                ],
                "output": gate.eval_producer.STEP_OUTPUTS[name],
            }
            for name in gate.eval_producer.STEP_NAMES
        },
    })
    source_text = (Path(gate.__file__).resolve().parents[1] / gate.IMPLEMENTATION_PATHS["source_config"]).read_text()
    runtime_text, count = re.subn(
        r"(?m)^(\s*adaptive_actor_kl:)\s*(?:false|true)\s*$", r"\1 true", source_text,
    )
    assert count == 1
    ab_plan["preregistered"] = {
        "schema_version": 1,
        "complete": True,
        "missing": [],
        "input_snapshots": {
            "model": model_snapshot,
            "training_data": training_snapshot,
            "validation_data": [{"suite": "heldout", "dataset": validation_snapshot}],
            "mask_metadata": {"path": str(metadata.resolve()), "sha256": metadata_hash},
            "mask_tree": {"path": str(masks.resolve()), "sha256": mask_hash},
            "frontier_manifest": {
                "path": str(frontier_manifest.resolve()), "sha256": gate.sha256(frontier_manifest),
            },
            "coverage_report": {
                "path": str(coverage_report.resolve()), "sha256": gate.sha256(coverage_report),
            },
            "filtered_manifest": {
                "path": str(filtered_manifest.resolve()), "sha256": gate.sha256(filtered_manifest),
            },
        },
        "implementation_sha256": implementation,
        "image_path_remap": None,
        "eval_contract": {
            "paired_metrics": copy.deepcopy(gate.FROZEN_PAIRED_METRICS),
            "benchmark_dataset_sha256": copy.deepcopy(gate.FORMAL_BENCHMARK_DATASET_SHA256),
            "benchmark_ordered_ids_sha256": copy.deepcopy(gate.FORMAL_BENCHMARK_ORDERED_IDS_SHA256),
            "benchmark_thresholds": {
                "pixmo_correct_min": gate.PIXMO_CORRECT_MIN,
                "pixmo_total": gate.PIXMO_TOTAL,
                "stepcount_correct_min": gate.STEPCOUNT_CORRECT_MIN,
                "stepcount_total": gate.STEPCOUNT_TOTAL,
            },
            "pipeline": {
                "producer": {
                    "path": str(
                        (Path(gate.__file__).resolve().parents[1]
                         / gate.IMPLEMENTATION_PATHS["eval_producer"]).resolve()
                    ),
                    "sha256": implementation["eval_producer"],
                },
                "recipe": {
                    "path": str(formal_eval_recipe.resolve()),
                    "sha256": gate.sha256(formal_eval_recipe),
                },
            },
            "paired_sample_universe": {
                "path": str(paired_universe.resolve()),
                "sha256": gate.sha256(paired_universe),
            },
        },
    }
    _write_json(ab_plan_path, ab_plan)
    ab_plan_hash = gate.sha256(ab_plan_path)
    for index, run in enumerate(value["runs"]):
        cell_id = f"seed{run['seed']}-{run['arm']}"
        run_dir = tmp_path / cell_id
        run_dir.mkdir()
        runtime_config = run_dir / "v37_config.yaml"
        runtime_config.write_text(runtime_text)
        manifest_output_path = run_dir / "v37_run_manifest.json"
        effective_environment_path = run_dir / "v37_effective_environment.json"
        training_evidence_path = run_dir / "v37_training_evidence.json"
        environment = run["manifest"]["audited_environment"]
        run["manifest"]["config"]["pilot_steps"] = gate.FORMAL_OPTIMIZER_STEPS
        environment.update({
            "CONFIG_PATH": str(runtime_config.resolve()),
            "MODEL_PATH": str(model.resolve()),
            "STEPCOUNT_TRAIN_DATA": str(training.resolve()),
            "STEPCOUNT_VAL_DATA": f"heldout::{validation.resolve()}",
            "STEPCOUNT_MASKS_METADATA": str(metadata.resolve()),
            "STEPCOUNT_MASKS_DIR": str(masks.resolve()),
            "V31_SAVE_CHECKPOINT_PATH": str(run_dir.resolve()),
            "V31_NNODES": "1",
            "V31_N_GPUS_PER_NODE": "8",
            "HOST_NUM": "1",
            "HOST_GPU_NUM": "8",
            "INDEX": "0",
            "V37_RUN_MANIFEST": str(manifest_output_path.resolve()),
            "V37_EFFECTIVE_ENVIRONMENT_PATH": str(effective_environment_path.resolve()),
            "V37_TRAINING_EVIDENCE_PATH": str(training_evidence_path.resolve()),
            "V37_TRAINING_EVIDENCE_REQUIRED": "1",
            "V37_AB_RUN_ID": ab_run_id,
            "V37_AB_PLAN_PATH": str(ab_plan_path.resolve()),
            "V37_AB_PLAN_SHA256": ab_plan_hash,
            "V37_AB_CELL_ID": cell_id,
            "V37_RUN_CLASS": "formal",
            "V37_DATA_MODE": "frontier_rl",
            "V37_RUN_PURPOSE": "formal_ab",
            "V37_PILOT_STEPS": "12",
            "V36_MAX_STEPS": "12",
            "BOK_TOTAL_STEPS": "12",
            "V37_CONTINUATION_MODE": "0",
            "V37_EVIDENCE_START_STEP": "0",
            "V37_FINALIZE_HF_CHECKPOINT": "1",
            "V37_EXPECTED_INITIAL_MODEL_SHA256": model_snapshot["sha256"],
            "V37_HF_MERGE_HOST_MEMORY_PREFLIGHT": "error",
            "V37_HF_MERGE_HOST_MEMORY_SAFETY_FACTOR": "4.0",
            "V37_REQUIRE_EXACT_RAY_GPUS": "1",
            "RAY_GPU_WAIT_TIMEOUT_SECONDS": "900",
            "RAY_STATUS_TIMEOUT_SECONDS": "10",
            "RAY_START_TIMEOUT_SECONDS": "60",
            "RAY_PLACEMENT_GROUP_TIMEOUT_SECONDS": "900",
            "INTERLEAVED_PROCESS_PROMPT_FILE": str(
                (Path(gate.__file__).resolve().parents[1]
                 / gate.IMPLEMENTATION_PATHS["process_prompt"]).resolve()
            ),
            "INTERLEAVED_PROCESS_PROMPT_SHA256": implementation["process_prompt"],
            "STEPCOUNT_HARDWARE_PROFILE": "h200",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_MODE": "offline",
            "ADAPTIVE_ACTOR_KL": "true",
            "USE_KL_LOSS": "false",
            "KL_TYPE": "adaptive",
            "KL_COEF": "0.08",
            "KL_TARGET": "0.15",
            "KL_HORIZON": "10000",
            "KL_PENALTY": "low_var_kl",
            "BOK_CORRECTNESS_FIRST": "1",
            "BOK_ALLWRONG_TERMINAL_ZERO": "1",
            "INTERLEAVED_ADAPTIVE_MAX_TURNS": "true",
            "INTERLEAVED_ADAPTIVE_MAX_TURNS_MARGIN": "2",
            "INTERLEAVED_MAX_TURNS": "53",
        })
        run["manifest"]["config"]["hf_merge"] = {
            "host_memory_preflight": "error",
            "host_memory_safety_factor": "4.0",
        }
        execution_environment_path = run_dir / "v37_execution_environment.json"
        execution_hash = hashlib.sha256(json.dumps(
            environment, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest()
        environment.update({
            "V37_EXECUTION_ENVIRONMENT_PATH": str(execution_environment_path.resolve()),
            "V37_EXECUTION_ENVIRONMENT_SHA256": execution_hash,
        })
        execution_environment_path.write_text(json.dumps({
            "schema_version": 1,
            "hash_contract": "canonical_environment_without_self_reference_v1",
            "environment_sha256": execution_hash,
            "environment": environment,
        }))
        effective_environment = dict(environment)
        effective_environment["RAY_ADDRESS"] = "auto"
        effective_canonical = json.dumps(
            effective_environment, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()
        effective_environment_path.write_text(json.dumps({
            "schema_version": 1,
            "contract": "v37_effective_pre_trainer_environment_v1",
            "environment_sha256": hashlib.sha256(effective_canonical).hexdigest(),
            "environment": effective_environment,
        }))
        run["manifest"]["continuation_config_sha256"] = (
            gate.continuation.continuation_config_sha256(run["manifest"]["config"])
        )
        run["manifest"]["config_sha256"] = config_sha(run["manifest"]["config"], environment)
        run["manifest"]["run_class"] = "formal"
        run["manifest"].update({
            "ab_preregistration": {
                "schema_version": 1,
                "ab_run_id": ab_run_id,
                "plan_path": str(ab_plan_path.resolve()),
                "plan_sha256": ab_plan_hash,
                "cell_id": cell_id,
            },
            "promotable_candidate": True,
            "run_purpose": "formal_ab",
            "model_path": str(model.resolve()),
            "train_data": str(training.resolve()),
            "validation_data": f"heldout::{validation.resolve()}",
            "mask_metadata": str(metadata.resolve()),
            "masks_dir": str(masks.resolve()),
            "dataset_manifest": str(frontier_manifest.resolve()),
            "dataset_manifest_sha256": gate.sha256(frontier_manifest),
            "metadata_coverage_report": str(coverage_report.resolve()),
            "metadata_coverage_report_sha256": gate.sha256(coverage_report),
            "filtered_manifest": str(filtered_manifest.resolve()),
            "filtered_manifest_sha256": gate.sha256(filtered_manifest),
            "dataset_sha256": training_snapshot["sha256"],
            "metadata_sha256": metadata_hash,
            "mask_tree_path": str(masks.resolve()),
            "mask_tree_sha256": mask_hash,
            "input_hashes": {
                "train": training_snapshot["sha256"],
                "validation_0": validation_snapshot["sha256"],
            },
            "input_snapshots": {
                "model": model_snapshot,
                "training_data": training_snapshot,
                "validation_data": [{"suite": "heldout", "dataset": validation_snapshot}],
                "mask_metadata": {"path": str(metadata.resolve()), "sha256": metadata_hash},
            },
            "implementation_sha256": implementation,
            "git_commit": git_commit,
            "git_status_sha256": git_status_sha256,
            "git_diff_sha256": git_diff_sha256,
            "config_path": str(runtime_config.resolve()),
            "config_file_sha256": gate.sha256(runtime_config),
            "source_config_sha256": implementation["source_config"],
            "reward_function": str(
                (Path(gate.__file__).resolve().parents[1] / gate.IMPLEMENTATION_PATHS["reward_function"]).resolve()
            ) + ":compute_score",
            "reward_function_sha256": implementation["reward_function"],
        })
        paired_eval_data_sha256 = hashlib.sha256(json.dumps(
            [{"suite": "heldout", "sha256": validation_snapshot["sha256"]}],
            sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest()
        run["manifest"]["paired_eval_data_sha256"] = paired_eval_data_sha256
        run["eval"]["data_sha256"] = paired_eval_data_sha256
        nonce = hashlib.sha256(f"{cell_id}:formal-eval".encode()).hexdigest()
        raw_sample_rows = formal_samples(run["arm"])
        run["eval"].pop("protocol_sha256", None)
        run["eval"].pop("eval_config_sha256", None)
        run["eval"].update({
            "schema_version": gate.FORMAL_PAIRED_SCHEMA_VERSION,
            "artifact_type": gate.FORMAL_PAIRED_ARTIFACT_TYPE,
            "producer_invocation_nonce": nonce,
            "sample_universe_sha256": gate.sha256(paired_universe),
            "samples": raw_sample_rows,
            "sample_set_sha256": gate._sample_set_sha256({
                (row["sample_id"], row["prompt_id"]): row for row in universe_rows
            }),
        })
        evidence_run_id = hashlib.sha256(json.dumps({
            "arm": run["manifest"]["arm"], "seed": run["manifest"]["seed"],
            "run_directory": str(Path(run["manifest"]["audited_environment"]["V37_RUN_MANIFEST"]).resolve().parent),
            "manifest_path": str(Path(run["manifest"]["audited_environment"]["V37_RUN_MANIFEST"]).resolve()),
            "config_sha256": run["manifest"]["config_sha256"],
            "runtime_config_sha256": run["manifest"]["config_file_sha256"],
            "source_config_sha256": run["manifest"]["source_config_sha256"],
            "runtime_environment": run["manifest"]["runtime_environment"],
            "execution_environment_sha256": execution_hash,
            "ab_preregistration": run["manifest"]["ab_preregistration"],
            "resume_evidence": run["manifest"].get("resume_evidence"),
            "git_commit": run["manifest"].get("git_commit"),
            "git_status_sha256": run["manifest"].get("git_status_sha256"),
            "git_diff_sha256": run["manifest"].get("git_diff_sha256"),
            "implementation_sha256": run["manifest"].get("implementation_sha256"),
            "initial_model_sha256": model_snapshot["sha256"],
            "dataset_sha256": training_snapshot["sha256"],
        }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        run["manifest"]["evidence_run_id"] = evidence_run_id
        run["manifest"].update({
            "execution_environment_schema": "v37_explicit_environment_v1",
            "execution_environment_path": str(execution_environment_path.resolve()),
            "execution_environment_sha256": execution_hash,
            "effective_environment_path": str(effective_environment_path.resolve()),
            "effective_environment_sha256": gate.sha256(effective_environment_path),
        })
        run["log"]["evidence_run_id"] = evidence_run_id
        run["log"]["execution_environment_sha256"] = execution_hash
        run["eval"]["evidence_run_id"] = evidence_run_id
        run["eval"]["execution_environment_sha256"] = execution_hash
        loader_observation = {
            "loader_contract": "verl.utils.dataset.RLHFDataset:v1",
            "loader_config": {"filter_overlong_num_proc": 64},
            "sample_count": 10,
        }
        external_image_snapshots, external_image_errors = preflight_module._external_image_snapshot_from_paths({
            "train": set(), "validation": set(), "forbidden": set(),
        })
        assert external_image_errors == []
        formal_contract = {
            "dataset_sha256": gate.tree_sha256(training),
            "metadata_sha256": metadata_hash,
            "mask_tree_sha256": mask_hash,
            "loader_observation": loader_observation,
            "external_image_snapshots": external_image_snapshots,
            "errors": [],
        }
        preflight = {
            "ok": True, "run_class": "formal", "data_mode": "frontier_rl",
            "errors": [], "sample_count": 10, "formal_contract": formal_contract,
            "data_path": str(training.resolve()),
            "validation_paths": [str(validation.resolve())],
            "forbidden_paths": [str(forbidden.resolve())],
            "formal_input_snapshots": {
                "data": gate.tree_sha256(training), "metadata": metadata_hash,
                "masks": mask_hash, "validation": [gate.tree_sha256(validation)],
                "forbidden": [gate.tree_sha256(forbidden)],
            },
        }
        preflight_path = run_dir / "v37_preflight.json"
        preflight_path.write_text(json.dumps(preflight))
        run["manifest"].update({
            "preflight_report": str(preflight_path),
            "preflight_report_sha256": gate.sha256(preflight_path),
            "preflight_summary": {
                "formal_contract": formal_contract,
                "formal_input_snapshots": preflight["formal_input_snapshots"],
            },
        })
        checkpoint = run_dir / "global_step_12"
        checkpoint.mkdir()
        (checkpoint / "state.bin").write_bytes(f"state:{run['arm']}:{run['seed']}".encode())
        eval_model = checkpoint / "actor" / "huggingface"
        eval_model.mkdir(parents=True)
        (eval_model / "model.safetensors").write_bytes(
            f"merged:{run['arm']}:{run['seed']}".encode("utf-8")
        )
        checkpoint_hash = gate.tree_sha256(checkpoint)
        checkpoint_id = f"{run['arm']}-seed{run['seed']}-step12"
        run["manifest"].update({
            "final_checkpoint_id": checkpoint_id,
            "final_checkpoint_path": str(checkpoint.resolve()),
            "final_checkpoint_sha256": checkpoint_hash,
            "training_evidence_path": str(training_evidence_path.resolve()),
            "training_evidence_sha256": None,
        })
        step_records = []
        for step in range(1, 13):
            step_records.append({
                "global_step": step,
                "optimizer_microsteps_attempted": 1,
                "optimizer_microsteps_executed": 1,
                "optimizer_microsteps_skipped": 0,
                "nonfinite_count_cumulative": 0,
                "oom_count_cumulative": 0,
                "sample_count": 10,
                "progress_degraded_count": 0,
                "cap_turn_exceeded_count": 0,
                "early_stop_count": 0,
                "selector_kl_contribution": 0.0,
                "action_rows_expected": 10 if run["arm"] == "progress" else 0,
                "action_rows_mapped": 10 if run["arm"] == "progress" else 0,
                "mask_rows_expected": 10,
                "mask_rows_scored": 10,
                "frontier_group_count": 10,
                "frontier_mixed_group_count": 8,
                "outcome_allwrong_group_count": 2,
                "correctness_sign_error_count": 0,
            })
        run["log"] = {
            "schema_version": training_evidence.SCHEMA_VERSION,
            "artifact_type": training_evidence.ARTIFACT_TYPE,
            "evidence_run_id": evidence_run_id,
            "execution_environment_sha256": execution_hash,
            "completed": True,
            "start_global_step": 0,
            "final_global_step": 12,
            "expected_optimizer_steps": 12,
            "final_checkpoint_id": checkpoint_id,
            "final_checkpoint_path": str(checkpoint.resolve()),
            "final_checkpoint_sha256": checkpoint_hash,
            "step_records": step_records,
            "records_sha256": training_evidence._sha(step_records),
            **training_evidence._summarize(step_records, True),
        }
        training_evidence.verify_payload(run["log"])
        run["eval"].update({
            "final_checkpoint_id": checkpoint_id,
            "final_checkpoint_path": str(checkpoint.resolve()),
            "final_checkpoint_sha256": checkpoint_hash,
        })
        if run["arm"] == "progress":
            for suite_key, suite_name, correct in (
                ("pixmo", "pixmo-test", 440),
                ("stepcount", "stepcount-500", 90),
            ):
                path = _materialize_benchmark_evidence(
                    tmp_path,
                    benchmark_bases[suite_name],
                    checkpoint,
                    index,
                    correct,
                    shared_implementations,
                )
                run[f"{suite_key}_eval"] = str(path)
                run[f"{suite_key}_eval_sha256"] = gate.sha256(path)
        training_evidence_path.write_text(json.dumps(run["log"]))
        run["manifest"]["training_evidence_sha256"] = gate.sha256(training_evidence_path)
        eval_path = tmp_path / f"{index}-eval.json"
        eval_path.write_text(json.dumps(run["eval"]))
        manifest_output_path.write_text(json.dumps(run["manifest"]))
        for key, path in (
            ("manifest", manifest_output_path),
            ("log", training_evidence_path),
            ("eval", eval_path),
        ):
            run[key] = str(path)
            run[f"{key}_sha256"] = gate.sha256(path)
        recipe_environment = {
            key: gate.eval_producer._render(value, {
                "cell_id": cell_id,
                "arm": run["arm"],
                "seed": str(run["seed"]),
                "invocation_nonce": nonce,
                "manifest": str(manifest_output_path.resolve()),
                "checkpoint": str(checkpoint.resolve()),
                "eval_model": str(eval_model.resolve()),
                "output": "",
                "output_dir": "",
            }, f"fixture environment {key}")
            for key, value in gate.eval_producer.verify_recipe(formal_eval_recipe)["environment"].items()
        }
        output_hashes = {"paired_eval": run["eval_sha256"]}
        if run["arm"] == "progress":
            output_hashes.update({
                "pixmo_benchmark": run["pixmo_eval_sha256"],
                "stepcount_benchmark": run["stepcount_eval_sha256"],
            })
        receipt_steps = []
        for step_name in gate.eval_producer.STEP_NAMES:
            if run["arm"] not in gate.eval_producer.STEP_ARMS[step_name]:
                continue
            receipt_staging = run_dir / ".v37_postprocess.fixture"
            receipt_step_dir = receipt_staging / f".{step_name}.fixture"
            receipt_output = receipt_step_dir / gate.eval_producer.STEP_OUTPUTS[step_name]
            receipt_argv = [
                str(formal_eval_program.resolve()),
                str(manifest_output_path.resolve()),
                str(checkpoint.resolve()),
                str(eval_model.resolve()),
                str(receipt_output),
                nonce,
            ]
            receipt_steps.append({
                "name": step_name,
                "program_path": str(formal_eval_program.resolve()),
                "program_sha256": gate.sha256(formal_eval_program),
                "argv": receipt_argv,
                "argv_sha256": hashlib.sha256(
                    gate.eval_producer._canonical_bytes(receipt_argv)
                ).hexdigest(),
                "output_path": str(receipt_output),
                "output_dir": str(receipt_step_dir),
                "output_name": gate.eval_producer.STEP_OUTPUTS[step_name],
                "output_sha256": output_hashes[step_name],
            })
        receipt_path = tmp_path / f"{index}-producer-receipt.json"
        receipt_producer_path = str(
            (Path(gate.__file__).resolve().parents[1]
             / gate.IMPLEMENTATION_PATHS["eval_producer"]).resolve()
        )
        receipt_eval_model_hash = gate.tree_sha256(eval_model)
        invocation_request_hash = gate.eval_producer.invocation_request_sha256(
            cell_id=cell_id, arm=run["arm"], seed=run["seed"],
            invocation_nonce=nonce,
            manifest_path=str(manifest_output_path.resolve()),
            manifest_sha256=run["manifest_sha256"],
            checkpoint_path=str(checkpoint.resolve()), checkpoint_sha256=checkpoint_hash,
            eval_model_path=str(eval_model.resolve()), eval_model_sha256=receipt_eval_model_hash,
            recipe_path=str(formal_eval_recipe.resolve()),
            recipe_sha256=gate.sha256(formal_eval_recipe),
            producer_path=receipt_producer_path,
            producer_sha256=implementation["eval_producer"],
        )
        _write_json(receipt_path, {
            "schema_version": gate.eval_producer.RECEIPT_SCHEMA_VERSION,
            "artifact_type": gate.eval_producer.RECEIPT_TYPE,
            "cell_id": cell_id,
            "arm": run["arm"],
            "seed": run["seed"],
            "invocation_nonce": nonce,
            "invocation_request_sha256": invocation_request_hash,
            "manifest": {"path": str(manifest_output_path.resolve()), "sha256": run["manifest_sha256"]},
            "checkpoint": {"path": str(checkpoint.resolve()), "sha256": checkpoint_hash},
            "eval_model": {
                "path": str(eval_model.resolve()),
                "sha256": receipt_eval_model_hash,
            },
            "recipe": {
                "path": str(formal_eval_recipe.resolve()),
                "sha256": gate.sha256(formal_eval_recipe),
            },
            "producer": {
                "path": receipt_producer_path,
                "sha256": implementation["eval_producer"],
            },
            "environment_sha256": hashlib.sha256(json.dumps(
                recipe_environment, sort_keys=True, separators=(",", ":"), allow_nan=False,
            ).encode()).hexdigest(),
            "steps": receipt_steps,
        })
        run["eval_receipt"] = str(receipt_path)
        run["eval_receipt_sha256"] = gate.sha256(receipt_path)
    return value


def artifact_payload(run, key):
    value = run[key]
    return json.loads(Path(value).read_text()) if isinstance(value, str) else value


def write_artifact(run, key, payload):
    value = run[key]
    if isinstance(value, str):
        path = Path(value)
        path.write_text(json.dumps(payload))
        run[f"{key}_sha256"] = gate.sha256(path)
    else:
        run[key] = payload


def reseal_paired_eval(run, evaluation):
    write_artifact(run, "eval", evaluation)
    receipt = artifact_payload(run, "eval_receipt")
    paired = next(step for step in receipt["steps"] if step["name"] == "paired_eval")
    paired["output_sha256"] = run["eval_sha256"]
    write_artifact(run, "eval_receipt", receipt)


def reseal_run(run):
    manifest = artifact_payload(run, "manifest")
    log = artifact_payload(run, "log")
    evaluation = artifact_payload(run, "eval")
    manifest["config_sha256"] = config_sha(manifest["config"], manifest["audited_environment"])
    if "continuation_config_sha256" in manifest:
        manifest["continuation_config_sha256"] = gate.continuation.continuation_config_sha256(
            manifest["config"]
        )
    identity_payload = {
        "arm": manifest["arm"], "seed": manifest["seed"],
        "run_directory": str(Path(manifest["audited_environment"]["V37_RUN_MANIFEST"]).resolve().parent),
        "manifest_path": str(Path(manifest["audited_environment"]["V37_RUN_MANIFEST"]).resolve()),
        "config_sha256": manifest["config_sha256"],
        "runtime_config_sha256": manifest.get("config_file_sha256"),
        "source_config_sha256": manifest.get("source_config_sha256"),
        "runtime_environment": manifest.get("runtime_environment"),
        "execution_environment_sha256": manifest.get("execution_environment_sha256"),
        "ab_preregistration": manifest.get("ab_preregistration"),
        "resume_evidence": manifest.get("resume_evidence"),
        "git_commit": manifest.get("git_commit"),
        "git_status_sha256": manifest.get("git_status_sha256"),
        "git_diff_sha256": manifest.get("git_diff_sha256"),
        "implementation_sha256": manifest.get("implementation_sha256"),
        "initial_model_sha256": manifest["input_snapshots"]["model"]["sha256"],
        "dataset_sha256": manifest["dataset_sha256"],
    }
    evidence_run_id = hashlib.sha256(json.dumps(
        identity_payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    manifest["evidence_run_id"] = evidence_run_id
    log["evidence_run_id"] = evidence_run_id
    evaluation["evidence_run_id"] = evidence_run_id
    write_artifact(run, "log", log)
    write_artifact(run, "eval", evaluation)
    write_artifact(run, "manifest", manifest)


def reseal_effective_environment(run, mutate):
    manifest_path = Path(run["manifest"])
    manifest = json.loads(manifest_path.read_text())
    effective_path = Path(manifest["effective_environment_path"])
    payload = json.loads(effective_path.read_text())
    mutate(payload["environment"])
    encoded = json.dumps(
        payload["environment"], sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()
    payload["environment_sha256"] = hashlib.sha256(encoded).hexdigest()
    effective_path.write_text(json.dumps(payload))
    manifest["effective_environment_sha256"] = gate.sha256(effective_path)
    manifest_path.write_text(json.dumps(manifest))
    run["manifest_sha256"] = gate.sha256(manifest_path)


def test_canary_computes_two_seed_paired_metrics_without_final_target():
    result = gate.evaluate(valid_spec())
    assert result["decision"] == "GO"
    assert set(result["computed_paired_metrics"]) == {"11", "22"}


@pytest.mark.parametrize("optimizer_steps", [1, 5])
def test_canary_rejects_optimizer_steps_outside_frozen_range(optimizer_steps):
    value = valid_spec()
    value["preregistered"]["optimizer_steps"] = optimizer_steps
    assert gate.evaluate(value)["decision"] == "NO-GO"


def test_canary_binds_manifest_and_environment_step_counts():
    value = valid_spec()
    value["runs"][0]["manifest"]["config"]["pilot_steps"] = 3
    assert gate.evaluate(value)["decision"] == "NO-GO"

    value = valid_spec()
    value["runs"][0]["manifest"]["audited_environment"]["BOK_TOTAL_STEPS"] = "3"
    assert gate.evaluate(value)["decision"] == "NO-GO"


def test_debug_is_never_promotable():
    value = valid_spec()
    value["run_class"] = "debug"
    for run in value["runs"]:
        run["manifest"]["run_class"] = "debug"
    assert gate.evaluate(value)["decision"] == "NO-GO"


def test_hard_failure_cannot_be_acked():
    value = valid_spec()
    value["ack"] = True
    value["runs"][0]["log"]["oom_count"] = 1
    result = gate.evaluate(value)
    assert result["decision"] == "NO-GO" and result["ack_overrode_failure"] is False


def test_non_allowlisted_config_difference_fails():
    value = valid_spec()
    value["runs"][1]["manifest"]["config"]["lr"] = 9e-7
    assert gate.evaluate(value)["decision"] == "NO-GO"


def test_continuation_config_identity_tamper_fails():
    value = valid_spec()
    value["runs"][0]["manifest"]["continuation_config_sha256"] = "0" * 64
    assert gate.evaluate(value)["decision"] == "NO-GO"


def test_config_seal_environment_and_evidence_identity_fail_closed():
    value = valid_spec()
    manifest = value["runs"][0]["manifest"]
    manifest["audited_environment"]["REWARD_UNDECLARED_DIFFERENCE"] = "1"
    manifest["config_sha256"] = config_sha(manifest["config"], manifest["audited_environment"])
    assert gate.evaluate(value)["decision"] == "NO-GO"


def test_arm_and_seed_allowlisted_values_are_bound_before_normalization():
    value = valid_spec()
    progress = value["runs"][1]
    progress["manifest"]["config"]["process_reward"] = "0"
    progress["manifest"]["audited_environment"]["ACTION_EVENT_REWARD_ENABLE"] = "0"
    reseal_run(progress)
    assert gate.evaluate(value)["decision"] == "NO-GO"

    value = valid_spec()
    baseline = value["runs"][0]
    baseline["manifest"]["config"]["data_seed"] = 999
    reseal_run(baseline)
    assert gate.evaluate(value)["decision"] == "NO-GO"

    value = valid_spec()
    value["runs"][0]["log"]["evidence_run_id"] = "f" * 64
    assert gate.evaluate(value)["decision"] == "NO-GO"

    value = valid_spec()
    manifest = value["runs"][0]["manifest"]
    for key in ("PYTHONHASHSEED", "V31_DATA_SEED", "V31_ROLLOUT_SEED", "V37_SEED"):
        manifest["audited_environment"][key] = "999"
    manifest["config_sha256"] = config_sha(manifest["config"], manifest["audited_environment"])
    assert gate.evaluate(value)["decision"] == "NO-GO"


def test_requires_exactly_two_seeds():
    value = valid_spec()
    value["preregistered"]["seeds"] = [11]
    assert gate.evaluate(value)["decision"] == "NO-GO"

    value = valid_spec()
    value["preregistered"]["seeds"] = ["11", "22"]
    assert gate.evaluate(value)["decision"] == "NO-GO"


def test_typo_direction_is_rejected():
    value = valid_spec()
    value["preregistered"]["paired_metrics"]["answer_exact"] = {"direction": "increaze"}
    result = gate.evaluate(value)
    assert result["decision"] == "NO-GO" and "unknown direction" in result["errors"][0]


def test_fractional_optimizer_steps_and_malformed_config_are_structured_no_go():
    value = valid_spec()
    value["runs"][0]["log"]["optimizer_steps"] = 12.5
    assert gate.evaluate(value)["decision"] == "NO-GO"
    value = valid_spec()
    value["runs"][0]["manifest"]["config"] = ["not", "a", "mapping"]
    result = gate.evaluate(value)
    assert result["decision"] == "NO-GO" and result["errors"]


def test_numeric_strings_duplicate_ids_and_missing_safety_evidence_fail_closed():
    value = valid_spec()
    value["runs"][0]["log"]["optimizer_steps"] = "12"
    assert gate.evaluate(value)["decision"] == "NO-GO"
    value = valid_spec()
    value["runs"][0]["eval"]["samples"][1]["sample_id"] = value["runs"][0]["eval"]["samples"][0]["sample_id"]
    assert gate.evaluate(value)["decision"] == "NO-GO"
    value = valid_spec()
    del value["runs"][0]["log"]["correctness_sign_error_count"]
    assert gate.evaluate(value)["decision"] == "NO-GO"


def test_frozen_sample_hash_is_recomputed():
    value = valid_spec()
    value["runs"][0]["eval"]["sample_set_sha256"] = "f" * 64
    assert gate.evaluate(value)["decision"] == "NO-GO"


def test_arbitrary_self_reported_delta_is_not_trusted():
    value = valid_spec()
    progress = value["runs"][1]["eval"]
    progress["samples"] = copy.deepcopy(value["runs"][0]["eval"]["samples"])
    progress["paired_metrics"] = {name: {"delta": 999, "ci_low": 998, "ci_high": 1000} for name in gate.REQUIRED_PAIRED_METRICS}
    assert gate.evaluate(value)["decision"] == "NO-GO"


def test_sample_protocol_and_prompt_identity_must_match():
    value = valid_spec()
    value["runs"][1]["eval"]["samples"][0]["prompt_id"] = "leaked-replacement"
    assert gate.evaluate(value)["decision"] == "NO-GO"


def test_final_formal_requires_path_hash_and_same_proven_checkpoint(tmp_path):
    value = materialize_formal(tmp_path, valid_spec())
    assert gate.evaluate(value)["decision"] == "GO"


def test_formal_paired_metrics_are_recomputed_even_when_hashes_are_resealed(tmp_path):
    value = materialize_formal(tmp_path, valid_spec())
    run = value["runs"][0]
    evaluation = artifact_payload(run, "eval")
    evaluation["samples"][0]["metrics"]["answer_exact"] = 1.0
    reseal_paired_eval(run, evaluation)
    result = gate.evaluate(value)
    assert result["decision"] == "NO-GO"
    assert "cached metric answer_exact" in result["errors"][0]


def test_formal_recompute_rejects_turn_overrun_and_impossible_unique_hits():
    row = formal_samples("progress")[0]
    ground_truth = gate.BUCKET_RANGES[gate.BUCKETS[0]][0]
    row["effective_max_turns"] = len(row["transcript"]) - 1
    row["configured_max_turns"] = row["effective_max_turns"]
    with pytest.raises(gate.GateError, match="exceeds effective_max_turns"):
        gate._recompute_formal_sample(
            row, "turn-overrun", ground_truth=ground_truth, bucket=gate.BUCKETS[0],
        )

    row = formal_samples("progress")[0]
    extra_index = len(row["point_events"]) + 1
    payload = (
        '<think>inspect</think><point>{"point_2d":[1,2],"label":"object",'
        f'"count_number":{extra_index}}}</point>'
    )
    raw_payload = gate.POINT_BLOCK_CAPTURE_RE.findall(payload)[0]
    row["transcript"].insert(-1, payload)
    row["point_events"].append({
        "point_index": extra_index,
        "matched_target_id": "impossible-extra-target",
        "point_payload_sha256": hashlib.sha256(raw_payload.encode()).hexdigest(),
    })
    row["num_rounds"] += 1
    with pytest.raises(gate.GateError, match="unique matched targets exceed authoritative GT"):
        gate._recompute_formal_sample(
            row, "impossible-hit", ground_truth=ground_truth, bucket=gate.BUCKETS[0],
        )


@pytest.mark.parametrize(
    "replacement",
    ("<POINT>", '<point class="action">'),
)
def test_formal_tag_grammar_matches_strict_winner_literals(replacement):
    row = formal_samples("progress")[0]
    transcript = list(row["transcript"])
    transcript[0] = transcript[0].replace("<point>", replacement, 1)
    structurally_valid, _, _, _ = gate._strict_transcript_facts(
        transcript, row["point_events"], "strict-tag-parity",
    )
    assert structurally_valid is False


def test_formal_paired_eval_cannot_spoof_gt_source_or_sample_subset(tmp_path):
    gt_case = materialize_formal(tmp_path / "gt", valid_spec())
    gt_run = gt_case["runs"][0]
    gt_eval = artifact_payload(gt_run, "eval")
    gt_eval["samples"][0]["ground_truth"] = 2
    reseal_paired_eval(gt_run, gt_eval)
    result = gate.evaluate(gt_case)
    assert result["decision"] == "NO-GO"
    assert "schema mismatch" in result["errors"][0]

    source_case = materialize_formal(tmp_path / "source", valid_spec())
    source_run = source_case["runs"][0]
    source_eval = artifact_payload(source_run, "eval")
    source_eval["samples"][0]["source_row_sha256"] = "9" * 64
    reseal_paired_eval(source_run, source_eval)
    result = gate.evaluate(source_case)
    assert result["decision"] == "NO-GO"
    assert "source row binding mismatch" in result["errors"][0]

    subset_case = materialize_formal(tmp_path / "subset", valid_spec())
    subset_run = subset_case["runs"][0]
    subset_eval = artifact_payload(subset_run, "eval")
    subset_eval["samples"].pop()
    reseal_paired_eval(subset_run, subset_eval)
    result = gate.evaluate(subset_case)
    assert result["decision"] == "NO-GO"
    assert "exactly cover the sample universe" in result["errors"][0]


def test_formal_sample_universe_and_invocation_nonce_are_unique_and_immutable(tmp_path):
    universe_case = materialize_formal(tmp_path / "universe", valid_spec())
    first_manifest = artifact_payload(universe_case["runs"][0], "manifest")
    plan = json.loads(Path(first_manifest["ab_preregistration"]["plan_path"]).read_text())
    universe_path = Path(plan["preregistered"]["eval_contract"]["paired_sample_universe"]["path"])
    universe = json.loads(universe_path.read_text())
    universe["rows"][0]["ground_truth"] += 1
    universe_path.write_text(json.dumps(universe))
    result = gate.evaluate(universe_case)
    assert result["decision"] == "NO-GO"
    assert "sample universe changed" in result["errors"][0]

    nonce_case = materialize_formal(tmp_path / "nonce-reuse", valid_spec())
    first_run, second_run = nonce_case["runs"][:2]
    first_nonce = artifact_payload(first_run, "eval_receipt")["invocation_nonce"]
    second_eval = artifact_payload(second_run, "eval")
    second_eval["producer_invocation_nonce"] = first_nonce
    write_artifact(second_run, "eval", second_eval)
    second_receipt = artifact_payload(second_run, "eval_receipt")
    second_receipt["invocation_nonce"] = first_nonce
    paired = next(step for step in second_receipt["steps"] if step["name"] == "paired_eval")
    paired["output_sha256"] = second_run["eval_sha256"]
    for step in second_receipt["steps"]:
        step["argv"][-1] = first_nonce
        step["argv_sha256"] = hashlib.sha256(
            gate.eval_producer._canonical_bytes(step["argv"])
        ).hexdigest()
    second_receipt["invocation_request_sha256"] = gate.eval_producer.invocation_request_sha256(
        cell_id=second_receipt["cell_id"], arm=second_receipt["arm"],
        seed=second_receipt["seed"], invocation_nonce=first_nonce,
        manifest_path=second_receipt["manifest"]["path"],
        manifest_sha256=second_receipt["manifest"]["sha256"],
        checkpoint_path=second_receipt["checkpoint"]["path"],
        checkpoint_sha256=second_receipt["checkpoint"]["sha256"],
        eval_model_path=second_receipt["eval_model"]["path"],
        eval_model_sha256=second_receipt["eval_model"]["sha256"],
        recipe_path=second_receipt["recipe"]["path"],
        recipe_sha256=second_receipt["recipe"]["sha256"],
        producer_path=second_receipt["producer"]["path"],
        producer_sha256=second_receipt["producer"]["sha256"],
    )
    write_artifact(second_run, "eval_receipt", second_receipt)
    result = gate.evaluate(nonce_case)
    assert result["decision"] == "NO-GO"
    assert "nonce is reused" in result["errors"][0]


def test_formal_eval_receipt_cannot_rebind_program_or_nonce(tmp_path):
    value = materialize_formal(tmp_path / "program", valid_spec())
    run = value["runs"][0]
    receipt = artifact_payload(run, "eval_receipt")
    receipt["steps"][0]["program_sha256"] = "9" * 64
    write_artifact(run, "eval_receipt", receipt)
    result = gate.evaluate(value)
    assert result["decision"] == "NO-GO"
    assert "provenance mismatch" in result["errors"][0]

    nonce_case = materialize_formal(tmp_path / "nonce", valid_spec())
    nonce_run = nonce_case["runs"][0]
    nonce_receipt = artifact_payload(nonce_run, "eval_receipt")
    nonce_receipt["invocation_nonce"] = "8" * 64
    write_artifact(nonce_run, "eval_receipt", nonce_receipt)
    result = gate.evaluate(nonce_case)
    assert result["decision"] == "NO-GO"
    assert "identity/nonce mismatch" in result["errors"][0]


def test_formal_effective_environment_is_fail_closed(tmp_path):
    tampered = materialize_formal(tmp_path / "tampered", valid_spec())
    first_manifest = json.loads(Path(tampered["runs"][0]["manifest"]).read_text())
    Path(first_manifest["effective_environment_path"]).write_text("{}")
    assert gate.evaluate(tampered)["decision"] == "NO-GO"

    drifted = materialize_formal(tmp_path / "drifted", valid_spec())
    reseal_effective_environment(
        drifted["runs"][0], lambda environment: environment.__setitem__("RAY_ADDRESS", "local"),
    )
    assert gate.evaluate(drifted)["decision"] == "NO-GO"

    remapped = materialize_formal(tmp_path / "remapped", valid_spec())
    reseal_effective_environment(
        remapped["runs"][0],
        lambda environment: environment.__setitem__(
            "STEPCOUNT_IMAGE_PATH_REMAP_JSON", '{"/old":"/new"}',
        ),
    )
    assert gate.evaluate(remapped)["decision"] == "NO-GO"

    wrong_initial_model = materialize_formal(tmp_path / "wrong-model-id", valid_spec())
    reseal_effective_environment(
        wrong_initial_model["runs"][0],
        lambda environment: environment.__setitem__(
            "V37_EXPECTED_INITIAL_MODEL_SHA256", "f" * 64,
        ),
    )
    assert gate.evaluate(wrong_initial_model)["decision"] == "NO-GO"


@pytest.mark.parametrize(
    ("suite", "path"),
    [
        (
            "pixmo",
            "/mnt/shared-storage-user/zhangchenhao/work/StepcountModel/dataset/eval/eval_dataset.json",
        ),
        (
            "stepcount",
            "/mnt/shared-storage-user/zhangchenhao/work/StepcountModel/dataset/eval/eval_stepcount_bench_500.json",
        ),
    ],
)
def test_production_benchmark_dataset_and_order_are_frozen(suite, path):
    dataset = Path(path)
    if not dataset.is_file():
        pytest.skip(f"canonical benchmark is not mounted: {dataset}")
    assert gate.sha256(dataset) == PRODUCTION_BENCHMARK_DATASET_SHA256[suite]
    rows = json.loads(dataset.read_text(encoding="utf-8"))
    ordered_ids = [str(row["id"]) for row in rows]
    assert hashlib.sha256(gate.canonical(ordered_ids).encode("utf-8")).hexdigest() == (
        PRODUCTION_BENCHMARK_ORDERED_IDS_SHA256[suite]
    )


def test_formal_rejects_noncanonical_benchmark_identity(tmp_path):
    value = materialize_formal(tmp_path, valid_spec())
    gate.FORMAL_BENCHMARK_DATASET_SHA256["pixmo"] = "f" * 64
    assert gate.evaluate(value)["decision"] == "NO-GO"


def test_cannot_splice_benchmarks_across_checkpoints(tmp_path):
    value = materialize_formal(tmp_path, valid_spec())
    progress = next(run for run in value["runs"] if run["arm"] == "progress")
    step_path = Path(progress["stepcount_eval"])
    payload = json.loads(step_path.read_text())
    payload["artifacts"]["checkpoint_tree"]["path"] = str(
        (tmp_path / "seed11-baseline" / "global_step_12").resolve()
    )
    step_path.write_text(json.dumps(payload))
    progress["stepcount_eval_sha256"] = gate.sha256(step_path)
    assert gate.evaluate(value)["decision"] == "NO-GO"


def test_formal_ab_plan_and_recomputed_benchmark_evidence_fail_closed(tmp_path):
    value = materialize_formal(tmp_path / "plan", valid_spec())
    plan_path = tmp_path / "plan" / "ab_plan.json"
    plan = json.loads(plan_path.read_text())
    plan["execution"]["scheduling"] = "parallel"
    _write_json(plan_path, plan)
    assert gate.evaluate(value)["decision"] == "NO-GO"

    value = materialize_formal(tmp_path / "raw", valid_spec())
    progress = next(run for run in value["runs"] if run["arm"] == "progress")
    descriptor = json.loads(Path(progress["pixmo_eval"]).read_text())
    raw_results = Path(descriptor["artifacts"]["raw_results"]["path"])
    rows = json.loads(raw_results.read_text())
    rows[0]["is_correct"] = not rows[0]["is_correct"]
    _write_json(raw_results, rows)
    assert gate.evaluate(value)["decision"] == "NO-GO"

    value = materialize_formal(tmp_path / "legacy", valid_spec())
    progress = next(run for run in value["runs"] if run["arm"] == "progress")
    legacy_path = Path(progress["pixmo_eval"])
    _write_json(legacy_path, {
        "schema_version": 1,
        "suite": "pixmo",
        "checkpoint_id": "claimed",
        "checkpoint_path": "/claimed",
        "checkpoint_sha256": "a" * 64,
        "correct": 529,
        "total": 529,
    })
    progress["pixmo_eval_sha256"] = gate.sha256(legacy_path)
    assert gate.evaluate(value)["decision"] == "NO-GO"


def test_formal_eval_recipe_is_frozen_before_training(tmp_path):
    value = materialize_formal(tmp_path, valid_spec())
    plan_path = tmp_path / "ab_plan.json"
    plan = json.loads(plan_path.read_text())
    recipe = Path(plan["preregistered"]["eval_contract"]["pipeline"]["recipe"]["path"])
    recipe.write_text(recipe.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert gate.evaluate(value)["decision"] == "NO-GO"


def test_formal_initial_model_and_preflight_are_bound_to_real_artifacts(tmp_path):
    gate_source = Path(gate.__file__).read_text(encoding="utf-8")
    expected = "f9e6b1e8031bdbc509d34249745cdcf75af85c320d6b88918444b1abb4f580a3"
    assert f'FORMAL_INITIAL_MODEL_SHA256 = "{expected}"' in gate_source
    for relative in (
        "examples/v37_strict_winner_step_rl_pilot.sh",
        "examples/rl_launch/run_v37_strict_ab.sh",
    ):
        assert expected in (Path(gate.__file__).resolve().parents[1] / relative).read_text()

    value = materialize_formal(tmp_path / "model", valid_spec())
    (tmp_path / "model" / "initial-model" / "model.bin").write_bytes(b"changed-after-launch")
    assert gate.evaluate(value)["decision"] == "NO-GO"

    oom_value = materialize_formal(tmp_path / "recovered-oom", valid_spec())
    oom_run = oom_value["runs"][0]
    oom_log = artifact_payload(oom_run, "log")
    for record in oom_log["step_records"]:
        record["oom_count_cumulative"] = 1
    oom_log["records_sha256"] = training_evidence._sha(oom_log["step_records"])
    oom_log.update(training_evidence._summarize(oom_log["step_records"], True))
    write_artifact(oom_run, "log", oom_log)
    oom_manifest = artifact_payload(oom_run, "manifest")
    oom_manifest["training_evidence_sha256"] = oom_run["log_sha256"]
    write_artifact(oom_run, "manifest", oom_manifest)
    assert gate.evaluate(oom_value)["decision"] == "NO-GO"

    value = materialize_formal(tmp_path / "external-image", valid_spec())
    first = value["runs"][0]
    manifest_path = Path(first["manifest"])
    manifest = json.loads(manifest_path.read_text())
    preflight_path = Path(manifest["preflight_report"])
    preflight_report = json.loads(preflight_path.read_text())
    image = tmp_path / "external-image" / "frozen-image.png"
    image.write_bytes(b"before")
    external_snapshot, external_errors = preflight_module._external_image_snapshot_from_paths({
        "train": {image.resolve()}, "validation": set(), "forbidden": set(),
    })
    assert external_errors == []
    preflight_report["formal_contract"]["external_image_snapshots"] = external_snapshot
    preflight_path.write_text(json.dumps(preflight_report))
    manifest["preflight_report_sha256"] = gate.sha256(preflight_path)
    manifest["preflight_summary"]["formal_contract"] = preflight_report["formal_contract"]
    manifest_path.write_text(json.dumps(manifest))
    first["manifest_sha256"] = gate.sha256(manifest_path)
    image.write_bytes(b"after")
    assert gate.evaluate(value)["decision"] == "NO-GO"

    value = materialize_formal(tmp_path / "preflight", valid_spec())
    first = value["runs"][0]
    manifest_path = Path(first["manifest"])
    manifest = json.loads(manifest_path.read_text())
    preflight_path = Path(manifest["preflight_report"])
    preflight = json.loads(preflight_path.read_text())
    preflight["formal_contract"]["loader_observation"]["loader_config"]["filter_overlong_num_proc"] = 1
    preflight_path.write_text(json.dumps(preflight))
    manifest["preflight_report_sha256"] = gate.sha256(preflight_path)
    manifest["preflight_summary"]["formal_contract"] = preflight["formal_contract"]
    manifest_path.write_text(json.dumps(manifest))
    first["manifest_sha256"] = gate.sha256(manifest_path)
    assert gate.evaluate(value)["decision"] == "NO-GO"


def test_formal_preflight_eval_and_implementation_cannot_be_resealed_to_unrelated_inputs(tmp_path):
    value = materialize_formal(tmp_path / "snapshot", valid_spec())
    first = value["runs"][0]
    manifest_path = Path(first["manifest"])
    manifest = json.loads(manifest_path.read_text())
    preflight_path = Path(manifest["preflight_report"])
    preflight = json.loads(preflight_path.read_text())
    preflight["formal_input_snapshots"]["validation"] = ["f" * 64]
    preflight_path.write_text(json.dumps(preflight))
    manifest["preflight_report_sha256"] = gate.sha256(preflight_path)
    manifest_path.write_text(json.dumps(manifest))
    first["manifest_sha256"] = gate.sha256(manifest_path)
    assert gate.evaluate(value)["decision"] == "NO-GO"


def test_runtime_config_manifest_path_and_validation_contract_are_recomputed(tmp_path):
    value = materialize_formal(tmp_path / "runtime", valid_spec())
    first = value["runs"][0]
    manifest = artifact_payload(first, "manifest")
    runtime_config = Path(manifest["config_path"])
    runtime_config.write_text(runtime_config.read_text() + "\n# hidden semantic drift\n")
    manifest["config_file_sha256"] = gate.sha256(runtime_config)
    write_artifact(first, "manifest", manifest)
    reseal_run(first)
    assert gate.evaluate(value)["decision"] == "NO-GO"

    value = materialize_formal(tmp_path / "path", valid_spec())
    first = value["runs"][0]
    manifest = artifact_payload(first, "manifest")
    manifest["audited_environment"]["V37_RUN_MANIFEST"] = str(tmp_path / "fake" / "manifest.json")
    write_artifact(first, "manifest", manifest)
    reseal_run(first)
    assert gate.evaluate(value)["decision"] == "NO-GO"

    value = materialize_formal(tmp_path / "paired", valid_spec())
    for run in value["runs"]:
        manifest = artifact_payload(run, "manifest")
        evaluation = artifact_payload(run, "eval")
        manifest["paired_eval_data_sha256"] = "f" * 64
        evaluation["data_sha256"] = "f" * 64
        write_artifact(run, "eval", evaluation)
        write_artifact(run, "manifest", manifest)
        reseal_run(run)
    assert gate.evaluate(value)["decision"] == "NO-GO"


def test_nonfinite_json_and_nested_symlinks_fail_closed(tmp_path):
    try:
        gate._decode_json('{"value": NaN}')
    except gate.GateError:
        pass
    else:
        raise AssertionError("NaN must be rejected")

    value = valid_spec()
    value["unused_nonfinite"] = float("nan")
    assert gate.evaluate(value)["decision"] == "NO-GO"

    target = tmp_path / "target"
    target.mkdir()
    (target / "state.bin").write_bytes(b"state")
    root = tmp_path / "root"
    root.mkdir()
    (root / "linked-dir").symlink_to(target, target_is_directory=True)
    try:
        gate.tree_sha256(root)
    except gate.GateError:
        pass
    else:
        raise AssertionError("nested directory symlink must be rejected")

    value = materialize_formal(tmp_path / "artifact-link", valid_spec())
    first = value["runs"][0]
    real_eval = Path(first["eval"])
    linked_eval = real_eval.with_name("linked-eval.json")
    linked_eval.symlink_to(real_eval)
    first["eval"] = str(linked_eval)
    assert gate.evaluate(value)["decision"] == "NO-GO"

    value = materialize_formal(tmp_path / "eval", valid_spec())
    first = value["runs"][0]
    eval_path = Path(first["eval"])
    evaluation = json.loads(eval_path.read_text())
    evaluation["data_sha256"] = "f" * 64
    eval_path.write_text(json.dumps(evaluation))
    first["eval_sha256"] = gate.sha256(eval_path)
    assert gate.evaluate(value)["decision"] == "NO-GO"

    value = materialize_formal(tmp_path / "implementation", valid_spec())
    first = value["runs"][0]
    manifest_path = Path(first["manifest"])
    manifest = json.loads(manifest_path.read_text())

    gate._validate_implementation_snapshot(manifest, "fixture")

    dirty_git = copy.deepcopy(manifest)
    dirty_git["git_status_sha256"] = "b" * 64
    with pytest.raises(gate.GateError, match="clean committed worktree"):
        gate._validate_implementation_snapshot(dirty_git, "fixture")

    forged_git = copy.deepcopy(manifest)
    forged_git["git_commit"] = "f" * 40
    with pytest.raises(gate.GateError, match="live Git identity"):
        gate._validate_implementation_snapshot(forged_git, "fixture")

    forged_prompt_binding = copy.deepcopy(manifest)
    forged_prompt_binding["audited_environment"][
        "INTERLEAVED_PROCESS_PROMPT_SHA256"
    ] = "f" * 64
    with pytest.raises(gate.GateError, match="process prompt runtime binding"):
        gate._validate_implementation_snapshot(forged_prompt_binding, "fixture")

    manifest["implementation_sha256"]["torch_functional"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest))
    first["manifest_sha256"] = gate.sha256(manifest_path)
    assert gate.evaluate(value)["decision"] == "NO-GO"


def test_jsonl_safety_aggregation_is_monotonic(tmp_path):
    path = tmp_path / "log.jsonl"
    rows = [
        {"optimizer_steps": 1, "oom_count": 1, "event_mapping_coverage": .5},
        {"optimizer_steps": 12, "oom_count": 0, "event_mapping_coverage": 1},
    ]
    aggregated = gate._aggregate_log_rows(rows, path)
    assert aggregated["oom_count"] == 1
    assert aggregated["event_mapping_coverage"] == .5


def test_mask_tree_change_after_manifest_fails(tmp_path):
    mask_tree = tmp_path / "masks"
    mask_tree.mkdir()
    mask = mask_tree / "a.bin"
    mask.write_bytes(b"before")
    value = valid_spec()
    expected = gate.tree_sha256(mask_tree)
    for run in value["runs"]:
        run["manifest"]["mask_tree_path"] = str(mask_tree)
        run["manifest"]["mask_tree_sha256"] = expected
    mask.write_bytes(b"after")
    assert gate.evaluate(value)["decision"] == "NO-GO"


def test_formal_constants_and_frozen_metric_rules_cannot_be_relaxed(tmp_path):
    for field, unsafe in (
        ("optimizer_steps", 1), ("frontier_mixed_min", 0.0), ("outcome_allwrong_max", 1.0),
    ):
        case = materialize_formal(tmp_path / field, valid_spec())
        case["preregistered"][field] = unsafe
        assert gate.evaluate(case)["decision"] == "NO-GO"
    case = materialize_formal(tmp_path / "margin", valid_spec())
    case["preregistered"]["paired_metrics"]["answer_exact"] = {
        "direction": "noninferiority", "beneficial_direction": "increase", "margin": 999,
    }
    assert gate.evaluate(case)["decision"] == "NO-GO"


def test_previous_false_go_one_step_vacuous_totals_shared_evidence_is_no_go(tmp_path):
    value = materialize_formal(tmp_path, valid_spec())
    value["preregistered"].update({
        "optimizer_steps": 1, "frontier_mixed_min": 0.0, "outcome_allwrong_max": 1.0,
    })
    for name, rule in value["preregistered"]["paired_metrics"].items():
        rule.clear()
        rule.update({
            "direction": "noninferiority",
            "beneficial_direction": "increase" if name in gate.HIGHER_BETTER else "decrease",
            "margin": 999,
        })
    first = value["runs"][0]
    for run in value["runs"][1:]:
        run["log"] = first["log"]
        run["log_sha256"] = first["log_sha256"]
        run["eval"] = first["eval"]
        run["eval_sha256"] = first["eval_sha256"]
    for progress in (run for run in value["runs"] if run["arm"] == "progress"):
        for suite in ("pixmo", "stepcount"):
            path = Path(progress[f"{suite}_eval"])
            payload = json.loads(path.read_text())
            payload.update({"correct": 1, "total": 1})
            path.write_text(json.dumps(payload))
            progress[f"{suite}_eval_sha256"] = gate.sha256(path)
    assert gate.evaluate(value)["decision"] == "NO-GO"


def test_minimal_mechanism_config_and_reused_checkpoint_are_no_go(tmp_path):
    value = valid_spec()
    value["runs"][0]["manifest"]["mechanism_config"] = {"estimator": "bok_grpo"}
    assert gate.evaluate(value)["decision"] == "NO-GO"

    formal = materialize_formal(tmp_path, valid_spec())
    first, second = formal["runs"][:2]
    first_manifest = json.loads(Path(first["manifest"]).read_text())
    second_manifest_path = Path(second["manifest"])
    second_manifest = json.loads(second_manifest_path.read_text())
    for key in ("final_checkpoint_id", "final_checkpoint_path", "final_checkpoint_sha256"):
        second_manifest[key] = first_manifest[key]
    second_manifest_path.write_text(json.dumps(second_manifest))
    second["manifest_sha256"] = gate.sha256(second_manifest_path)
    assert gate.evaluate(formal)["decision"] == "NO-GO"
