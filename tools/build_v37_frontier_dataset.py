#!/usr/bin/env python3
"""Build a provenance-locked V37 frontier-RL dataset.

Each JSONL candidate row must identify the prompt, mining seed, candidate, and
generation evidence.  Canary/formal audits require SHA256 evidence for the
model checkpoint, generation and sampling configurations, and response bytes.
Formal builds are fixed to two seeds, M=32, and bucket ratios 40/10/20/20/10.
The output directory is published as one atomic, complete artifact set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import stat
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from verl.utils.path_remap import parse_image_path_remap, remap_image_path
except ModuleNotFoundError:  # Direct execution from tools/.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from verl.utils.path_remap import parse_image_path_remap, remap_image_path

BUCKETS = ((2, 10), (11, 20), (21, 30), (31, 40), (41, 50))
LABELS = tuple(f"{low}-{high}" for low, high in BUCKETS)
RATIOS = (0.40, 0.10, 0.20, 0.20, 0.10)
TERMINATIONS = {"answer", "cap", "eos", "error", "cancelled"}
SHA256_RE = re.compile(r"[0-9a-f]{64}")
CANDIDATE_ONLY_COLUMNS = {
    "response", "responses", "trajectory", "trajectories", "sampled_response",
    "candidate_id", "raw_success", "trajectory_quality", "process", "termination",
    "declared_seed", "response_sha256", "candidate_content_sha256",
}


class ValidationError(ValueError):
    """The requested build is not sufficiently proven or internally valid."""


@dataclass(frozen=True)
class Candidate:
    prompt_id: str
    seed: str
    declared_seed: str
    candidate_id: str
    answer_exact: bool
    raw_success: bool
    quality: float
    unique_hit: int
    duplicate: int
    miss: int
    cap: bool
    termination: str
    model_checkpoint_sha256: str | None
    generation_config_sha256: str | None
    sampling_config_sha256: str | None
    response_sha256: str | None
    generation_config: Mapping[str, Any] | None
    sampling_config: Mapping[str, Any] | None
    source_row_sha256: str | None
    source_image_sha256: tuple[str, ...] | None
    rendered_prompt_sha256: str | None
    rendered_prompt: str | None


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _loads_json(text: str, where: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValidationError(f"non-finite JSON constant: {value}")
            ),
        )
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValidationError(f"{where}: invalid JSON: {exc}") from exc


def _id(value: Any, name: str) -> str:
    if value is None or isinstance(value, (bool, list, dict)):
        raise ValidationError(f"{name} must be a scalar")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationError(f"{name} must be finite")
        if value.is_integer():
            value = int(value)
    result = str(value).strip()
    if not result:
        raise ValidationError(f"{name} must be non-empty")
    return result


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValidationError(f"{name} must be finite")
    return result


def _count(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValidationError(f"{name} must be a nonnegative integer")
    return value


def _sha(value: Any, name: str, *, required: bool) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value.lower()) or set(value.lower()) == {"0"}:
        raise ValidationError(f"{name} must be a known lowercase SHA256")
    return value.lower()


def _sha_list(value: Any, name: str, *, required: bool) -> tuple[str, ...] | None:
    if value is None and not required:
        return None
    if not isinstance(value, list) or not value:
        raise ValidationError(f"{name} must be a nonempty SHA256 list")
    result = tuple(_sha(item, f"{name}[{index}]", required=True) for index, item in enumerate(value))
    return result  # type: ignore[return-value]


def _provenance_value(row: Mapping[str, Any], *names: str) -> Any:
    provenance = row.get("generation_provenance")
    for name in names:
        if name in row:
            return row[name]
        if isinstance(provenance, dict) and name in provenance:
            return provenance[name]
    return None


def parse_candidate(
    row: Any,
    source: str = "audit",
    line: int = 0,
    *,
    require_provenance: bool = False,
    require_source_binding: bool = False,
) -> Candidate:
    where = f"{source}:{line}"
    if not isinstance(row, dict):
        raise ValidationError(f"{where}: expected object")
    required = {
        "schema_version", "prompt_id", "seed", "candidate_id", "answer_exact",
        "raw_success", "trajectory_quality", "process", "termination",
    }
    missing = sorted(required - row.keys())
    if missing:
        raise ValidationError(f"{where}: missing {missing}")
    if row["schema_version"] != 1:
        raise ValidationError(f"{where}: schema_version must be 1")
    if type(row["answer_exact"]) is not bool or type(row["raw_success"]) is not bool:
        raise ValidationError(f"{where}: success fields must be boolean")
    process = row["process"]
    if not isinstance(process, dict) or not {"unique_hit", "duplicate", "miss", "cap"} <= process.keys():
        raise ValidationError(f"{where}: invalid process object")
    if type(process["cap"]) is not bool:
        raise ValidationError(f"{where}: cap must be boolean")
    termination = row["termination"]
    if termination not in TERMINATIONS:
        raise ValidationError(f"{where}: invalid termination")
    if row["raw_success"] and (not row["answer_exact"] or process["cap"] or termination != "answer"):
        raise ValidationError(f"{where}: inconsistent raw_success")
    if process["cap"] != (termination == "cap"):
        raise ValidationError(f"{where}: cap/termination mismatch")
    quality = _number(row["trajectory_quality"], "trajectory_quality")
    if not 0 <= quality <= 1:
        raise ValidationError(f"{where}: trajectory_quality outside [0,1]")
    seed = _id(row["seed"], "seed")
    declared_seed = _id(_provenance_value(row, "declared_seed", "generation_seed") if require_provenance else _provenance_value(row, "declared_seed", "generation_seed") or seed, "declared_seed")
    if declared_seed != seed:
        raise ValidationError(f"{where}: declared seed does not match row seed")
    model_hash = _sha(_provenance_value(row, "model_checkpoint_sha256", "checkpoint_sha256"), "model_checkpoint_sha256", required=require_provenance)
    generation_hash = _sha(_provenance_value(row, "generation_config_sha256", "generation_config_hash"), "generation_config_sha256", required=require_provenance)
    sampling_hash = _sha(_provenance_value(row, "sampling_config_sha256", "sampling_hash"), "sampling_config_sha256", required=require_provenance)
    response_hash = _sha(_provenance_value(row, "response_sha256", "candidate_content_sha256", "content_sha256"), "response_sha256", required=require_provenance)
    source_row_hash = _sha(
        _provenance_value(row, "source_row_sha256"),
        "source_row_sha256", required=require_source_binding,
    )
    source_image_hashes = _sha_list(
        _provenance_value(row, "source_image_sha256"),
        "source_image_sha256", required=require_source_binding,
    )
    rendered_prompt_hash = _sha(
        _provenance_value(row, "rendered_prompt_sha256"),
        "rendered_prompt_sha256", required=require_source_binding,
    )
    rendered_prompt = _provenance_value(row, "rendered_prompt")
    if require_source_binding and (not isinstance(rendered_prompt, str) or not rendered_prompt):
        raise ValidationError(f"{where}: nonempty rendered_prompt is required")
    if rendered_prompt is not None:
        if not isinstance(rendered_prompt, str) or not rendered_prompt:
            raise ValidationError(f"{where}: rendered_prompt must be a nonempty string")
        if rendered_prompt_hash is not None and hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest() != rendered_prompt_hash:
            raise ValidationError(f"{where}: rendered_prompt SHA256 mismatch")
    raw_response = next((row.get(name) for name in ("response", "candidate_response", "content") if row.get(name) is not None), None)
    generation_config = _provenance_value(row, "generation_config")
    sampling_config = _provenance_value(row, "sampling_config")
    if require_provenance:
        if not isinstance(raw_response, str) or not raw_response:
            raise ValidationError(f"{where}: nonempty raw response text is required")
        if not isinstance(generation_config, dict) or not isinstance(sampling_config, dict):
            raise ValidationError(f"{where}: raw generation_config and sampling_config objects are required")
        embedded_seed = next(
            (generation_config.get(name) for name in ("seed", "generation_seed", "declared_seed") if name in generation_config),
            next((sampling_config.get(name) for name in ("seed", "generation_seed", "declared_seed") if name in sampling_config), None),
        )
        if embedded_seed is None or _id(embedded_seed, "embedded config seed") != declared_seed:
            raise ValidationError(f"{where}: embedded config seed does not match declared seed")
        for config, expected, name in (
            (generation_config, generation_hash, "generation_config"),
            (sampling_config, sampling_hash, "sampling_config"),
        ):
            encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
            if hashlib.sha256(encoded).hexdigest() != expected:
                raise ValidationError(f"{where}: raw {name} SHA256 mismatch")
    if raw_response is not None and response_hash is not None:
        if isinstance(raw_response, str):
            response_bytes = raw_response.encode("utf-8")
        else:
            try:
                response_bytes = json.dumps(raw_response, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
            except (TypeError, ValueError) as exc:
                raise ValidationError(f"{where}: response content is not canonical JSON") from exc
        if hashlib.sha256(response_bytes).hexdigest() != response_hash:
            raise ValidationError(f"{where}: response content SHA256 mismatch")
    return Candidate(
        _id(row["prompt_id"], "prompt_id"), seed, declared_seed,
        _id(row["candidate_id"], "candidate_id"), row["answer_exact"], row["raw_success"],
        quality, _count(process["unique_hit"], "unique_hit"),
        _count(process["duplicate"], "duplicate"), _count(process["miss"], "miss"),
        process["cap"], termination, model_hash, generation_hash, sampling_hash, response_hash,
        generation_config, sampling_config, source_row_hash, source_image_hashes,
        rendered_prompt_hash, rendered_prompt,
    )


def load_audit(
    path: Path, *, require_provenance: bool = False, require_source_binding: bool = False,
) -> tuple[str, dict[str, list[Candidate]]]:
    groups: dict[str, list[Candidate]] = defaultdict(list)
    seeds: set[str] = set()
    raw_records: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = _loads_json(line, f"{path}:{number}")
            record = json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False)
            if record in raw_records:
                raise ValidationError(f"{path}:{number}: duplicate JSON record")
            raw_records.add(record)
            candidate = parse_candidate(
                raw, str(path), number, require_provenance=require_provenance,
                require_source_binding=require_source_binding,
            )
            groups[candidate.prompt_id].append(candidate)
            seeds.add(candidate.seed)
    if not groups or len(seeds) != 1:
        raise ValidationError(f"{path}: audit must be nonempty and contain one seed")
    return next(iter(seeds)), dict(groups)


def _one(values: set[str | None], name: str, seed: str) -> str:
    if None in values or len(values) != 1:
        raise ValidationError(f"seed {seed}: {name} must be present and constant across the audit")
    return next(iter(values))  # type: ignore[return-value]


def validate_audits(
    paths: Sequence[Path],
    prompt_ids: set[str],
    m: int = 32,
    *,
    require_provenance: bool = False,
    source_bindings: Mapping[str, Mapping[str, Any]] | None = None,
    expected_model_checkpoint_sha256: str | None = None,
) -> tuple[tuple[str, str], dict[str, dict[str, list[Candidate]]], dict[str, Any]]:
    if len(paths) != 2:
        raise ValidationError("exactly two audit files are required")
    if source_bindings is not None and set(source_bindings) != prompt_ids:
        raise ValidationError("source binding prompt set mismatch")
    loaded = [
        load_audit(
            path, require_provenance=require_provenance,
            require_source_binding=source_bindings is not None,
        )
        for path in paths
    ]
    seeds = tuple(item[0] for item in loaded)
    if seeds[0] == seeds[1]:
        raise ValidationError("audit seeds must be distinct")
    output: dict[str, dict[str, list[Candidate]]] = defaultdict(dict)
    provenance: list[tuple[str, str, str]] = []
    mining_configs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    content_fingerprints: list[str] = []
    candidate_ids: set[str] = set()
    response_hashes: set[str] = set()
    for seed, groups in loaded:
        if set(groups) != prompt_ids:
            raise ValidationError(f"seed {seed}: prompt set mismatch")
        all_rows = [row for rows in groups.values() for row in rows]
        if require_provenance:
            model_hash = _one({row.model_checkpoint_sha256 for row in all_rows}, "model checkpoint hash", seed)
            generation_hash = _one({row.generation_config_sha256 for row in all_rows}, "generation config hash", seed)
            sampling_hash = _one({row.sampling_config_sha256 for row in all_rows}, "sampling config hash", seed)
            provenance.append((model_hash, generation_hash, sampling_hash))
            generation_configs = {
                json.dumps(row.generation_config, sort_keys=True, separators=(",", ":"), allow_nan=False)
                for row in all_rows
            }
            sampling_configs = {
                json.dumps(row.sampling_config, sort_keys=True, separators=(",", ":"), allow_nan=False)
                for row in all_rows
            }
            if len(generation_configs) != 1 or len(sampling_configs) != 1:
                raise ValidationError(f"seed {seed}: raw mining configs must be constant across the audit")
            generation_config = json.loads(next(iter(generation_configs)))
            sampling_config = json.loads(next(iter(sampling_configs)))
            if set(name for name in generation_config if name in {"generation_seed", "declared_seed"}):
                raise ValidationError("formal generation config must use exactly the declared 'seed' field")
            if generation_config.get("seed") is None or _id(generation_config["seed"], "generation config seed") != seed:
                raise ValidationError(f"seed {seed}: generation config must contain the matching seed field")
            if any(name in sampling_config for name in ("seed", "generation_seed", "declared_seed")):
                raise ValidationError("sampling config must not carry an additional seed field")
            mining_configs.append((generation_config, sampling_config))
        fingerprint = hashlib.sha256()
        for prompt_id in sorted(groups):
            rows = groups[prompt_id]
            if len(rows) != m or len({row.candidate_id for row in rows}) != m:
                raise ValidationError(f"prompt {prompt_id} seed {seed}: require {m} unique candidates")
            if require_provenance and any(row.response_sha256 is None for row in rows):
                raise ValidationError(f"prompt {prompt_id} seed {seed}: response hashes must be present")
            for row in rows:
                if row.candidate_id in candidate_ids:
                    raise ValidationError(f"duplicate candidate_id across audits: {row.candidate_id}")
                candidate_ids.add(row.candidate_id)
                if require_provenance:
                    assert row.response_sha256 is not None
                    if row.response_sha256 in response_hashes:
                        raise ValidationError("identical response content copied across candidates")
                    response_hashes.add(row.response_sha256)
                if source_bindings is not None:
                    binding = source_bindings[prompt_id]
                    expected_images = tuple(binding["source_image_sha256"])
                    if (
                        row.source_row_sha256 != binding["source_row_sha256"]
                        or row.source_image_sha256 != expected_images
                        or row.rendered_prompt_sha256 != binding["rendered_prompt_sha256"]
                        or row.rendered_prompt != binding["rendered_prompt"]
                    ):
                        raise ValidationError(
                            f"prompt {prompt_id} seed {seed}: audit/source row, image, or prompt binding mismatch"
                        )
            for response_hash in sorted(row.response_sha256 or "" for row in rows):
                fingerprint.update(prompt_id.encode() + b"\0" + response_hash.encode() + b"\n")
            output[prompt_id][seed] = rows
        content_fingerprints.append(fingerprint.hexdigest())
    if require_provenance:
        if provenance[0][0] != provenance[1][0]:
            raise ValidationError("mining audits must use the same model checkpoint hash")
        if (
            expected_model_checkpoint_sha256 is not None
            and provenance[0][0] != expected_model_checkpoint_sha256
        ):
            raise ValidationError("mining audit checkpoint hash does not match the live checkpoint tree")
        if provenance[0][2] != provenance[1][2]:
            raise ValidationError("mining audits must use the same sampling config hash")
        if content_fingerprints[0] == content_fingerprints[1]:
            raise ValidationError("audits have identical response content; independent generation is not proven")
        first_generation, first_sampling = mining_configs[0]
        second_generation, second_sampling = mining_configs[1]
        normalized_first = {key: value for key, value in first_generation.items() if key != "seed"}
        normalized_second = {key: value for key, value in second_generation.items() if key != "seed"}
        if canonical_json(normalized_first) != canonical_json(normalized_second):
            raise ValidationError("mining generation config differs across seeds outside the declared seed field")
        if canonical_json(first_sampling) != canonical_json(second_sampling):
            raise ValidationError("mining sampling config differs across seeds")
    source_binding_payload = None
    source_binding_sha256 = None
    if source_bindings is not None:
        source_binding_payload = {
            prompt_id: {
                "source_row_sha256": source_bindings[prompt_id]["source_row_sha256"],
                "source_image_sha256": list(source_bindings[prompt_id]["source_image_sha256"]),
                "rendered_prompt_sha256": source_bindings[prompt_id]["rendered_prompt_sha256"],
            }
            for prompt_id in sorted(source_bindings)
        }
        source_binding_sha256 = hashlib.sha256(
            canonical_json(source_binding_payload).encode("utf-8")
        ).hexdigest()
    mining_contract = {
        "model_checkpoint_sha256": provenance[0][0] if require_provenance else None,
        "generation_config_by_seed": {
            seed: mining_configs[index][0] for index, seed in enumerate(seeds)
        } if require_provenance else {},
        "sampling_config": mining_configs[0][1] if require_provenance else None,
        "seed_difference_allowlist": ["seed"],
        "source_binding_contract": (
            "source_row_image_rendered_prompt_v1" if source_bindings is not None else None
        ),
        "source_binding_sha256": source_binding_sha256,
    }
    return (seeds[0], seeds[1]), dict(output), mining_contract


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValidationError("value is not canonical finite JSON") from exc


def classify_prompt(
    groups: Mapping[str, Sequence[Candidate]],
    winner_min: int = 3,
    winner_max: int = 20,
    min_quality: float = 0.55,
    max_duplicate_rate: float = 0.25,
    max_cap_rate: float = 0.25,
    min_quality_range: float = 0.10,
) -> str | None:
    """Classify outcome-frontier or zero-winner process-hard prompts.

    Process-hard is deliberately independent of successful trajectories: both
    seeds must have zero raw winners and still exhibit repeatable positive
    process signal plus nontrivial quality variation.
    """
    if len(groups) != 2:
        return None
    if all(winner_min <= sum(row.raw_success for row in rows) <= winner_max for rows in groups.values()):
        return "outcome-frontier"
    if any(any(row.raw_success for row in rows) for rows in groups.values()):
        return None
    rows = [row for values in groups.values() for row in values]
    points = sum(row.unique_hit + row.duplicate + row.miss for row in rows)
    reliable_signal = all(
        max(row.quality for row in values) >= min_quality
        and max(row.quality for row in values) - min(row.quality for row in values) >= min_quality_range
        and sum(row.unique_hit for row in values) > 0
        for values in groups.values()
    )
    if (
        reliable_signal
        and sum(row.duplicate for row in rows) / max(1, points) <= max_duplicate_rate
        and sum(row.cap for row in rows) / len(rows) <= max_cap_rate
    ):
        return "process-hard"
    return None


def largest_remainder(total: int, ratios: Sequence[float]) -> list[int]:
    if total < 0 or any(not math.isfinite(value) or value < 0 for value in ratios) or not math.isclose(sum(ratios), 1, abs_tol=1e-9):
        raise ValidationError("invalid quota ratios")
    raw = [total * value for value in ratios]
    quotas = [math.floor(value) for value in raw]
    for index in sorted(range(len(quotas)), key=lambda item: (-(raw[item] - quotas[item]), item))[: total - sum(quotas)]:
        quotas[index] += 1
    return quotas


def select_ids(
    eligible: Mapping[str, Sequence[str]],
    sample_count: int,
    bucket_ratios: Sequence[float] = RATIOS,
    outcome_ratio: float = 0.75,
    selection_seed: int = 37,
    formal: bool = True,
    *,
    require_complete: bool | None = None,
) -> tuple[list[str], dict[str, dict[str, int]]]:
    complete = formal if require_complete is None else require_complete
    rng = random.Random(selection_seed)
    selected: list[str] = []
    stats: dict[str, dict[str, int]] = {}
    for label, quota in zip(LABELS, largest_remainder(sample_count, bucket_ratios)):
        outcome_quota, process_quota = largest_remainder(quota, (outcome_ratio, 1 - outcome_ratio))
        outcome = sorted(eligible.get(f"{label}|outcome-frontier", ()))
        process = sorted(eligible.get(f"{label}|process-hard", ()))
        rng.shuffle(outcome)
        rng.shuffle(process)
        if complete and (len(outcome) < outcome_quota or len(process) < process_quota):
            raise ValidationError(
                f"insufficient quota {label}: outcome {len(outcome)}/{outcome_quota}, "
                f"process {len(process)}/{process_quota}"
            )
        picked_outcome = outcome[:outcome_quota]
        picked_process = process[:process_quota]
        selected.extend(picked_outcome + picked_process)
        stats[label] = {
            "quota": quota,
            "outcome-frontier": len(picked_outcome),
            "process-hard": len(picked_process),
        }
    if complete and len(selected) != sample_count:
        raise ValidationError("quota selection incomplete")
    rng.shuffle(selected)
    return selected, stats


def path_sha256(path: Path) -> str:
    absolute = Path(os.path.abspath(os.path.expanduser(str(path))))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = os.lstat(current)
        except (FileNotFoundError, NotADirectoryError) as exc:
            raise ValidationError(f"cannot hash missing path: {current}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ValidationError(f"cannot hash symlinked path component: {current}")
    if absolute.is_dir():
        for root, directories, filenames in os.walk(absolute, followlinks=False):
            for name in directories + filenames:
                candidate = Path(root) / name
                if candidate.is_symlink():
                    raise ValidationError(f"cannot hash tree containing symlink: {candidate}")
    path = absolute
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValidationError(f"nothing to hash: {path}")
    def file_hash(item: Path) -> str:
        digest = hashlib.sha256()
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    if path.is_file():
        return file_hash(path)
    entries = [
        {
            "path": item.relative_to(path).as_posix(),
            "size": item.stat().st_size,
            "sha256": file_hash(item),
        }
        for item in files
    ]
    encoded = json.dumps(entries, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _answer(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, dict):
        for key in ("answer", "count_number", "final_count", "N", "total_count"):
            if key in value:
                parsed = _answer(value[key])
                if parsed is not None:
                    return parsed
        return None
    if type(value) is not int:
        return None
    return value


def _bucket(value: int) -> str | None:
    return next((label for (low, high), label in zip(BUCKETS, LABELS) if low <= value <= high), None)


def _valid_prompt(value: Any, row_index: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"source row {row_index} requires a nonempty prompt string")
    return value


def _valid_image_item(value: Any, row_index: int) -> None:
    path: Any = None
    payload: Any = None
    if isinstance(value, (str, os.PathLike)):
        path = os.fspath(value)
    elif isinstance(value, dict):
        path = next((value.get(key) for key in ("path", "image_path", "file_name") if value.get(key) is not None), None)
        payload = value.get("bytes")
    if isinstance(path, (str, os.PathLike)) and os.fspath(path).strip():
        return
    if isinstance(payload, (bytes, bytearray, memoryview)) and len(payload) > 0:
        return
    raise ValidationError(f"source row {row_index} image lacks a nonempty payload/path")


def _canonical_source_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_source_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_source_value(item) for item in value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        payload = bytes(value)
        return {
            "__bytes_sha256__": hashlib.sha256(payload).hexdigest(),
            "__bytes_length__": len(payload),
        }
    if isinstance(value, float) and not math.isfinite(value):
        raise ValidationError("source row contains a non-finite number")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValidationError(f"source row contains unsupported value type: {type(value).__name__}")


def _source_row_sha256(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        canonical_json(_canonical_source_value(row)).encode("utf-8")
    ).hexdigest()


def _source_image_sha256(
    item: Any, roots: tuple[Path, ...], mappings: tuple[tuple[str, str], ...], row_index: int,
) -> str:
    path: Any = None
    payload: Any = None
    if isinstance(item, (str, os.PathLike)):
        path = os.fspath(item)
    elif isinstance(item, Mapping):
        path = next(
            (item.get(key) for key in ("path", "image_path", "file_name") if item.get(key) is not None),
            None,
        )
        payload = item.get("bytes")
    if isinstance(payload, (bytes, bytearray, memoryview)) and len(payload) > 0:
        return hashlib.sha256(bytes(payload)).hexdigest()
    if not isinstance(path, (str, os.PathLike)) or not os.fspath(path).strip():
        raise ValidationError(f"source row {row_index} image bytes/path unavailable")
    raw_path = remap_image_path(os.fspath(path), mappings)
    expanded = Path(os.path.expandvars(os.path.expanduser(raw_path)))
    candidates = (expanded,) if expanded.is_absolute() else tuple(root / expanded for root in roots)
    for candidate in candidates:
        if candidate.is_file():
            return path_sha256(candidate)
    raise ValidationError(f"source row {row_index} image is missing after path remap: {path}")


def build_source_bindings(
    table: Any, source: Path, image_roots: Sequence[Path] = (),
) -> dict[str, dict[str, Any]]:
    roots = (source if source.is_dir() else source.parent, *(Path(root) for root in image_roots))
    mappings = parse_image_path_remap()
    names = list(table.column_names)
    bindings: dict[str, dict[str, Any]] = {}
    for index in range(table.num_rows):
        row = {name: table[name][index].as_py() for name in names}
        prompt_id = _id(row.get("prompt_id"), f"source prompt_id at row {index}")
        prompt = _valid_prompt(row.get("prompt"), index)
        images = row.get("images")
        _validate_training_source_row(prompt, images, index)
        assert isinstance(images, (list, tuple))
        bindings[prompt_id] = {
            "source_row_sha256": _source_row_sha256(row),
            "source_image_sha256": [
                _source_image_sha256(item, roots, mappings, index) for item in images
            ],
            # The RL source prompt is the exact user-rendered prompt consumed by
            # the mining harness before tokenizer chat-template expansion.
            "rendered_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "rendered_prompt": prompt,
        }
    return bindings


def _validate_training_source_row(prompt: Any, images: Any, row_index: int) -> None:
    prompt_text = _valid_prompt(prompt, row_index)
    if not isinstance(images, (list, tuple)):
        raise ValidationError(f"source row {row_index} images must be a nonempty list for RLHFDataset")
    items = images
    if not items:
        raise ValidationError(f"source row {row_index} requires at least one image")
    if prompt_text.count("<image>") != len(items):
        raise ValidationError(f"source row {row_index} prompt/image placeholder count mismatch")
    for item in items:
        _valid_image_item(item, row_index)


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _known_expected(value: str | None, name: str) -> str:
    result = _sha(value, name, required=True)
    assert result is not None
    return result


def build(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    if output_dir.exists() or output_dir.is_symlink():
        raise ValidationError(
            f"output directory already exists; refusing mixed publication: {output_dir}"
        )
    formal = args.run_class == "formal"
    proven = args.run_class in {"canary", "formal"}
    if formal:
        if tuple(args.bucket_ratios) != RATIOS:
            raise ValidationError("formal bucket ratios are locked to 0.40,0.10,0.20,0.20,0.10")
        if args.candidates_per_seed != 32:
            raise ValidationError("formal mining is locked to M=32 per seed")
        locked = (
            args.winner_min == 3 and args.winner_max == 20
            and math.isclose(args.outcome_ratio, .75, rel_tol=0, abs_tol=1e-12)
            and math.isclose(args.process_min_quality, .55, rel_tol=0, abs_tol=1e-12)
            and math.isclose(args.process_min_quality_range, .10, rel_tol=0, abs_tol=1e-12)
            and math.isclose(args.process_max_duplicate_rate, .25, rel_tol=0, abs_tol=1e-12)
            and math.isclose(args.process_max_cap_rate, .25, rel_tol=0, abs_tol=1e-12)
        )
        if not locked:
            raise ValidationError("formal mining thresholds are locked (winners 3..20 and process-hard quality controls)")
        expected_source = _known_expected(args.expected_source_sha256, "expected source SHA256")
        if not isinstance(args.expected_audit_sha256, list) or len(args.expected_audit_sha256) != 2:
            raise ValidationError("formal requires exactly two expected audit SHA256 values")
        expected_audits = [_known_expected(value, "expected audit SHA256") for value in args.expected_audit_sha256]
        expected_mining_checkpoint = _known_expected(
            args.expected_mining_checkpoint_sha256,
            "expected mining checkpoint SHA256",
        )
    else:
        expected_source = args.expected_source_sha256
        expected_audits = args.expected_audit_sha256
        expected_mining_checkpoint = args.expected_mining_checkpoint_sha256

    if proven and args.mining_checkpoint is None:
        raise ValidationError("canary/formal requires --mining-checkpoint")
    mining_checkpoint_before = (
        path_sha256(args.mining_checkpoint) if args.mining_checkpoint is not None else None
    )
    if (
        expected_mining_checkpoint is not None
        and mining_checkpoint_before != expected_mining_checkpoint
    ):
        raise ValidationError("mining checkpoint SHA256 mismatch")

    try:
        import pyarrow as pa
        import pyarrow.dataset as ds
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required") from exc

    source_before = path_sha256(args.source)
    if expected_source and source_before != expected_source:
        raise ValidationError("source SHA256 mismatch")

    table = ds.dataset(args.source, format="parquet").to_table()
    names = set(table.column_names)
    required_source_columns = {"prompt_id", "sample_id", "prompt", "answer", "images"}
    if not required_source_columns <= names:
        raise ValidationError(
            "source must be directly RLHFDataset-consumable and requires "
            "prompt_id, sample_id, prompt, answer, and images"
        )
    candidate_columns = names & CANDIDATE_ONLY_COLUMNS
    if candidate_columns:
        raise ValidationError(
            "frontier prompt source contains candidate response/audit columns: "
            + ",".join(sorted(candidate_columns))
        )
    id_column = "prompt_id"
    ids: list[str] = []
    answers: list[int] = []
    positions: dict[str, int] = {}
    identifier_values = {
        name: table[name].to_pylist() for name in ("prompt_id", "sample_id") if name in names
    }
    seen_identifiers = {name: set() for name in identifier_values}
    prompt_values = table["prompt"].to_pylist()
    image_values = table["images"].to_pylist()
    for index, (raw_id, raw_answer) in enumerate(zip(table[id_column].to_pylist(), table["answer"].to_pylist())):
        _validate_training_source_row(prompt_values[index], image_values[index], index)
        prompt_id = _id(raw_id, "source id")
        answer = _answer(raw_answer)
        if prompt_id in positions or answer is None or _bucket(answer) is None:
            raise ValidationError(f"invalid/duplicate source row {index}")
        positions[prompt_id] = index
        for name, values in identifier_values.items():
            identifier = _id(values[index], f"source {name}")
            if identifier in seen_identifiers[name]:
                raise ValidationError(f"duplicate source {name} at row {index}")
            seen_identifiers[name].add(identifier)
        ids.append(prompt_id)
        answers.append(answer)

    source_bindings = build_source_bindings(table, args.source, args.image_root or ()) if proven else None
    if source_bindings is not None and set(source_bindings) != set(ids):
        raise ValidationError("source binding set differs from validated source prompt IDs")

    # Validate the training contract before touching mining artifacts.  This
    # keeps an invalid source diagnosable even when audit paths are stale or
    # unavailable, and matches the builder's fail-fast source boundary.
    audit_before = [path_sha256(path) for path in args.audit]
    if expected_audits and (len(expected_audits) != 2 or audit_before != list(expected_audits)):
        raise ValidationError("audit SHA256 mismatch")

    seeds, audits, mining_config = validate_audits(
        args.audit, set(ids), args.candidates_per_seed, require_provenance=proven,
        source_bindings=source_bindings,
        expected_model_checkpoint_sha256=mining_checkpoint_before,
    )
    if proven:
        assert args.mining_checkpoint is not None and mining_checkpoint_before is not None
        mining_config.update({
            "model_checkpoint_path": str(args.mining_checkpoint.resolve()),
            "model_checkpoint_sha256": mining_checkpoint_before,
        })
    eligible: dict[str, list[str]] = defaultdict(list)
    histograms = {seed: Counter() for seed in seeds}
    for prompt_id, answer in zip(ids, answers):
        category = classify_prompt(
            audits[prompt_id], args.winner_min, args.winner_max,
            args.process_min_quality, args.process_max_duplicate_rate,
            args.process_max_cap_rate, args.process_min_quality_range,
        )
        for seed in seeds:
            histograms[seed][sum(row.raw_success for row in audits[prompt_id][seed])] += 1
        if category:
            eligible[f"{_bucket(answer)}|{category}"].append(prompt_id)

    chosen, stats = select_ids(
        eligible, args.sample_count, args.bucket_ratios, args.outcome_ratio,
        args.selection_seed, formal=formal, require_complete=args.run_class in {"canary", "formal"},
    )
    selected = table.take(pa.array([positions[prompt_id] for prompt_id in chosen], type=pa.int64()))

    # Snapshot again after every input has been consumed.  A concurrent mutation
    # makes the build invalid even when an initial expected hash matched.
    source_after = path_sha256(args.source)
    audit_after = [path_sha256(path) for path in args.audit]
    mining_checkpoint_after = (
        path_sha256(args.mining_checkpoint) if args.mining_checkpoint is not None else None
    )
    if (
        source_after != source_before
        or audit_after != audit_before
        or mining_checkpoint_after != mining_checkpoint_before
    ):
        raise ValidationError("input mutated while the build was reading it")

    if output_dir.exists() or output_dir.is_symlink():
        raise ValidationError(f"output directory appeared while building: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging.", dir=output_dir.parent))
    try:
        parquet_path = staging / "frontier_rl.parquet"
        ids_path = staging / "selected_ids.json"
        manifest_path = staging / "selection_manifest.json"
        pq.write_table(selected, parquet_path)
        _fsync_file(parquet_path)
        _write_json(ids_path, chosen)
        artifact_hashes = {
            "frontier_rl.parquet": path_sha256(parquet_path),
            "selected_ids.json": path_sha256(ids_path),
        }
        manifest = {
            "schema_version": 3,
            "data_mode": "frontier_rl",
            "run_class": args.run_class,
            "publication": "atomic_directory_v1",
            "source_path": str(args.source.resolve()),
            "source_sha256": source_before,
            "audit_paths": [str(path.resolve()) for path in args.audit],
            "audit_sha256": {str(path.resolve()): digest for path, digest in zip(args.audit, audit_before)},
            "seeds": list(seeds),
            "candidates_per_seed": args.candidates_per_seed,
            "requested_sample_count": args.sample_count,
            "selected_sample_count": len(chosen),
            "selection_seed": args.selection_seed,
            "bucket_ratios": dict(zip(LABELS, args.bucket_ratios)),
            "outcome_ratio": args.outcome_ratio,
            "process_ratio": 1 - args.outcome_ratio,
            "classifier_thresholds": {
                "winner_min": args.winner_min,
                "winner_max": args.winner_max,
                "process_min_quality": args.process_min_quality,
                "process_min_quality_range": args.process_min_quality_range,
                "process_max_duplicate_rate": args.process_max_duplicate_rate,
                "process_max_cap_rate": args.process_max_cap_rate,
            },
            "mining_generation_contract": mining_config,
            "selection": stats,
            "eligible_counts": {key: len(value) for key, value in sorted(eligible.items())},
            "seed_stats": {seed: {"winner_count_histogram": dict(histograms[seed])} for seed in seeds},
            "selected_ids_sha256": hashlib.sha256("\n".join(chosen).encode()).hexdigest(),
            "artifact_sha256": artifact_hashes,
            "output_schema": str(selected.schema),
            "source_schema_sha256": hashlib.sha256(str(table.schema).encode()).hexdigest(),
            "output_schema_sha256": hashlib.sha256(str(selected.schema).encode()).hexdigest(),
            "candidate_fields_injected": False,
            "published_artifacts": ["frontier_rl.parquet", "selected_ids.json", "selection_manifest.json"],
        }
        _write_json(manifest_path, manifest)
        complete_hashes = {
            **artifact_hashes,
            "selection_manifest.json": path_sha256(manifest_path),
        }
        complete = {
            "schema_version": 1,
            "publication": "atomic_directory_v1",
            "published_artifacts": ["frontier_rl.parquet", "selected_ids.json", "selection_manifest.json"],
            "artifact_sha256": complete_hashes,
        }
        _write_json(staging / "_COMPLETE.json", complete)
        directory_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(staging, output_dir)
        parent_fd = os.open(output_dir.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--source", type=Path, required=True)
    result.add_argument("--audit", type=Path, action="append", required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--run-class", choices=("debug", "canary", "formal"), required=True)
    result.add_argument("--sample-count", type=int, required=True)
    result.add_argument("--selection-seed", type=int, default=37)
    result.add_argument("--candidates-per-seed", type=int, default=32)
    result.add_argument("--winner-min", type=int, default=3)
    result.add_argument("--winner-max", type=int, default=20)
    result.add_argument("--bucket-ratios", default=".4,.1,.2,.2,.1")
    result.add_argument("--outcome-ratio", type=float, default=.75)
    result.add_argument("--process-min-quality", type=float, default=.55)
    result.add_argument("--process-min-quality-range", type=float, default=.10)
    result.add_argument("--process-max-duplicate-rate", type=float, default=.25)
    result.add_argument("--process-max-cap-rate", type=float, default=.25)
    result.add_argument("--expected-source-sha256")
    result.add_argument("--expected-audit-sha256", action="append")
    result.add_argument("--mining-checkpoint", type=Path)
    result.add_argument("--expected-mining-checkpoint-sha256")
    result.add_argument("--image-root", type=Path, action="append")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    try:
        args.bucket_ratios = tuple(float(value) for value in args.bucket_ratios.split(","))
        if (
            len(args.audit) != 2 or len(args.bucket_ratios) != 5 or args.sample_count < 1
            or any(not math.isfinite(value) for value in args.bucket_ratios)
            or not .7 <= args.outcome_ratio <= .8
            or not 0 <= args.winner_min <= args.winner_max <= args.candidates_per_seed
            or not 0 <= args.process_min_quality <= 1
            or not 0 <= args.process_min_quality_range <= 1
            or not 0 <= args.process_max_duplicate_rate <= 1
            or not 0 <= args.process_max_cap_rate <= 1
        ):
            raise ValidationError("invalid audits, quotas, sample count, outcome ratio, or winner bounds")
        manifest = build(args)
    except (ValidationError, RuntimeError, OSError, ValueError) as exc:
        argument_parser.exit(2, f"[V37-frontier][ERROR] {exc}\n")
    print(json.dumps({"ok": True, "selected": manifest["selected_sample_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
