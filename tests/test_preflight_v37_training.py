# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import hashlib
import os
import subprocess
import sys
import types
from copy import deepcopy
from pathlib import Path

import pytest
from tools import preflight_v37_training as preflight_module


REPO_ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = REPO_ROOT / "tools" / "preflight_v37_training.py"
UNIFORM_RATIOS = "0.2,0.2,0.2,0.2,0.2"
PYARROW_PYTHON = os.environ.get("V37_TEST_PYARROW_PYTHON", sys.executable)

_pyarrow_probe = subprocess.run(
    [PYARROW_PYTHON, "-c", "import pyarrow.parquet"],
    text=True,
    capture_output=True,
    check=False,
)
HAS_PYARROW = _pyarrow_probe.returncode == 0


def test_numeric_ids_are_canonical_across_fields() -> None:
    keys: set[str] = set()
    preflight_module._add_identifier(keys, 17)
    preflight_module._add_identifier(keys, "17")
    preflight_module._add_identifier(keys, 17.0)
    assert keys == {"canonical_id:17"}


def test_image_representation_normalization_supports_single_and_list() -> None:
    assert tuple(preflight_module._image_items("one.png")) == ("one.png",)
    assert tuple(preflight_module._image_items(["a.png", "b.png"])) == ("a.png", "b.png")
    assert preflight_module._image_path({"image_path": "one.png"}) == "one.png"
    assert preflight_module._embedded_image_bytes({"bytes": b"png"}) == b"png"


def test_formal_locks_loader_filter_num_proc_to_64(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    result = preflight_module.main([
        str(tmp_path / "data.parquet"),
        "--metadata", str(tmp_path / "metadata.json"),
        "--masks-dir", str(tmp_path / "masks"),
        "--run-class", "formal",
        "--loader-filter-num-proc", "63",
    ])
    assert result == 2
    assert "--loader-filter-num-proc is locked to 64" in capsys.readouterr().err


def test_non_formal_loader_filter_num_proc_remains_configurable(tmp_path: Path) -> None:
    args = preflight_module.build_parser().parse_args([
        str(tmp_path / "data.parquet"),
        "--metadata", str(tmp_path / "metadata.json"),
        "--masks-dir", str(tmp_path / "masks"),
        "--run-class", "canary",
        "--loader-filter-num-proc", "7",
    ])
    assert args.loader_filter_num_proc == 7


def _require_pyarrow() -> None:
    if not HAS_PYARROW:
        pytest.skip("pyarrow is unavailable; set V37_TEST_PYARROW_PYTHON to a compatible Python")


def _fixture_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    _require_pyarrow()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    metadata = tmp_path / "masks_metadata.json"
    metadata.write_text("[]\n", encoding="utf-8")
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    return data_dir, metadata, masks_dir


def _write_parquet(
    data_dir: Path,
    answers: list[str],
    image_paths: list[Path | None],
    *,
    embedded: bool = False,
    corrupt_embedded: bool = False,
    prompt_ids: list[str] | None = None,
) -> None:
    writer = """
import io
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

output = Path(sys.argv[1])
answers = json.loads(sys.argv[2])
paths = json.loads(sys.argv[3])
embedded = json.loads(sys.argv[4])
corrupt_embedded = json.loads(sys.argv[5])
prompt_ids = json.loads(sys.argv[6])
payload = None
if embedded:
    if corrupt_embedded:
        payload = b"corrupt-image"
    else:
        buffer = io.BytesIO()
        Image.new("RGB", (2, 2), color=(12, 34, 56)).save(buffer, format="PNG")
        payload = buffer.getvalue()
image_type = pa.list_(pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())]))
images = [[{"bytes": payload, "path": path}] for path in paths]
table = pa.table(
    {
        "answer": pa.array(answers, type=pa.string()),
        "images": pa.array(images, type=image_type),
    }
)
if prompt_ids is not None:
    table = table.append_column("prompt_id", pa.array(prompt_ids, type=pa.string()))
pq.write_table(table, output, row_group_size=2)
"""
    subprocess.run(
        [
            PYARROW_PYTHON,
            "-c",
            writer,
            str(data_dir / "train.parquet"),
            json.dumps(answers),
            json.dumps([None if path is None else str(path) for path in image_paths]),
            json.dumps(embedded),
            json.dumps(corrupt_embedded),
            json.dumps(prompt_ids),
        ],
        text=True,
        capture_output=True,
        check=True,
    )


def _run(data_dir: Path, metadata: Path, masks_dir: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            PYARROW_PYTHON,
            str(PREFLIGHT),
            str(data_dir),
            "--metadata",
            str(metadata),
            "--masks-dir",
            str(masks_dir),
            *extra,
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def _tree_hash(path: Path) -> str:
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    entries = [{
        "path": item.relative_to(path).as_posix(),
        "size": item.stat().st_size,
        "sha256": hashlib.sha256(item.read_bytes()).hexdigest(),
    } for item in files]
    return hashlib.sha256(json.dumps(
        entries, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _formal_files(tmp_path: Path, data_dir: Path, metadata: Path, masks_dir: Path, image: Path) -> tuple[Path, Path]:
    subprocess.run([PYARROW_PYTHON, "-c", "from PIL import Image; import sys; Image.new('RGB',(8,8),'red').save(sys.argv[1]); Image.new('L',(8,8),1).save(sys.argv[2])", str(image), str(masks_dir / "mask.png")], check=True)
    metadata.write_text(json.dumps([{"image": image.name, "mask": "mask.png"}]), encoding="utf-8")
    filtered = tmp_path / "filtered.json"
    filtered.write_text(json.dumps({"dataset_sha256": _tree_hash(data_dir), "sample_count": 1, "answer_buckets": {"2-10": 0, "11-20": 1, "21-30": 0, "31-40": 0, "41-50": 0}}), encoding="utf-8")
    coverage = tmp_path / "coverage.json"
    coverage.write_text(json.dumps({
        "dataset_sha256": _tree_hash(data_dir), "metadata_sha256": _tree_hash(metadata),
        "mask_tree_sha256": _tree_hash(masks_dir), "coverage": 1.0,
        "sample_count": 1, "mapped_sample_count": 1, "missing_sample_count": 0,
        "ambiguous_sample_count": 0, "dimension_mismatch_count": 0,
        "unreadable_mask_count": 0, "mapping_key_normalization": "nfkc_basename_casefold_v1",
    }), encoding="utf-8")
    return filtered, coverage


def _formal_required_args(data_dir: Path, metadata: Path, masks_dir: Path) -> list[str]:
    data_hash = _tree_hash(data_dir)
    entrypoint = REPO_ROOT / "examples" / "v32_sparse_0_10_stable_drfix.sh"
    return [
        "--expected-ratios", "0.4,0.1,0.2,0.2,0.1", "--require-val-all-buckets",
        "--val-data", str(data_dir), "--forbidden-data", str(data_dir),
        "--expected-data-sha256", data_hash,
        "--expected-metadata-sha256", _tree_hash(metadata),
        "--expected-masks-sha256", _tree_hash(masks_dir),
        "--expected-val-sha256", data_hash, "--expected-forbidden-sha256", data_hash,
        "--loader-filter-num-proc", "64",
        "--training-entrypoint", str(entrypoint),
        "--expected-training-entrypoint-sha256", _tree_hash(entrypoint),
    ]


def _external_snapshots(
    data_dir: Path,
    val_paths: tuple[Path, ...] = (),
    forbidden_paths: tuple[Path, ...] = (),
) -> tuple[dict[str, object], list[str]]:
    _require_pyarrow()
    try:
        import pyarrow.parquet as pq
    except ImportError:
        script = """
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
from tools import preflight_v37_training as preflight

snapshot, errors = preflight._collect_external_image_snapshots(
    Path(sys.argv[1]),
    tuple(Path(item) for item in json.loads(sys.argv[2])),
    tuple(Path(item) for item in json.loads(sys.argv[3])),
    pq,
    2,
    (),
)
print(json.dumps({"snapshot": snapshot, "errors": errors}))
"""
        result = subprocess.run(
            [
                PYARROW_PYTHON, "-c", script, str(data_dir),
                json.dumps([str(path) for path in val_paths]),
                json.dumps([str(path) for path in forbidden_paths]),
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        payload = json.loads(result.stdout)
        return payload["snapshot"], payload["errors"]
    return preflight_module._collect_external_image_snapshots(
        data_dir, val_paths, forbidden_paths, pq, 2, ()
    )


def test_external_image_snapshot_is_deterministic_grouped_and_excludes_embedded(
    tmp_path: Path,
) -> None:
    data_dir, _, _ = _fixture_paths(tmp_path)
    val_dir = tmp_path / "validation"
    forbidden_dir = tmp_path / "forbidden"
    val_dir.mkdir()
    forbidden_dir.mkdir()
    train_image = tmp_path / "train-image.png"
    val_image = tmp_path / "validation-image.png"
    forbidden_image = tmp_path / "forbidden-image.png"
    train_image.write_bytes(b"train-image")
    val_image.write_bytes(b"validation-image")
    forbidden_image.write_bytes(b"forbidden-image")
    _write_parquet(data_dir, ["11"], [train_image])
    _write_parquet(val_dir, ["12"], [val_image])
    _write_parquet(forbidden_dir, ["13"], [forbidden_image])
    (forbidden_dir / "train.parquet").rename(forbidden_dir / "external.parquet")
    _write_parquet(
        forbidden_dir, ["14"], [Path("embedded-only.png")], embedded=True
    )

    snapshot, errors = _external_snapshots(
        data_dir, (val_dir,), (forbidden_dir,)
    )
    repeated, repeated_errors = _external_snapshots(
        data_dir, (val_dir,), (forbidden_dir,)
    )

    assert errors == repeated_errors == []
    assert snapshot == repeated
    assert snapshot["schema_version"] == 1
    assert snapshot["snapshot_contract"] == "external_image_resolved_path_sha256_v1"
    assert snapshot["embedded_bytes_contract"] == "covered_by_source_parquet_sha256"
    for split, image in (
        ("train", train_image),
        ("validation", val_image),
        ("forbidden", forbidden_image),
    ):
        assert snapshot[split]["file_count"] == 1
        assert snapshot[split]["files"] == [{
            "resolved_path": str(image.resolve()),
            "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        }]
    assert "embedded-only.png" not in json.dumps(snapshot)
    assert preflight_module.verify_external_image_snapshots(snapshot) == []


def test_external_image_snapshot_revalidation_fails_closed_after_change_or_missing(
    tmp_path: Path,
) -> None:
    data_dir, _, _ = _fixture_paths(tmp_path)
    image = tmp_path / "external.png"
    image.write_bytes(b"before")
    _write_parquet(data_dir, ["11"], [image])
    snapshot, errors = _external_snapshots(data_dir)
    assert errors == []

    image.write_bytes(b"after")
    assert "external image snapshot does not match current files" in (
        preflight_module.verify_external_image_snapshots(snapshot)
    )
    image.unlink()
    replacement = tmp_path / "replacement.png"
    replacement.write_bytes(b"before")
    image.symlink_to(replacement)
    verification_errors = preflight_module.verify_external_image_snapshots(snapshot)
    assert any("symlink" in error for error in verification_errors)
    image.unlink()
    verification_errors = preflight_module.verify_external_image_snapshots(snapshot)
    assert any("missing" in error for error in verification_errors)


def test_external_image_snapshot_rejects_symlink_and_missing_paths(tmp_path: Path) -> None:
    data_dir, _, _ = _fixture_paths(tmp_path)
    target = tmp_path / "target.png"
    target.write_bytes(b"target")
    symlink = tmp_path / "linked.png"
    symlink.symlink_to(target)
    _write_parquet(data_dir, ["11", "12"], [symlink, tmp_path / "missing.png"])

    snapshot, errors = _external_snapshots(data_dir)

    assert snapshot["train"]["file_count"] == 0
    assert any("symlink" in error for error in errors)
    assert any("missing or not a regular file" in error for error in errors)


def test_formal_report_seals_external_image_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "train.parquet"
    validation = tmp_path / "validation.parquet"
    forbidden = tmp_path / "forbidden.json"
    metadata = tmp_path / "metadata.json"
    masks = tmp_path / "masks"
    masks.mkdir()
    for path, payload in (
        (data, b"train"), (validation, b"validation"),
        (forbidden, b"[]"), (metadata, b"[]"), (masks / "mask.png", b"mask"),
    ):
        path.write_bytes(payload)
    snapshot, snapshot_errors = preflight_module._external_image_snapshot_from_paths({
        "train": set(), "validation": set(), "forbidden": set(),
    })
    assert snapshot_errors == []
    monkeypatch.setattr(
        preflight_module, "_collect_external_image_snapshots",
        lambda *args, **kwargs: (deepcopy(snapshot), []),
    )
    monkeypatch.setattr(preflight_module, "validate", lambda **kwargs: {"ok": True, "errors": []})
    monkeypatch.setattr(preflight_module, "_formal_contract", lambda *args, **kwargs: {"errors": []})
    monkeypatch.setattr(preflight_module, "_print_report", lambda report: None)
    fake_pyarrow = types.ModuleType("pyarrow")
    fake_pyarrow.__path__ = []
    fake_parquet = types.ModuleType("pyarrow.parquet")
    fake_pyarrow.parquet = fake_parquet
    monkeypatch.setitem(sys.modules, "pyarrow", fake_pyarrow)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", fake_parquet)
    report_path = tmp_path / "report.json"

    result = preflight_module.main([
        str(data), "--metadata", str(metadata), "--masks-dir", str(masks),
        "--run-class", "formal", "--data-mode", "frontier_rl",
        "--expected-ratios", "0.4,0.1,0.2,0.2,0.1", "--require-val-all-buckets",
        "--val-data", str(validation), "--forbidden-data", str(forbidden),
        "--expected-data-sha256", _tree_hash(data),
        "--expected-metadata-sha256", _tree_hash(metadata),
        "--expected-masks-sha256", _tree_hash(masks),
        "--expected-val-sha256", _tree_hash(validation),
        "--expected-forbidden-sha256", _tree_hash(forbidden),
        "--loader-filter-num-proc", "64", "--json-out", str(report_path),
    ])

    assert result == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["formal_contract"]["external_image_snapshots"] == snapshot


def test_preflight_success_and_json_report(tmp_path: Path) -> None:
    data_dir, metadata, masks_dir = _fixture_paths(tmp_path)
    image = tmp_path / "image.jpg"
    image.write_bytes(b"not-decoded-by-preflight")
    _write_parquet(data_dir, ["2", "11", "21", "31", "41"], [image] * 5)
    report_path = tmp_path / "report.json"

    result = _run(
        data_dir,
        metadata,
        masks_dir,
        "--expected-ratios",
        UNIFORM_RATIOS,
        "--tolerance",
        "0",
        "--json-out",
        str(report_path),
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["sample_count"] == 5
    assert report["image_path_existence_rate"] == 1.0
    assert report["image_payload_available_rate"] == 1.0
    assert report["metadata_coverage"] == "not_checked"
    assert [report["answer_buckets"][label]["count"] for label in ("2-10", "11-20", "21-30", "31-40", "41-50")] == [1] * 5


def test_canary_requires_disk_images_to_decode(tmp_path: Path) -> None:
    data_dir, metadata, masks_dir = _fixture_paths(tmp_path)
    image = tmp_path / "corrupt.png"
    image.write_bytes(b"not an image")
    _write_parquet(data_dir, ["11"], [image])
    result = _run(data_dir, metadata, masks_dir, "--run-class", "canary")
    assert result.returncode != 0
    assert "neither embedded bytes nor an existing path" in result.stderr


def test_preflight_fails_ratio_check(tmp_path: Path) -> None:
    data_dir, metadata, masks_dir = _fixture_paths(tmp_path)
    image = tmp_path / "image.jpg"
    image.touch()
    _write_parquet(data_dir, ["2"] * 5, [image] * 5)

    result = _run(data_dir, metadata, masks_dir, "--expected-ratios", UNIFORM_RATIOS, "--tolerance", "0.05")

    assert result.returncode == 1
    assert "answer buckets exceed ratio tolerance" in result.stderr


def test_preflight_accepts_embedded_image_with_identifier_only(tmp_path: Path) -> None:
    data_dir, metadata, masks_dir = _fixture_paths(tmp_path)
    _write_parquet(
        data_dir,
        ["11"],
        [Path("sequence-id.jpg")],
        embedded=True,
    )
    report_path = tmp_path / "report.json"

    result = _run(data_dir, metadata, masks_dir, "--json-out", str(report_path))

    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["embedded_image_count"] == 1
    assert report["existing_image_path_count"] == 0
    assert report["image_payload_available_rate"] == 1.0


def test_preflight_rejects_corrupt_embedded_image(tmp_path: Path) -> None:
    data_dir, metadata, masks_dir = _fixture_paths(tmp_path)
    _write_parquet(
        data_dir,
        ["11"],
        [Path("sequence-id.jpg")],
        embedded=True,
        corrupt_embedded=True,
    )

    result = _run(data_dir, metadata, masks_dir)

    assert result.returncode == 1
    assert "embedded images are not decodable" in result.stderr


def test_preflight_rejects_train_val_identity_overlap(tmp_path: Path) -> None:
    data_dir, metadata, masks_dir = _fixture_paths(tmp_path)
    val_dir = tmp_path / "val"
    val_dir.mkdir()
    train_image = tmp_path / "same-sequence_1.jpg"
    val_image = tmp_path / "same-sequence_9.jpg"
    train_image.touch()
    val_image.touch()
    _write_parquet(data_dir, ["11"], [train_image])
    _write_parquet(val_dir, ["11"], [val_image])

    result = _run(data_dir, metadata, masks_dir, "--val-data", str(val_dir))

    assert result.returncode == 1
    assert "train/validation identity overlap detected" in result.stderr


def test_preflight_includes_prompt_id_in_leakage_identity(tmp_path: Path) -> None:
    data_dir, metadata, masks_dir = _fixture_paths(tmp_path)
    val_dir = tmp_path / "val"
    val_dir.mkdir()
    train_image = tmp_path / "train.png"
    val_image = tmp_path / "val.png"
    train_image.touch()
    val_image.touch()
    _write_parquet(data_dir, ["11"], [train_image], prompt_ids=["shared-prompt"])
    _write_parquet(val_dir, ["12"], [val_image], prompt_ids=["shared-prompt"])
    result = _run(data_dir, metadata, masks_dir, "--val-data", str(val_dir))
    assert result.returncode == 1
    assert "train/validation identity overlap detected" in result.stderr


def test_preflight_rejects_renamed_benchmark_image_by_content(tmp_path: Path) -> None:
    data_dir, metadata, masks_dir = _fixture_paths(tmp_path)
    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    _write_parquet(data_dir, ["11"], [Path("renamed-train.jpg")], embedded=True)
    _write_parquet(benchmark_dir, ["12"], [Path("different-benchmark-name.jpg")], embedded=True)

    result = _run(
        data_dir,
        metadata,
        masks_dir,
        "--forbidden-data",
        str(benchmark_dir),
    )

    assert result.returncode == 1
    assert "train/benchmark identity overlap detected" in result.stderr


def test_preflight_checks_json_benchmark_image_content(tmp_path: Path) -> None:
    data_dir, metadata, masks_dir = _fixture_paths(tmp_path)
    train_image = tmp_path / "renamed-train.jpg"
    benchmark_image = tmp_path / "benchmark-original.jpg"
    train_image.write_bytes(b"same-image-content")
    benchmark_image.write_bytes(b"same-image-content")
    _write_parquet(data_dir, ["11"], [train_image])
    benchmark_json = tmp_path / "benchmark.json"
    benchmark_json.write_text(
        json.dumps([{"id": "benchmark-1", "image_path": str(benchmark_image)}]),
        encoding="utf-8",
    )

    result = _run(
        data_dir,
        metadata,
        masks_dir,
        "--forbidden-data",
        str(benchmark_json),
    )

    assert result.returncode == 1
    assert "train/benchmark identity overlap detected" in result.stderr


def test_preflight_requires_val_image_path_for_mask_reward(tmp_path: Path) -> None:
    data_dir, metadata, masks_dir = _fixture_paths(tmp_path)
    val_dir = tmp_path / "val"
    val_dir.mkdir()
    train_image = tmp_path / "train-image.jpg"
    train_image.touch()
    _write_parquet(data_dir, ["11"], [train_image])
    _write_parquet(val_dir, ["12"], [None], embedded=True)

    result = _run(data_dir, metadata, masks_dir, "--val-data", str(val_dir))

    assert result.returncode == 1
    assert "validation samples have no image path for mask-aware reward" in result.stderr


def test_preflight_validates_val_answer_and_image_payload(tmp_path: Path) -> None:
    data_dir, metadata, masks_dir = _fixture_paths(tmp_path)
    val_dir = tmp_path / "val"
    val_dir.mkdir()
    train_image = tmp_path / "train-image.jpg"
    train_image.touch()
    _write_parquet(data_dir, ["11"], [train_image])
    _write_parquet(
        val_dir,
        ["not-an-answer"],
        [Path("corrupt-val.jpg")],
        embedded=True,
        corrupt_embedded=True,
    )

    result = _run(data_dir, metadata, masks_dir, "--val-data", str(val_dir))

    assert result.returncode == 1
    assert "validation samples have an invalid answer" in result.stderr
    assert "validation samples have no decodable image payload" in result.stderr


def test_preflight_can_require_all_val_buckets(tmp_path: Path) -> None:
    data_dir, metadata, masks_dir = _fixture_paths(tmp_path)
    val_dir = tmp_path / "val"
    val_dir.mkdir()
    train_image = tmp_path / "train-image.jpg"
    val_image = tmp_path / "val-image.jpg"
    train_image.touch()
    val_image.touch()
    _write_parquet(data_dir, ["11"], [train_image])
    _write_parquet(val_dir, ["11"], [val_image])

    result = _run(
        data_dir,
        metadata,
        masks_dir,
        "--val-data",
        str(val_dir),
        "--require-val-all-buckets",
    )

    assert result.returncode == 1
    assert "validation data must cover every answer bucket" in result.stderr


def test_preflight_fails_when_image_is_missing(tmp_path: Path) -> None:
    data_dir, metadata, masks_dir = _fixture_paths(tmp_path)
    present = tmp_path / "present.jpg"
    present.touch()
    missing = tmp_path / "missing.jpg"
    _write_parquet(data_dir, ["2", "11"], [present, missing])

    result = _run(data_dir, metadata, masks_dir)

    assert result.returncode == 1
    assert "neither embedded bytes nor an existing path" in result.stderr


def test_preflight_fails_on_invalid_answer(tmp_path: Path) -> None:
    data_dir, metadata, masks_dir = _fixture_paths(tmp_path)
    image = tmp_path / "image.jpg"
    image.touch()
    _write_parquet(data_dir, ["2", "not-an-answer"], [image, image])

    result = _run(data_dir, metadata, masks_dir)

    assert result.returncode == 1
    assert "invalid or out-of-range answer" in result.stderr


def test_formal_frontier_and_rft_schema_are_mutually_exclusive(tmp_path: Path) -> None:
    data_dir, metadata, masks_dir = _fixture_paths(tmp_path); image = tmp_path / "image.png"
    _write_parquet(data_dir, ["11"], [image]); filtered, coverage = _formal_files(tmp_path, data_dir, metadata, masks_dir, image)
    # Add a response column, which is forbidden for frontier RL.
    subprocess.run([PYARROW_PYTHON, "-c", "import pyarrow.parquet as pq,pyarrow as pa,sys; p=sys.argv[1];t=pq.read_table(p);pq.write_table(t.append_column('response',pa.array(['winner'])),p)", str(data_dir / "train.parquet")], check=True)
    required = _formal_required_args(data_dir, metadata, masks_dir)
    result = _run(data_dir, metadata, masks_dir, "--run-class", "formal", "--data-mode", "frontier_rl", "--entrypoint-contract", "frontier_rl", "--filtered-manifest", str(filtered), "--coverage-report", str(coverage), *required)
    assert result.returncode == 1 and "forbids response/trajectory" in result.stderr
    result = _run(data_dir, metadata, masks_dir, "--run-class", "formal", "--data-mode", "strict_winner_rft", "--entrypoint-contract", "strict_winner_rft", "--training-entrypoint", "strict_winner_rft.py", "--filtered-manifest", str(filtered), "--coverage-report", str(coverage), *required)
    assert "strict_winner_rft requires a response" not in result.stderr


def test_formal_rejects_missing_or_corrupt_mask_and_coverage_hash(tmp_path: Path) -> None:
    data_dir, metadata, masks_dir = _fixture_paths(tmp_path); image = tmp_path / "image.png"
    _write_parquet(data_dir, ["11"], [image]); filtered, coverage = _formal_files(tmp_path, data_dir, metadata, masks_dir, image)
    (masks_dir / "mask.png").write_bytes(b"corrupt")
    metadata.write_text(json.dumps([{"image": image.name, "mask": "mask.png"}, {"image": image.name, "mask": "other.png"}]))
    payload = json.loads(coverage.read_text()); payload["dataset_sha256"] = "wrong"; coverage.write_text(json.dumps(payload))
    required = _formal_required_args(data_dir, metadata, masks_dir)
    result = _run(data_dir, metadata, masks_dir, "--run-class", "formal", "--data-mode", "frontier_rl", "--entrypoint-contract", "frontier_rl", "--filtered-manifest", str(filtered), "--coverage-report", str(coverage), *required)
    assert result.returncode == 1
    assert "coverage report dataset_sha256 mismatch" in result.stderr
    assert "mask is not decodable" in result.stderr
    assert "basename ambiguity" in result.stderr


def test_formal_rejects_custom_ratio_contract(tmp_path: Path) -> None:
    data_dir, metadata, masks_dir = _fixture_paths(tmp_path); image = tmp_path / "image.png"
    _write_parquet(data_dir, ["11"], [image]); filtered, coverage = _formal_files(tmp_path, data_dir, metadata, masks_dir, image)
    required = _formal_required_args(data_dir, metadata, masks_dir)
    required[1] = "0.2,0.2,0.2,0.2,0.2"
    result = _run(data_dir, metadata, masks_dir, "--run-class", "formal", "--data-mode", "frontier_rl", "--entrypoint-contract", "frontier_rl", "--filtered-manifest", str(filtered), "--coverage-report", str(coverage), *required)
    assert result.returncode == 1
    assert "formal expected ratios are locked" in result.stderr


def test_formal_locks_ratio_tolerance_and_phash_threshold(tmp_path: Path) -> None:
    data_dir, metadata, masks_dir = _fixture_paths(tmp_path)
    result = _run(data_dir, metadata, masks_dir, "--run-class", "formal", "--tolerance", "0.1")
    assert result.returncode == 2 and "ratio tolerance is locked" in result.stderr
    result = _run(data_dir, metadata, masks_dir, "--run-class", "formal", "--phash-hamming-threshold", "5")
    assert result.returncode == 2 and "pHash Hamming threshold is locked" in result.stderr


def test_formal_rejects_near_duplicate_and_unreadable_forbidden_image(tmp_path: Path) -> None:
    data_dir, metadata, masks_dir = _fixture_paths(tmp_path)
    train_image = tmp_path / "train.png"
    val_image = tmp_path / "val-near.png"
    subprocess.run([
        PYARROW_PYTHON, "-c",
        "from PIL import Image; import sys; a=Image.new('RGB',(32,32),'red');a.save(sys.argv[1]);a.putpixel((0,0),(254,0,0));a.save(sys.argv[2])",
        str(train_image), str(val_image),
    ], check=True)
    _write_parquet(data_dir, ["11"], [train_image], prompt_ids=["train-prompt"])
    filtered, coverage = _formal_files(tmp_path, data_dir, metadata, masks_dir, train_image)
    val_dir = tmp_path / "val"
    val_dir.mkdir()
    _write_parquet(val_dir, ["12"], [val_image], prompt_ids=["val-prompt"])
    forbidden = tmp_path / "forbidden.json"
    forbidden.write_text(json.dumps([{"prompt_id": "benchmark-prompt", "image_path": str(tmp_path / "missing.png")}]))
    result = _run(
        data_dir, metadata, masks_dir,
        "--run-class", "formal", "--data-mode", "frontier_rl", "--entrypoint-contract", "frontier_rl",
        "--filtered-manifest", str(filtered), "--coverage-report", str(coverage),
        "--expected-ratios", "0.4,0.1,0.2,0.2,0.1", "--require-val-all-buckets",
        "--val-data", str(val_dir), "--forbidden-data", str(forbidden),
        "--expected-data-sha256", _tree_hash(data_dir),
        "--expected-metadata-sha256", _tree_hash(metadata),
        "--expected-masks-sha256", _tree_hash(masks_dir),
        "--expected-val-sha256", _tree_hash(val_dir),
        "--expected-forbidden-sha256", _tree_hash(forbidden),
        "--loader-filter-num-proc", "64",
    )
    assert result.returncode == 1
    assert "train/validation perceptual near-duplicate overlap detected" in result.stderr
    assert "formal forbidden benchmark unreadable image" in result.stderr
