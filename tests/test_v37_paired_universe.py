import hashlib
import json
from pathlib import Path

import pytest

from tools import v37_gate as gate
from tools import v37_paired_universe as universe


def test_identity_and_answer_parsing_are_unambiguous() -> None:
    assert universe._identifier("sample-1", "sample") == "sample-1"
    assert universe._answer("+005", "answer") == 5
    for value in ("", " sample", 7, True):
        with pytest.raises(universe.UniverseError):
            universe._identifier(value, "sample")
    for value in ("5 objects", "5.0", 5.0, True, 1, 51):
        with pytest.raises(universe.UniverseError):
            universe._answer(value, "answer")


def test_build_binds_sorted_rows_data_hash_and_all_buckets(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    data = tmp_path / "heldout.parquet"
    answers = [41, 2, 31, 11, 21]
    pq.write_table(pa.table({
        "sample_id": [f"s-{answer}" for answer in answers],
        "prompt_id": [f"p-{answer}" for answer in answers],
        "answer": answers,
    }), data)
    output = tmp_path / "paired-universe.json"
    payload = universe.build(data=data, output=output)
    assert output.is_file()
    assert payload["sample_count"] == 5
    assert payload["bucket_counts"] == {bucket: 1 for bucket in gate.BUCKETS}
    assert [row["sample_id"] for row in payload["rows"]] == sorted(
        row["sample_id"] for row in payload["rows"]
    )
    assert payload["data_sha256"] == gate.content_snapshot(data, parquet_only=True)["sha256"]
    assert payload["ordered_rows_sha256"] == hashlib.sha256(
        universe._canonical(payload["rows"])
    ).hexdigest()
    assert json.loads(output.read_text()) == payload
    with pytest.raises(universe.UniverseError, match="already exists"):
        universe.build(data=data, output=output)


def test_dangling_output_symlink_is_rejected_before_pyarrow(tmp_path: Path) -> None:
    output = tmp_path / "paired-universe.json"
    output.symlink_to(tmp_path / "missing")
    with pytest.raises(universe.UniverseError, match="already exists"):
        universe.build(data=tmp_path / "missing.parquet", output=output)


def test_gate_recomputes_ids_and_gt_from_authoritative_parquet_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = tmp_path / "heldout.parquet"
    data.write_bytes(b"synthetic parquet snapshot")
    snapshot = gate.content_snapshot(data, parquet_only=True)
    authoritative = []
    for index, bucket in enumerate(gate.BUCKETS):
        ground_truth = gate.BUCKET_RANGES[bucket][0]
        projection = {
            "sample_id": f"s-{index}", "prompt_id": f"p-{index}",
            "bucket": bucket, "ground_truth": ground_truth,
        }
        authoritative.append({
            **projection,
            "source_row_sha256": hashlib.sha256(
                gate.eval_producer._canonical_bytes(projection)
            ).hexdigest(),
        })
    forged = json.loads(json.dumps(authoritative))
    forged[1]["ground_truth"] += 1
    forged_projection = {
        key: forged[1][key] for key in ("sample_id", "prompt_id", "bucket", "ground_truth")
    }
    forged[1]["source_row_sha256"] = hashlib.sha256(
        gate.eval_producer._canonical_bytes(forged_projection)
    ).hexdigest()
    artifact = tmp_path / "universe.json"
    payload = {
        "schema_version": gate.PAIRED_UNIVERSE_SCHEMA_VERSION,
        "artifact_type": gate.PAIRED_UNIVERSE_ARTIFACT_TYPE,
        "data_path": str(data.resolve()),
        "data_sha256": snapshot["sha256"],
        "sample_count": len(forged),
        "bucket_counts": {bucket: 1 for bucket in gate.BUCKETS},
        "ordered_rows_sha256": hashlib.sha256(gate.canonical(forged).encode()).hexdigest(),
        "rows": forged,
    }
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        gate, "_recompute_paired_source_rows", lambda _: json.loads(json.dumps(authoritative)),
    )
    with pytest.raises(gate.GateError, match="differ from authoritative parquet"):
        gate._load_paired_sample_universe(
            {"path": str(artifact.resolve()), "sha256": gate.sha256(artifact)},
            [{"suite": "heldout", "dataset": snapshot}],
            "test",
            verify_source_rows=True,
        )
