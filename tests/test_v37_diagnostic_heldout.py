import hashlib
import json
import inspect
from pathlib import Path

import pytest


pytest.importorskip("pyarrow")
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from tools import build_v37_diagnostic_heldout as builder  # noqa: E402


def test_fixed_500_quota_contract():
    assert builder.LABELS == ("2-10", "11-20", "21-30", "31-40", "41-50")
    assert builder.QUOTAS == {
        "2-10": 200, "11-20": 50, "21-30": 100, "31-40": 100, "41-50": 50,
    }
    assert sum(builder.QUOTAS.values()) == 500
    assert 1 <= builder.PARQUET_BATCH_SIZE <= 16


def test_large_row_groups_are_never_materialized_by_builder():
    source = "\n".join((
        inspect.getsource(builder.benchmark_rows),
        inspect.getsource(builder.scan_sources),
        inspect.getsource(builder.write_selected),
    ))
    assert "iter_batches" in source
    assert "read_row_group" not in source


def test_embedded_bytes_are_exact_evidence_without_resolvable_path(tmp_path: Path):
    payload = b"embedded-image"
    keys = builder.image_keys(
        [{"bytes": payload, "path": "missing-image.jpg"}],
        [tmp_path],
        require_exact=True,
    )
    assert f"sha256:{hashlib.sha256(payload).hexdigest()}" in keys
    with pytest.raises(builder.BuildError, match="neither embedded bytes"):
        builder.image_keys([{"bytes": None, "path": "missing.jpg"}], [tmp_path], require_exact=True)


def test_write_selected_streams_and_normalizes_cross_source_schema_metadata(tmp_path: Path):
    image_type = pa.list_(pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())]))
    schema = pa.schema([
        pa.field("images", image_type), pa.field("problem", pa.string()), pa.field("answer", pa.string()),
    ])
    paths = []
    for index, metadata in enumerate(({b"huggingface": b"mock"}, None)):
        path = tmp_path / f"source-{index}.parquet"
        table = pa.Table.from_arrays([
            pa.array([[{"bytes": f"image-{index}".encode(), "path": f"{index}.jpg"}]], type=image_type),
            pa.array(["count"]),
            pa.array([str(2 + index)]),
        ], schema=schema).replace_schema_metadata(metadata)
        pq.write_table(table, path)
        paths.append(path)
    selected = [
        builder.Candidate(index, str(index), path, path.name, 0, 0, 0, 2 + index, "2-10", index, ())
        for index, path in enumerate(paths)
    ]
    output = tmp_path / "heldout.parquet"
    builder.write_selected(selected, output)
    result = pq.read_table(output)
    assert result.num_rows == 2
    assert result.schema.metadata is None


def test_focused_manifest_exclusion_uses_source_file_and_local_idx(tmp_path: Path):
    manifest = tmp_path / "selection_manifest.json"
    manifest.write_text(json.dumps({"selected_indices": [
        {"source_file": "nested/part-1.parquet", "local_idx": 7},
        {"source_file": "part-2.parquet", "local_idx": 11},
    ]}), encoding="utf-8")
    assert builder.focused_exclusions(manifest) == {
        ("part-1.parquet", 7), ("part-2.parquet", 11),
    }


def test_builder_rejects_existing_output(tmp_path: Path):
    output = tmp_path / "heldout"
    output.mkdir()
    args = builder.parser().parse_args([
        "--benchmark", str(tmp_path / "benchmark.json"),
        "--output-dir", str(output),
    ])
    with pytest.raises(builder.BuildError, match="overwrite"):
        builder.build(args)
