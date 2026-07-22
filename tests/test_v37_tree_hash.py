import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gate = _load("v37_tree_gate", "tools/v37_gate.py")
preflight = _load("v37_tree_preflight", "tools/preflight_v37_training.py")
frontier = _load("v37_tree_frontier", "tools/build_v37_frontier_dataset.py")
postprocess = _load("v37_tree_postprocess", "tools/v37_postprocess.py")
benchmark = _load("v37_tree_benchmark", "tools/v37_benchmark_evidence.py")
eval_producer = _load("v37_tree_eval_producer", "tools/v37_eval_producer.py")
continuation = _load("v37_tree_continuation", "tools/v37_continuation.py")
training = _load("v37_tree_training", "verl/utils/v37_training_evidence.py")


def _canonical_expected(root: Path) -> str:
    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        payload = path.read_bytes()
        entries.append({
            "path": path.relative_to(root).as_posix(),
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
    encoded = json.dumps(
        entries, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _all_hashes(root: Path) -> list[str]:
    return [
        gate.tree_sha256(root),
        preflight._tree_sha256(root),
        frontier.path_sha256(root),
        postprocess._tree_sha256(root),
        benchmark.tree_sha256(root),
        eval_producer._tree_sha256(root),
        continuation.tree_sha256(root),
        training.tree_sha256(root),
    ]


def test_all_v37_tree_hash_implementations_share_canonical_contract(tmp_path):
    root = tmp_path / "artifact"
    (root / "nested").mkdir(parents=True)
    (root / "alpha").write_bytes(b"a\x00b")
    (root / "nested" / "beta").write_bytes(b"payload")
    expected = _canonical_expected(root)
    assert _all_hashes(root) == [expected] * 8


def test_canonical_tree_hash_has_unambiguous_file_boundaries(tmp_path):
    one = tmp_path / "one"
    two = tmp_path / "two"
    one.mkdir()
    two.mkdir()
    (one / "a").write_bytes(b"b\x00c")
    (two / "a").write_bytes(b"")
    (two / "b").write_bytes(b"c")
    assert len(set(_all_hashes(one))) == 1
    assert len(set(_all_hashes(two))) == 1
    assert _all_hashes(one)[0] != _all_hashes(two)[0]


def test_benchmark_descriptor_names_the_v2_hash_contract():
    assert benchmark.TREE_HASH_ALGORITHM == "v37-canonical-tree-manifest-sha256-v2"


@pytest.mark.parametrize("dangling", [False, True])
def test_all_v37_tree_hash_implementations_reject_symlinked_tree_content(tmp_path, dangling):
    root = tmp_path / "artifact"
    root.mkdir()
    outside = tmp_path / "outside.bin"
    if not dangling:
        outside.write_bytes(b"outside")
    os.symlink(outside, root / "linked.bin")
    for function in (
        gate.tree_sha256, preflight._tree_sha256, frontier.path_sha256,
        postprocess._tree_sha256, benchmark.tree_sha256, eval_producer._tree_sha256,
        continuation.tree_sha256, training.tree_sha256,
    ):
        try:
            function(root)
        except (ValueError, frontier.ValidationError):
            pass
        else:
            raise AssertionError(f"{function.__module__} accepted symlinked tree content")
