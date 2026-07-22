import json
import importlib.util
from pathlib import Path

import pytest

from tools import preflight_v37_training as preflight


REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("v37_path_remap_test", REPO / "verl/utils/path_remap.py")
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
parse_image_path_remap = _MODULE.parse_image_path_remap
remap_image_path = _MODULE.remap_image_path
remap_image_payload = _MODULE.remap_image_payload


def test_path_remap_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("STEPCOUNT_IMAGE_PATH_REMAP_JSON", raising=False)
    original = "/old/images/a.jpg"
    assert remap_image_path(original) == original


def test_longest_prefix_and_path_boundary_are_deterministic(monkeypatch):
    monkeypatch.setenv(
        "STEPCOUNT_IMAGE_PATH_REMAP_JSON",
        json.dumps({"/old": "/new", "/old/images": "/fast/images"}),
    )
    assert remap_image_path("/old/images/a.jpg") == "/fast/images/a.jpg"
    assert remap_image_path("/old/other/a.jpg") == "/new/other/a.jpg"
    assert remap_image_path("/older/images/a.jpg") == "/older/images/a.jpg"
    assert remap_image_path("relative/a.jpg") == "relative/a.jpg"


def test_payload_remap_does_not_mutate_hf_image_dict(monkeypatch):
    monkeypatch.setenv("STEPCOUNT_IMAGE_PATH_REMAP_JSON", '{"/old":"/new"}')
    source = {"bytes": None, "path": "/old/a.jpg"}
    remapped = remap_image_payload(source)
    assert source["path"] == "/old/a.jpg"
    assert remapped == {"bytes": None, "path": "/new/a.jpg"}


def test_chained_and_identity_mappings_are_rejected():
    with pytest.raises(ValueError, match="chained remaps are forbidden"):
        parse_image_path_remap('{"/old":"/new","/new":"/final"}')
    with pytest.raises(ValueError, match="identity mappings"):
        parse_image_path_remap('{"/same":"/same"}')


def test_formal_rejects_unsealed_path_expansion(monkeypatch):
    monkeypatch.setenv("V37_RUN_CLASS", "formal")
    with pytest.raises(ValueError, match="cannot contain"):
        remap_image_path("$DATA/image.png", ())
    with pytest.raises(ValueError, match="cannot contain"):
        remap_image_path("~/image.png", ())


def test_formal_rejects_relative_external_path_but_not_embedded_bytes(monkeypatch):
    monkeypatch.setenv("V37_RUN_CLASS", "formal")
    mappings = parse_image_path_remap('{"/old":"/new"}')
    with pytest.raises(ValueError, match="must be absolute"):
        remap_image_path("images/a.png", mappings)
    payload = {"path": "images/a.png", "bytes": b"authoritative"}
    assert remap_image_payload(payload, mappings) == payload


@pytest.mark.parametrize(
    "raw",
    [
        "{}",
        "[]",
        '{"relative":"/new"}',
        '{"/old":"relative"}',
        '{"/":"/new"}',
        '{"/old":"/new","/old":"/other"}',
        '{"/old":NaN}',
    ],
)
def test_invalid_mapping_fails_closed(raw):
    with pytest.raises(ValueError):
        parse_image_path_remap(raw)


def test_loader_and_preflight_share_one_remap_implementation():
    dataset_source = (REPO / "verl/utils/dataset.py").read_text(encoding="utf-8")
    preflight_source = (REPO / "tools/preflight_v37_training.py").read_text(encoding="utf-8")
    assert "from .path_remap import remap_image_payload" in dataset_source
    assert '"v37_path_remap"' in preflight_source
    assert "remap_image_path(raw_path, parse_image_path_remap())" in preflight_source
    process_body = dataset_source.split("def process_image", 1)[1].split("def process_video", 1)[0]
    assert "remap_image_payload(image)" not in process_body


def test_preflight_resolves_the_same_remapped_file(monkeypatch, tmp_path):
    target_root = tmp_path / "new-images"
    target_root.mkdir()
    target = target_root / "a.jpg"
    target.write_bytes(b"image-bytes")
    monkeypatch.setenv(
        "STEPCOUNT_IMAGE_PATH_REMAP_JSON",
        json.dumps({"/old/images": str(target_root)}),
    )
    resolved = preflight._resolve_external_image_path("/old/images/a.jpg", (tmp_path,))
    assert resolved == target.resolve()
    assert preflight._resolve_existing_path("/old/images/a.jpg", (tmp_path,)) == target.resolve()
    assert f"image_sha256:{preflight._file_sha256(str(target.resolve()))}" in (
        preflight._identity_keys("/old/images/a.jpg", (tmp_path,))
    )
