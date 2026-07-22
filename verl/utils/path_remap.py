"""Strict, opt-in remapping for portable external image paths."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


IMAGE_PATH_REMAP_ENV = "STEPCOUNT_IMAGE_PATH_REMAP_JSON"


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"{IMAGE_PATH_REMAP_ENV} contains duplicate key {key!r}.")
        result[key] = value
    return result


def parse_image_path_remap(raw: str | None = None) -> tuple[tuple[str, str], ...]:
    """Parse canonical absolute-prefix mappings, longest source first."""
    value = os.environ.get(IMAGE_PATH_REMAP_ENV, "") if raw is None else raw
    if not value:
        return ()
    try:
        decoded = json.loads(
            value,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {item!r}")
            ),
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Invalid {IMAGE_PATH_REMAP_ENV}: {exc}") from exc
    if not isinstance(decoded, dict) or not decoded:
        raise ValueError(f"{IMAGE_PATH_REMAP_ENV} must be a nonempty JSON object.")

    mappings: list[tuple[str, str]] = []
    for source, target in decoded.items():
        if not isinstance(source, str) or not isinstance(target, str) or not source or not target:
            raise ValueError(f"{IMAGE_PATH_REMAP_ENV} keys and values must be nonempty strings.")
        source_path = Path(os.path.expanduser(source))
        target_path = Path(os.path.expanduser(target))
        if not source_path.is_absolute() or not target_path.is_absolute():
            raise ValueError(f"{IMAGE_PATH_REMAP_ENV} prefixes must be absolute paths.")
        source_text = os.path.normpath(str(source_path))
        target_text = os.path.normpath(str(target_path))
        if source_text == os.path.sep:
            raise ValueError(f"{IMAGE_PATH_REMAP_ENV} cannot remap the filesystem root.")
        if source_text == target_text:
            raise ValueError(f"{IMAGE_PATH_REMAP_ENV} cannot contain identity mappings.")
        mappings.append((source_text, target_text))
    ordered = tuple(sorted(mappings, key=lambda item: (-len(item[0]), item[0])))
    sources = tuple(source for source, _ in ordered)
    for _, target in ordered:
        for candidate in sources:
            if target == candidate or target.startswith(candidate + os.path.sep):
                raise ValueError(
                    f"{IMAGE_PATH_REMAP_ENV} target {target!r} is inside source {candidate!r}; "
                    "chained remaps are forbidden."
                )
    return ordered


def remap_image_path(path: str, mappings: tuple[tuple[str, str], ...] | None = None) -> str:
    """Apply one longest-boundary prefix mapping to an absolute image path."""
    if not isinstance(path, str) or not path:
        raise ValueError("image path must be a nonempty string")
    formal = os.environ.get("V37_RUN_CLASS") == "formal"
    if formal and ("$" in path or path.startswith("~")):
        raise ValueError("formal V37 image paths must be absolute and cannot contain ~ or $VAR")
    expanded = os.path.normpath(os.path.expanduser(os.path.expandvars(path)))
    if formal and not os.path.isabs(expanded):
        raise ValueError("formal V37 external image paths must be absolute")
    parsed = parse_image_path_remap() if mappings is None else mappings
    if not parsed:
        return path
    if not os.path.isabs(expanded):
        return path
    for source, target in parsed:
        if expanded == source:
            return target
        prefix = source + os.path.sep
        if expanded.startswith(prefix):
            return os.path.join(target, expanded[len(prefix):])
    return expanded


def remap_image_payload(
    image: Any, mappings: tuple[tuple[str, str], ...] | None = None
) -> Any:
    """Remap string or HF Image dict paths without mutating dataset rows."""
    if isinstance(image, str):
        return remap_image_path(image, mappings)
    if isinstance(image, Mapping):
        result = dict(image)
        # Embedded bytes are authoritative; the optional source path is not
        # consumed by the image loader and must not become a false dependency.
        if result.get("bytes") is not None:
            return result
        path = result.get("path")
        if isinstance(path, str) and path:
            result["path"] = remap_image_path(path, mappings)
        return result
    return image
