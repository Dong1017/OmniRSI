from __future__ import annotations

import json
import re
from pathlib import Path
from typing import BinaryIO

import yaml


def _frontmatter(stream: BinaryIO) -> dict:
    if stream.readline().strip() != b"---":
        raise ValueError("SKILL.md must start with YAML frontmatter")
    lines = []
    for line in stream:
        if line.strip() == b"---":
            break
        lines.append(line)
    else:
        raise ValueError("Unclosed SKILL.md frontmatter")
    try:
        record = yaml.safe_load(b"".join(lines).decode("utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid skill YAML: {exc}") from exc
    if not isinstance(record, dict):
        raise ValueError("Skill frontmatter must be a mapping")
    return record


def _directory(root: Path, name: str) -> Path:
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
        raise ValueError(f"Invalid skill name: {name}")
    directory = (root / name).resolve()
    if not directory.is_relative_to(root.resolve()):
        raise ValueError("Skill must be inside the selected root")
    if not (directory / "SKILL.md").is_file():
        raise ValueError(f"Unknown skill: {name}")
    return directory


def _metadata(directory: Path) -> dict:
    # 二进制逐行读取，只解析 frontmatter，不解码正文。
    with (directory / "SKILL.md").open("rb") as stream:
        record = _frontmatter(stream)
    for key in ("name", "description"):
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise ValueError(f"Skill requires {key}: {directory.name}")
    if record["name"] != directory.name:
        raise ValueError(f"Skill name disagrees with directory: {directory.name}")
    sidecar = directory / "metadata.json"
    if sidecar.is_file():
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError(f"Skill metadata must be a mapping: {directory.name}")
        for key in record.keys() & metadata.keys():
            if record[key] != metadata[key]:
                raise ValueError(f"Conflicting skill metadata field {key}: {directory.name}")
        if metadata.get("profile") == "local-skill-v1":
            if any(key in metadata for key in ("name", "description")):
                raise ValueError("Local skill name and description belong in SKILL.md")
            if set(record) != {"name", "description"}:
                raise ValueError("Local skill structured fields belong in metadata.json")
        record.update(metadata)
    if "status" in record and not isinstance(record["status"], str):
        raise ValueError(f"Skill status must be a string: {directory.name}")
    if "tags" in record and (
        not isinstance(record["tags"], list)
        or any(not isinstance(tag, str) for tag in record["tags"])
    ):
        raise ValueError(f"Skill tags must be a string list: {directory.name}")
    return {key: record[key] for key in ("name", "description", "status", "tags") if key in record}


def skill_records(root: Path) -> list[dict]:
    """Project metadata only; this projection is not an authoritative matrix."""
    if not root.is_dir():
        raise ValueError(f"Skill root is not a directory: {root}")
    return [
        _metadata(_directory(root, path.parent.name))
        for path in sorted(root.glob("*/SKILL.md"))
    ]


def get_skill(root: Path, name: str) -> dict:
    """Read selected metadata and resource names, never resource contents."""
    directory = _directory(root, name)
    record = _metadata(directory)
    record["resources"] = [
        path.relative_to(directory).as_posix()
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "SKILL.md"
    ]
    return record


def skill_instructions(root: Path, name: str) -> str:
    """Load selected instructions. Reading does not authorize their commands."""
    directory = _directory(root, name)
    _metadata(directory)
    with (directory / "SKILL.md").open("rb") as stream:
        _frontmatter(stream)
        return stream.read().decode("utf-8").lstrip("\r\n")


def skill_resource(root: Path, name: str, resource: str) -> Path:
    """Resolve one explicitly selected resource inside its skill directory."""
    directory = _directory(root, name)
    path = (directory / resource).resolve()
    if not path.is_relative_to(directory) or path == directory / "SKILL.md":
        raise ValueError("Resource must be inside the selected skill; use show for instructions")
    if not path.is_file():
        raise ValueError(f"Unknown skill resource: {resource}")
    return path


def skill_reference(root: Path, name: str, reference: str) -> Path:
    """Resolve a declared checkout reference without escaping the skills root."""
    directory = _directory(root, name)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if reference not in metadata.get("references", []):
        raise ValueError(f"Undeclared skill reference: {reference}")
    if "\\" in reference or Path(reference).is_absolute():
        raise ValueError("Reference must use a relative POSIX path")
    path = (directory / reference).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Reference must stay inside the skills root")
    if not path.is_file():
        raise ValueError(f"Missing skill reference: {reference}")
    return path
