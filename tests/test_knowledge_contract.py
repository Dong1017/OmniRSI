import json
from pathlib import Path

import pytest

from omnirsi.skill_catalog import skill_records, skill_reference

ROOT = Path(__file__).resolve().parents[1] / "skills"


def test_local_profiles_preserve_core_and_structured_ownership():
    records = skill_records(ROOT)
    normalized = []
    for record in records:
        sidecar = ROOT / record["name"] / "metadata.json"
        if not sidecar.exists():
            continue
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if metadata.get("profile") != "local-skill-v1":
            continue
        normalized.append(record["name"])
        assert "name" not in metadata and "description" not in metadata
        assert metadata["status"] == record["status"]
        assert metadata["kind"] == "skill"
        for readset in metadata["provenance_readsets"]:
            assert (sidecar.parent / readset).is_file()
    assert len(normalized) == 43


def test_contradictory_legacy_metadata_fails(tmp_path):
    directory = tmp_path / "selected"
    directory.mkdir()
    (directory / "SKILL.md").write_text(
        "---\nname: selected\ndescription: test\nstatus: candidate\n---\nProcedure\n"
    )
    (directory / "metadata.json").write_text('{"status":"active"}')
    with pytest.raises(ValueError, match="Conflicting.*status"):
        skill_records(tmp_path)


def test_two_real_consumers_resolve_one_authoritative_case():
    reference = "../inference-optimization-casebook/cases/admission-drain-occupancy.md"
    first = skill_reference(ROOT, "inference-benchmark-design", reference)
    second = skill_reference(ROOT, "dense-performance-evidence", reference)
    assert first == second
    assert "## Observation" in first.read_text(encoding="utf-8")
    assert "## Interpretation" in first.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="Undeclared"):
        skill_reference(ROOT, "inference-benchmark-design", "../../README.md")


@pytest.mark.parametrize("reference", ["../../outside.md", "missing.md"])
def test_declared_reference_escape_and_missing_dependency_fail(tmp_path, reference):
    directory = tmp_path / "selected"
    directory.mkdir()
    (directory / "SKILL.md").write_text("---\nname: selected\ndescription: test\n---\n")
    (directory / "metadata.json").write_text(json.dumps({"references": [reference]}))
    with pytest.raises(ValueError, match="inside|Missing"):
        skill_reference(tmp_path, "selected", reference)
