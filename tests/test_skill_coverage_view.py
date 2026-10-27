"""The reverse coverage view stays reconciled with the registry it indexes."""

import json
from pathlib import Path

from tools import build_skill_coverage as coverage

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / coverage.OUTPUT


def test_view_lists_every_registered_skill_exactly_once():
    registry = json.loads((PROJECT / coverage.INDEX).read_text(encoding="utf-8"))["skills"]
    rows, unregistered, documents, dead, _ = coverage.collect(PROJECT)
    assert [row["id"] for row in rows] == sorted(registry)
    assert len({row["id"] for row in rows}) == len(rows)
    # Unregistered documents are listed separately and never enter the count.
    assert not set(unregistered) & set(registry)
    assert len(rows) + len(unregistered) == documents
    assert dead == []


def test_view_rows_copy_owner_state_without_promotion():
    rows, _, _, _, _ = coverage.collect(PROJECT)
    for row in rows:
        assert row["task"]
        assert isinstance(row["layers"], dict)
        assert isinstance(row["rm"], list)
        if not row["sidecar"]:
            assert row["status"] == "unknown"
            continue
        assert row["status"] == "candidate"
        assert row["validation"] == "not_run"


def test_layer_stages_are_unioned_not_repeated():
    rows, _, _, _, _ = coverage.collect(PROJECT)
    for row in rows:
        for stages in row["layers"].values():
            assert len(stages) == len(set(stages))


def test_regeneration_is_byte_identical():
    first = coverage.collect(PROJECT)
    second = coverage.collect(PROJECT)
    assert first[-1] == second[-1], "input digest is not stable"
    assert coverage.render(*first) == coverage.render(*second)


def test_committed_view_matches_its_sources():
    rendered = coverage.render(*coverage.collect(PROJECT))
    assert OUTPUT.read_text(encoding="utf-8") == rendered, (
        "docs/skill-coverage.md is stale; rerun tools/build_skill_coverage.py"
    )


def test_input_digest_tracks_sources_not_git_state():
    rows, unregistered, documents, dead, digest = coverage.collect(PROJECT)
    rendered = coverage.render(rows, unregistered, documents, dead, digest)
    assert digest in rendered
    assert coverage.render(rows, unregistered, documents, dead, "0" * 64) != rendered
