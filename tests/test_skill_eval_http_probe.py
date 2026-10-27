import copy
import io
import json
import re
import sys
from pathlib import Path
from urllib.error import HTTPError

import pytest
import yaml

from tools import skill_eval_http_probe as probe
from tools.skill_eval_pr import publication_http_error


class Fake:
    def __init__(self, failure=None):
        self.failure, self.calls, self.comments = failure, [], {}

    def request(self, method, route, data=None):
        self.calls.append((method, route, data))
        if route == "/issues/3":
            return {"number": 3, "state": "open", "locked": False}
        if route == "/pulls/4":
            return {"number": 4, "state": "open", "locked": False,
                    "head": {"sha": "a" * 40, "repo": {"full_name": probe.REPOSITORY}},
                    "base": {"sha": "b" * 40}}
        if method == "GET" and "?" in route:
            number = int(route.split("/")[2])
            return [copy.deepcopy(c) for c in self.comments.values()
                    if c["issue_url"].endswith(f"/{number}")]
        if method == "POST":
            number = int(route.split("/")[2])
            if ((self.failure == "403" and number == 3)
                    or (self.failure == "pr-403" and number == 4)):
                raise publication_http_error(HTTPError(
                    "https://example.invalid", 403, "private", {},
                    io.BytesIO(b'{"message":"Resource not accessible by integration"}')),
                    method, route)
            identifier = number * 100
            self.comments[identifier] = {
                "id": identifier, "body": data["body"],
                "issue_url": f"https://api.github.com/repos/{probe.REPOSITORY}/issues/{number}",
                "html_url": f"https://github.com/{probe.REPOSITORY}/issues/{number}#issuecomment-{identifier}",
                "user": {"login": "github-actions[bot]"}}
            if self.failure == "uncertain-post":
                raise OSError("FAKE_SECRET https://private.example.invalid")
            result = copy.deepcopy(self.comments[identifier])
            if self.failure == "wrong-owner":
                result["user"]["login"] = "example-other-user"
            return result
        identifier = int(route.rsplit("/", 1)[1])
        if method == "PATCH":
            self.comments[identifier]["body"] = data["body"]
            if self.failure == "uncertain-patch":
                raise OSError("FAKE_SECRET")
        if self.failure == "readback-failure" and method == "GET":
            raise OSError("FAKE_SECRET https://private.example.invalid")
        if self.failure == "readback-403" and method == "GET":
            raise publication_http_error(HTTPError(
                "https://example.invalid", 403, "private", {},
                io.BytesIO(b'{"message":"Resource not accessible by integration"}')),
                method, route)
        return copy.deepcopy(self.comments[identifier])


def context():
    return {"GITHUB_REPOSITORY": probe.REPOSITORY, "GITHUB_EVENT_NAME": "push",
            "GITHUB_REF": probe.BRANCH, "GITHUB_RUN_ATTEMPT": "1",
            "GITHUB_RUN_ID": "99", "GITHUB_SHA": "a" * 40}


@pytest.mark.parametrize("key,value", [
    ("GITHUB_REPOSITORY", "example/other"), ("GITHUB_EVENT_NAME", "workflow_dispatch"),
    ("GITHUB_EVENT_NAME", "pull_request_target"), ("GITHUB_REF", "refs/heads/main"),
    ("GITHUB_RUN_ATTEMPT", "2"), ("GITHUB_RUN_ID", "x"), ("GITHUB_SHA", "main")])
def test_probe_context_rejects_other_targets_events_and_reruns(key, value):
    env = context()
    probe.validate_context(env)
    env[key] = value
    with pytest.raises(ValueError):
        probe.validate_context(env)


def test_probe_success_verifies_both_owned_comment_updates_with_fixed_budget():
    client = Fake()
    api = probe.Budget(client, 3, 4)
    report = probe.compare(api, "https://example.invalid/run", "a" * 40)
    assert report["state"] == "observations-complete"
    assert (api.reads, api.writes) == (8, 4)
    assert len(client.calls) == 12
    assert all(r["post"]["readback_verified"] and r["patch"]["readback_verified"]
               for r in report["results"])
    with pytest.raises(ValueError):
        api.request("POST", "/issues/3/comments", {"body": probe.MARKER})
    assert len(client.calls) == 12


def test_probe_denied_post_is_not_retried_or_patched_and_other_target_is_compared():
    client = Fake("403")
    api = probe.Budget(client, 3, 4)
    report = probe.compare(api, "https://example.invalid/run", "a" * 40)
    assert report["results"][0]["post"]["http_status"] == 403
    assert report["results"][0]["patch"]["state"] == "skipped-post-rejected"
    assert report["results"][1]["patch"]["readback_verified"]
    assert sum(m == "POST" and p == "/issues/3/comments" for m, p, _ in client.calls) == 1
    assert (api.reads, api.writes) == (6, 3)


@pytest.mark.parametrize("failure,writes", [("uncertain-post", 1), ("uncertain-patch", 2)])
def test_probe_uncertain_writes_reconcile_only_by_read_then_stop(failure, writes):
    api = probe.Budget(Fake(failure), 3, 4)
    report = probe.compare(api, "https://example.invalid/run", "a" * 40)
    assert report["state"] == "stopped-uncertain-write"
    assert api.writes == writes
    assert len(report["results"]) == 1
    assert "FAKE_SECRET" not in json.dumps(report)
    assert "private.example" not in json.dumps(report)


def test_probe_refuses_to_patch_another_owner_or_repeat_prior_diagnostic():
    api = probe.Budget(Fake("wrong-owner"), 3, 4)
    report = probe.compare(api, "https://example.invalid/run", "a" * 40)
    assert report["state"] == "stopped-invalid-create-receipt"
    assert api.writes == 1
    repeated = probe.Budget(api.client, 3, 4)
    with pytest.raises(ValueError, match="Prior diagnostic"):
        probe.compare(repeated, "https://example.invalid/another", "b" * 40)
    assert repeated.writes == 0


@pytest.mark.parametrize("method,route,data", [
    ("POST", "/issues/999/comments", {"body": probe.MARKER}),
    ("PATCH", "/issues/comments/999", {"body": probe.MARKER}),
    ("POST", "/issues/3/comments", {"body": "ordinary comment"}),
    ("POST", "/issues/3/comments", {"body": probe.MARKER, "token": "FAKE_SECRET"}),
    ("DELETE", "/issues/3/comments", None)])
def test_probe_rejects_unowned_routes_methods_and_payloads_before_transport(method, route, data):
    client = Fake()
    with pytest.raises(ValueError):
        probe.Budget(client, 3, 4).request(method, route, data)
    assert client.calls == []


@pytest.mark.parametrize("failure", ["readback-failure", "readback-403"])
def test_probe_main_preserves_partial_receipts_when_readback_fails(
        failure, tmp_path, monkeypatch, capsys):
    env = context()
    env["GITHUB_TOKEN"] = "FAKE_SECRET"
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(probe, "GitHub", lambda *args: Fake(failure))
    monkeypatch.setattr(probe.subprocess, "check_output", lambda *args, **kwargs: "b" * 40)
    output = tmp_path / "receipt.json"
    monkeypatch.setattr(sys, "argv",
                        ["probe", "--issue", "3", "--pr", "4", "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        probe.main()
    assert error.value.code == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["requests"]["writes"] == 1
    assert report["results"][0]["post"]["state"] == "response-returned"
    assert "FAKE_SECRET" not in output.read_text(encoding="utf-8") + capsys.readouterr().out


def test_pr_only_probe_ignores_previous_issue_receipt_and_uses_two_writes_four_reads():
    client = Fake()
    client.request("POST", "/issues/3/comments", {"body": probe.MARKER + " old issue receipt"})
    client.calls.clear()
    api = probe.Budget(client, 3, 4, pr_only=True)
    report = probe.compare(api, "https://example.invalid/pr-only", "b" * 40)
    assert report["state"] == "observations-complete"
    assert report["request_budget"] == {"reads": 4, "writes": 2}
    assert len(report["results"]) == 1
    assert report["results"][0]["target_kind"] == "pr"
    assert report["results"][0]["patch"]["readback_verified"]
    assert (api.reads, api.writes) == (4, 2)
    assert all("/issues/3" not in route for _, route, _ in client.calls)
    with pytest.raises(ValueError, match="budget"):
        api.request("POST", "/issues/4/comments", {"body": probe.MARKER})
    assert len(client.calls) == 6


def test_pr_only_post_rejection_stops_without_patch_or_retry():
    client = Fake("pr-403")
    api = probe.Budget(client, 3, 4, pr_only=True)
    report = probe.compare(api, "https://example.invalid/pr-only", "b" * 40)
    row, = report["results"]
    assert row["post"]["http_status"] == 403
    assert row["patch"]["state"] == "skipped-post-rejected"
    assert (api.reads, api.writes) == (2, 1)
    assert sum(method == "POST" for method, _, _ in client.calls) == 1
    assert not any(method == "PATCH" for method, _, _ in client.calls)


@pytest.mark.parametrize("failure,writes", [("uncertain-post", 1), ("uncertain-patch", 2)])
def test_pr_only_uncertain_write_readback_never_resumes_writes(failure, writes):
    api = probe.Budget(Fake(failure), 3, 4, pr_only=True)
    report = probe.compare(api, "https://example.invalid/pr-only", "b" * 40)
    assert report["state"] == "stopped-uncertain-write"
    assert api.writes == writes
    assert api.reads <= 4
    assert "FAKE_SECRET" not in json.dumps(report)


@pytest.mark.parametrize("method,route,data", [
    ("GET", "/issues/3", None), ("GET", "/issues/3/comments?per_page=100", None),
    ("POST", "/issues/3/comments", {"body": probe.MARKER})])
def test_pr_only_probe_cannot_access_or_write_issue_target(method, route, data):
    client = Fake()
    with pytest.raises(ValueError, match="allowlist"):
        probe.Budget(client, 3, 4, pr_only=True).request(method, route, data)
    assert client.calls == []


@pytest.mark.parametrize("grant", ["read", "write"])
def test_pr_only_workflow_preserves_other_grants_and_requires_distinct_run_marker(grant):
    path = Path(__file__).resolve().parents[1] / ".github/workflows/skill-eval-http-diagnostic.yml"
    text = path.read_text(encoding="utf-8")
    assert len(re.findall(r"pull-requests: (?:read|write)", text)) == 1
    text = re.sub(r"pull-requests: (?:read|write)", "pull-requests: " + grant, text)
    workflow = yaml.load(text, Loader=yaml.BaseLoader)
    assert set(workflow["on"]) == {"push"}
    assert workflow["permissions"] == {}
    job = workflow["jobs"]["compare"]
    assert job["permissions"] == {"contents": "read", "actions": "read",
                                  "pull-requests": grant, "issues": "write"}
    assert "[run-http-pr-write-once]" in job["if"]
    assert "github.run_attempt == 1" in job["if"]
    assert any("--pr-only" in step.get("run", "") for step in job["steps"])
    assert "[run-http-pr-write-once]" not in "test: restore diagnostic pull request read permission"
