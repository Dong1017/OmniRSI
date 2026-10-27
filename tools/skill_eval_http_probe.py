"""One bounded fork-only HTTP comparison; never a Skill Eval execution."""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

from tools.skill_eval_pr import GitHub, GitHubHTTPError

REPOSITORY = "Dong1017/OmniRSI"
BRANCH = "refs/heads/codex/issue101-omni-http"
MARKER = "<!-- omnirsi-http-diagnostic:v1 -->"


def validate_context(context):
    if (context.get("GITHUB_REPOSITORY") != REPOSITORY
            or context.get("GITHUB_EVENT_NAME") != "push"
            or context.get("GITHUB_REF") != BRANCH
            or context.get("GITHUB_RUN_ATTEMPT") != "1"
            or not re.fullmatch(r"[0-9]+", context.get("GITHUB_RUN_ID", ""))
            or not re.fullmatch(r"[0-9a-f]{40}", context.get("GITHUB_SHA", ""))):
        raise ValueError("Outside the single authorized fork diagnostic context")


class Budget:
    def __init__(self, client, issue, pr):
        if type(issue) is not int or type(pr) is not int or min(issue, pr) < 1 or issue == pr:
            raise ValueError("Expected distinct positive diagnostic targets")
        self.client, self.issue, self.pr = client, issue, pr
        self.reads = self.writes = 0
        self.owned = set()

    def request(self, method, route, data=None):
        collections = {f"/issues/{n}/comments" for n in (self.issue, self.pr)}
        comment_paths = {f"/issues/comments/{n}" for n in self.owned}
        reads = {f"/issues/{self.issue}", f"/pulls/{self.pr}"} | comment_paths
        reads |= {path + "?per_page=100" for path in collections}
        if method == "GET" and route in reads and data is None:
            if self.reads >= 8:
                raise ValueError("Read budget exhausted")
            self.reads += 1
        elif ((method == "POST" and route in collections)
              or (method == "PATCH" and route in comment_paths)):
            if (self.writes >= 4 or type(data) is not dict or set(data) != {"body"}
                    or type(data["body"]) is not str or not data["body"].startswith(MARKER)):
                raise ValueError("Outside the diagnostic write budget or body contract")
            self.writes += 1
        else:
            raise ValueError("Outside the diagnostic route allowlist")
        return self.client.request(method, route, data)


def owned_comment(comment, number, body):
    if not isinstance(comment, dict):
        return False
    identifier = comment.get("id")
    if type(identifier) is not int or identifier < 1:
        return False
    urls = {f"https://github.com/{REPOSITORY}/{kind}/{number}#issuecomment-{identifier}"
            for kind in ("issues", "pull")}
    return (comment.get("user", {}).get("login") == "github-actions[bot]"
            and comment.get("issue_url") == f"https://api.github.com/repos/{REPOSITORY}/issues/{number}"
            and comment.get("html_url") in urls and comment.get("body") == body)


def attempt(api, method, route, row, data=None):
    row.update(method=method, route=route)
    try:
        result = api.request(method, route, data)
    except GitHubHTTPError as error:
        row.update(state="http-rejected" if 400 <= error.code < 500 else "uncertain",
                   diagnostic=str(error), http_status=error.code)
        return None
    except Exception:
        # Unknown transport/client exceptions may include private URLs or credentials.
        row["state"] = "uncertain"
        return None
    row["state"] = "response-returned"
    return result


def compare(api, run_url, revision, report=None):
    if report is None:
        report = {}
    report.update(repository=REPOSITORY, event="push", revision=revision,
                  run_url=run_url, request_budget={"reads": 8, "writes": 4},
                  results=[], state="incomplete")
    # Check both targets and prior receipts before any write. Do not guess through
    # a full first page or resume a previous partial run.
    for kind, number in (("issue", api.issue), ("pr", api.pr)):
        info = api.request("GET", f"/issues/{number}" if kind == "issue" else f"/pulls/{number}")
        if (info.get("number") != number or info.get("state") != "open"
                or info.get("locked") is not False
                or (kind == "issue" and "pull_request" in info)
                or (kind == "pr" and info.get("head", {}).get("repo", {}).get("full_name")
                    != REPOSITORY)):
            raise ValueError("Diagnostic target identity/state mismatch")
        if kind == "pr":
            report["pr_identity"] = {key: info[key]["sha"] for key in ("base", "head")}
        comments = api.request("GET", f"/issues/{number}/comments?per_page=100")
        if (not isinstance(comments, list) or len(comments) >= 100
                or any(not isinstance(c, dict) or not isinstance(c.get("body"), str)
                       or c["body"].startswith(MARKER) for c in comments)):
            raise ValueError("Prior diagnostic or ambiguous comment history; refusing replay")
    for kind, number in (("issue", api.issue), ("pr", api.pr)):
        row = {"target_kind": kind, "target_number": number, "post": {}, "patch": {}}
        report["results"].append(row)
        body = (f"{MARKER}\nDiagnostic only: {kind} comment POST/PATCH "
                "using the trusted publisher HTTP client.\n"
                f"Run: {run_url}\nRevision: {revision}\n"
                "No model/GPU execution or Skill Eval efficacy claim. Phase: create.")
        created = attempt(api, "POST", f"/issues/{number}/comments", row["post"], {"body": body})
        if row["post"]["state"] == "http-rejected":
            row["patch"]["state"] = "skipped-post-rejected"
            continue
        if row["post"]["state"] == "uncertain":
            # Reconcile once with reads only, then stop all further writes.
            matches = api.request("GET", f"/issues/{number}/comments?per_page=100")
            row["post"]["readback_matches"] = [c["html_url"] for c in matches
                                                if owned_comment(c, number, body)]
            report["state"] = "stopped-uncertain-write"
            return report
        if not owned_comment(created, number, body):
            report["state"] = "stopped-invalid-create-receipt"
            return report
        identifier = created["id"]
        api.owned.add(identifier)
        route = f"/issues/comments/{identifier}"
        if not owned_comment(api.request("GET", route), number, body):
            report["state"] = "stopped-create-readback-mismatch"
            return report
        row["comment_url"] = created["html_url"]
        row["post"]["readback_verified"] = True
        updated = body.replace("Phase: create.", "Phase: update.")
        attempt(api, "PATCH", route, row["patch"], {"body": updated})
        if row["patch"]["state"] == "http-rejected":
            continue
        row["patch"]["readback_verified"] = owned_comment(
            api.request("GET", route), number, updated)
        if row["patch"]["state"] == "uncertain":
            report["state"] = "stopped-uncertain-write"
            return report
        if not row["patch"]["readback_verified"]:
            report["state"] = "stopped-update-readback-mismatch"
            return report
    report["state"] = "observations-complete"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issue", required=True, type=int)
    parser.add_argument("--pr", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    validate_context(os.environ)
    client = Budget(GitHub(REPOSITORY, os.environ["GITHUB_TOKEN"]), args.issue, args.pr)
    run_url = f"https://github.com/{REPOSITORY}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
    report = {"state": "stopped-preflight-or-read-failure", "run_url": run_url}
    try:
        compare(client, run_url, os.environ["GITHUB_SHA"], report)
    except GitHubHTTPError as error:
        report.update(state="stopped-preflight-or-read-failure", http_status=error.code,
                      diagnostic=str(error))
    except Exception:
        # Keep any prior write receipts; never render unknown exception text.
        report["state"] = "stopped-preflight-or-read-failure"
    report["requests"] = {"reads": client.reads, "writes": client.writes}
    report["publisher_blob"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD:tools/skill_eval_pr.py"], text=True).strip()
    text = json.dumps(report, sort_keys=True, indent=2)
    args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    verified = (report["state"] == "observations-complete" and len(report["results"]) == 2
                and all(row["post"].get("readback_verified")
                        and row["patch"].get("readback_verified") for row in report["results"]))
    raise SystemExit(0 if verified else 1)


if __name__ == "__main__":
    main()
