import argparse
import json
import os
import re
import sys
from fnmatch import fnmatchcase
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

CONTEXT = "knowledge-model-policy"
DECLARATION = "knowledge-model-policy"
APPROVAL = "knowledge-policy-approval"
SHA = re.compile(r"[0-9a-f]{40}")
ROOT = Path(__file__).resolve().parents[1]
# Exit codes: published results are final, a rejection is final, but stale work must be rerun.
EXIT_COMPLIANT = 0
EXIT_NONCOMPLIANT = 1
EXIT_RERUN = 2
# Bounded in-run re-evaluation before the run reports the work as unfinished.
ATTEMPTS = 3
API_ERRORS = (ValueError, KeyError, TypeError, OSError)
# Outcome kinds, reported separately so evaluator noise is never read as a rejection.
KINDS = ("compliant", "noncompliant", "evaluator-error", "deferred", "closed",
         "publication-failed")
FAILING_KINDS = {"noncompliant", "evaluator-error", "deferred", "publication-failed"}


def summarize(results):
    counts = {kind: sum(result["kind"] == kind for result in results) for kind in KINDS}
    return {"kind_counts": counts, "results": len(results),
            "failed": sum(result["kind"] in FAILING_KINDS for result in results),
            "rerun": any(result["rerun"] for result in results)}


def exit_code(results):
    # Unfinished (rerun) work keeps open PRs fail-closed; a real rejection alone exits 1.
    if any(result["rerun"] for result in results):
        return EXIT_RERUN
    return EXIT_NONCOMPLIANT if any(result["kind"] in FAILING_KINDS for result in results) \
        else EXIT_COMPLIANT


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def json_block(body, tag):
    opening = rf"^```{re.escape(tag)}[ \t]*\r?$"
    blocks = re.findall(opening + r"\n(.*?)^```[ \t]*\r?$", body or "", re.M | re.S)
    if len(re.findall(opening, body or "", re.M)) != 1 or len(blocks) != 1:
        raise ValueError(f"Exactly one complete {tag} JSON block is required")
    value = json.loads(blocks[0], object_pairs_hook=unique_object)
    if not isinstance(value, dict):
        raise ValueError(f"{tag} must be a JSON object")
    return value


def fields(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= value.keys():
        raise ValueError("Missing structured declaration fields")
    if value.keys() - set(required) - set(optional):
        raise ValueError("Unknown structured declaration fields")


def text(value):
    return isinstance(value, str) and bool(value.strip())


def path_set(value):
    if not isinstance(value, list) or not value or any(not text(p) for p in value):
        raise ValueError("Scope must be a nonempty list of repository paths")
    if len(set(value)) != len(value):
        raise ValueError("Duplicate scope paths")
    for path in value:
        if "\\" in path or any(part in ("", ".", "..") for part in path.split("/")):
            raise ValueError("Scope must use repository-relative POSIX paths")
    return set(value)


def changed_paths(files, expected):
    if type(expected) is not int or expected < 0 or len(files) != expected:
        raise ValueError("Incomplete changed-file API results")
    names = [item["filename"] for item in files]
    if len(set(names)) != expected:
        raise ValueError("Duplicate changed-file API results")
    paths = set()
    for item in files:
        paths |= path_set([item["filename"]])
        if item["status"] == "renamed":
            paths |= path_set([item["previous_filename"]])
    return paths


def matching(paths, patterns):
    return {path for path in paths if any(fnmatchcase(path, p) for p in patterns)}


def file_page_numbers(count):
    if type(count) is not int or not 0 <= count <= 3000:
        raise ValueError("Changed-file count is unavailable or exceeds the GitHub API limit")
    return range(1, (count + 99) // 100 + 1)


def assemble_file_pages(pages, count):
    pages = list(pages)
    expected_pages = len(file_page_numbers(count))
    if len(pages) != expected_pages:
        raise ValueError("Incomplete changed-file API pagination")
    for index, page in enumerate(pages):
        size = min(100, count - index * 100)
        if not isinstance(page, list) or len(page) != size:
            raise ValueError("Incomplete changed-file API pagination")
    files = [item for page in pages for item in page]
    changed_paths(files, count)
    return files


def declaration(body, head):
    value = json_block(body, DECLARATION)
    fields(value, ("schema_version", "head_sha", "human_only", "material_ai_authors"),
           ("exception",))
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("Unsupported declaration schema")
    if value["head_sha"] != head or not SHA.fullmatch(head):
        raise ValueError("Declaration does not cover the current PR head")
    authors = value["material_ai_authors"]
    if type(value["human_only"]) is not bool or not isinstance(authors, list):
        raise ValueError("Invalid human-only declaration or author list")
    if value["human_only"] != (not authors):
        raise ValueError("Human-only declaration conflicts with material AI authorship")
    for author in authors:
        fields(author, ("tool", "model_id", "effort", "role"))
        if any(not text(author[key]) for key in author) or author["role"] != "author":
            raise ValueError("Material AI authors require an author role and explicit attribution")
    if "exception" in value:
        fields(value["exception"], ("paths", "reason"))
        path_set(value["exception"]["paths"])
        if not text(value["exception"]["reason"]):
            raise ValueError("Exception reason is required")
    return value


def model_errors(authors, policy):
    identities = {}
    for canonical, model in policy["models"].items():
        for name in (canonical, *model["aliases"]):
            if name in identities:
                raise ValueError("Ambiguous model identity in trusted policy")
            identities[name] = model
    errors = []
    for index, author in enumerate(authors, 1):
        model = identities.get(author["model_id"])
        if model is None or model["suspended"]:
            errors.append(f"Material AI author {index}: model is unapproved or suspended")
        elif author["effort"] not in {"xhigh", "max"} & set(model["efforts"]):
            errors.append(f"Material AI author {index}: effort is not allowed")
    return errors


def approved(comments, policy, contributors, number, head, kind, paths, reason=None):
    owners = {name.casefold() for name in
              policy["approval_owners"] + policy["independent_approvers"]}
    latest = {}
    for comment in sorted(comments, key=lambda item: item["id"]):
        login = comment["user"]["login"].casefold()
        if login in owners and re.search(rf"^```{APPROVAL}[ \t]*\r?$",
                                         comment.get("body") or "", re.M):
            latest[login] = comment
    # 只信任维护者账号的最新批准声明；提交者自己的文字不能提供授权。
    for login, comment in latest.items():
        if login in {name.casefold() for name in contributors}:
            continue
        try:
            value = json_block(comment["body"], APPROVAL)
            fields(value, ("schema_version", "pull_request", "head_sha", "decision",
                           "no_material_involvement", "approvals"))
            if (type(value["schema_version"]) is not int or value["schema_version"] != 1
                    or type(value["pull_request"]) is not int or value["pull_request"] != number
                    or value["head_sha"] != head or value["decision"] != "approve"
                    or value["no_material_involvement"] is not True):
                continue
            if not isinstance(value["approvals"], list):
                continue
            kinds = set()
            matches = False
            for approval in value["approvals"]:
                fields(approval, ("kind", "paths", "reason"))
                if approval["kind"] not in {"exception", "policy-change"}:
                    raise ValueError("Unknown approval kind")
                if approval["kind"] in kinds or not text(approval["reason"]):
                    raise ValueError("Duplicate approval kind or missing reason")
                kinds.add(approval["kind"])
                scope = path_set(approval["paths"])
                if (approval["kind"] == kind and scope == paths
                        and (reason is None or approval["reason"] == reason)):
                    matches = True
            if matches:
                return True
        except (ValueError, TypeError):
            # 无效批准不能回退到同一账号的历史批准。
            continue
    return False


def assess(pr, paths, policy, comments=(), contributors=()):
    protected = matching(paths, policy["protected_paths"])
    controls = matching(paths, policy["policy_paths"])
    errors = []
    if controls and not approved(comments, policy, contributors, pr["number"], pr["head"]["sha"],
                                 "policy-change", controls):
        errors.append("Policy changes require explicit independent owner approval")
    if protected:
        try:
            value = declaration(pr.get("body"), pr["head"]["sha"])
            denied = model_errors(value["material_ai_authors"], policy)
            request = value.get("exception")
            if request is not None:
                if path_set(request["paths"]) != protected:
                    errors.append("Exception scope must exactly cover the protected changed paths")
                elif not approved(comments, policy, contributors, pr["number"], pr["head"]["sha"],
                                  "exception", protected, request["reason"]):
                    errors.append("Exception requires explicit independent owner approval")
                else:
                    denied = []
            errors.extend(denied)
        except (ValueError, TypeError):
            errors.append("Missing, invalid or stale knowledge author declaration")
    applicable = bool(protected or controls)
    # Pure metadata: same state, but a kind so a rejection is never confused with evaluator noise.
    return {
        "state": "failure" if errors else "success",
        "kind": "noncompliant" if errors else "compliant",
        "description": ("Declaration or policy approval failed" if errors else
                        "Declaration compliance only" if applicable else "Not applicable"),
        "errors": errors,
        "protected_paths": sorted(protected),
        "policy_paths": sorted(controls),
    }


class GitHub:
    def __init__(self, repository, token):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            raise ValueError("Expected an owner/repository identifier")
        self.root = f"https://api.github.com/repos/{repository}/"
        self.token = token

    def request(self, route, payload=None):
        headers = {"Accept": "application/vnd.github+json", "User-Agent": CONTEXT,
                   "X-GitHub-Api-Version": "2022-11-28"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        data = None if payload is None else json.dumps(payload).encode()
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = Request(self.root + route, data=data, headers=headers)
        with urlopen(request, timeout=30) as response:
            return json.load(response), response.headers.get("Link", "")

    def get(self, route):
        return self.request(route)[0]

    def pages(self, route):
        separator = "&" if "?" in route else "?"
        page = 1
        result = []
        while True:
            query = urlencode({"per_page": 100, "page": page})
            items, links = self.request(route + separator + query)
            if not isinstance(items, list):
                raise ValueError("Expected a paginated GitHub API list")
            result.extend(items)
            if 'rel="next"' not in links:
                return result
            if len(items) != 100:
                raise ValueError("Incomplete GitHub API pagination")
            page += 1

    def status(self, head, state, description):
        payload = {"state": state, "context": CONTEXT, "description": description}
        run_id = os.environ.get("GITHUB_RUN_ID")
        if run_id:
            repository_url = self.root.replace("https://api.github.com/repos/", "https://github.com/")
            payload["target_url"] = f"{repository_url}actions/runs/{run_id}"
        self.request(f"statuses/{head}", payload)


def snapshot(pr):
    return (pr["head"]["sha"], pr["base"]["ref"], pr["base"]["sha"],
            pr.get("body"), pr["changed_files"], pr["state"])


def require_unique_head(prs, number, head):
    # Commit status 由 SHA/context 共享；其他 base 的 open PR 也不能复用它。
    peers = {pr["number"] for pr in prs
             if pr["head"]["sha"] == head}
    if peers != {number}:
        raise ValueError("Head must belong to exactly one open PR; shared heads are ineligible")


def deferred(reason):
    # Not a verdict: the target must be re-evaluated before anything is published.
    return {"state": "failure", "kind": "deferred", "description": "Evaluation deferred",
            "errors": [reason], "retry": True}


def evaluation_failure(error):
    message = (f"GitHub API HTTP {error.code}" if isinstance(error, HTTPError)
               else "GitHub API unavailable" if isinstance(error, OSError)
               else str(error) if isinstance(error, ValueError)
               else "Invalid GitHub API response")
    return {"state": "failure", "kind": "evaluator-error",
            "description": "Policy evaluation failed", "errors": [message], "retry": True}


def refused(reason):
    # Deterministic refusal by admission metadata, not an evaluator failure; still fail-closed.
    return {"state": "failure", "kind": "noncompliant",
            "description": "Eligibility refused", "errors": [reason], "retry": False}


def read_pr_result(api, pr, policy, policy_sha, open_prs):
    number = pr["number"]
    head = pr["head"]["sha"]
    comments = []
    try:
        require_unique_head(open_prs, number, head)
    except ValueError as error:
        return refused(str(error)), comments
    base_route = f"git/ref/heads/{policy['base_branch']}"
    if api.get(base_route)["object"]["sha"] != policy_sha:
        # The trusted checkout moved: this verdict belongs to no publishable base.
        return deferred("Trusted policy checkout is stale; rerun on the current base"), comments
    try:
        pages = [api.get(f"pulls/{number}/files?per_page=100&page={page}")
                 for page in file_page_numbers(pr["changed_files"])]
        files = assemble_file_pages(pages, pr["changed_files"])
        paths = changed_paths(files, pr["changed_files"])
        controls = matching(paths, policy["policy_paths"])
        protected = matching(paths, policy["protected_paths"])
        request = None
        if protected:
            try:
                request = declaration(pr.get("body"), head).get("exception")
            except (ValueError, TypeError):
                pass
        comments, commits = [], []
        contributors = {pr["user"]["login"]}
        if controls or request is not None:
            if pr["commits"] > 250:
                raise ValueError("Commit attribution exceeds the GitHub PR commits API limit")
            commits = api.pages(f"pulls/{number}/commits")
            if len(commits) != pr["commits"]:
                raise ValueError("Incomplete commit attribution API results")
            for commit in commits:
                for field in ("author", "committer"):
                    if commit.get(field):
                        contributors.add(commit[field]["login"])
            comments = api.pages(f"issues/{number}/comments")
        result = assess(pr, paths, policy, comments, contributors)
        # Anything that moved since the snapshot invalidates this verdict; the base ref must
        # still match the trusted checkout, or the verdict belongs to no publishable base.
        if snapshot(api.get(f"pulls/{number}")) != snapshot(pr):
            return deferred("PR changed during evaluation; rerun required"), comments
        if api.get(base_route)["object"]["sha"] != policy_sha:
            return deferred("Base advanced during evaluation; rerun required"), comments
        if (controls or request is not None) and comments != api.pages(f"issues/{number}/comments"):
            # Re-read approvals that were read, not only approvals that changed the verdict.
            return deferred("Approval comments changed during evaluation; rerun required"), comments
    except API_ERRORS as error:
        result = evaluation_failure(error)
    return result, comments


def admission_snapshot(pr):
    # list PR API 不含 changed_files；head/base/body/state 相同才可复用已验证的文件列表。
    return (pr["head"]["sha"], pr["base"]["ref"], pr["base"]["sha"],
            pr.get("body"), pr["state"])


def addressable(pr, policy):
    return pr["state"] == "open" and pr["base"]["ref"] == policy["base_branch"]


def check_prs(api, policy, policy_sha, dry_run, numbers=None):
    open_prs = api.pages("pulls?state=open")
    if numbers is None:
        numbers = [pr["number"] for pr in open_prs if addressable(pr, policy)]
    published = {}
    retry = list(numbers)
    # Results are published only from a fresh scan, and only for a target that still matches
    # the evaluated metadata. Stale or moved work is re-evaluated instead of being written to
    # a captured head; a closed target receives no publication at all.
    for attempt in range(ATTEMPTS):
        staged, retry_next = [], []
        for number in retry:
            pr = api.get(f"pulls/{number}")
            head = pr["head"]["sha"]
            if not SHA.fullmatch(head):
                raise ValueError("Invalid PR head")
            if not addressable(pr, policy):
                published[number] = {"head": head, "state": "success", "kind": "closed",
                                     "description": "Not open against the policy branch",
                                     "errors": [], "retry": False}
                continue
            # No pending write: a run that stops early must not leave a stale pending status
            # on a head that merges or otherwise leaves the sweep.
            result, comments = read_pr_result(api, pr, policy, policy_sha, open_prs)
            if result.pop("retry", False) and attempt + 1 < ATTEMPTS:
                retry_next.append(number)
            else:
                staged.append((pr, result, comments))
        open_prs = api.pages("pulls?state=open")
        current = {pr["number"]: pr for pr in open_prs}
        for pr, result, comments in staged:
            number, head = pr["number"], pr["head"]["sha"]
            live = current.get(number)
            if live is None or not addressable(live, policy):
                published[number] = {"head": head, "state": "success", "kind": "closed",
                                     "description": "Closed before publication", "errors": [],
                                     "retry": False}
                continue
            if admission_snapshot(live) != admission_snapshot(pr):
                retry_next.append(number)
                continue
            if comments and comments != api.pages(f"issues/{number}/comments"):
                # Approvals read for this verdict must still be current at publication.
                retry_next.append(number)
                continue
            try:
                require_unique_head(open_prs, number, head)
            except ValueError as error:
                result = refused(str(error))
            # Only a fresh, fully revalidated verdict reaches a head; a shared head is refused
            # above, so each head is written at most once per round.
            if not dry_run:
                try:
                    api.status(head, result["state"], result["description"])
                except API_ERRORS as error:
                    result = {**result, "kind": "publication-failed",
                              "errors": [*result["errors"],
                                         evaluation_failure(error)["errors"][0]]}
            published[number] = {"head": head, **result}
        retry = sorted(set(retry_next) - set(published))
        if not retry:
            break
    # After the attempt budget, an unfinished target stays failed and asks for a fresh run.
    for number in retry:
        published.setdefault(number, {"head": "", "state": "failure", "kind": "deferred",
                                      "description": "Evaluation incomplete", "errors": []})
    results = []
    for number in sorted(published):
        item = published[number]
        results.append({"pull_request": number, "head_sha": item["head"],
                        "policy_sha": policy_sha, "dry_run": dry_run,
                        "rerun": item["kind"] == "deferred", "state": item["state"],
                        "kind": item["kind"], "description": item["description"],
                        "errors": item["errors"]})
    return results


def check_pr(api, number, policy, policy_sha, dry_run):
    return check_prs(api, policy, policy_sha, dry_run, [number])[0]


def refresh_event(event, event_name):
    if event_name == "issue_comment":
        return "pull_request" in event["issue"]
    if event_name in {"pull_request_target", "push", "workflow_dispatch"}:
        return True
    raise ValueError("Unsupported policy event; merge queue requires an explicit integration")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check declared knowledge authorship")
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY"), required=False)
    parser.add_argument("--event-path", default=os.environ.get("GITHUB_EVENT_PATH"))
    parser.add_argument("--policy", type=Path, default=ROOT / ".github/knowledge-model-policy.json")
    parser.add_argument("--policy-sha", required=True)
    parser.add_argument("--pull-request", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not SHA.fullmatch(args.policy_sha):
        raise ValueError("Expected the trusted policy checkout SHA")
    token = os.environ.get("GITHUB_TOKEN")
    if not args.dry_run and not token:
        raise ValueError("GITHUB_TOKEN is required to publish policy statuses")
    policy = json.loads(args.policy.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    if policy["schema_version"] != 1:
        raise ValueError("Unsupported policy schema")
    api = GitHub(args.repository or "", token)
    if args.pull_request is not None:
        if args.pull_request <= 0:
            raise ValueError("PR number must be positive")
        numbers = [args.pull_request]
    else:
        event = json.loads(Path(args.event_path).read_text(encoding="utf-8"))
        # 所有触发共用一个 writer；待运行事件被合并时仍覆盖全部 open PR。
        numbers = None if refresh_event(event, os.environ["GITHUB_EVENT_NAME"]) else []
    results = check_prs(api, policy, args.policy_sha, args.dry_run, numbers)
    for result in results:
        print(json.dumps(result, ensure_ascii=True))
    print(json.dumps(summarize(results), ensure_ascii=True))
    return exit_code(results)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, TypeError, OSError) as error:
        message = (f"GitHub API HTTP {error.code}" if isinstance(error, HTTPError)
                   else "Required input or GitHub API unavailable" if isinstance(error, OSError)
                   else "Invalid policy or event input")
        print(message, file=sys.stderr)
        sys.exit(1)
