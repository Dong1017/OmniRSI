#!/usr/bin/env bash
# Run the knowledge admission checker against the current trusted main.
# Exit 2 means the trusted base advanced during evaluation: refetch main and evaluate again.
set -uo pipefail

attempts=3
for attempt in $(seq 1 "$attempts"); do
  git fetch --quiet origin main || exit 1
  git checkout --quiet origin/main || exit 1
  status=0
  python tools/check_knowledge_model_policy.py --policy-sha "$(git rev-parse HEAD)" "$@" || status=$?
  if [ "$status" -ne 2 ]; then
    exit "$status"
  fi
  echo "Trusted base advanced during evaluation; retrying against current main (${attempt}/${attempts})" >&2
done
echo "Trusted base kept advancing; ending this run as failed" >&2
exit 1
