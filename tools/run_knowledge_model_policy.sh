#!/usr/bin/env bash
# Run the knowledge admission checker against the current trusted main.
# Exit 2 means the work could not be completed (stale base, unreadable evidence); this runner
# is the single retry owner, so it refetches main and runs the checker again.
# Each attempt is told its position so the checker starts further along the target list: an
# unchanged retry would spend the same bounded time on the same prefix and starve the tail again.
set -uo pipefail

attempts=3
for attempt in $(seq 1 "$attempts"); do
  git fetch --quiet origin main || exit 1
  git checkout --quiet origin/main || exit 1
  status=0
  python tools/check_knowledge_model_policy.py --policy-sha "$(git rev-parse HEAD)" \
    --attempt "$attempt" --attempts "$attempts" "$@" || status=$?
  if [ "$status" -ne 2 ]; then
    exit "$status"
  fi
  echo "Admission work incomplete; retrying against current main (${attempt}/${attempts})" >&2
done
echo "Admission work still incomplete after ${attempts} attempts" >&2
exit 1
