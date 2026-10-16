#!/usr/bin/env bash
# The checks CI's lint job runs, in the same order, before work goes into
# dev (the one automated run is on the pull request from dev into master).
# Give test files or directories to run them too, on every core:
#
#   scripts/check.sh                                  # lint only
#   scripts/check.sh repyability/tests/test_units.py  # lint, then those tests
set -euo pipefail
cd "$(dirname "$0")/.."

black --check repyability
isort --check-only repyability
python -m flake8 repyability
mypy repyability

if [ "$#" -gt 0 ]; then
    python -m pytest -n auto -q "$@"
fi
