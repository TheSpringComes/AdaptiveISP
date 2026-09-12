#!/usr/bin/env bash
# debug/smoke/run.sh — run all smoke tests in order, exit 0 on all pass.
# Target wall-clock: < 60s (no real data, no pretrained YOLO).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

TESTS=(
  "debug/smoke/test_imports.py"
  "debug/smoke/test_operators.py"
  "debug/smoke/test_pipeline.py"
  "debug/smoke/test_front_isp.py"
  "debug/smoke/test_calibration.py"
  "debug/smoke/test_controller.py"
  "debug/smoke/test_end_to_end.py"
)

pass=0
fail=0
failures=()
for t in "${TESTS[@]}"; do
  if python "$t"; then
    pass=$((pass + 1))
  else
    fail=$((fail + 1))
    failures+=("$t")
  fi
done

echo
echo "smoke summary: ${pass} passed, ${fail} failed"
if (( fail > 0 )); then
  printf '  failed: %s\n' "${failures[@]}"
  exit 1
fi
