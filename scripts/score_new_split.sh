#!/usr/bin/env bash
# Score already-finished experiments on a test-only split added after they ran (e.g. midas_test),
# once per run (ledger-enforced), then re-evaluate them.
#   PY=.venv/Scripts/python bash scripts/score_new_split.sh midas_test e1_effb0 e1_r50 ...
set -uo pipefail
PY=${PY:-python}
SPLIT=$1; shift
mkdir -p runs/logs
for exp in "$@"; do
  for d in runs/"$exp"/s*/; do
    [ -f "$d/best.pt" ] || continue
    $PY -m skincancer.predict --run "${d%/}" --splits "$SPLIT" --final >> "runs/logs/score_${SPLIT}.log" 2>&1 \
      || echo "  skipped ${d%/} (already scored or failed; see runs/logs/score_${SPLIT}.log)"
  done
  $PY -m skincancer.evaluate --exp "$exp" >> runs/logs/evaluate.log 2>&1 && echo "scored $exp on $SPLIT"
done
