#!/usr/bin/env bash
# A reference model (DECISIONS.md, section 19): never a checkpoint, scored once on every set.
# Trains three seeds and scores them on the test-only sets (scripts/run.sh), averages the seeds on
# every set, then compares with checkpoint 3 everywhere.
#   PY=.venv/Scripts/python bash scripts/run_reference.sh resnet18-imagenet
set -uo pipefail
EXP=${1:?usage: run_reference.sh <name>}
PY=${PY:-python}
REF=${REF:-skinnet-c3}
LOGS=runs/$EXP/logs
export PY PYTHONUNBUFFERED=1

bash scripts/run.sh "configs/$EXP.yaml" "$EXP" || exit 1
for s in ${SEEDS:-0 1 2}; do
  [ -f "runs/$EXP/s$s/summary.json" ] || { echo "  FAILED: seed $s did not finish, see $LOGS/s$s.log"; exit 1; }
done
$PY -m skincancer.derive ensemble --exp "$EXP" --splits val test ak_side midas_val > "$LOGS/derive.log" 2>&1 \
  && $PY -m skincancer.derive ensemble --exp "$EXP" --final \
       --splits ddi_test dermacon_test dermacon_other midas_test >> "$LOGS/derive.log" 2>&1 \
  || { echo "  FAILED: see $LOGS/derive.log"; exit 1; }
$PY -m skincancer.compare --a "$REF" --b "$EXP" --splits val midas_val test midas_test ddi_test dermacon_test \
  | tee "$LOGS/compare.log"
echo "[$(date +%H:%M)] $EXP reference done"
