#!/usr/bin/env bash
# Train three seeds, score each once on the test-only sets, then compute metrics.
# Safe to re-run: finished seeds are skipped and the ledger refuses double scoring.
#   PY=.venv/Scripts/python bash scripts/run.sh                    # Windows venv
#   PY=python bash scripts/run.sh configs/skinnet-c3.yaml skinnet-c3   # config and model name
set -uo pipefail
PY=${PY:-python}
CFG=${1:-configs/skinnet.yaml}
EXP=${2:-skinnet}
SEEDS=${SEEDS:-"0 1 2"}
FINAL="ddi_test dermacon_test dermacon_other midas_test"
LOGS=runs/$EXP/logs
mkdir -p "$LOGS"
export PYTHONUNBUFFERED=1

for s in $SEEDS; do
  echo "[$(date +%H:%M)] train $EXP seed $s"
  $PY -m skincancer.train --config "$CFG" --seed "$s" --set "name=$EXP" > "$LOGS/s$s.log" 2>&1 \
    || echo "  FAILED: see $LOGS/s$s.log"
done
for s in $SEEDS; do
  [ -f "runs/$EXP/s$s/summary.json" ] && $PY -m skincancer.predict --run "runs/$EXP/s$s" \
    --splits $FINAL --final >> "$LOGS/final.log" 2>&1
done
$PY -m skincancer.evaluate --exp "$EXP" | tee "$LOGS/evaluate.log"
echo "[$(date +%H:%M)] done"
