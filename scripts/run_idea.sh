#!/usr/bin/env bash
# Step 1 for an idea: train its seeds, average them, and compare with the current checkpoint
# on the validation sets only. Test-only sets are never touched here (scripts/run.sh does that,
# once an idea has passed step 1).
#   PY=.venv/Scripts/python bash scripts/run_idea.sh n            # configs/skinnet-idea-n.yaml
#   REF=skinnet-c3 SEEDS="0 1 2" PY=python bash scripts/run_idea.sh m
# Safe to re-run: seeds with a finished summary.json are skipped.
set -uo pipefail
PY=${PY:-python}
LETTER=${1:?usage: run_idea.sh <letter>}
EXP=skinnet-idea-$LETTER
CFG=configs/$EXP.yaml
REF=${REF:-skinnet-c3}
SEEDS=${SEEDS:-"0 1 2"}
VAL="val midas_val"
LOGS=runs/$EXP/logs
mkdir -p "$LOGS"
export PYTHONUNBUFFERED=1

for s in $SEEDS; do
  if [ -f "runs/$EXP/s$s/summary.json" ]; then echo "[$(date +%H:%M)] $EXP seed $s already done"; continue; fi
  echo "[$(date +%H:%M)] train $EXP seed $s"
  $PY -m skincancer.train --config "$CFG" --seed "$s" > "$LOGS/s$s.log" 2>&1 \
    || { echo "  FAILED: see $LOGS/s$s.log"; exit 1; }
  echo "[$(date +%H:%M)] $EXP seed $s done: $(tail -n 1 "$LOGS/s$s.log")"
done
$PY -m skincancer.derive ensemble --exp "$EXP" --splits val test ak_side midas_val > "$LOGS/derive.log" 2>&1
$PY -m skincancer.compare --a "$REF" --b "$EXP" --splits $VAL | tee "$LOGS/compare_val.log"
echo "[$(date +%H:%M)] $EXP step 1 done"
