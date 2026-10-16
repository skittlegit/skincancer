#!/usr/bin/env bash
# Every experiment in order, 3 seeds each, with an evaluate + commit + push checkpoint after
# each block. Safe to re-run: finished runs are skipped and the ledger refuses double scoring.
#   PY=.venv/Scripts/python bash scripts/run_all.sh          # Windows venv
#   PY=python bash scripts/run_all.sh                         # Kaggle / Colab
# Blocks can be run alone: bash scripts/run_all.sh e3 e4
set -uo pipefail
PY=${PY:-python}
SEEDS="0 1 2"
FINAL="ddi_test dermacon_test dermacon_other pad_dark_test midas_test"
LOGS=runs/logs
mkdir -p "$LOGS"
export PYTHONUNBUFFERED=1

train() {  # train <config> <seed> [--set k=v ...]
  local cfg=$1 seed=$2; shift 2
  local tag; tag=$(basename "$cfg" .yaml)_s${seed}$(printf '_%s' "$@" | tr -c 'A-Za-z0-9._-' '_')
  echo "[$(date +%H:%M)] train $tag"
  $PY -m skincancer.train --config "$cfg" --seed "$seed" "$@" > "$LOGS/$tag.log" 2>&1 || echo "  FAILED: see $LOGS/$tag.log"
}

final() {  # final <exp>: score each seed once on the external test sets (ledger-enforced)
  for s in $SEEDS; do
    [ -f "runs/$1/s$s/best.pt" ] && $PY -m skincancer.predict --run "runs/$1/s$s" --splits $FINAL --final \
      >> "$LOGS/final_$1.log" 2>&1
  done
}

evaluate() { $PY -m skincancer.evaluate --exp "$@" >> "$LOGS/evaluate.log" 2>&1; }

checkpoint() {  # commit + push results and the test-only ledger
  git add -f results/*/metrics.csv runs/final_ledger.csv 2>/dev/null
  if ! git diff --cached --quiet; then
    git commit -q -m "Results: $1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q && echo "  pushed: $1"
  fi
}

block_e1() {
  for s in $SEEDS; do train configs/e1_effb0.yaml $s; done
  final e1_effb0; evaluate e1_effb0
  $PY -m skincancer.evaluate --exp e1_effb0 --prior-mix all >> "$LOGS/evaluate.log" 2>&1   # E4 prior shift
  checkpoint "E1/E2 EfficientNet-B0 baseline (+E4 prior-shift sweep)"
  for s in $SEEDS; do train configs/e1_r50.yaml $s; done
  final e1_r50; evaluate e1_r50
  checkpoint "E1/E2 ResNet50 baseline"
}

block_e3() {
  for c in cc clahe crop all; do
    for s in $SEEDS; do train configs/e3_$c.yaml $s; done
    final e3_$c; evaluate e3_$c
  done
  checkpoint "E3 image processing ablation"
}

block_e4() {
  for s in $SEEDS; do train configs/e4_indian_loss.yaml $s; done
  final e4_indian_loss; evaluate e4_indian_loss
  checkpoint "E4 Indian-prior weighted loss"
}

block_e5() {
  for f in 10 25 50 100; do   # equal optimizer steps per fraction (D18)
    for s in $SEEDS; do train configs/e5_finetune_steps.yaml $s --set data.fraction=$f; done
    final e5s_f$f; evaluate e5s_f$f
  done
  checkpoint "E5 localized fine-tuning curve (equal steps)"
}

block_e6() {
  echo "[$(date +%H:%M)] fusion"
  $PY -m skincancer.fusion --source-exp e1_effb0 --seeds $SEEDS --final $FINAL > "$LOGS/e6_fusion.log" 2>&1
  evaluate e6_fusion e6_embed_mlp e6_texture_lr
  checkpoint "E6 texture fusion"
}

block_e7() {
  for l in 100.0 1000.0 10000.0; do
    for s in $SEEDS; do train configs/e7_coral.yaml $s --set train.coral_lambda=$l; done
    final e7_coral_l$l; evaluate e7_coral_l$l
  done
  checkpoint "E7 Deep CORAL lambda sweep"
}

# E8 is chosen by hand on target_val after E3-E7 (see configs/e8_combined.yaml).
BLOCKS=${*:-e1 e3 e4 e5 e6 e7}
for b in $BLOCKS; do "block_$b"; done
echo "[$(date +%H:%M)] all done"
