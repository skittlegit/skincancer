#!/usr/bin/env bash
# Follow-up queue after run_all.sh (D15), with a commit + push checkpoint after each step:
#   1. score E3-E7 on midas_test (added after they were queued)
#   2. E1-strong: EfficientNet-B0 at 320 px + flip TTA, 3 seeds
#   3. E1-FM: Derm Foundation embeddings + heads (skipped until Hugging Face access works)
#   4. seed ensembles for every finished experiment
#   PY=.venv/Scripts/python bash scripts/run_extras.sh            # waits for run_all to finish
#   PY=.venv/Scripts/python bash scripts/run_extras.sh e1s fm     # chosen steps only, no wait
set -uo pipefail
PY=${PY:-python}
PYTF=${PYTF:-.venv-tf/Scripts/python}
SEEDS="0 1 2"
FINAL="ddi_test dermacon_test dermacon_other pad_dark_test midas_test"
LOGS=runs/logs
mkdir -p "$LOGS"
export PYTHONUNBUFFERED=1

checkpoint() {
  git add -f results/*/metrics.csv runs/final_ledger.csv 2>/dev/null
  if ! git diff --cached --quiet; then
    git commit -q -m "Results: $1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q && echo "  pushed: $1"
  fi
}

step_midas() {
  echo "[$(date +%H:%M)] midas_test for E3-E7"
  bash scripts/score_new_split.sh midas_test e3_clahe e3_crop e3_all e4_indian_loss \
    e5_f10 e5_f25 e5_f50 e5_f100 e7_coral_l100.0 e7_coral_l1000.0 e7_coral_l10000.0
  # E6 heads have no checkpoint: refit them (deterministic per seed; checked against the
  # saved source_val predictions) and score midas_test only.
  for n in e6_fusion e6_embed_mlp e6_texture_lr; do
    for s in $SEEDS; do cp "runs/$n/s$s/preds/source_val.csv" "runs/$n/s$s/source_val.before.csv" 2>/dev/null; done
  done
  $PY -m skincancer.fusion --source-exp e1_effb0 --seeds $SEEDS --final midas_test >> "$LOGS/e6_fusion.log" 2>&1 \
    || echo "  FAILED: e6 midas (see $LOGS/e6_fusion.log)"
  $PY - <<'EOF'
import glob, numpy as np, pandas as pd
for f in sorted(glob.glob("runs/e6_*/s*/source_val.before.csv")):
    a, b = pd.read_csv(f), pd.read_csv(f.replace("source_val.before.csv", "preds/source_val.csv"))
    cols = [c for c in a if c.startswith("p_")]
    print(f, "max |diff| vs saved source_val preds:", float(np.abs(a[cols].values - b[cols].values).max()))
EOF
  $PY -m skincancer.evaluate --exp e3_clahe e3_crop e3_all e4_indian_loss e5_f10 e5_f25 e5_f50 e5_f100 \
    e6_fusion e6_embed_mlp e6_texture_lr e7_coral_l100.0 e7_coral_l1000.0 e7_coral_l10000.0 >> "$LOGS/evaluate.log" 2>&1
  $PY -m skincancer.evaluate --exp e1_effb0 --prior-mix all >> "$LOGS/evaluate.log" 2>&1
  checkpoint "E3-E7 on MIDAS"
}

step_e1s() {
  for s in $SEEDS; do
    echo "[$(date +%H:%M)] train e1_effb0_320_s$s"
    $PY -m skincancer.train --config configs/e1_effb0_320.yaml --seed $s > "$LOGS/e1_effb0_320_s$s.log" 2>&1 \
      || echo "  FAILED: see $LOGS/e1_effb0_320_s$s.log"
  done
  for s in $SEEDS; do
    [ -f "runs/e1_effb0_320/s$s/best.pt" ] && $PY -m skincancer.predict --run "runs/e1_effb0_320/s$s" \
      --splits $FINAL --final >> "$LOGS/final_e1_effb0_320.log" 2>&1
  done
  $PY -m skincancer.evaluate --exp e1_effb0_320 >> "$LOGS/evaluate.log" 2>&1
  checkpoint "E1-strong (320 px + flip TTA)"
}

step_fm() {
  if ! "$PYTF" -c "from huggingface_hub import hf_hub_download; hf_hub_download('google/derm-foundation', 'saved_model.pb')" \
      > "$LOGS/fm_access.log" 2>&1; then
    echo "[$(date +%H:%M)] E1-FM skipped: no Hugging Face access to google/derm-foundation yet (see $LOGS/fm_access.log)"
    return
  fi
  echo "[$(date +%H:%M)] Derm Foundation embeddings"
  "$PYTF" scripts/derm_foundation_embed.py > "$LOGS/fm_embed.log" 2>&1 || { echo "  FAILED: see $LOGS/fm_embed.log"; return; }
  echo "[$(date +%H:%M)] E1-FM heads"
  $PY -m skincancer.foundation --seeds $SEEDS --final $FINAL > "$LOGS/fm_heads.log" 2>&1 \
    || { echo "  FAILED: see $LOGS/fm_heads.log"; return; }
  $PY -m skincancer.evaluate --exp e1_fm_lr e1_fm_mlp >> "$LOGS/evaluate.log" 2>&1
  checkpoint "E1-FM (Derm Foundation embeddings)"
}

step_ens() {
  echo "[$(date +%H:%M)] seed ensembles"
  exps=$(for d in runs/*/s1/preds; do e=${d#runs/}; echo "${e%%/*}"; done | grep -v "^pilot")
  $PY -m skincancer.evaluate --exp $exps --ensemble >> "$LOGS/ensemble.log" 2>&1
  checkpoint "seed ensembles"
}

if [ $# -eq 0 ]; then
  until grep -q "all done" runs/run_all.out 2>/dev/null; do sleep 60; done
  STEPS="midas e1s fm ens"
else
  STEPS="$*"
fi
for s in $STEPS; do "step_$s"; done
echo "[$(date +%H:%M)] extras done"
