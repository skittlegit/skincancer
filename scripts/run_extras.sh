#!/usr/bin/env bash
# Follow-up queue after run_all.sh (D15), with a commit + push checkpoint after each step:
#   1. score E3-E7 on midas_test (added after they were queued; superseded E5 runs skipped)
#   2. E5 with equal optimizer steps per fraction (D18), 4 fractions x 3 seeds
#   3. E1 + Derm7pt: E1 with Derm7pt clinical photos added to source_train only (D16)
#   4. E8 ResNet50 (D23; E8 EfficientNet-B0 is e5s_f100)
#   5. E1-strong: EfficientNet-B0 at 320 px + flip TTA, 3 seeds
#   6. E1-FM: Derm Foundation embeddings + heads (skipped until Hugging Face access works)
#   7. seed ensembles for every finished experiment
#   PY=.venv/Scripts/python bash scripts/run_extras.sh            # waits for run_all to finish
#   PY=.venv/Scripts/python bash scripts/run_extras.sh d7 e1s fm  # chosen steps only, no wait
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

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && { if [ -n "${NO_PUSH:-}" ]; then echo "  committed (not pushed): $1"; else git push -q && echo "  pushed: $1"; fi; }
  fi
}

step_retry() {  # re-run every run_all block that logged a FAILED training run (e.g. out of memory)
  local blocks
  # a failure counts only if that run was not trained again later in the log (e.g. by a restart)
  blocks=$(awk '/ train /{t=$NF; delete bad[t]} /FAILED: see runs\/logs\//{f=$NF; sub(/.*\/logs\//,"",f); sub(/\.log$/,"",f); bad[f]=1}
      END{for (f in bad) print f}' runs/run_all.out 2>/dev/null \
    | sed -E 's/^(e1)_.*/\1/; s/^(e[3-7])[a-z]?_.*/\1/' | sort -u | tr '\n' ' ')
  [ -n "$blocks" ] || return 0
  for d in runs/*/s*/; do   # drop crashed partial runs so they retrain cleanly
    [ -f "$d/best.pt" ] && [ ! -f "$d/summary.json" ] && [ ! -d "$d/preds" ] && rm -rf "$d"
  done
  echo "[$(date +%H:%M)] retry blocks: $blocks"
  bash scripts/run_all.sh $blocks >> runs/run_all_retry.out 2>&1
  grep -E "FAILED|committed|pushed" runs/run_all_retry.out | tail -5
}

step_repair() {  # runs that crashed after saving best.pt (e.g. a segfault while predicting)
  step_retry       # first re-run blocks with crashed training runs (the chain calls repair first)
  for d in runs/*/s*/; do
    d=${d%/}
    [ -f "$d/best.pt" ] && [ -f "$d/summary.json" ] || continue
    missing=""
    for sp in source_val source_test target_val target_test ak_side; do
      [ -f "$d/preds/$sp.csv" ] || missing="$missing $sp"
    done
    [ -n "$missing" ] || continue
    echo "[$(date +%H:%M)] repair $d:$missing"
    $PY -m skincancer.predict --run "$d" --splits $missing >> "$LOGS/repair.log" 2>&1 || echo "  FAILED: see $LOGS/repair.log"
    e=${d#runs/}; e=${e%%/*}
    $PY -m skincancer.evaluate --exp "$e" >> "$LOGS/evaluate.log" 2>&1
  done
  checkpoint "repair missing predictions"
}

step_midas() {
  echo "[$(date +%H:%M)] midas_test for E3-E7"
  bash scripts/score_new_split.sh midas_test e3_clahe e3_crop e3_all e4_indian_loss \
    e7_coral_l100.0 e7_coral_l1000.0 e7_coral_l10000.0
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
  $PY -m skincancer.evaluate --exp e3_clahe e3_crop e3_all e4_indian_loss \
    e6_fusion e6_embed_mlp e6_texture_lr e7_coral_l100.0 e7_coral_l1000.0 e7_coral_l10000.0 >> "$LOGS/evaluate.log" 2>&1
  $PY -m skincancer.evaluate --exp e1_effb0 --prior-mix all >> "$LOGS/evaluate.log" 2>&1
  checkpoint "E3-E7 on MIDAS"
}

train_arm() {  # train_arm <config> <exp> <message>: 3 seeds, score once, evaluate, checkpoint
  for s in $SEEDS; do
    echo "[$(date +%H:%M)] train ${2}_s$s"
    $PY -m skincancer.train --config "$1" --seed $s > "$LOGS/${2}_s$s.log" 2>&1 || echo "  FAILED: see $LOGS/${2}_s$s.log"
  done
  for s in $SEEDS; do
    [ -f "runs/$2/s$s/best.pt" ] && $PY -m skincancer.predict --run "runs/$2/s$s" \
      --splits $FINAL --final >> "$LOGS/final_$2.log" 2>&1
  done
  $PY -m skincancer.evaluate --exp "$2" >> "$LOGS/evaluate.log" 2>&1
  checkpoint "$3"
}

step_d7() {
  [ -f data/extra/derm7pt.csv ] || { echo "[$(date +%H:%M)] E1 + Derm7pt skipped: data/extra/derm7pt.csv missing"; return; }
  train_arm configs/e1_effb0_d7.yaml e1_effb0_d7 "E1 + Derm7pt (training-only extra data)"
}

step_e5s() {
  for f in 10 25 50 100; do
    for s in $SEEDS; do
      echo "[$(date +%H:%M)] train e5s_f${f}_s$s"
      $PY -m skincancer.train --config configs/e5_finetune_steps.yaml --seed $s --set data.fraction=$f \
        > "$LOGS/e5s_f${f}_s$s.log" 2>&1 || echo "  FAILED: see $LOGS/e5s_f${f}_s$s.log"
    done
    for s in $SEEDS; do
      [ -f "runs/e5s_f$f/s$s/best.pt" ] && $PY -m skincancer.predict --run "runs/e5s_f$f/s$s" \
        --splits $FINAL --final >> "$LOGS/final_e5s_f$f.log" 2>&1
    done
    $PY -m skincancer.evaluate --exp "e5s_f$f" >> "$LOGS/evaluate.log" 2>&1
  done
  checkpoint "E5 fine-tuning curve, equal steps per fraction (D18)"
}

step_e1s() { train_arm configs/e1_effb0_320.yaml e1_effb0_320 "E1-strong (320 px + flip TTA)"; }

step_e8() {  # D23: E8 = the E5 recipe; on EfficientNet-B0 that is e5s_f100, so only ResNet50 trains
  train_arm configs/e8_r50.yaml e8_r50 "E8 ResNet50 (E5 recipe, D23)"
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
  STEPS="repair midas e5s d7 e8 e1s fm ens"
else
  STEPS="$*"
fi
for s in $STEPS; do "step_$s"; done
echo "[$(date +%H:%M)] extras done"
