#!/usr/bin/env bash
# Every experiment in order, 3 seeds each. Safe to re-run: finished runs are skipped.
# Run sections separately on Kaggle/Colab sessions; each section only needs the ones above it.
set -euo pipefail
SEEDS="0 1 2"
T="python -m skincancer.train"
FINAL="ddi_test dermacon_test dermacon_other pad_dark_test"

# --- E1 / E2: source-only baselines (E2 = their target / test-only scores) --------------
for s in $SEEDS; do $T --config configs/e1_effb0.yaml --seed $s; $T --config configs/e1_r50.yaml --seed $s; done

# --- E3: image processing, one step at a time and all together -------------------------
for c in cc clahe crop all; do for s in $SEEDS; do $T --config configs/e3_$c.yaml --seed $s; done; done

# --- E4: Indian priors (weighted loss; prior-shift needs no training, see evaluate) ------
for s in $SEEDS; do $T --config configs/e4_indian_loss.yaml --seed $s; done

# --- E5: localized fine-tuning curve (0% = E1) ------------------------------------------
for f in 10 25 50 100; do for s in $SEEDS; do
  $T --config configs/e5_finetune.yaml --seed $s --set data.fraction=$f; done; done

# --- E6: texture fusion on cached features ----------------------------------------------
python -m skincancer.fusion --source-exp e1_effb0 --seeds $SEEDS

# --- E7: Deep CORAL; pick λ on target_val; λ=1000 is the untuned label-free number (D13) --
for l in 100.0 1000.0 10000.0; do for s in $SEEDS; do
  $T --config configs/e7_coral.yaml --seed $s --set train.coral_lambda=$l; done; done

# --- Evaluate everything on the non-final splits ----------------------------------------
python -m skincancer.evaluate --exp e1_effb0 e1_r50 e3_cc e3_clahe e3_crop e3_all e4_indian_loss \
  e5_f10 e5_f25 e5_f50 e5_f100 e6_fusion e6_embed_mlp e6_texture_lr \
  e7_coral_l100.0 e7_coral_l1000.0 e7_coral_l10000.0
python -m skincancer.evaluate --exp e1_effb0 --prior-mix all

# --- E8 and the test-only sets: by hand, once, after choosing on target_val --------------
#   edit configs/e8_combined.yaml, train it for both backbones, then for each finished method:
#   for s in $SEEDS; do python -m skincancer.predict --run runs/<exp>/s$s --splits $FINAL --final; done
#   python -m skincancer.fusion --source-exp e1_effb0 --seeds $SEEDS --final $FINAL
#   python -m skincancer.evaluate --exp <exp> ...
