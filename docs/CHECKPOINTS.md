# Checkpoints

A checkpoint is a model we have decided to keep and build on. Each new round starts from the latest checkpoint and has to beat it to become the next one. Scores are the mean of three seeds; details and confidence intervals are in [RESULTS.md](RESULTS.md).

| | Baseline | Checkpoint 1 = checkpoint 2 |
| --- | --- | --- |
| Date | 12 Nov 2026 | 12 Nov 2026 |
| Config | `configs/skinnet.yaml` | `configs/skinnet_src.yaml` |
| What it is | SkinNet from random weights, class-weighted loss | Baseline, but cancer and not cancer are drawn equally often inside each dataset |
| Git tag | — | `checkpoint-1`, `checkpoint-2` |
| Weights | `runs/skinnet/s{0,1,2}/best.pt` | `checkpoints/checkpoint-1/s{0,1,2}/best.pt` |
| Our test split: AUC | 0.887 | 0.890 |
| Our test split: AUC, cancers vs benign lookalikes | 0.723 | 0.783 |
| False alarms, PAD-UFES-20 non-cancers | 89% | 63% |
| False alarms, benign lookalikes | 81% | 65% |
| MIDAS: AUC | 0.603 | 0.628 |
| DDI: AUC | 0.654 | 0.623 |
| DermaCon-IN tumours: AUC | 0.825 | 0.793 |

False alarms use the threshold that catches 90% of cancers on the validation split.

## Checkpoint 1

Balanced drawing weakened the "smartphone photo means cancer" shortcut and improved the hard distinction on our own test split. The outside sets (MIDAS, DDI, DermaCon-IN) did not move: every difference from the baseline is inside the confidence intervals.

Two other variants were tried in the same round (hard lesion types drawn more often, and random loss of image quality). Both lost to this one on the validation split and were deleted.

## Checkpoint 2

The same model as checkpoint 1. The third round tried three ways to improve it: Derm7pt added to training, images at 320 pixels, and both. On the validation split none beat checkpoint 1 (0.857, 0.856 and 0.862 against 0.869 on within-source AUC), so by the rule fixed beforehand it carries forward unchanged. The three variants were not scored on the test split or the outside sets, and were deleted.

Nothing was retrained, so the weights are the ones in `checkpoints/checkpoint-1/`.

## How to use a checkpoint

```bash
# score images with checkpoint 1, seed 0 (weights are local, not in git)
python -m skincancer.predict --run checkpoints/checkpoint-1/s0 --splits val
# return the code to the state that produced it
git checkout checkpoint-1
```

The weights are not in the repository (`checkpoints/` is git-ignored). Retraining from the tagged code with seeds 0, 1 and 2 reproduces them.
