# Checkpoints

A checkpoint is a model we have decided to keep and build on. Each new idea starts from the latest checkpoint and has to beat it to become the next one. A checkpoint also records everything that was tried on the way to it, so the same experiments are not run twice.

Scores are the mean of three seeds. Details and confidence intervals are in [RESULTS.md](RESULTS.md).

## Checkpoint 1 (12 Nov 2026): the current model

| | |
| --- | --- |
| Task | Cancer or not cancer, from a clinical skin photo |
| Network | SkinNet: our own residual CNN, 2.8M parameters, trained from random weights |
| Training data | 12,562 images (2,220 cancer) from Fitzpatrick17k and PAD-UFES-20 |
| Training | 60 epochs at 224 × 224, AdamW, about 20 minutes per seed on an RTX 4060 |
| What sets it apart from the baseline | Inside each dataset, cancer and not cancer are drawn equally often, so the kind of photo no longer predicts the class |
| Config | `configs/skinnet_src.yaml` |
| Git tag | `checkpoint-1` |
| Weights | `checkpoints/checkpoint-1/s{0,1,2}/best.pt` (local, git-ignored) |

| Score | Baseline | Checkpoint 1 |
| --- | --- | --- |
| Our test split: AUC | 0.887 | 0.890 |
| Our test split: AUC, cancers vs benign lookalikes | 0.723 | 0.783 |
| Cancers missed | 9% | 8% |
| False alarms, all non-cancers | 42% | 40% |
| False alarms, PAD-UFES-20 non-cancers | 89% | 63% |
| False alarms, benign lookalikes | 81% | 65% |
| MIDAS (outside): AUC | 0.603 | 0.628 |
| DDI (outside): AUC | 0.654 | 0.623 |
| DermaCon-IN tumours (outside): AUC | 0.825 | 0.793 |

Missed cancers and false alarms use the threshold that catches 90% of cancers on the validation split.

**Where it stands.** On photos like its training data it is a usable screen. On photos from other hospitals it is little better than the baseline, and not far above guessing on DDI and MIDAS. That gap is the open problem.

## What was tried to get here: do not repeat

Seven models were trained on the way to checkpoint 1. The six that were not kept are listed with the reason, so nobody spends time on them again unless something else has changed (see the last section).

| # | What was tried | Compared on | Result | Verdict |
| --- | --- | --- | --- | --- |
| 0 | **Baseline.** Every image once per epoch, class-weighted loss | test and outside sets, 3 seeds | Test AUC 0.887; MIDAS 0.60, DDI 0.65. Learned "smartphone photo means cancer": 89% false alarms on PAD-UFES-20 non-cancers | Replaced by A |
| A | **Balanced drawing inside each dataset** | test and outside sets, 3 seeds | Shortcut weakened (89% → 63%), cancers vs lookalikes 0.72 → 0.78. Outside sets unchanged | **Kept: checkpoint 1** |
| B | A + hard lesion types drawn more often (share ∝ square root of count) | validation, 1 seed | Within-source AUC 0.847 against A's 0.869. Fewer false alarms on lookalikes (53% against 64%) but more on everything else | Discarded |
| C | B + random loss of resolution and JPEG quality in training | validation, 1 seed | 0.853 against 0.869. Same trade-off as B | Discarded |
| D | A + Derm7pt (958 atlas photos: 294 cancers, 664 benign) as training-only data | validation, 1 seed | 0.857 against 0.869 | Discarded |
| E | A at 320 × 320 pixels | validation, 1 seed | 0.856 against 0.869; stopped early with no gain | Discarded |
| F | A + Derm7pt at 320 × 320 pixels | validation, 1 seed | 0.862 against 0.869 | Discarded |

The deciding score on validation was the within-source AUC: the AUC inside each dataset, averaged. Seeds of one model differ by about 0.01 on it, so B to F are "no better than A", not "clearly worse".

**What these seven runs established**

1. **Balancing cancer and not cancer inside each dataset is worth keeping.** It removes the reward for recognising the kind of photo. Every later model should start with it.
2. **Drawing hard lesion types more often is a trade, not a gain.** It cuts false alarms on benign lookalikes by about a fifth and raises them elsewhere. It is only worth revisiting if mistaking moles for cancer becomes the main complaint.
3. **Degrading image quality in training does not help.**
4. **Derm7pt does not help.** One more atlas of moles and melanomas adds little the network had not seen. Its list is still on disk (`data/extra/derm7pt.csv`) and was checked: none of its images duplicates a validation or test image.
5. **Larger images do not help at this network size.** Image detail is not what limits the model.
6. **Nothing done with these two data sources moved the outside scores.** The gap between 0.89 at home and about 0.62 outside comes from how different our training images are from other hospitals' photos.
7. **Choosing models on our own validation split cannot find outside gains,** because that split comes from the same two sources as training.

**Also settled along the way** (reasons are in [DECISIONS.md](../DECISIONS.md))

- A five-class version (BCC / SCC / melanoma / benign / other) was set up and dropped before any result, in favour of cancer / not cancer as stage one.
- Skin-tone analysis is left for a later stage.
- Training on the CPU takes about 7 minutes per epoch; on the GPU, 18 seconds. Always train on the GPU.

## Ideas not tried yet

Each idea has a letter, continuing from A to F above. The rule is the same for every one:

- **If it works, it becomes the next checkpoint.** "Works" means it beats the current checkpoint by more than the seed-to-seed spread on the score fixed beforehand, and the gain shows on the outside sets, not only on our own test split.
- **If it does not, it is discarded:** its config and trained models are deleted and it moves to the "do not repeat" table with its result and the reason.
- One idea at a time where possible, so a gain or a loss can be traced to one change.

**Setup step, before any lettered idea.** Set aside 30% of MIDAS, by patient, as an outside validation set. Nothing trains on it; models are chosen on it. The other 70% of MIDAS and all of DDI stay untouched tests. This is a change to how we measure, so it is not an experiment that can pass or fail, but without it the ideas below are judged on a split that cannot see outside gains. It cannot be undone, and is waiting for a decision.

| # | Idea | What it targets | Cost | Expected effect |
| --- | --- | --- | --- | --- |
| G | **Average the three seeds** into one prediction | Noise in a single model | Minutes, no retraining | Small gain everywhere |
| H | **Test-time flips:** predict on the image and its mirror images, average | Sensitivity to framing | Minutes, no retraining | Small gain everywhere |
| I | **Add SCIN to training:** about 10,000 phone photos from the public, almost all not cancer | False alarms on everyday photos; a third kind of source | Download, label mapping, duplicate check, one training round | Fewer false alarms outside; no new cancers |
| J | **Add part of MIDAS to training** (the share left after the outside validation and test parts) | Biopsy-confirmed hospital photos, the closest we have to the outside test images | No download; shrinks the MIDAS test set further | The most direct attack on the outside gap |
| K | **Add ISIC 2019 to training:** about 8,500 cancers, 12,900 nevi | Far more cancers | Large download; dermoscopy, a different kind of image | Uncertain: may not carry over to ordinary photos |
| L | **Self-supervised pretraining on our own images:** learn general skin features from all our photos without labels, then train cancer / not cancer. No outside weights | Getting more from the images we have | New training code; a few GPU hours | A real gain is plausible; the standard remedy for a small labelled set |
| M | **A larger network** (about four times the parameters) | Capacity | One training round, longer epochs | Modest at home, little outside |
| N | **A longer schedule** (more than 60 epochs) | Training was still improving slightly at epoch 60 | One longer training round | Small |

Counts for SCIN and ISIC 2019 are from memory and must be confirmed at download. Data details are in [DATA_PLAN.md](DATA_PLAN.md).

**Suggested order:** the setup step, then G and H together (same day), then I and J, then L, then M and N. K last, as a with-and-without experiment.

**Later stages, not ideas for this one:** which type of cancer it is, and whether darker skin needs its own model or one combined model.

**A limit to keep in mind.** The largest known lever for performance on new hospitals is starting from pretrained weights, which this project has ruled out. A from-scratch network of this size on this much data is likely to stay well below what pretrained models reach on DDI and MIDAS, whatever comes of G to N.

**When a discarded idea is worth another look:** if the training data changes a lot (new sources) or the network changes a lot (much larger), results B to F no longer apply as they stand, because they were all measured with this data and this network.

## How to use a checkpoint

```bash
# score images with checkpoint 1, seed 0 (weights are local, not in git)
python -m skincancer.predict --run checkpoints/checkpoint-1/s0 --splits val
# return the code to the state that produced it
git checkout checkpoint-1
```

The weights are not in the repository (`checkpoints/` is git-ignored). Retraining from the tagged code with seeds 0, 1 and 2 reproduces them.
