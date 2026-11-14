# Checkpoints

The full list of ideas by letter, kept and discarded, is in [IDEAS.md](IDEAS.md).

A checkpoint is a model we have decided to keep and build on. Each new idea starts from the latest checkpoint and has to beat it to become the next one. A checkpoint also records everything that was tried on the way to it, so the same experiments are not run twice.

Scores are the mean of three seeds. Details and confidence intervals are in [RESULTS.md](RESULTS.md).

## Names

| Model | Name in `results/` and `runs/` | Built from |
| --- | --- | --- |
| Baseline | `skinnet` | `configs/skinnet.yaml` |
| Checkpoint 1 | `skinnet-c1` | `configs/skinnet-c1.yaml` |
| Checkpoint 2 | `skinnet-c2` | `skinnet-c1`'s three seeds, predicting with test-time flips, and their average |
| Checkpoint 3 | `skinnet-c3` | `configs/skinnet-c3.yaml` |

One name per model, the same for its config, run folder and result folder. A run folder `runs/<name>/` holds the seeds (`s0`, `s1`, `s2`) and their average (`avg`); `results/<name>/metrics.csv` is the model's full score table. An idea being tested is `skinnet-idea-<letter>`; if kept it is renamed to the next checkpoint, `skinnet-c4`, and if discarded it is deleted. The rule is in [DECISIONS.md](../DECISIONS.md), section 15.

## Checkpoint 3 (13 Nov 2026): the current model

Checkpoint 1's recipe trained again with photos from a second kind of source: 552 biopsy-confirmed hospital photos from MIDAS. Predicts the same way as checkpoint 2 (test-time flips, three-seed average).

| | |
| --- | --- |
| Training data | 13,114 images: the 12,562 of checkpoint 1 plus 552 from MIDAS (`midas_train`) |
| Training | 60 epochs, early stopping off, about 20 minutes per seed |
| Config | `configs/skinnet-c3.yaml` |
| Run folder | `runs/skinnet-c3`: seeds in `s0`, `s1`, `s2`, their average in `avg` |
| Weights | `checkpoints/checkpoint-3/` (local, git-ignored) |
| Git tag | `checkpoint-3` |

| Score | Checkpoint 2 | Checkpoint 3 | Paired difference [95% interval] |
| --- | --- | --- | --- |
| DDI (outside, never seen): AUC | 0.629 | **0.657** | +0.028 [+0.007, +0.048] |
| MIDAS test part (414 images): AUC | 0.645 | **0.698** | +0.053 [+0.020, +0.089] |
| DermaCon-IN tumours (outside): AUC | 0.826 | 0.841 | +0.015 [−0.058, +0.090] |
| Our test split: AUC | **0.904** | 0.898 | −0.007 [−0.012, −0.001] |
| Our test split: AUC, cancers vs benign lookalikes | **0.797** | 0.775 | |
| Our test split: cancers missed / false alarms | 9% / 33% | 10% / 34% | |
| DDI: cancers missed / false alarms | 21% / 59% | 16% / 67% | |
| MIDAS test: cancers missed / false alarms | 50% / 31% | 33% / 41% | |
| DermaCon-IN everyday conditions called cancer | **24%** | 28% | |

**Where it stands.** It is the first model here shown to do better on a source it has never seen: DDI. The MIDAS gain is larger, but the model has now seen that hospital's photos, so it says less. It costs a little at home. The threshold fixed on our own validation split still does not carry over: outside, fewer cancers are missed but there are more false alarms.

**How it became a checkpoint.** The rule written beforehand (DECISIONS.md, sections 11 and 12) asked for a supported gain on MIDAS test or DDI, which it has on both, and no supported loss on our own test split, which it missed by a small margin (−0.007). The team decided the trade is worth taking, because performance on new hospitals is the open problem. That is a decision made after seeing the result, recorded here as such.

**From now on** every new idea is compared with checkpoint 3 after the same treatment (three seeds, 60 epochs, test-time flips, seed average), and chosen on the MIDAS validation part as well as our own.

## Checkpoint 2 (12 Nov 2026)

Checkpoint 1's three trained models, used better when predicting. Nothing was retrained.

| | |
| --- | --- |
| What it is | For each image, each of the three seeds predicts on the image and its three mirror images (left-right, up-down, both); the twelve cancer probabilities are averaged into one |
| Weights | The same as checkpoint 1; copies with the prediction settings are in `checkpoints/checkpoint-2/` (local, git-ignored) |
| Run folder | `runs/skinnet-c2`: seeds in `s0`, `s1`, `s2`, their average in `avg` |
| Git tag | `checkpoint-2` |
| Built with | `python -m skincancer.derive` (commands below) |

| Score | Baseline | Checkpoint 1 | Checkpoint 2 |
| --- | --- | --- | --- |
| Our test split: AUC | 0.887 | 0.890 | **0.904** |
| Our test split: AUC, cancers vs benign lookalikes | 0.723 | 0.783 | **0.797** |
| Cancers missed | 9% | 8% | 9% |
| False alarms, all non-cancers | 42% | 40% | **33%** |
| False alarms, PAD-UFES-20 non-cancers | 89% | 63% | 57% |
| False alarms, benign lookalikes | 81% | 65% | 61% |
| MIDAS (outside): AUC | 0.603 | 0.628 | 0.634 |
| DDI (outside): AUC | 0.654 | 0.623 | 0.629 |
| DermaCon-IN tumours (outside): AUC | 0.825 | 0.793 | 0.826 |
| DermaCon-IN everyday conditions called cancer | 35% | 34% | **24%** |

Missed cancers and false alarms use the threshold that catches 90% of cancers on the validation split.

**Where it stands.** On photos like its training data it is a clear step up from checkpoint 1: +0.015 AUC and a sixth fewer false alarms, a gain the paired comparison supports. On photos from other hospitals it is not shown to be better: MIDAS and DDI each moved by +0.006, which cannot be told from no change. The outside gap is still the open problem.

**How it became a checkpoint.** The rule written before the test (DECISIONS.md, section 10) asked for a supported gain on MIDAS or DDI, and by that rule it was first discarded. On seeing the result the team decided that a free, supported gain at home with nothing worse anywhere counts as working, and restored it. That was a change of rule after the result, recorded here so the outside numbers are not read as better than they are.

**Until checkpoint 3** every new idea was compared with checkpoint 2, and got the same treatment before the comparison: three seeds, test-time flips, seed average.

## Checkpoint 1 (12 Nov 2026)

| | |
| --- | --- |
| Task | Cancer or not cancer, from a clinical skin photo |
| Network | SkinNet: our own residual CNN, 2.8M parameters, trained from random weights |
| Training data | 12,562 images (2,220 cancer) from Fitzpatrick17k and PAD-UFES-20 |
| Training | 60 epochs at 224 × 224, AdamW, about 20 minutes per seed on an RTX 4060 |
| What sets it apart from the baseline | Inside each dataset, cancer and not cancer are drawn equally often, so the kind of photo no longer predicts the class |
| Config | `configs/skinnet-c1.yaml` |
| Git tag | `checkpoint-1` |
| Weights | `checkpoints/checkpoint-1/s{0,1,2}/best.pt` (local, git-ignored) |

It weakened the "smartphone photo means cancer" shortcut (false alarms on PAD-UFES-20 non-cancers 89% → 63%) and improved the hard distinction, cancer against a benign lookalike (0.72 → 0.78). Its scores are in the table above. Checkpoint 2 uses its weights unchanged.

## What was tried to get here: do not repeat

Seven models were trained on the way to checkpoint 1, and two ideas for using it differently led to checkpoint 2. Everything that was not kept is listed with the reason, so nobody spends time on them again unless something else has changed (see the last section).

| # | What was tried | Compared on | Result | Verdict |
| --- | --- | --- | --- | --- |
| 0 | **Baseline.** Every image once per epoch, class-weighted loss | test and outside sets, 3 seeds | Test AUC 0.887; MIDAS 0.60, DDI 0.65. Learned "smartphone photo means cancer": 89% false alarms on PAD-UFES-20 non-cancers | Replaced by A |
| A | **Balanced drawing inside each dataset** | test and outside sets, 3 seeds | Shortcut weakened (89% → 63%), cancers vs lookalikes 0.72 → 0.78. Outside sets unchanged | **Kept: checkpoint 1** |
| B | A + hard lesion types drawn more often (share ∝ square root of count) | validation, 1 seed | Within-source AUC 0.847 against A's 0.869. Fewer false alarms on lookalikes (53% against 64%) but more on everything else | Discarded |
| C | B + random loss of resolution and JPEG quality in training | validation, 1 seed | 0.853 against 0.869. Same trade-off as B | Discarded |
| D | A + Derm7pt (958 atlas photos: 294 cancers, 664 benign) as training-only data | validation, 1 seed | 0.857 against 0.869 | Discarded |
| E | A at 320 × 320 pixels | validation, 1 seed | 0.856 against 0.869; stopped early with no gain | Discarded |
| F | A + Derm7pt at 320 × 320 pixels | validation, 1 seed | 0.862 against 0.869 | Discarded |
| G | Average the three seeds of checkpoint 1 into one prediction | validation | Within-source AUC 0.882 against 0.868 | Kept as part of G+H |
| H | Test-time flips: predict on the image and its three mirror images, average | validation | 0.879 against 0.868 | Kept as part of G+H |
| J | Checkpoint 1's recipe with 552 MIDAS training photos added | validation only (ours and the MIDAS validation part) | MIDAS validation AUC 0.689 → 0.782, but within-source AUC on our validation split 0.888 → 0.876, a supported loss, mostly one seed stopped early at epoch 25 | Discarded at the first step; not scored on test sets. Replaced by O |
| O | J with early stopping off (all seeds run 60 epochs) | test and outside sets, paired comparison with checkpoint 2 | MIDAS test 0.645 → 0.698 and DDI 0.629 → 0.657, both supported. Our test split 0.904 → 0.898, a small supported loss | **Kept: checkpoint 3** (see above for how) |
| I | Checkpoint 3's recipe with 2,907 SCIN phone photos of everyday conditions added as not cancer | validation only (ours and the MIDAS validation part), paired comparison with checkpoint 3 | Our validation split: within-source AUC 0.882 → 0.902, supported. MIDAS validation part: AUC 0.793 → 0.729, a supported loss | Discarded at the first step; not scored on test sets |
| M | Checkpoint 3's recipe with the network at twice the width (11.2 million parameters against 2.8 million) | validation only, paired comparison with checkpoint 3 | Our validation split: within-source AUC 0.882 → 0.885, not supported. MIDAS validation part: AUC 0.793 → 0.773, a supported loss | Discarded at the first step (14 Nov); not scored on test sets |
| N | Checkpoint 3's recipe trained for 120 epochs, not 60 | validation, then test and outside sets, paired comparison with checkpoint 3 | Our test split: AUC 0.898 → 0.930, supported. MIDAS test 0.698 → 0.733, not supported. DDI 0.657 → 0.592, a supported loss | Discarded (14 Nov): the extra training learns our home datasets, not other hospitals |
| G+H | Both together | test and outside sets, paired comparison | Our test split: AUC 0.890 → 0.904, false alarms 40% → 33%, a supported gain. MIDAS +0.006 and DDI +0.006, neither supported | **Kept: checkpoint 2** (see above for how) |

The deciding score on validation was the within-source AUC: the AUC inside each dataset, averaged. Seeds of one model differ by about 0.01 on it, so B to F are "no better than A", not "clearly worse".

**What these runs established**

1. **Balancing cancer and not cancer inside each dataset is worth keeping.** It removes the reward for recognising the kind of photo. Every later model should start with it.
2. **Drawing hard lesion types more often is a trade, not a gain.** It cuts false alarms on benign lookalikes by about a fifth and raises them elsewhere. It is only worth revisiting if mistaking moles for cancer becomes the main complaint.
3. **Degrading image quality in training does not help.**
4. **Derm7pt does not help.** One more atlas of moles and melanomas adds little the network had not seen. Its list is still on disk (`data/extra/derm7pt.csv`) and was checked: none of its images duplicates a validation or test image.
5. **Larger images do not help at this network size.** Image detail is not what limits the model.
6. **Averaging seeds and test-time flips are a real but home-only gain.** Together they add about 0.015 AUC on our own test split and cut false alarms there by a sixth, at no training cost. On MIDAS and DDI the change is +0.006, too small to tell from zero. They are now the standard way of predicting (checkpoint 2), but they do not make a model travel better.
7. **Nothing done with these two data sources moved the outside scores.** The gap between 0.89 at home and about 0.62 outside comes from how different our training images are from other hospitals' photos.
8. **Choosing models on our own validation split cannot find outside gains,** because that split comes from the same two sources as training.
9. **Photos from another hospital in training are the first thing that moved an outside score.** Adding 552 MIDAS photos raised AUC on DDI, a source never seen, by 0.028, and on the MIDAS test part by 0.053, at a cost of 0.007 at home. More sources of training photos is the direction that works.
10. **Early stopping can cut a good run short.** With the learning rate still high the validation score is noisy; one seed stopped at epoch 25 would have improved by 0.03 if left to finish. Train all 60 epochs and keep the best weights.
11. **Easy not-cancer photos help at home and hurt on tumours.** SCIN's everyday rashes raised our own validation score by 0.02 and lowered the MIDAS validation score by 0.06. New data has to contain the hard distinction, cancer against a benign growth, to help outside. SCIN at a much smaller share of the draws is untested.

**Also settled along the way** (reasons are in [DECISIONS.md](../DECISIONS.md))

- A five-class version (BCC / SCC / melanoma / benign / other) was set up and dropped before any result, in favour of cancer / not cancer as stage one.
- Skin-tone analysis is left for a later stage.
- Training on the CPU takes about 7 minutes per epoch; on the GPU, 18 seconds. Always train on the GPU.

## Ideas not tried yet

Each idea has a letter, continuing from A to H above. The rule is the same for every one:

- **If it works, it becomes the next checkpoint.** "Works" means a gain over the current checkpoint that the paired comparison supports, with nothing clearly worse elsewhere. Whether the gain is at home or on the outside sets is stated with it; a gain on the outside sets is the one we are after.
- **If it does not, it is discarded:** its config and trained models are deleted and it moves to the "do not repeat" table with its result and the reason.
- One idea at a time where possible, so a gain or a loss can be traced to one change.

**Setup step: done (13 Nov 2026).** MIDAS is split by patient into a training part (552 images), an outside validation part (414, never trained on, used to choose models) and a test part (414). DDI is untouched. Ideas are now chosen on the outside validation part as well as our own.

| # | Idea | What it targets | Cost | Expected effect |
| --- | --- | --- | --- | --- |
| K | **Add ISIC 2019 to training:** about 8,500 cancers, 12,900 nevi | Far more cancers | Large download; dermoscopy, a different kind of image | Uncertain: may not carry over to ordinary photos |
| L | **Self-supervised pretraining on our own images:** learn general skin features from all our photos without labels, then train cancer / not cancer. No outside weights | Getting more from the images we have | New training code; a few GPU hours | A real gain is plausible; the standard remedy for a small labelled set |
| Q | **More biopsy-confirmed tumour photos from new sources** (clinical, not dermoscopy) | The outside gap: the only lever that has moved DDI | Finding and checking sources | The most likely to help outside, if sources exist |
| P | **Pretrained (ImageNet) weights** | The outside gap | About 3.5 hours for two ResNet-18 models (ImageNet start and random start). Run as a reference comparison only, never a checkpoint (team decision, 14 Nov) | Large, by everything published |

Counts for SCIN and ISIC 2019 are from memory and must be confirmed at download. Data details are in [DATA_PLAN.md](DATA_PLAN.md).

**Order (14 Nov 2026):** N and M are done, both discarded (table above). Q is running (MILK10k, section 18), then L (widened to unlabelled photos already on disk), then P as a reference comparison only: ResNet-18 from ImageNet weights and from random weights, never a checkpoint. K only if Q becomes a checkpoint with a gain outside. From Q on, step 1 needs a supported gain on the MIDAS validation part. Reasons are in [DECISIONS.md](../DECISIONS.md), sections 16 and 19.

**Later stages, not ideas for this one:** which type of cancer it is, and whether darker skin needs its own model or one combined model.

**A limit to keep in mind.** The largest known lever for performance on new hospitals is starting from pretrained weights, which this project has ruled out. A from-scratch network of this size on this much data is likely to stay well below what pretrained models reach on DDI and MIDAS, whatever comes of I to N.

**When a discarded idea is worth another look:** if the training data changes a lot (new sources) or the network changes a lot (much larger), results B to H no longer apply as they stand, because they were all measured with this data and this network.

## How to use a checkpoint

```bash
# checkpoint 1, one seed (weights are local, not in git)
python -m skincancer.predict --run checkpoints/checkpoint-1/s0 --splits val

# checkpoint 2: rebuild it from checkpoint 1's runs
python -m skincancer.derive tta --exp skinnet-c1 --name skinnet-c2
for s in 0 1 2; do python -m skincancer.predict --run runs/skinnet-c2/s$s --splits val test; done
python -m skincancer.derive ensemble --exp skinnet-c2 --splits val test
python -m skincancer.compare --a skinnet-c1 --b skinnet-c2 --splits val test

# checkpoint 3: train it (flips are in its config), then average the seeds
for s in 0 1 2; do python -m skincancer.train --config configs/skinnet-c3.yaml --seed $s; done
python -m skincancer.derive ensemble --exp skinnet-c3 --splits val test midas_val

# return the code to the state that produced a checkpoint
git checkout checkpoint-3
```

The weights are not in the repository (`checkpoints/` is git-ignored). Retraining from the tagged code with seeds 0, 1 and 2 reproduces them.
