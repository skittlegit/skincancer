# Decisions

The rules this project runs by. The code enforces them. Changing a rule means changing it here, in the code, and re-running whatever it touches.

## 1. Fresh start (12 Nov 2026)

The project was restarted on 12 Nov 2026. The classifier is now one convolutional network of our own (SkinNet), trained from random weights. Nothing in it is pretrained and no layer is frozen.

- Everything from before the restart was removed from the working tree: the pretrained-backbone fine-tuning runs, image-processing ablation, Indian-prior loss, texture fusion, Deep CORAL, Derm Foundation heads, model ensembles, the "not a tumour" gates, and their results, figures and write-ups.
- That work is kept in git under the tag `pre-fresh-start` (`git checkout pre-fresh-start`). Its local run folders, results and figures were moved to `archive/pre-fresh-start/`, which git ignores.
- The label map and the near-duplicate groups are the same files as before. The labels became cancer / not cancer and the train / validation / test splits were rebuilt the same day (sections 2 and 3). The three outside datasets used for testing (DDI, MIDAS, DermaCon-IN) hold the same images and the same cancer labels as before.
- The ledger of test-set looks starts empty. The test sets themselves are not new: the earlier experiments were scored on them, and those scores were seen. No choice in this file was made from them.

## 2. Classes and labels

**Stage one is one question: is this image a skin cancer or not?** Two classes, `cancer` and `not_cancer` (decided 12 Nov 2026).

Not part of stage one, by decision: which type of cancer it is, and how results differ between lighter and darker skin. Whether darker skin needs its own model or one combined model serves everyone is the next stage's question. Skin type stays recorded in the manifest for that, but it plays no part in the labels, the splits or the reported results now.

| Class | What is in it | Images in train / val / test |
| --- | --- | --- |
| `cancer` | Basal cell carcinoma, squamous cell carcinoma (with Bowen's disease), melanoma, and rare skin cancers (Kaposi sarcoma, mycosis fungoides, DFSP, Merkel cell and others) | 3,176 |
| `not_cancer` | Skin conditions that are not tumours (psoriasis, eczema, acne, scabies, lupus and about a hundred more); benign tumours that resemble cancer (nevi, seborrheic keratosis, dermatofibroma, solar lentigo, pyogenic granuloma); other benign growths and birthmarks (cysts, syringoma, epidermal nevi, port wine stains) | 14,776 |

- **Actinic keratosis is in neither class.** It is precancerous, so calling it either would be wrong. It is never trained on. AK images form the `ak_side` set, where we report how many the model calls cancer.
- **Diagnosis mapping.** Ordered rules in `src/skincancer/labels.py` map each dataset's diagnosis strings to a tumour label (`label_4class`). `data/label_map.csv` holds every (dataset, diagnosis) pair, was reviewed row by row ([docs/LABEL_MAP_REVIEW.md](docs/LABEL_MAP_REVIEW.md)), and is the source of truth. `build_manifest` turns it into `label_cancer`, the column the model is trained on, and `label_fine`, a finer description (BCC, melanoma, non-tumour and so on) used only to break results down.
- **Where "not a tumour" comes from.** Fitzpatrick17k marks each of its diagnoses malignant, benign or non-neoplastic; that marking decides its `not_cancer` images. DDI uses its biopsy result for every image. DermaCon-IN's everyday clinic conditions count as not cancer.
- **Excluded:** 17 Fitzpatrick17k images its own quality check marked as wrongly labelled, 2 DermaCon-IN images of insufficient quality, 593 MIDAS images whose diagnosis does not say whether they are cancer ("benign-other", "malignant-other", controls), and AK images of patients who also have another lesion in the data.
- Images with no recorded skin type are kept (915 of them in train / val / test), since stage one does not use skin type.

## 3. Splits

Frozen in `data/manifest.csv` (`split` column) and `data/splits/`. Rebuilt on 12 Nov 2026 for the two-class design; nobody re-splits after this.

| Split | Images | Cancer | What it is |
| --- | --- | --- | --- |
| `train` | 12,562 | 2,220 | Fitzpatrick17k + PAD-UFES-20 |
| `val` | 2,695 | 478 | same sources |
| `test` | 2,695 | 478 | same sources |
| `ddi_test` | 656 | 171 | DDI (Stanford, biopsy-confirmed) |
| `midas_test` | 1,380 | 802 | MRA-MIDAS clinical photos (Stanford, biopsy-confirmed), tumours only |
| `dermacon_test` | 65 | 19 | DermaCon-IN (India), tumours |
| `dermacon_other` | 5,379 | 0 | DermaCon-IN, everyday conditions |
| `ak_side` | 703 | none labelled | Actinic keratoses |

- Train / val / test is a 70 / 15 / 15 split, stratified by type of lesion and by dataset, so every split has the same mix (17.7% cancer in each).
- **No leakage.** Near-duplicate images (perceptual hash distance ≤ 2, or embedding cosine ≥ 0.90) and images of the same patient form a group. A group never spans two splits; `make_splits` asserts this. The duplicate search used an ImageNet EfficientNet-B0 as a similarity measure. That is data preparation, and it is the only place pretrained weights appear in this repository.
- **The three outside test datasets (DDI, MIDAS, DermaCon-IN) hold the same images and cancer labels as before the restart.** The rebuilt manifest was checked against the previous one: paths, diagnoses, tumour labels, skin types and groups are identical for all 27,081 rows.
- **DDI images never appear in a figure** (licence terms). The repository stays private because the manifest carries DDI metadata.

## 4. What trains, what selects, what tests

- **Training data:** `train`.
- **Selection:** `val` picks the checkpoint and stops training. Test splits never choose anything.
- **Main result:** `test`, overall, per dataset, and per type of lesion (which cancers are missed, which harmless conditions raise false alarms).
- **Test-only sets** (`ddi_test`, `dermacon_test`, `dermacon_other`, `midas_test`) are never scored during training. `predict --final` scores them once per finished run and records it in `runs/final_ledger.csv`; a second look is refused.
- **Known limits of this design, stated before any run:**
  - **The two sources differ in how much cancer they hold.** 70% of PAD-UFES-20 images are cancer against 13% of Fitzpatrick17k images, and PAD-UFES-20 images are smartphone photos while Fitzpatrick17k images are atlas photos. The network can score well by recognising the kind of photo instead of the lesion. `test` is therefore reported per dataset, and the outside sets (different hospitals, cameras and countries) are the real check.
  - **Most `not_cancer` images are easy.** 82% of them are rashes and other non-tumour conditions that look little like a tumour. A high overall score can hide a weak result on the hard part, telling a cancer from a benign tumour that resembles it. The per-type breakdown and MIDAS (tumours only) show that part.
  - The classes are unbalanced: 18% of `train` is cancer. The loss is class-balanced (section 6).

## 5. The network: SkinNet

Defined in `src/skincancer/models.py`. A residual CNN, small enough to train on a laptop CPU.

| Part | Layers | Output for a 224 × 224 input |
| --- | --- | --- |
| Stem | 3 × (3×3 conv, batch norm, ReLU), widths 16, 16, 32, the first with stride 2; then 3×3 max-pool, stride 2 | 32 × 56 × 56 |
| Stage 1 | 2 residual blocks, 32 channels | 32 × 56 × 56 |
| Stage 2 | 2 residual blocks, 64 channels, first one stride 2 | 64 × 28 × 28 |
| Stage 3 | 2 residual blocks, 128 channels, first one stride 2 | 128 × 14 × 14 |
| Stage 4 | 2 residual blocks, 256 channels, first one stride 2 | 256 × 7 × 7 |
| Head | global average pool, dropout 0.3, linear to 2 classes | 2 |

- A residual block is two 3×3 convolutions, each followed by batch norm, with a skip connection around them. Where a block changes resolution or width, the skip is average-pool then 1×1 convolution.
- 2.8 million parameters, 17 convolution layers on the main path.
- Weights start random (Kaiming normal). The last batch norm of each block starts at zero, so every block begins as the identity. Each seed gives a different start.

## 6. Training

Fixed on 12 Nov 2026, before any SkinNet run was scored on a test set. The config is `configs/skinnet.yaml`.

| Setting | Value |
| --- | --- |
| Layers trained | all, from the first step |
| Optimiser | AdamW, learning rate 1e-3, weight decay 0.05 |
| Schedule | 3 epochs linear warm-up, then cosine decay to zero at epoch 60 |
| Stopping | best `val` AUC is kept; stop after 15 epochs without improvement |
| Loss | cross-entropy, label smoothing 0.1, class-balanced weights |
| Batch | 32 images, 224 × 224 |
| Augmentation | random resized crop (40–100% of the image), horizontal and vertical flips, 90° turns, rotation ±30°, brightness and contrast ±20%, light blur, 1–3 erased patches |
| Not used | hue or saturation changes: they change apparent skin tone |
| Seeds | 0, 1, 2 |

- Class-balanced weights give each class the same total weight in the loss, so the 10,342 `not_cancer` images do not outvote the 2,220 cancers.
- Evaluation images are resized to 256 on the shorter side and centre-cropped to 224.
- One epoch is 12,562 images, about 6 minutes on the development laptop (no GPU), so a seed takes up to 6 hours.
- **Expectation, stated in advance:** the score on `test` will look good, mostly because rashes are easy to tell from cancers and because of the photo-type shortcut above. Expect clearly lower scores on DDI, MIDAS and DermaCon-IN; those are the numbers to judge the model by.

## 7. Metrics and statistics

- **AUC** for cancer vs not cancer, on every set that holds both.
- **Missed cancers and false alarms at a fixed operating point ("op90"):** per seed, the threshold on P(cancer) that catches 90% of cancers in `val`, applied unchanged to every other set. Missed cancers are the share of cancers below the threshold; false alarms the share of non-cancers at or above it. The same two rates at the plain 0.5 threshold are also reported.
- **Per type of lesion** on `test`, MIDAS and DermaCon-IN: the missed-cancer rate for each cancer type, and the false-alarm rate for each kind of non-cancer.
- On `ak_side` and `dermacon_other`, the share of images called cancer.
- Every number is the mean of the three seeds, with a 95% confidence interval from 1,000 bootstrap resamples of groups (not single images). Subgroups under 10 images are not reported.

## 8. Second round: training that the shortcut cannot exploit (12 Nov 2026)

Written after the baseline results ([docs/RESULTS.md](docs/RESULTS.md)) and before any of these models was trained. No new data is used.

**What the baseline showed.** It learned that a smartphone photo usually means cancer (89% false alarms on PAD-UFES-20 non-cancers), and it could not tell a cancer from a benign tumour that resembles it (81% false alarms on those).

**Three variants**, each changing only how training images are drawn and augmented. The network, optimiser, schedule and data are the baseline's.

| Variant | Config | Change |
| --- | --- | --- |
| A | `skinnet-c1` | Inside each dataset, cancer and not cancer are drawn equally often, so the kind of photo no longer predicts the class |
| B | `skinnet-c1type` | As A, and inside each dataset × class a lesion type's share of the draws is proportional to the square root of its image count. Benign lookalikes go from 6% to 11% of the images drawn, rare cancers from 7% to 9% |
| C | `skinnet-c1type_q` | As B, plus random loss of resolution and JPEG quality during training, so image quality says less about the source |

- Balanced drawing replaces the class-weighted loss (the loss is unweighted for A, B and C). An epoch still draws 12,562 images, now with replacement.
- **A new selection score: within-source AUC.** The AUC is computed inside each dataset and the two are averaged. Telling the datasets apart cannot raise it. A, B and C pick their checkpoint on it. The baseline picked on plain AUC, so its within-source AUC on `val` is reported next to theirs for reference.

**How one is chosen.**
1. Train seed 0 of A, B and C.
2. Compare them on `val` only: within-source AUC first; the AUC on cancers against benign lookalikes (`hard` subset) breaks a tie of less than 0.005.
3. Train seeds 1 and 2 of the winner only, and score that one model on `test` and the outside sets, once.

The other two variants are never scored on `test` or the outside sets. Training writes their `test` predictions to disk like every run, but those files are not evaluated.

**What would count as helping:** on the outside sets, a higher AUC on MIDAS and DDI than the baseline's 0.60 and 0.65; on our own `test` split, fewer false alarms on PAD-UFES-20 non-cancers and on benign lookalikes at the same share of cancers caught. Plain AUC on `test` may fall, since part of the baseline's 0.887 came from the shortcut.

**If none helps,** the next steps are in [docs/DATA_PLAN.md](docs/DATA_PLAN.md) (Derm7pt first) and then larger images.

**Outcome (12 Nov 2026):** A was chosen and became checkpoint 1 ([docs/CHECKPOINTS.md](docs/CHECKPOINTS.md)). B and C were discarded: their configs, trained models and the two code options only they used were removed. The comparison is kept in [docs/RESULTS.md](docs/RESULTS.md).

## 9. Third round: more hard cases and larger images (12 Nov 2026)

Written after the second-round results and before any of these models was trained.

**What the second round showed.** Balanced drawing (variant A) weakened the shortcut and improved the hard distinction on our own test split, but the outside sets did not move (MIDAS 0.63, DDI 0.62). Variant A's training setup is the starting point for everything below.

**Derm7pt as training-only data.** 958 clinical photos from an Italian atlas with a usable label: 294 cancers (252 melanomas, 4 of them skin metastases, and 42 BCCs) and 664 benign tumours, mostly nevi. The duplicate search found none of them among the 13,573 validation and test images. It has no SCC and records no skin type.
- It is kept outside the manifest (`data/extra/derm7pt.csv`, built by `python -m skincancer.data.extra derm7pt`) and joins `train` only. `val`, `test` and the outside sets are untouched, so models with and without it are scored on exactly the same images.
- Any Derm7pt image that near-duplicates a validation or test image is dropped, with the same thresholds as the main duplicate search.
- In the balanced drawing it counts as a third dataset: cancer and not cancer are drawn equally often inside it.
- Derm7pt images never appear in a figure (licence terms), and its list is not committed.

**Three variants.**

| Variant | Config | Change from variant A |
| --- | --- | --- |
| D | `skinnet_d7` | Derm7pt added to training |
| E | `skinnet_320` | Images at 320 × 320 pixels instead of 224 × 224 (evaluation: shorter side to 366, centre crop 320) |
| F | `skinnet_d7_320` | Both |

**How one is chosen:** the rule of section 8. Seed 0 of D, E and F is compared with seed 0 of variant A (checkpoint 1) on `val`: within-source AUC first, and the AUC on cancers against benign lookalikes breaks a tie of less than 0.005. If none beats checkpoint 1 by 0.005 or more on within-source AUC, checkpoint 1 stays the current model and nothing new is scored outside. Otherwise the winner gets seeds 1 and 2, is scored once on `test` and the outside sets, and becomes the next checkpoint.

**What would count as helping:** a higher AUC on MIDAS and DDI than variant A's 0.63 and 0.62. An improvement on our own test split alone would repeat the second round's outcome.

**Outcome (12 Nov 2026):** none of D, E and F beat checkpoint 1 on `val` (0.857, 0.856 and 0.862 against 0.869), so checkpoint 1 stays the current model and nothing new was scored on `test` or the outside sets. A full list of what has been tried, so it is not repeated, is in [docs/CHECKPOINTS.md](docs/CHECKPOINTS.md). The three variant configs and their trained models were removed. The code for training-only extra data (`data/extra.py`, `data.extra_train`) is kept for the next data steps; no config uses it now.

## 10. Ideas G and H: averaging seeds and test-time flips (12 Nov 2026)

Written before either was computed. Both change only how checkpoint 1's three trained models are used to predict; nothing is retrained.

- **G, seed average** (`skinnet-c1_ens`): the three seeds' cancer probabilities are averaged into one prediction per image.
- **H, test-time flips** (the seeds in `runs/skinnet-c2`): each seed predicts on the image and its three mirror images (left-right, up-down, both) and averages the four.
- **G + H** (`skinnet-c2`): the seed average of H's predictions.

**Step 1, validation only.** Within-source AUC on `val` for all three, against checkpoint 1's 0.868 (mean of its seeds). A candidate passes with a gain of 0.005 or more. If none passes, G and H are discarded and nothing is scored outside.

**Step 2, one candidate.** The passing candidate with the highest validation score is scored once on `test` and the outside sets. Averaging existing outside predictions is still a new look at those sets, so it is recorded in the ledger like any other.

**Step 3, does it become a checkpoint?** Differences this small cannot be read from two separate confidence intervals, so the comparison is paired: the same resampled images are scored by both models and the difference in AUC gets its own 95% interval (`python -m skincancer.compare`). The candidate becomes checkpoint 2 if that interval is above zero on MIDAS or on DDI and is not below zero on the other. Otherwise it is discarded, even if it gained on our own test split.

**Outcome (12 Nov 2026):** all three passed step 1 (within-source AUC on `val` 0.882, 0.879 and 0.888 against 0.868). G + H was scored outside. Its paired gain in AUC was +0.006 [−0.001, 0.014] on MIDAS and +0.006 [−0.004, 0.016] on DDI. Both intervals include zero, so it was discarded as a checkpoint, although it gained +0.015 on our own test split. The paired comparison tool (`skincancer.compare`) is kept, because every later idea is judged the same way. The tool that built G and H was removed with them.

**Amended the same day:** G + H was restored and is checkpoint 2. The rule above was written before the test and G + H did not meet it; the decision to keep it anyway was made after seeing the result, on the grounds that the gain on our own test split is supported, costs nothing, and nothing is worse anywhere. This is recorded as a change of rule, not as G + H having passed. From here on an idea "works" if the paired comparison supports a gain over the current checkpoint and nothing is clearly worse elsewhere; where the gain is (at home or outside) is always stated with it. The tool that builds G + H (`skincancer.derive`) is back in the repository.

## 11. MIDAS split three ways; idea J (12 Nov 2026)

Written before the split was made and before idea J was trained.

**Why.** Every model so far was chosen on a validation split drawn from the same two sources as training, and no choice made that way moved the outside scores. Two things were missing: outside photos to choose models on, and outside-like photos to train on. MIDAS can supply both, at the price of a smaller MIDAS test set.

**The split.** MIDAS's 1,380 biopsy-confirmed tumour photos are split by patient, stratified by type of lesion:

| Part | Share | Use |
| --- | --- | --- |
| `midas_train` | 40% | May be added to training (idea J). Unused unless an idea says so |
| `midas_val` | 30% | Outside validation: never trained on, used to choose between models |
| `midas_test` | 30% | Test only, scored once per model as before |

- This cannot be undone: once models are chosen on `midas_val` or trained on `midas_train`, those images are no longer a clean test. MIDAS scores reported before this section were on all 1,380 images; from here `midas_test` means the 30%.
- DDI stays whole and untouched. It is now the only test set from a source no model has trained on or been chosen on.
- `train`, `val`, `test` and the other sets do not change. The rebuild is checked against the previous manifest.
- A caution on reading results: a model trained on `midas_train` has seen that hospital's photos, so a gain on `midas_val` or `midas_test` shows it learned that source. Only a gain on DDI shows it travels to a source it has never seen.

**Idea J** (its config was removed with it; idea O's config holds the same recipe): checkpoint 1's recipe with `midas_train` added to the training images. In the balanced drawing MIDAS counts as a third dataset. Three seeds, then test-time flips and the seed average, the same treatment as checkpoint 2.

**How it is judged,** each time by paired comparison with checkpoint 2:

1. On `midas_val`. It passes if the gain in AUC is supported (the interval is above zero) and the within-source AUC on our own `val` is not lower by a supported amount.
2. If it passes, it is scored once on `test`, `midas_test`, DDI and DermaCon-IN.
3. It becomes checkpoint 3 if the gain is supported on `midas_test` or DDI and nothing is lower by a supported amount on `test` or DDI. Where the gain is will be stated with it.

If it fails step 1, it is discarded and nothing new is scored on the test sets.

**Outcome of idea J (13 Nov 2026): failed step 1, not scored on any test set.** Paired comparison with checkpoint 2, both after test-time flips and the seed average:

| Set | Score | Checkpoint 2 | Idea J | Difference [95% interval] | Supported |
| --- | --- | --- | --- | --- | --- |
| `midas_val` | AUC | 0.689 | 0.782 | +0.093 [+0.062, +0.121] | yes |
| `val` | AUC | 0.901 | 0.893 | −0.008 [−0.014, −0.001] | yes |
| `val` | Within-source AUC | 0.888 | 0.876 | −0.012 [−0.024, −0.001] | yes |

The gain on `midas_val` is large and supported, but the within-source AUC on our own `val` is lower by a supported amount, which the rule does not allow. The rule is applied as written.

What the training logs show (validation only): seeds 0 and 1 ran all 60 epochs and reached 0.867 and 0.866 within-source AUC on `val`, against 0.869, 0.872 and 0.865 for checkpoint 1's seeds. Seed 2 was stopped by the patience rule at epoch 40 with its best at epoch 25 (0.846), while the learning rate was still high; no other seed in the project has stopped before epoch 46. Most of the loss at home is that one run.

## 12. Idea O: idea J with early stopping off (13 Nov 2026)

Written before idea O was trained.

**Why.** Idea J did what it was meant to do on the outside validation photos, and lost at home mainly because one seed was cut short. Early stopping saves time; it is not part of what makes the model good, and the best weights are kept whichever epoch they come from. Turning it off removes that source of chance.

**Idea O** (`configs/skinnet-c3.yaml`): idea J with `patience: 0`, so every seed trains for all 60 epochs. Seeds 0 and 1 of idea J already ran all 60 epochs, so with early stopping off they would be the same runs; they are reused, and only seed 2 is trained again. Then test-time flips and the seed average, as before.

**How it is judged:** exactly as idea J (section 11), by paired comparison with checkpoint 2. Step 1 on `midas_val` and `val`; only if it passes, scored once on `test`, `midas_test`, DDI and DermaCon-IN; checkpoint 3 if the gain is supported on `midas_test` or DDI and nothing is lower by a supported amount on `test` or DDI.

**Cautions recorded now.**

- This idea was thought of after seeing idea J's validation result. That is what validation sets are for, and no test set has been looked at for J or O. But `val` and `midas_val` have now been used twice for this choice, so step 3 on the test sets is the real check.
- If idea O also fails step 1, the trade is real and not chance: MIDAS training photos help on MIDAS and cost a little at home. Whether that trade is worth taking is then a decision for the team, not something to settle by trying more variants.
- As for J: O has seen MIDAS's hospital. Only a gain on DDI shows it travels.

**Outcome of idea O (13 Nov 2026): the gain conditions are met, the no-loss condition is not; checkpoint 3 is not declared and the decision is with the team.** Seed 2 with early stopping off reached 0.875 within-source AUC on `val` (best at epoch 47). Paired comparison with checkpoint 2:

| Set | Score | Checkpoint 2 | Idea O | Difference [95% interval] | Supported |
| --- | --- | --- | --- | --- | --- |
| `midas_val` (outside validation) | AUC | 0.689 | 0.793 | +0.104 [+0.071, +0.134] | yes |
| `val` | AUC | 0.901 | 0.898 | −0.003 [−0.008, +0.003] | no |
| `val` | Within-source AUC | 0.888 | 0.882 | −0.006 [−0.014, +0.002] | no |
| `midas_test` | AUC | 0.645 | 0.698 | +0.053 [+0.020, +0.089] | yes |
| DDI | AUC | 0.629 | 0.657 | +0.028 [+0.007, +0.048] | yes |
| DermaCon-IN tumours | AUC | 0.826 | 0.841 | +0.015 [−0.058, +0.090] | no |
| `test` | AUC | 0.904 | 0.898 | −0.007 [−0.012, −0.001] | yes |
| `test` | Within-source AUC | 0.883 | 0.876 | −0.007 [−0.020, +0.006] | no |

- Step 1 passed: the gain on `midas_val` is supported and nothing on `val` is lower by a supported amount. Idea O was then scored once on the test sets.
- Step 3, gain: supported on `midas_test` and on DDI. DDI is a source no model here has trained on or been chosen on, so this is the first gain in the project shown to travel.
- Step 3, no loss: AUC on our own `test` is lower by 0.007, and the interval just excludes zero. The rule says nothing may be lower by a supported amount on `test`, so by the rule as written idea O is not a checkpoint.
- The rule is not changed here. Keeping idea O would be a change of rule after the result, as with checkpoint 2, and is for the team to decide. Until then checkpoint 2 stays the current model and idea O's runs are kept.

**Decided 13 Nov 2026: idea O is checkpoint 3.** The team accepted a loss of 0.007 AUC on our own test split for supported gains on DDI (+0.028) and the MIDAS test part (+0.053). This is a decision taken after seeing the result, not idea O passing the rule as written. Idea J's runs and config are deleted. New ideas are compared with checkpoint 3, train for all 60 epochs (`patience: 0`), and are chosen on `midas_val` together with our own `val`.

## 13. Idea I: SCIN photos in training (13 Nov 2026)

Written before the photos were downloaded and before idea I was trained.

**Why.** Checkpoint 3 showed that photos from a new kind of source are what moves the outside scores. Its remaining weakness outside is false alarms: at the fixed threshold it calls 67% of DDI's non-cancers and 28% of DermaCon-IN's everyday conditions cancer. SCIN is about 5,000 cases of everyday skin conditions photographed by the public on their phones, the kind of photo a real user would send.

**What is used.**

- Only cases the dermatologists could grade, and only as not cancer. SCIN has no biopsies, so its few suspected cancers are not trusted as cancers.
- A case is left out if a cancer, a pre-cancer or an uncertain growth appears anywhere in its list of possible diagnoses, however low its weight.
- One photo per case (the first), so SCIN does not swamp the other sources. About 3,000 photos.
- Training only. SCIN is kept outside the manifest like Derm7pt was, so every validation and test image stays exactly the same. Photos that near-duplicate a validation or test image are dropped.

**Idea I** (`configs/skinnet_scin.yaml`): checkpoint 3's recipe with these photos added. Three seeds, 60 epochs, test-time flips, seed average.

**A risk named now.** SCIN adds phone photos that are all not cancer. The first baseline learned "phone photo means cancer" from PAD-UFES-20; this could teach the opposite shortcut. It would show as more missed cancers and a lower within-source AUC on PAD-UFES-20.

**How it is judged,** by paired comparison with checkpoint 3:

1. On validation (`val` and `midas_val`). SCIN's benefit is on everyday photos, which no validation set outside our own has, so this step only checks for harm: it passes if neither the within-source AUC on `val` nor the AUC on `midas_val` is lower by a supported amount.
2. If it passes, it is scored once on `test`, `midas_test`, DDI and DermaCon-IN.
3. It becomes checkpoint 4 if at least one of these holds: a supported gain in AUC on DDI; a supported gain in AUC on `midas_test`; or the share of DermaCon-IN everyday conditions called cancer falls by at least 5 points with its interval clear of checkpoint 3's. And none of the AUCs on `test`, `midas_test` or DDI is lower by a supported amount.

If it fails step 1 it is discarded and not scored on the test sets. If it meets the gain condition but narrowly misses the no-loss one, the result is reported and the team decides, as with checkpoint 3.

**Outcome of idea I (13 Nov 2026): failed step 1, discarded, not scored on any test set.** 2,907 SCIN photos were used (2 dropped as near-duplicates of evaluation images); the training set was 16,021 images. Paired comparison with checkpoint 3, both after test-time flips and the seed average:

| Set | Score | Checkpoint 3 | Idea I | Difference [95% interval] | Supported |
| --- | --- | --- | --- | --- | --- |
| `midas_val` (outside validation) | AUC | 0.793 | 0.729 | −0.064 [−0.096, −0.032] | yes |
| `val` | AUC | 0.898 | 0.913 | +0.015 [+0.008, +0.023] | yes |
| `val` | Within-source AUC | 0.882 | 0.902 | +0.020 [+0.006, +0.037] | yes |

The AUC on `midas_val` is lower by a supported amount, which the rule does not allow. The gain on our own `val` is real but is the kind of gain the project has learned not to trust on its own: `val` is full of everyday conditions like SCIN's, and MIDAS is all tumours. Its config and runs are deleted. The SCIN photos stay on disk and the code that lists them stays in `data/extra.py`.

## 14. Ideas M and N: a larger network, a longer schedule (13 Nov 2026)

Written before either was trained.

**Why.** Both ask whether checkpoint 3 is held back by the model and its training and not only by the data. Neither needs new data, and both can now be chosen on the MIDAS validation part.

- **Idea M** (`configs/skinnet-idea-m.yaml`): the same network at twice the width (64 channels in the first stage, not 32), which is about four times the parameters (about 11 million against 2.8 million). Everything else as checkpoint 3.
- **Idea N** (`configs/skinnet-idea-n.yaml`): the same network trained for 120 epochs, not 60, with the same warm-up and the cosine schedule stretched to match. Everything else as checkpoint 3.

Each gets three seeds, test-time flips and the seed average, and is judged on its own by paired comparison with checkpoint 3:

1. On validation. It passes if the gain is supported on the AUC of `midas_val` or on the within-source AUC of `val`, and neither of those two is lower by a supported amount.
2. An idea that passes is scored once on `test`, `midas_test`, DDI and DermaCon-IN. If both pass, only the one with the higher AUC on `midas_val` is scored; the other is kept as a later idea to try on top of the new checkpoint.
3. It becomes checkpoint 4 if the gain is supported on at least one of `test`, `midas_test` or DDI and none of those three is lower by a supported amount. Where the gain is will be stated with it.

An idea that fails step 1 is discarded and not scored on the test sets. If one meets the gain condition and narrowly misses the no-loss one, it is reported and the team decides.

**Expectation recorded now.** Idea E (larger images) suggested detail is not the limit, and 13,000 images is little for an 11-million-parameter network trained from scratch, so M may overfit and do no better. N is the more likely of the two to help, since two of checkpoint 3's three seeds had their best epoch after epoch 45.

**Outcome of idea M (14 Nov 2026): failed step 1, discarded, not scored on any test set.** 11.2 million parameters; the seeds' best epochs were 41, 42 and 59 (within-source AUC on `val` 0.866, 0.866, 0.863, against checkpoint 3's 0.867, 0.866, 0.875). Paired comparison with checkpoint 3, both after test-time flips and the seed average:

| Set | Score | Checkpoint 3 | Idea M | Difference [95% interval] | Supported |
| --- | --- | --- | --- | --- | --- |
| `midas_val` (outside validation) | AUC | 0.793 | 0.773 | −0.020 [−0.037, −0.002] | yes |
| `val` | AUC | 0.898 | 0.897 | −0.001 [−0.007, +0.005] | no |
| `val` | Within-source AUC | 0.882 | 0.885 | +0.004 [−0.006, +0.013] | no |

No gain anywhere and a supported loss on `midas_val`, as expected: four times the parameters on the same 13,000 images learns no more that carries to other hospitals. Its config and runs are deleted. The width is not what holds the network back.

**Outcome of idea N, step 1 (14 Nov 2026): passed.** The seeds' best epochs were 80, 107 and 91 (within-source AUC on `val` 0.897, 0.898, 0.895), all after checkpoint 3's whole schedule had ended. Paired comparison with checkpoint 3, both after test-time flips and the seed average:

| Set | Score | Checkpoint 3 | Idea N | Difference [95% interval] | Supported |
| --- | --- | --- | --- | --- | --- |
| `midas_val` (outside validation) | AUC | 0.793 | 0.795 | +0.002 [−0.037, +0.042] | no |
| `val` | AUC | 0.898 | 0.930 | +0.032 [+0.023, +0.042] | yes |
| `val` | Within-source AUC | 0.882 | 0.921 | +0.039 [+0.024, +0.056] | yes |

The gain on `val` is supported and `midas_val` is not lower, so by rule 1 N goes on to the test sets. The gain is at home only so far: on the outside validation part it is flat. N was then scored once on the test sets.

**Outcome of idea N, step 3 (14 Nov 2026): failed, discarded.** Scored once on the test-only sets (ledger, 14 Nov):

| Set | Score | Checkpoint 3 | Idea N | Difference [95% interval] | Supported |
| --- | --- | --- | --- | --- | --- |
| `test` | AUC | 0.898 | 0.930 | +0.032 [+0.022, +0.041] | yes |
| `test` | Within-source AUC | 0.876 | 0.924 | +0.048 [+0.030, +0.071] | yes |
| `midas_test` | AUC | 0.698 | 0.733 | +0.034 [−0.010, +0.075] | no |
| DDI | AUC | 0.657 | 0.592 | −0.064 [−0.096, −0.035] | yes |
| DermaCon-IN tumours | AUC | 0.841 | 0.814 | −0.028 [−0.175, +0.097] | no |

The gain condition is met on `test`; the no-loss condition fails on DDI by a wide margin, not narrowly, so this is not a case for the team and N is discarded. Its config and runs are deleted. The reading: the second half of the longer schedule taught the network more about our two home datasets (the largest gain at home in the project) and less that holds on a hospital it has never seen. DDI falls below checkpoint 2 (0.629) and the baseline (0.65). It also answers the question from section 16: more training on the same photos widens the gap to other hospitals, it does not close it.

**A suggestion for the team (adopted the same day, section 19).** Twice now (ideas I and N) a gain on `val` alone has come with a loss outside: `val` is two datasets we train on, and it rewards learning them better. Step 1 could require a supported gain on `midas_val` and not only on `val`, which would have stopped N before it spent a look at the test sets. Changing the rule is the team's call.

**Renamed 13 Nov 2026.** The kept models are now named after their checkpoint: `skinnet` (baseline), `skinnet-c1`, `skinnet-c2`, `skinnet-c3`. Configs, run folders, result folders, the test-set ledger and every mention in the documents, including earlier sections of this file, use the new names. Only names changed: no model was retrained or re-scored. The old names were `skinnet_src` (now `skinnet-c1`), `skinnet_src_tta_ens` (`skinnet-c2`) and `skinnet_midas_full_tta_ens` (`skinnet-c3`); git history before this date uses them.

## 15. Naming rule (13 Nov 2026)

One name per model, used for its config, its run folder and its result folder.

| What | Name | Example |
| --- | --- | --- |
| Baseline | `skinnet` | `configs/skinnet.yaml`, `runs/skinnet/`, `results/skinnet/` |
| A checkpoint | `skinnet-c<number>` | `skinnet-c3` |
| An idea being tested | `skinnet-idea-<letter>` | `skinnet-idea-n` |

- A run folder holds the seeds (`s0`, `s1`, `s2`) and, once built, their average (`avg`). There are no separate folders for steps on the way to a model: no `-train`, `-tta` or `-ens` names.
- An idea that is kept is renamed to the next checkpoint name everywhere (config, runs, results, ledger, documents). An idea that is discarded is deleted.
- Git tags stay `checkpoint-<number>`.
- No model was retrained or re-scored for this change. `evaluate` and `compare` read `avg` when it exists, and the seeds otherwise. The pause on 13 Nov stopped ideas N and M before any seed finished; both start again from seed 0.
- Logs sit inside the model's run folder (`runs/<name>/logs/s0.log`, `final.log`, `evaluate.log`). `runs/` holds only model folders and the test-set ledger; `results/` holds only one folder per kept model.

## 16. Ideas P and Q, and a wider idea L (14 Nov 2026)

Written before any of them was started.

**Why.** Our supervisor's comment: after 60 epochs the outside scores are still about 0.6, which is poor. Checkpoint 3's training logs say the cause is not too little training. On our own validation split the within-source AUC was still rising at epoch 50 (about 0.85 to 0.87) and flattens only as the cosine schedule takes the learning rate to zero. The training loss ends at about 0.48, against a floor of about 0.20 with label smoothing 0.1, so the network is not memorising its training set. At home it scores about 0.90; on photos from other hospitals 0.66 (DDI) and 0.70 (MIDAS test). The gap is how far our training photos are from other hospitals' photos. Ideas N and M test the training side; P, Q and the wider L aim at the gap itself.

- **P. Pretrained weights.** Start from weights learned on ImageNet and train every layer. It is the largest known lever for performance on new hospitals. It breaks section 1 (from random weights, no pretrained weights anywhere in the classifier), which is the team's own rule, so it is listed and not run: the team and the supervisor decide whether the rule changes. If it does, P gets its own section with its rules before anything is trained.
- **Q. More biopsy-confirmed tumour photos from new sources.** Clinical photos (not dermoscopy) of cancers and of the benign growths that resemble them, from hospitals not yet in training. Idea O is the only change so far that moved a score on a source never seen (DDI), and it did so with photos of exactly this kind; idea I showed that new photos without tumours do not help. First step: find public sets of this kind, check their licences and their overlap with our evaluation images. None is downloaded yet.
- **L, widened.** Self-supervised pretraining on our own training photos (section 14's idea L) plus unlabelled photos already on disk that no evaluation set uses: SCIN and the Derm7pt clinical photos. Never on any image of `val`, `test`, `midas_val`, `midas_test`, DDI or DermaCon-IN, or on a near-duplicate of one. It keeps the from-scratch rule: no outside weights.

**Order.** N and M first (section 14), then L, then Q. P only if the rule is changed.

**Resumed 14 Nov 2026.** N and M were restarted from seed 0, N first, M after it. They run with `scripts/run_idea.sh`, new today: it trains the seeds, averages them and compares with checkpoint 3 on `val` and `midas_val` only, so no test set is scored before step 1 is passed. The existing `scripts/run.sh` scores the test-only sets and is for a model that has passed step 1.

## 17. Idea L: self-supervised pretraining on our own photos (14 Nov 2026)

Written before the pretraining was run.

**What.** SkinNet first learns from photos without their labels, then is trained on cancer / not cancer exactly as checkpoint 3 was. The only change from checkpoint 3 is where the weights start: from this pretraining instead of at random. No outside weights are used, so section 1 still holds.

- **Pretraining** (`python -m skincancer.pretrain --config configs/skinnet-idea-l.yaml`): SimCLR. Two random views of each photo (crop, flips, rotations, brightness, contrast and saturation changes, blur; no hue change, as in training) pass through SkinNet and a small projection head; the loss pulls the two views of a photo together and pushes the other photos in the batch apart. 200 epochs, 256 photos per step, AdamW at 1e-3 with a 10-epoch warm-up and cosine decay, temperature 0.2. One pretraining (seed 0), shared by the three training seeds. The weights after the last epoch are used; nothing is chosen on a validation set.
- **Photos used:** `train` and `midas_train` of the manifest (13,114) and the extra sets on disk, Derm7pt clinical (958) and SCIN (2,907): about 17,000. Their labels are not read. The extra sets already exclude near-duplicates of every validation and test image (`data/extra.py`). The code refuses any evaluation split.
- **Training:** `configs/skinnet-idea-l.yaml`, checkpoint 3's recipe with `model.init` pointing at the pretrained backbone. Same learning rate, schedule, epochs, sampling and test-time flips, so a difference can be traced to the starting weights.
- Weights and logs stay in the model's run folder: `runs/skinnet-idea-l/pretrain/` holds the pretraining, `s0`–`s2` and `avg` the trained model, as for every model.

**How it is judged:** as ideas M and N (section 14), by paired comparison with checkpoint 3. Step 1 on `midas_val` and `val`; only if it passes, scored once on `test`, `midas_test`, DDI and DermaCon-IN; checkpoint 4 (or the next free number) if the gain is supported on at least one of `test`, `midas_test` or DDI and none is lower by a supported amount. If N or M has become checkpoint 4 by then, L is still judged against checkpoint 3 as written here, and whether to combine the two is a separate idea. (Neither did. Step 1 was tightened on 14 Nov before L was trained: it now needs a supported gain on `midas_val`. See section 19.)

**Expectation recorded now.** Self-supervised pretraining usually helps most when the unlabelled set is much larger than the labelled one. Here it is only about a third larger, so a small gain is the likely outcome. Its value for the outside sets is that it learns from SCIN's phone photos and Derm7pt without their labels steering it, which idea I showed can hurt.

## 18. Idea Q: MILK10k's biopsied clinical photos in training (14 Nov 2026)

Written after the photos were downloaded and before idea Q was trained. No model has been trained or scored with them.

**The source.** MILK10k (ISIC archive, collection 425; the MILK study team; CC BY-NC 4.0): 5,240 lesions, each with one clinical close-up and one dermoscopic photo, from five centres: Medical University of Vienna, Ankara University, the University of Queensland's Mayne Academy of General Practice, Memorial Sloan Kettering, and a practice in Skopje. None is Stanford (DDI, MIDAS) or in India (DermaCon-IN), so DDI stays a source no model has seen, with no shared patients.

**What is used** (`python -m skincancer.data.extra milk10k`, written to `data/extra/milk10k.csv`):

- The clinical close-ups only, one per lesion. The dermoscopic photos are a different kind of image and are not used.
- Only the 5,016 lesions confirmed by histopathology. The 224 that rest on one clinician's opinion are left out.
- Cancer: ISIC-DX says malignant and the most specific diagnosis maps to a cancer type under the label map (BCC, SCC including in situ and keratoacanthoma, melanoma including in situ). Not cancer: ISIC-DX says benign (nevi, seborrheic and lichen-planus-like keratoses, solar lentigines, dermatofibromas and others). Actinic keratoses (123, never trained on, as in the manifest) and indeterminate lesions are left out.
- Training only, kept outside the manifest, like Derm7pt and SCIN, so every validation and test image stays the same. Photos that near-duplicate a validation or test image are dropped.
- No patient ids are published, so photos are grouped by lesion.
- **Built 14 Nov:** 4,908 photos may be trained on, 3,620 cancers (2,520 BCC, 652 SCC, 448 melanoma) and 1,288 not cancer. 332 were left out (actinic keratoses, indeterminate lesions, not biopsied, or a diagnosis outside the label map). None near-duplicates a validation or test image.

**Why it may help where earlier data did not.** It is the hard distinction, cancers against the benign growths that resemble them, all biopsied, from five new centres and in large numbers (several times MIDAS's 552). Idea O showed photos of that kind from one new hospital moved DDI; idea I showed new photos without tumours do not.

**A risk named now.** Cancers are about 70% of its photos and half of all its photos are BCC. Balanced drawing inside each dataset (checkpoint 1) removes the reward for recognising MILK10k's kind of photo, but the network may still lean towards BCC-like features.

**Idea Q** (`configs/skinnet-idea-q.yaml`, now `configs/skinnet-c4.yaml`): checkpoint 3's recipe with MILK10k added. Three seeds, 60 epochs, test-time flips, seed average. Judged exactly as ideas M, N and L (section 14), by paired comparison with checkpoint 3: step 1 on `midas_val` and `val`; only if it passes, scored once on `test`, `midas_test`, DDI and DermaCon-IN; checkpoint 4 (or the next number) if the gain is supported on at least one of `test`, `midas_test` or DDI and none is lower by a supported amount. (Step 1 was tightened on 14 Nov, before Q's result was known: it now needs a supported gain on `midas_val`. See section 19.)

**Order changed.** Q runs before L, after M (section 16 had L first). Q takes about an hour and a half, L about five and a half, and Q is the more likely of the two to move the outside scores.

## 19. Team decisions after ideas M and N (14 Nov 2026)

Decided by the team on 14 Nov, after ideas M and N (section 14) and before idea Q's result was known.

**1. Step 1 now needs a gain outside.** From idea Q on, an idea passes step 1 only if the AUC on `midas_val` is higher by a supported amount and the within-source AUC on `val` is not lower by a supported amount. A gain on `val` alone no longer passes. Steps 2 and 3 are unchanged.

- Why: the outside scores are the model's main weakness. Twice (ideas I and N) a gain on `val` came with a loss outside, and N spent a look at DDI, the only source no model has trained or been chosen on, before failing there.
- The cost, accepted: `midas_val` has 414 images, so an outside gain smaller than about 0.04 cannot be shown as supported and will be discarded with the rest.
- It applies to ideas Q and L, and to every idea after them.

**1, corrected the same day: a gain on everything.** The team's answer had been read as above; the team corrected it after Q's validation result was known, and before any test set was scored for Q. The rule meant is a gain on every set, not a supported gain on one:

- **Step 1:** higher on both `midas_val` (AUC) and `val` (within-source AUC), and neither lower by a supported amount. A gain on one set alone does not pass, so ideas I and N would still have failed (`midas_val` −0.064 and +0.002).
- **Step 3 (checkpoint):** higher on every test set, `test`, `midas_test` and DDI, at least one of them by a supported amount, and none lower by a supported amount. DermaCon-IN tumours (65 images) is reported and must not be lower by a supported amount, but is too small to need a rise.
- This replaces the version above and section 14's steps 1 and 3 for Q, L and every idea after them.

**2. Idea P runs as a reference comparison, not as a candidate.** The from-scratch rule (section 1) stays for every checkpoint. P measures what that rule costs, so the team and the supervisor can decide with numbers in hand.

- SkinNet has no ImageNet weights, so P uses torchvision's ResNet-18 (11.2 million parameters). Two models, so a difference can be put on the weights and not on the network:
  - `resnet18-imagenet` (`configs/resnet18-imagenet.yaml`): starts from torchvision's ImageNet weights (IMAGENET1K_V1), every layer trained.
  - `resnet18-scratch` (`configs/resnet18-scratch.yaml`): the same network from random weights.
- Both use checkpoint 3's recipe unchanged (MIDAS training share, balanced drawing inside each dataset, AdamW at 1e-3 with warm-up and cosine decay, 60 epochs, test-time flips, three seeds and their average), so the only difference from checkpoint 3 is the network, and between the two only the starting weights. The learning rate is not tuned for the ImageNet start; if it is lower than expected, that is the first thing to check.
- Both are scored once on every set, test sets included, whatever their validation scores: they are measurements, not candidates. They are reported next to checkpoint 3 with the paired comparison and are never renamed to a checkpoint. They are named after their network, not `skinnet-`, because they are not SkinNet.
- The images are normalised with ImageNet's mean and spread already (`data/dataset.py`), so nothing else changes.
- They run after idea L.

**3. Idea K only if Q helps.** ISIC 2019 is downloaded and tried only if idea Q becomes a checkpoint with a supported gain on `midas_test` or DDI. A gain at home alone does not count. If Q fails, K is dropped.

**Outcome of idea Q, step 1 (14 Nov 2026): passes under the corrected rule (section 19, "a gain on everything"): higher on both validation sets. It would have failed the first reading of section 19.** 18,022 training images (4,908 from MILK10k). The seeds ended at within-source AUC on `val` 0.876, 0.894, 0.881. Paired comparison with checkpoint 3, both after test-time flips and the seed average:

| Set | Score | Checkpoint 3 | Idea Q | Difference [95% interval] | Supported |
| --- | --- | --- | --- | --- | --- |
| `midas_val` (outside validation) | AUC | 0.793 | 0.818 | +0.026 [−0.005, +0.059] | no |
| `val` | AUC | 0.898 | 0.900 | +0.002 [−0.007, +0.010] | no |
| `val` | Within-source AUC | 0.882 | 0.905 | +0.024 [+0.009, +0.039] | yes |

- Under the first reading of section 19 (a supported gain on `midas_val`) Q would have failed: the gain there is +0.026 and its interval reaches below zero. Under the corrected rule it passes: it is higher on both validation sets, and it is the only idea since checkpoint 3 to rise on both at once.
- Q was then scored once on the test sets.

**Outcome of idea Q, step 3 (14 Nov 2026): narrowly misses "higher on every test set" (DDI −0.007, not supported); reported, the team decides.** Scored once on the test-only sets (ledger, 14 Nov):

| Set | Score | Checkpoint 3 | Idea Q | Difference [95% interval] | Supported |
| --- | --- | --- | --- | --- | --- |
| `test` | AUC | 0.898 | 0.903 | +0.005 [−0.003, +0.013] | no |
| `test` | Within-source AUC | 0.876 | 0.907 | +0.031 [+0.014, +0.052] | yes |
| `midas_test` | AUC | 0.698 | 0.750 | +0.052 [+0.004, +0.098] | yes |
| DDI | AUC | 0.657 | 0.650 | −0.007 [−0.034, +0.021] | no |
| DermaCon-IN tumours | AUC | 0.841 | 0.818 | −0.023 [−0.088, +0.030] | no |

- Supported gains on MIDAS test (the largest outside gain since checkpoint 3) and on our own test split; nothing lower by a supported amount anywhere.
- DDI is flat: 0.007 lower, well inside its interval. Under section 19's corrected step 3 (higher on every test set) that misses; under section 18's rule as first written (a supported gain on one, no supported loss) Q would be checkpoint 4.
- **Team decision, 14 Nov: accepted. Idea Q is checkpoint 4 (`skinnet-c4`).** The miss on DDI is 0.007, well inside its interval, and nothing is lower by a supported amount anywhere; the team took it as the narrow miss section 14 leaves to the team. Config, runs, ledger rows and results were renamed to `skinnet-c4`; git tag `checkpoint-4`. New ideas are compared with checkpoint 4. Idea L, specified against checkpoint 3 before checkpoint 4 existed, is judged against checkpoint 3 as written in section 17, and its comparison with checkpoint 4 is reported alongside. By section 19, decision 3, idea K now goes ahead.

## 20. Idea K: ISIC 2019 dermoscopy in training (14 Nov 2026)

Written before the images were downloaded and before anything was trained. K goes ahead because idea Q became checkpoint 4 with a supported gain on `midas_test` (section 19, decision 3).

**The source.** The ISIC 2019 challenge training set (ISIC archive collection 65, "Challenge 2019: Training"): 25,331 dermoscopic images from HAM10000 (Vienna, Queensland), BCN_20000 (Barcelona) and Memorial Sloan Kettering. Each image carries its own Creative Commons licence (CC-0, CC BY or CC BY-NC), recorded per image in `data/extra/isic2019.csv`. Some ISIC 2019 lesions may be the same people as MILK10k's (Vienna, Queensland and MSK are in both); both are training-only, so this cannot leak into a score. No ISIC centre is Stanford or in India.

**What is used** (`python -m skincancer.data.extra isic2019`):

- Labels as MILK10k (section 18): cancer when ISIC-DX says malignant and the most specific diagnosis maps to a cancer type (melanoma, BCC, SCC); not cancer when it says benign (nevi, benign keratoses, dermatofibromas, vascular lesions); actinic keratoses and indeterminate lesions left out. Unlike MILK10k a biopsy is not required: the challenge's benign ground truth also rests on follow-up and expert consensus, which is the standard reading of this set.
- **A subsample, not the whole set.** One image per lesion, then 2,500 cancers and 2,500 not cancer drawn at random with seed 0. The balanced drawing (checkpoint 1) keeps each dataset's share of the draws, so all of ISIC 2019 would be about 57% of every epoch, all of it dermoscopy, a different kind of image from every evaluation set. Idea I showed how a large share of one new kind of photo can crowd out the hard cases. At 5,000 images ISIC 2019 is about 22% of the draws, close to MILK10k's share, and a seed takes about 40 minutes, not 70. If K helps, a larger share is a separate idea.
- Stored resized to 512 pixels on the short side (some originals are 6,000 pixels wide); training crops to 224 either way.
- Training only, outside the manifest; images that near-duplicate a validation or test image are dropped.

**Idea K** (`configs/skinnet-idea-k.yaml`): checkpoint 4's recipe with ISIC 2019's subsample added. Three seeds, 60 epochs, test-time flips, seed average. Judged by section 19's rule against checkpoint 4: step 1 needs a rise on both `midas_val` and `val` (within-source) with neither lower by a supported amount; then scored once on the test sets; checkpoint 5 if higher on `test`, `midas_test` and DDI, at least one by a supported amount, and none lower by a supported amount (DermaCon-IN only must not be lower by a supported amount).

**Expectation recorded now.** Dermoscopy is a close-up through a lens with polarised light: no skin texture around the lesion, no glare, a fixed distance. It is uncertain whether what the network learns there carries to phone and clinic photos. The most likely gain is on melanoma against nevi, where ISIC 2019 adds far more examples than all other sources together; the most likely harm is that it weakens what MILK10k taught.

**Order.** K trains after the queue already running (idea L, then the two ResNet-18 references), around midnight.
