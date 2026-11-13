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
