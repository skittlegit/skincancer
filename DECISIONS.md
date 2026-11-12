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
| A | `skinnet_src` | Inside each dataset, cancer and not cancer are drawn equally often, so the kind of photo no longer predicts the class |
| B | `skinnet_srctype` | As A, and inside each dataset × class a lesion type's share of the draws is proportional to the square root of its image count. Benign lookalikes go from 6% to 11% of the images drawn, rare cancers from 7% to 9% |
| C | `skinnet_srctype_q` | As B, plus random loss of resolution and JPEG quality during training, so image quality says less about the source |

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
