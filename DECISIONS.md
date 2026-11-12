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
