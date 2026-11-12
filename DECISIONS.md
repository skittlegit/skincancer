# Decisions

The rules this project runs by. The code enforces them. Changing a rule means changing it here, in the code, and re-running whatever it touches.

## 1. Fresh start (12 Nov 2026)

The project was restarted on 12 Nov 2026. The classifier is now one convolutional network of our own (SkinNet), trained from random weights. Nothing in it is pretrained and no layer is frozen.

- Everything from before the restart was removed from the working tree: the pretrained-backbone fine-tuning runs, image-processing ablation, Indian-prior loss, texture fusion, Deep CORAL, Derm Foundation heads, model ensembles, the "not a tumour" gates, and their results, figures and write-ups.
- That work is kept in git under the tag `pre-fresh-start` (`git checkout pre-fresh-start`). Its local run folders, results and figures were moved to `archive/pre-fresh-start/`, which git ignores.
- The label map and the near-duplicate groups are the same files as before. The manifest gained a fifth class and the train / validation / test splits were rebuilt the same day (sections 2 and 3). The six external test sets are unchanged.
- The ledger of test-set looks starts empty. The test sets themselves are not new: the earlier experiments were scored on them, and those scores were seen. No choice in this file was made from them.

## 2. Classes and labels

Five classes: `bcc`, `scc`, `mel`, `benign`, `other`. The first three are malignant.

- **`other` means a skin condition that is not a tumour:** psoriasis, eczema, lichen planus, acne, scabies, lupus and about a hundred more. It is every image Fitzpatrick17k itself marks non-neoplastic (12,080 images). Added on 12 Nov 2026 so the network trains on about 10,500 images instead of 2,000, and so it can say "not a tumour" instead of being forced to pick one.
- **Benign means tumour lookalikes only:** melanocytic nevi of all kinds, seborrheic keratosis (and dermatosis papulosa nigra), dermatofibroma, solar lentigo, pyogenic granuloma.
- **Diagnosis mapping.** Ordered rules in `src/skincancer/labels.py` map each dataset's diagnosis strings to a tumour label (`label_4class`). `data/label_map.csv` holds every (dataset, diagnosis) pair, was reviewed row by row ([docs/LABEL_MAP_REVIEW.md](docs/LABEL_MAP_REVIEW.md)), and is the source of truth. `build_manifest` then adds `other` to make `label_5class`, the column the model is trained on.
- **Actinic keratosis is in no class.** It is precancerous, so it is never trained on. AK images form the `ak_side` set, where we report what the model calls them.
- **Excluded:** rare cancers (Kaposi sarcoma, DFSP, mycosis fungoides, Merkel cell and others), and 1,662 Fitzpatrick17k images of tumours and birthmarks outside the four tumour classes (cysts, syringoma, epidermal and sebaceous nevi, port wine stains and similar). They are neither one of our tumour classes nor "not a tumour". Bowen's disease counts as SCC.
- **Unknown skin type is dropped** (565 Fitzpatrick17k and 804 PAD-UFES-20 images), as are 17 Fitzpatrick17k rows its own QC marked as wrongly labelled.
- Only Fitzpatrick17k supplies `other`. DermaCon-IN's everyday conditions are not labelled `other`, because that dataset does not say which of its diagnoses are tumours; they stay a test set of images that are not cancer (`dermacon_other`).

## 3. Splits

Frozen in `data/manifest.csv` (`split` column) and `data/splits/`. Rebuilt once on 12 Nov 2026 for the five-class design; nobody re-splits after this.

| Split | Images | What it is |
| --- | --- | --- |
| `train` / `val` / `test` | 10,533 / 2,261 / 2,261 | Fitzpatrick17k (all skin types) + PAD-UFES-20 (skin types I–III), five classes, split 70 / 15 / 15 |
| `pad_dark_test` | 60 | PAD-UFES-20, skin types IV–VI (biopsy-confirmed, Brazil) |
| `ddi_test` | 656 | DDI (Stanford, biopsy-confirmed), scored cancer vs not |
| `midas_test` | 1,380 | MRA-MIDAS clinical photos (Stanford, biopsy-confirmed), tumours only |
| `dermacon_test` / `dermacon_other` | 65 / 5,379 | DermaCon-IN (India): tumours / everyday conditions that are not cancer |
| `ak_side` | 311 | Actinic keratoses |

Images in `train` / `val` / `test`:

| | BCC | SCC | Melanoma | Benign | Other | All |
| --- | --- | --- | --- | --- | --- | --- |
| Train | 981 | 503 | 416 | 458 | 8,175 | 10,533 |
| Val | 211 | 110 | 89 | 99 | 1,752 | 2,261 |
| Test | 211 | 109 | 89 | 100 | 1,752 | 2,261 |
| Test, skin types I–III / IV / V–VI | 194 / 12 / 5 | 89 / 11 / 9 | 73 / 9 / 7 | 83 / 10 / 7 | 1,172 / 317 / 263 | 1,611 / 359 / 291 |

- **Stratified** by class × skin-type bucket, so every split has the same mix.
- **No leakage.** Near-duplicate images (perceptual hash distance ≤ 2, or embedding cosine ≥ 0.90) and images of the same patient form a group. A group never spans two splits; `make_splits` asserts this. The duplicate search used an ImageNet EfficientNet-B0 as a similarity measure. That is data preparation, and it is the only place pretrained weights appear in this repository.
- **The six external test sets hold the same images as before the restart.** The rebuilt manifest was checked against the previous one: paths, diagnoses, tumour labels, skin types and groups are identical for all 27,081 rows.
- **Skin-type buckets** are I–III / IV / V–VI. DDI only provides I–II / III–IV / V–VI and is reported in those.
- **DDI images never appear in a figure** (licence terms). The repository stays private because the manifest carries DDI metadata.

## 4. What trains, what selects, what tests

- **Training data:** `train`, all skin types together.
- **Selection:** `val` picks the checkpoint and stops training. Test splits never choose anything.
- **Main result:** `test`, reported overall and separately for skin types I–III, IV and V–VI.
- **Test-only sets** (`ddi_test`, `dermacon_test`, `dermacon_other`, `pad_dark_test`, `midas_test`) are never scored during training. `predict --final` scores them once per finished run and records it in `runs/final_ledger.csv`; a second look is refused.
- **Known limits of this design, stated before any run:**
  - There are few dark-skin tumours to test on: 21 skin-type V–VI cancers and 32 type IV cancers in `test`. Per-skin-type tumour results will have wide confidence intervals; DDI and `pad_dark_test` carry most of the dark-skin evidence.
  - Every `other` image is an atlas photo from Fitzpatrick17k, while 814 of the 2,358 training tumours are PAD-UFES-20 smartphone photos. The network could learn "smartphone photo means tumour". `test` is reported by dataset to show this, and `dermacon_other` (Indian clinic photos, no tumours) checks it on a third source.
  - The classes are unbalanced: 78% of `train` is `other`. The loss is class-balanced (section 6) and the headline metric, macro AUC, weighs every class equally.

## 5. The network: SkinNet

Defined in `src/skincancer/models.py`. A residual CNN, small enough to train on a laptop CPU.

| Part | Layers | Output for a 224 × 224 input |
| --- | --- | --- |
| Stem | 3 × (3×3 conv, batch norm, ReLU), widths 16, 16, 32, the first with stride 2; then 3×3 max-pool, stride 2 | 32 × 56 × 56 |
| Stage 1 | 2 residual blocks, 32 channels | 32 × 56 × 56 |
| Stage 2 | 2 residual blocks, 64 channels, first one stride 2 | 64 × 28 × 28 |
| Stage 3 | 2 residual blocks, 128 channels, first one stride 2 | 128 × 14 × 14 |
| Stage 4 | 2 residual blocks, 256 channels, first one stride 2 | 256 × 7 × 7 |
| Head | global average pool, dropout 0.3, linear to 5 classes | 5 |

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
| Stopping | best `val` macro AUC is kept; stop after 15 epochs without improvement |
| Loss | cross-entropy, label smoothing 0.1, class-balanced weights |
| Batch | 32 images, 224 × 224 |
| Augmentation | random resized crop (40–100% of the image), horizontal and vertical flips, 90° turns, rotation ±30°, brightness and contrast ±20%, light blur, 1–3 erased patches |
| Not used | hue or saturation changes: they change apparent skin tone |
| Seeds | 0, 1, 2 |

- Class-balanced weights give each class the same total weight in the loss, so the 8,175 `other` images do not outvote the 416 melanomas.
- Evaluation images are resized to 256 on the shorter side and centre-cropped to 224.
- One epoch is 10,533 images, about 5 minutes on the development laptop (no GPU), so a seed takes up to 5 hours.
- **Expectation, stated in advance:** telling `other` from tumours should be the easy part, since the classes look different and there is plenty of data. Telling the four tumour classes apart rests on only about 2,400 training images, and without pretraining that is where the network will be weakest.

## 7. Metrics and statistics

- **Macro AUC** (one-vs-rest, averaged over the classes present) on `val`, `test`, `pad_dark_test` and MIDAS, with recall and precision per class. **Binary AUC** (cancer vs not, using P(BCC) + P(SCC) + P(melanoma)) on every set, and the only score on DDI and DermaCon-IN.
- The external tumour sets hold no `other` images, so their macro AUC averages over the four tumour classes; a tumour there called `other` counts as an error.
- **Missed cancers and false alarms at a fixed operating point ("op90"):** per seed, the threshold on P(malignant) that catches 90% of cancers in `val`, applied unchanged to every other set.
- On `ak_side` and `dermacon_other`, which hold no cancers, the share of images called malignant and the share called each class.
- Every number is the mean of the three seeds, with a 95% confidence interval from 1,000 bootstrap resamples of groups (not single images). Subgroups under 10 images are not reported.
