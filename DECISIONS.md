# Decisions

The rules this project runs by. The code enforces them. Changing a rule means changing it here, in the code, and re-running whatever it touches.

## 1. Fresh start (12 Nov 2026)

The project was restarted on 12 Nov 2026. The classifier is now one convolutional network of our own (SkinNet), trained from random weights. Nothing in it is pretrained and no layer is frozen.

- Everything from before the restart was removed from the working tree: the pretrained-backbone fine-tuning runs, image-processing ablation, Indian-prior loss, texture fusion, Deep CORAL, Derm Foundation heads, model ensembles, the "not a tumour" gates, and their results, figures and write-ups.
- That work is kept in git under the tag `pre-fresh-start` (`git checkout pre-fresh-start`). Its local run folders, results and figures were moved to `archive/pre-fresh-start/`, which git ignores.
- The data foundation is unchanged: the label map, manifest, near-duplicate groups and splits are the same files as before. Sections 2 to 4 restate the rules they were built with.
- The ledger of test-set looks starts empty. The test sets themselves are not new: the earlier experiments were scored on them, and those scores were seen. No choice in this file was made from them.

## 2. Classes and labels

Four classes: `bcc`, `scc`, `mel`, `benign`. The first three are malignant.

- **Diagnosis mapping.** Ordered rules in `src/skincancer/labels.py` map each dataset's diagnosis strings. `data/label_map.csv` holds every (dataset, diagnosis) pair with its label, was reviewed row by row ([docs/LABEL_MAP_REVIEW.md](docs/LABEL_MAP_REVIEW.md)), and is the source of truth. `build_manifest` refuses to run if a diagnosis is unmapped.
- **Benign means tumour lookalikes only:** melanocytic nevi of all kinds, seborrheic keratosis (and dermatosis papulosa nigra), dermatofibroma, solar lentigo, pyogenic granuloma.
- **Actinic keratosis is not benign.** It is precancerous, so it is never trained on. AK images form the `ak_side` set, where we report the share the model calls malignant.
- **Excluded:** rare cancers (Kaposi sarcoma, DFSP, mycosis fungoides, Merkel cell and others), non-melanocytic "nevi", and everything Fitzpatrick17k marks non-neoplastic. Bowen's disease counts as SCC.
- **Unknown skin type is dropped** from the trainable pools, as are Fitzpatrick17k rows its own QC marked as wrongly labelled.
- **BCC + SCC merge rule.** `count_check` recommends three classes (`nmsc / mel / benign`) if `target_test` holds fewer than 15 SCC images. It holds 23, so four classes are kept.

## 3. Splits

Frozen in `data/manifest.csv` (`split` column) and `data/splits/`. Nobody re-splits.

| Split | Images | What it is |
| --- | --- | --- |
| `source_train` / `source_val` / `source_test` | 2,022 / 435 / 435 | Fitzpatrick17k + PAD-UFES-20, skin types I–III, split 70 / 15 / 15 |
| `target_train` / `target_val` / `target_test` | 257 / 86 / 86 | Fitzpatrick17k, skin types IV–VI, split 60 / 20 / 20 |
| `pad_dark_test` | 60 | PAD-UFES-20, skin types IV–VI (biopsy-confirmed, Brazil) |
| `ddi_test` | 656 | DDI (Stanford, biopsy-confirmed), scored cancer vs not |
| `midas_test` | 1,380 | MRA-MIDAS clinical photos (Stanford, biopsy-confirmed) |
| `dermacon_test` / `dermacon_other` | 65 / 5,379 | DermaCon-IN (India): tumours / everyday non-tumour conditions |
| `ak_side` | 311 | Actinic keratoses |

- **No leakage.** Near-duplicate images (perceptual hash distance ≤ 2, or embedding cosine ≥ 0.90) and images of the same patient form a group. A group never spans two splits; `make_splits` asserts this. The duplicate search used an ImageNet EfficientNet-B0 as a similarity measure. That is data preparation, and it is the only place pretrained weights appear in this repository.
- **Skin-type buckets** are I–III / IV / V–VI. DDI only provides I–II / III–IV / V–VI and is reported in those.
- **DDI and Derm7pt images never appear in a figure** (licence terms). The repository stays private because the manifest carries DDI metadata.

## 4. What trains, what selects, what tests

- **Training data:** `source_train` (2,022 images, skin types I–III). The question stays the one the project started with: how well does a model trained on lighter skin hold up on darker skin and on Indian images?
- **Selection:** `source_val` picks the checkpoint and stops training. `target_val` is only for choosing between finished models. Test splits never choose anything.
- **Test-only sets** (`ddi_test`, `dermacon_test`, `dermacon_other`, `pad_dark_test`, `midas_test`) are never scored during training. `predict --final` scores them once per finished run and records it in `runs/final_ledger.csv`; a second look is refused.
- `data.train_splits` and `data.val_splits` in the config are lists, so training on more data (for example adding `target_train`) is a config change. Doing so changes the question, so record it here first.

## 5. The network: SkinNet

Defined in `src/skincancer/models.py`. A residual CNN, sized for a training set of about 2,000 images.

| Part | Layers | Output for a 224 × 224 input |
| --- | --- | --- |
| Stem | 3 × (3×3 conv, batch norm, ReLU), widths 16, 16, 32, the first with stride 2; then 3×3 max-pool, stride 2 | 32 × 56 × 56 |
| Stage 1 | 2 residual blocks, 32 channels | 32 × 56 × 56 |
| Stage 2 | 2 residual blocks, 64 channels, first one stride 2 | 64 × 28 × 28 |
| Stage 3 | 2 residual blocks, 128 channels, first one stride 2 | 128 × 14 × 14 |
| Stage 4 | 2 residual blocks, 256 channels, first one stride 2 | 256 × 7 × 7 |
| Head | global average pool, dropout 0.3, linear to 4 classes | 4 |

- A residual block is two 3×3 convolutions, each followed by batch norm, with a skip connection around them. Where a block changes resolution or width, the skip is average-pool then 1×1 convolution.
- 2.8 million parameters, 17 convolution layers on the main path.
- Weights start random (Kaiming normal). The last batch norm of each block starts at zero, so every block begins as the identity. Each seed gives a different start.

## 6. Training

Fixed on 12 Nov 2026, before any SkinNet run was scored. The config is `configs/skinnet.yaml`.

| Setting | Value |
| --- | --- |
| Layers trained | all, from the first step |
| Optimiser | AdamW, learning rate 1e-3, weight decay 0.05 |
| Schedule | 5 epochs linear warm-up, then cosine decay to zero at epoch 150 |
| Stopping | best `source_val` macro AUC is kept; stop after 30 epochs without improvement |
| Loss | cross-entropy, label smoothing 0.1, class-balanced weights |
| Batch | 32 images, 224 × 224 |
| Augmentation | random resized crop (40–100% of the image), horizontal and vertical flips, 90° turns, rotation ±30°, brightness and contrast ±20%, light blur, 1–3 erased patches |
| Not used | hue or saturation changes: they change apparent skin tone |
| Seeds | 0, 1, 2 |

- Class-balanced weights give each class the same total weight in the loss, so the 902 BCC images do not outvote the 337 melanomas.
- Evaluation images are resized to 256 on the shorter side and centre-cropped to 224.
- **Expectation, stated in advance:** with 2,022 training images and no pretraining, SkinNet will probably score below the earlier ImageNet-pretrained EfficientNet-B0 (macro AUC 0.89 on `source_test`, in the tag). Small from-scratch networks on a few thousand medical images usually trail pretrained ones by a wide margin.

## 7. Metrics and statistics

- **Macro AUC** (one-vs-rest, averaged over the four classes) on the four-class sets. **Binary AUC** (cancer vs not) on DDI and DermaCon-IN.
- **Missed cancers and false alarms at a fixed operating point ("op90"):** per seed, the threshold on P(malignant) that catches 90% of cancers in `source_val`, applied unchanged to every other set.
- On `ak_side` and `dermacon_other`, which hold no cancers, the share of images called malignant.
- Every number is the mean of the three seeds, with a 95% confidence interval from 1,000 bootstrap resamples of groups (not single images). Subgroups under 10 images are not reported.
