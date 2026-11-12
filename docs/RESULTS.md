# Results

Stage one: is the image a skin cancer or not? Every number is the mean of three seeds with a 95% confidence interval. The rules are in [DECISIONS.md](../DECISIONS.md); the full table is [results/skinnet/metrics.csv](../results/skinnet/metrics.csv).

## Baseline (12 Nov 2026)

SkinNet, 2.8M parameters, trained from random weights on 12,562 images (2,220 cancer) from Fitzpatrick17k and PAD-UFES-20. 60 epochs, 18 seconds per epoch on an RTX 4060. The three seeds reached validation AUC 0.883, 0.874 and 0.877.

**The model works on images like its training data and much less well on images from elsewhere.** AUC is 0.89 on our own test split, 0.65 on DDI and 0.60 on MIDAS.

| Test set | Images (cancers) | AUC | Missed cancers | False alarms |
| --- | --- | --- | --- | --- |
| Our test split | 2,695 (478) | 0.887 [0.87, 0.90] | 9% [7, 11] | 42% [40, 44] |
| DDI (Stanford, biopsy-confirmed) | 656 (171) | 0.654 [0.61, 0.70] | 14% [10, 19] | 66% [63, 70] |
| MIDAS (Stanford, biopsy-confirmed tumours) | 1,380 (802) | 0.603 [0.57, 0.64] | 32% [28, 35] | 55% [52, 59] |
| DermaCon-IN tumours (India) | 65 (19) | 0.825 [0.63, 0.96] | 16% [0, 41] | 43% [32, 56] |

Missed cancers and false alarms use one fixed threshold per seed: the one that catches 90% of the cancers in the validation split. AUC 0.5 is guessing and 1.0 is perfect.

On sets with no cancers, the share of images called cancer at that threshold:

| Set | Images | Called cancer |
| --- | --- | --- |
| DermaCon-IN everyday conditions (India) | 5,379 | 35% [34, 36] |
| Actinic keratosis (precancerous, never trained on) | 703 | 91% [90, 93] |

## What the breakdown shows

**1. The photo-type shortcut is real.** On the PAD-UFES-20 part of our test split the model calls nearly everything cancer: it misses 0.4% of cancers but raises a false alarm on 89% of the non-cancers. PAD-UFES-20 images are smartphone photos and 70% of them are cancer, so the model has learned that this kind of photo usually means cancer. This was predicted before the run (DECISIONS.md, section 4).

| Part of our test split | Images (cancers) | AUC | Missed cancers | False alarms |
| --- | --- | --- | --- | --- |
| Fitzpatrick17k | 2,459 (314) | 0.850 [0.83, 0.87] | 13% [10, 16] | 41% [39, 43] |
| PAD-UFES-20 | 236 (164) | 0.864 [0.80, 0.92] | 0.4% [0, 1] | 89% [81, 96] |

The overall 0.887 is higher than either part alone because telling the two sources apart already separates much of the cancer from the non-cancer.

**2. The model cannot yet tell a cancer from a benign tumour that resembles it.** False alarms on our test split, by kind of non-cancer:

| Kind of non-cancer | Images | False alarms |
| --- | --- | --- |
| Not a tumour (psoriasis, eczema, acne ...) | 1,810 | 38% [36, 40] |
| Other benign growths (cysts, birthmarks ...) | 249 | 51% [45, 56] |
| Benign tumours that resemble cancer (nevi, seborrheic keratosis ...) | 158 | 81% [75, 86] |

MIDAS holds only tumours, cancers against benign lookalikes, which is why it scores lowest.

**3. Common cancers are caught on familiar images, rare ones are not.** Missed cancers on our test split: BCC 4%, melanoma 5%, SCC 7%, rare cancers 39% (51 images). On MIDAS the same three common cancers are each missed about 31% of the time.

**4. DermaCon-IN tumours look better (AUC 0.825), but with 19 cancers the interval runs from 0.63 to 0.96.** It is not evidence either way.

## Reading

This is a baseline, and it behaves as expected for a small network trained from random weights on this data: a good score at home that does not travel. The two weaknesses it shows are the two that [DATA_PLAN.md](DATA_PLAN.md) targets, more hard cases (benign tumours and biopsy-confirmed cancers) and smartphone photos that are not cancer.

Training was still improving slowly when it ended (best epochs 54 to 56 of 60), so a longer schedule or a larger network may add a little. Neither would fix the shortcut.

The test-only sets have now been scored once by this model, as recorded in `runs/final_ledger.csv`.

## Second round: training the shortcut cannot exploit (12 Nov 2026)

No new data. Three variants changed only how training images are drawn and augmented; one was chosen on the validation split and only that one was scored here (DECISIONS.md, section 8).

**Choosing the variant (validation split, one seed each).** The deciding score is the AUC measured inside each dataset and averaged, which telling the datasets apart cannot raise.

| | Baseline | A: balanced inside each dataset | B: A + hard types drawn more often | C: B + quality augmentation |
| --- | --- | --- | --- | --- |
| Within-source AUC | 0.850 | **0.869** | 0.847 | 0.853 |
| AUC, cancers vs benign lookalikes | 0.713 | 0.777 | 0.790 | 0.785 |
| False alarms on benign lookalikes | 72% | 64% | 53% | 49% |
| False alarms overall | 36% | 39% | 41% | 43% |

A was chosen. B and C halve the false alarms on benign lookalikes but raise them on everything else; they were not scored on the test sets.

**Variant A against the baseline, three seeds each.**

| | Baseline | Variant A |
| --- | --- | --- |
| Our test split: AUC | 0.887 [0.87, 0.90] | 0.890 [0.87, 0.91] |
| Our test split: within-source AUC | 0.857 [0.83, 0.89] | 0.865 [0.84, 0.89] |
| Our test split: AUC, cancers vs benign lookalikes | 0.723 [0.67, 0.77] | 0.783 [0.74, 0.82] |
| False alarms, PAD-UFES-20 non-cancers | 89% [81, 96] | 63% [53, 74] |
| False alarms, benign lookalikes | 81% [75, 86] | 65% [58, 72] |
| False alarms, all non-cancers | 42% [40, 44] | 40% [38, 41] |
| Missed cancers, all | 9% [7, 11] | 8% [6, 10] |
| Missed rare cancers (51 images) | 39% [28, 50] | 22% [12, 32] |
| DDI: AUC | 0.654 [0.61, 0.70] | 0.623 [0.58, 0.66] |
| MIDAS: AUC | 0.603 [0.57, 0.64] | 0.628 [0.59, 0.66] |
| MIDAS: missed cancers / false alarms | 32% / 55% | 41% / 40% |
| DermaCon-IN tumours: AUC | 0.825 [0.63, 0.96] | 0.793 [0.62, 0.95] |
| DermaCon-IN everyday conditions called cancer | 35% | 34% |
| Actinic keratoses called cancer | 91% | 80% |

**Reading.**

- **At home it worked.** The photo-type shortcut is weaker (false alarms on PAD-UFES-20 non-cancers fell from 89% to 63%) and the model is better at the hard distinction (AUC on cancers against lookalikes up from 0.72 to 0.78), without missing more cancers.
- **Outside nothing changed.** MIDAS rose a little and DDI fell a little; both differences are inside the confidence intervals. On MIDAS the model traded false alarms for missed cancers, which is a shift of the threshold and not a better model.
- **So the shortcut was not what held the outside scores down.** The gap between 0.89 at home and 0.62 outside is mostly the difference between our training images and biopsy-confirmed hospital photos, which no reweighting of the same data can close.

Variant A's training setup is kept for the next rounds. Next: add Derm7pt to training, then larger images.

## Third round: more hard cases and larger images (12 Nov 2026)

Three variants built on checkpoint 1, compared with it on the validation split, one seed each (DECISIONS.md, section 9). None was scored on the test split or the outside sets.

| | Checkpoint 1 | D: + Derm7pt | E: 320-pixel images | F: both |
| --- | --- | --- | --- | --- |
| Within-source AUC (the deciding score) | **0.869** | 0.857 | 0.856 | 0.862 |
| AUC, cancers vs benign lookalikes | 0.777 | 0.757 | 0.745 | 0.770 |
| False alarms on benign lookalikes | 64% | 62% | 62% | 60% |
| False alarms on PAD-UFES-20 non-cancers | 58% | 51% | 56% | 53% |
| False alarms overall | 39% | 41% | 42% | 40% |
| Training images | 12,562 | 13,520 | 12,562 | 13,520 |

**No variant beat checkpoint 1,** so it stays the current model. Everything tried so far is summarised in [CHECKPOINTS.md](CHECKPOINTS.md).

**Reading.**

- Adding 958 Derm7pt photos (294 cancers, 664 benign tumours) did not help. They come from one more atlas and are almost all moles and melanomas, so they add little the network had not seen.
- Larger images did not help either. The variant at 320 pixels stopped early at epoch 56 with no gain; whatever limits the model, it is not image detail at this network size.
- The differences from checkpoint 1 (0.007 to 0.013) are about the size of the seed-to-seed spread (0.01), so "no better" is the supported conclusion, not "worse".
- Three rounds now point the same way: changes to how we use these two sources move the score on these two sources at most. The outside gap needs training images from other kinds of source, and a way to measure outside performance while choosing models. Both are in [DATA_PLAN.md](DATA_PLAN.md).
