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
