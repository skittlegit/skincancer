# Data plan: what to add after the baseline

Status (12 Nov 2026): Derm7pt was tried as training-only data in the third round and did not help ([RESULTS.md](RESULTS.md)); it is not in the training data. Nothing else here has been started.

## Why more data

The baseline trains on 12,562 images, of which 2,220 are cancer. Two weaknesses are known before any result:

1. **Few hard cases.** Most not-cancer images are rashes and similar conditions that look little like a tumour. The hard part of the task is telling a cancer from a benign tumour that resembles it, and training holds only 733 such benign lookalikes.
2. **A shortcut.** 70% of the PAD-UFES-20 images are cancer against 13% of the Fitzpatrick17k images, and the two are different kinds of photo (smartphone against atlas). The network can score well by recognising the kind of photo.

## Candidates

Counts marked "about" are from memory and must be confirmed at download.

| Dataset | What it adds | On disk? | Cost or risk | Confidence it helps |
| --- | --- | --- | --- | --- |
| Derm7pt, clinical photos | 248 melanomas, 42 BCCs, 664 benign nevi | Yes | None: it is not a test set. No SCC. | High |
| MRA-MIDAS, 70% by patient | About 560 cancers and 400 benign tumours, all biopsy-confirmed | Yes | Shrinks our largest outside tumour test to 30%. Cannot be undone once a model has trained on it. | Fairly high |
| SCIN (Google) | About 10,000 phone photos from the public, almost all not cancer | No | Adds no cancers. Labels come from dermatologists reading photos, not biopsies. | High, for weakness 2 |
| ISIC 2019 | About 8,500 cancers and 12,900 nevi | No | Dermoscopy images (a magnifying scope pressed on the skin), which look nothing like ordinary photos. May not help on clinical photos. | Unsure |

Not planned: Dermofit (paid licence, about 400 cancers) and MED-NODE (170 images).

## Proposed order

Each step is one training run, compared with the run before it on the same test sets. A step is kept only if it improves the outside scores (DDI, the held-back part of MIDAS, DermaCon-IN), not just the score on our own `test` split.

1. **Baseline.** Fitzpatrick17k + PAD-UFES-20 only. Running now.
2. **Add Derm7pt and 70% of MIDAS.** Both are on disk. Training cancers go from 2,220 to about 3,070, and most of the additions are hard cases.
3. **Add SCIN.** Check whether false alarms on phone photos of harmless conditions fall (DermaCon-IN everyday conditions is the test for this).
4. **Add ISIC 2019,** as a with-and-without experiment.

## Open decisions

- **MIDAS:** move 70% into training, or keep all of it as a test set? DDI stays an untouched, biopsy-confirmed outside test either way.
- **Dermoscopy:** mix ISIC images into the same model, or keep the project on clinical photos only?
- **Every added dataset needs the same checks as the current ones:** diagnoses mapped through the label map, a near-duplicate search against the test sets, and splits by patient. Adding data means rebuilding `train` / `val` / `test`, so earlier runs must be repeated on the new splits to stay comparable.

## Not covered here

Skin tone. Almost all of these datasets are lighter skin, so none of them fixes the shortage of dark-skin cancers. Whether darker skin needs its own model or one combined model is the question for stage two ([DECISIONS.md](../DECISIONS.md), section 2).
