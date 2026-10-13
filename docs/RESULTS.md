# Results log

All results here are **data v1**: Fitzpatrick17k has 1,919 of 4,491 neoplastic images (see DECISIONS.md D6). Every run is re-done on data v2 once the authors' full image set arrives. Numbers are the mean over seeds 0, 1 and 2, with group-bootstrap 95% CIs (D8). Full tables are in `results/<exp>/metrics.csv`.

## E1 / E2: source-only baseline, EfficientNet-B0 (14 Oct 2026)

| Test set | AUC [95% CI] | Missed cancers | Benign called malignant | n (malignant) |
| --- | --- | --- | --- | --- |
| Source test, FST I–III | macro 0.864 [0.827, 0.902] | 4.2% | 49.5% | 271 (240) |
| Fitzpatrick17k IV–VI (target test) | macro 0.866 [0.786, 0.927] | 17.8% | 44.4% | 51 (45) |
| PAD-UFES-20 IV–VI (Brazil, biopsied) | macro 0.828 [0.719, 0.905] | 5.9% | 47.4% | 60 (34) |
| DDI, FST I–II | binary 0.682 [0.599, 0.754] | 17.0% | 56.0% | 208 (49) |
| DDI, FST III–IV | binary 0.670 [0.601, 0.740] | 24.8% | 55.7% | 241 (74) |
| **DDI, FST V–VI** | **binary 0.502 [0.398, 0.598]** | **51.4%** | 47.6% | 207 (48) |
| DermaCon-IN (India), lookalikes + cancers | binary 0.690 [0.493, 0.873] | 29.8% | 41.3% | 65 (19) |
| DermaCon-IN, non-cancer clinic images | — | — | **30.8% called malignant** | 5,379 (0) |
| Actinic keratosis (side analysis) | — | — | 91.6% called malignant | 246 |

**Reading (RQ1):**
- On biopsy-proven external images (DDI), performance falls with skin tone and reaches **chance on FST V–VI**, where the model misses about half the cancers (51% vs 17% on FST I–II). This matches Daneshjou et al. (2022): 0.72 → 0.57 for a HAM10000 model.
- Within Fitzpatrick17k there is **no gap** (0.864 vs 0.866). Its IV–VI images come from the same textbook atlases as the training images, so the domain shift is mostly skin tone, not source. DDI (another institution, real clinic photos) is the meaningful test.
- On Indian clinic data, almost a third of ordinary non-cancer images are called malignant. That would be a heavy false-alarm load in practice.
- The benign-lookalike class is the weak point everywhere (false-alarm rates of 40–56%). Data v1 has few benign training images (143); data v2 adds about 460.
