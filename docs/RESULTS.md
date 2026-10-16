# Results

All results use the full Fitzpatrick17k set (D21). Numbers are the mean over seeds 0, 1 and 2, with group-bootstrap 95% CIs (D8). "op90" rates use the threshold that catches 90% of cancers on each seed's source val, applied unchanged to every set (D17). Full tables are in `results/<exp>/metrics.csv`.

## E1 / E2: source-only baseline, EfficientNet-B0

| Test set | AUC [95% CI] | Missed cancers (op90) | n |
| --- | --- | --- | --- |
| Source val (selection), FST I–III | macro 0.894 [0.87, 0.92] | — | 435 |
| Source test, FST I–III | macro 0.893 [0.87, 0.92]; binary 0.903 | — | 435 |
| Fitzpatrick17k IV–VI (target test) | macro 0.888 [0.84, 0.94] | — | 86 |
| PAD-UFES-20 IV–VI | macro 0.828 [0.73, 0.91] | — | 60 |
| DDI FST I–II | binary 0.645 [0.56, 0.73] | — | 208 |
| DDI FST III–IV | binary 0.658 [0.58, 0.74] | — | 241 |
| **DDI FST V–VI** | **binary 0.526 [0.43, 0.62]** | 36% [24, 50] | 207 |
| DermaCon-IN cancers vs lookalikes | binary 0.706 [0.52, 0.87] | — | 65 |
| DermaCon-IN non-cancer called malignant | 47% (argmax), 43% (op90) | — | 5,379 |
| MIDAS FST I–III | binary 0.685 [0.65, 0.72] | 40% [37, 44] | 1,322 |
| Actinic keratosis called malignant | 90% | — | 311 |

**Reading:**
- With the full training set, in-domain performance rises: source test 0.89, and target test 0.89 with no gap between light and dark skin within the same atlases.
- At other institutions the model is no better: DDI V–VI is still at chance (0.53, CI includes 0.5), and DDI I–II (0.65) and MIDAS (0.69) are slightly below what the partial-download development run showed. More atlas data makes the model better at the atlases, not at other clinics.
- False alarms on Indian clinic images are high: almost half of DermaCon-IN's non-cancer images are called malignant.

## E1 / E2: ResNet50

| | Source val | Target val | Source test | Target test | PAD-UFES-20 IV–VI | DDI I–II | DDI III–IV | DDI V–VI | DDI V–VI missed (op90) | DermaCon-IN | DermaCon-IN non-cancer called malignant | MIDAS I–III |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EfficientNet-B0 | 0.894 | 0.916 | 0.893 | 0.888 | 0.828 | 0.645 | 0.658 | 0.526 | 36% | 0.706 | 47% | 0.685 |
| ResNet50 (layer4 unfrozen) | 0.819 | 0.805 | 0.790 | 0.793 | 0.893 | 0.749 | 0.769 | 0.581 | 45% | 0.757 | 34% | 0.645 |

**Reading:** ResNet50 is clearly weaker in-domain (source test 0.79 vs 0.89) but better at DDI (I–II 0.75, V–VI 0.58) and DermaCon-IN (0.76). With only its last block fine-tuned it stays closer to its ImageNet features, so it fits the atlases less and travels a little better. DDI V–VI still falls well below DDI I–II for both backbones, so **the skin-tone gap is not architecture-specific**.
