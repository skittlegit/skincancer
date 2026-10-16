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

## E3: image processing ablation (EfficientNet-B0)

Each variant changes only the preprocessing of E1 (training and evaluation). Binary AUC unless marked; macro AUC for the selection sets.

| | Source val (macro) | Target val (macro) | Target test (macro) | PAD-UFES-20 IV–VI (macro) | DDI I–II | DDI V–VI | DDI V–VI missed (op90) | DermaCon-IN | DermaCon-IN non-cancer called malignant (op90) | MIDAS I–III |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 (none) | 0.894 | 0.916 | 0.888 | 0.828 | 0.645 | 0.526 | 36% | 0.706 | 43% | 0.685 |
| Colour constancy | 0.900 | 0.917 | 0.889 | 0.870 | 0.631 | 0.519 | 37% | 0.715 | 42% | 0.677 |
| CLAHE | 0.888 | 0.896 | 0.881 | 0.835 | 0.655 | 0.551 | 26% | 0.708 | 43% | 0.701 |
| Lesion crop | 0.885 | 0.897 | 0.870 | 0.773 | 0.609 | 0.527 | 46% | 0.624 | 44% | 0.662 |
| All three | 0.886 | 0.895 | 0.861 | 0.892 | 0.624 | 0.505 | 45% | 0.727 | 29% | 0.678 |

**Reading:**
- No preprocessing closes the DDI skin-tone gap: DDI V–VI stays between 0.51 and 0.55 for every variant, all with CIs covering 0.5.
- CLAHE gives the best DDI V–VI (0.55, fewer missed cancers at op90: 26% vs 36%) but is slightly worse on both selection sets, so by the protocol it is not selected.
- Lesion crop hurts nearly everywhere (DermaCon-IN 0.62, PAD-UFES-20 0.77): the crop often cuts lesion context on clinical photos.
- Combining all three gives the fewest DermaCon-IN false alarms (29%) but the weakest target test.
- Colour constancy ties E1 on target val (0.917 vs 0.916, well inside the CI), so E8 keeps no preprocessing (D22).

## E4: Indian-prior weighted loss (EfficientNet-B0)

E1 retrained with class weights set from the Indian consensus prior (more BCC/SCC, fewer melanomas, D7), no preprocessing.

| | Source val (macro) | Target val (macro) | Target test (macro) | PAD-UFES-20 IV–VI (macro) | DDI I–II | DDI III–IV | DDI V–VI | DDI V–VI missed (op90) | DermaCon-IN | DermaCon-IN non-cancer called malignant (argmax / op90) | MIDAS I–III |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | 0.894 | 0.916 | 0.888 | 0.828 | 0.645 | 0.658 | 0.526 | 36% | 0.706 | 47% / 43% | 0.685 |
| E4 | 0.893 | 0.915 | 0.887 | 0.825 | 0.648 | 0.674 | 0.518 | 38% | 0.738 [0.54, 0.91] | 41% / 43% | 0.701 |

**Reading:**
- The selection sets do not move (target val 0.915 vs 0.916), so the loss change neither helps nor hurts in-domain.
- DermaCon-IN rises slightly (0.71 → 0.74) and MIDAS a little (0.69 → 0.70), but both CIs overlap E1's widely.
- With the default argmax decision, fewer Indian non-cancer images are called malignant (47% → 41%). At the matched operating point (op90) the rate is the same 43%: the weighting mostly moves the decision threshold rather than separating the classes better.
- DDI V–VI stays at chance (0.52). Re-weighting the classes does not address the skin-tone gap.
