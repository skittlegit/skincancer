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
| MIDAS FST I–III | binary 0.685 [0.65, 0.72] | 40% [37, 44] | 1,327 |
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

## E5: fine-tuning on Fitzpatrick17k FST IV–VI labels (EfficientNet-B0)

E1 fine-tuned on 10/25/50/100% of target_train (28 to 257 images), with the same number of optimizer steps at every fraction (D18). Fraction 0 is E1.

| Target labels used | Source val (macro) | Target val (macro) | Target test (macro) | PAD-UFES-20 IV–VI (macro) | DDI I–II | DDI V–VI | DDI V–VI missed (op90) | DermaCon-IN | DermaCon-IN non-cancer called malignant (argmax / op90) | MIDAS I–III |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0% (E1) | 0.894 | 0.916 | 0.888 | 0.828 | 0.645 | 0.526 | 36% | 0.706 | 47% / 43% | 0.685 |
| 10% | 0.874 | 0.917 | 0.885 | 0.840 | 0.617 | 0.524 | 31% | 0.703 | 35% / 48% | 0.676 |
| 25% | 0.874 | 0.933 | 0.911 | 0.843 | 0.635 | 0.538 | 35% | 0.715 | 31% / 45% | 0.698 |
| 50% | 0.871 | 0.937 | 0.921 | 0.836 | 0.647 | 0.526 | 38% | 0.724 | 36% / 46% | 0.703 |
| 100% | 0.872 | 0.948 | 0.936 [0.89, 0.97] | 0.820 | 0.664 | 0.521 | 34% | 0.701 | 42% / 51% | 0.708 |

Figure: `reports/figures/e5_curve.png` (`python -m skincancer.report curve --ref e1_effb0 --prefix e5s_f`).

**Reading:**
- Dark-skin labels from the same atlas help on that atlas: target test rises steadily from 0.89 to 0.94 with the number of labels.
- They do not transfer to other clinics: DDI V–VI stays at 0.52–0.54 at every fraction, and DermaCon-IN and PAD-UFES-20 do not move beyond noise. The FST IV–VI gap on DDI is a dataset shift, not only a skin-tone shift that dark-skin atlas images can fix.
- Source val drops by about 0.02 after fine-tuning (mild forgetting).
- The op90 false-alarm rate on DermaCon-IN rises (43% → 51%) because the op90 threshold is set on source val, where the fine-tuned model is slightly worse. At the argmax decision it falls (47% → 31–42%).

## E6: texture features and CNN + texture fusion

Heads trained on frozen E1 features, per seed: an MLP on the CNN embedding alone (the control), logistic regression on handcrafted texture features alone, and the same MLP on embedding + texture (fusion).

| | Source val (macro) | Target val (macro) | Target test (macro) | PAD-UFES-20 IV–VI (macro) | DDI I–II | DDI V–VI | DDI V–VI missed (op90) | DermaCon-IN | DermaCon-IN non-cancer called malignant (argmax / op90) | MIDAS I–III |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | 0.894 | 0.916 | 0.888 | 0.828 | 0.645 | 0.526 | 36% | 0.706 | 47% / 43% | 0.685 |
| Embedding MLP (control) | 0.881 | 0.906 | 0.884 | 0.795 | 0.643 | 0.519 | 42% | 0.715 | 47% / 36% | 0.656 |
| Texture only | 0.773 | 0.727 | 0.675 | 0.853 | 0.632 | 0.545 | 19% | 0.484 | 54% / 54% | 0.608 |
| Fusion | 0.883 | 0.914 | 0.889 | 0.804 | 0.649 | 0.518 | 46% | 0.717 | 44% / 33% | 0.662 |

**Reading:**
- Adding texture to the embedding gives a small, non-significant gain over the control (target val 0.914 vs 0.906) and matches E1, not better.
- Texture alone is weak in-domain (target val 0.73) but loses less at other clinics: PAD-UFES-20 0.85 and DDI V–VI 0.55, the best DDI V–VI so far, though the CI still covers 0.5. Its low missed-cancer rate (19%) comes with 54% false alarms, so it is a looser threshold, not a better classifier.
- Fusion has the fewest DermaCon-IN false alarms at op90 so far (33%) but misses more DDI V–VI cancers (46%): the trade-off moves, it does not improve.

## E7: Deep CORAL with unlabelled FST IV–VI images (EfficientNet-B0)

CORAL aligns the feature covariance of source images with unlabelled target_train images during training. λ grid from D13; λ chosen on target val (D23).

| λ | Source val (macro) | Target val (macro) | Target test (macro) | PAD-UFES-20 IV–VI (macro) | DDI I–II | DDI V–VI | DDI V–VI missed (op90) | DermaCon-IN | DermaCon-IN non-cancer called malignant (argmax / op90) | MIDAS I–III |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 (E1) | 0.894 | 0.916 | 0.888 | 0.828 | 0.645 | 0.526 | 36% | 0.706 | 47% / 43% | 0.685 |
| 100 | 0.888 | 0.909 | 0.886 | 0.843 | 0.636 | 0.536 | 32% | 0.712 | 45% / 42% | 0.701 |
| 1000 | 0.892 | 0.912 | 0.892 | 0.861 | 0.637 | 0.535 | 39% | 0.716 | 45% / 37% | 0.711 |
| **10000** (chosen) | 0.892 | 0.914 | 0.890 | 0.882 [0.81, 0.94] | 0.651 | 0.552 [0.45, 0.65] | 34% | 0.699 | 47% / 42% | 0.715 |

**Reading:**
- On the selection sets CORAL is level with E1 at every λ (target val 0.909–0.914 vs 0.916), so by the protocol it is not selected for E8.
- The external sets move in CORAL's favour as λ grows: PAD-UFES-20 0.83 → 0.88, MIDAS 0.69 → 0.72, DDI V–VI 0.53 → 0.55. Each change is inside its CI, but the direction is consistent across three independent clinics, which E3–E6 did not show.
- DDI V–VI remains near chance (CI 0.45–0.65), so alignment to Fitzpatrick17k's dark-skin images narrows the clinic gap a little without closing the skin-tone gap.

## E8: chosen combination (D23)

Only E5 fine-tuning beats E1 on target val, so E8 is the E5 recipe alone. On EfficientNet-B0 that is the E5 100% row above (target test 0.936, DDI V–VI 0.521). E8 ResNet50 is trained after the follow-up runs and added here.

## Stronger baselines: E1 + Derm7pt clinical photos (D16)

E1 with Derm7pt's 954 clinical photos (FST I–III assumed, deduplicated against every evaluation split) added to source_train only.

| | Source val (macro) | Target val (macro) | Target test (macro) | PAD-UFES-20 IV–VI (macro) | DDI I–II | DDI V–VI | DDI V–VI missed (op90) | DermaCon-IN | DermaCon-IN non-cancer called malignant (argmax / op90) | MIDAS I–III |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | 0.894 | 0.916 | 0.888 | 0.828 | 0.645 | 0.526 | 36% | 0.706 | 47% / 43% | 0.685 |
| E1 + Derm7pt | 0.890 | 0.912 | 0.893 | 0.814 | 0.637 | **0.600 [0.50, 0.70]** | 33% | 0.638 | 64% / 60% | 0.712 |

**Reading:**
- The selection sets do not change (target val 0.912 vs 0.916).
- DDI V–VI rises from 0.53 to 0.60, the largest gain of any method so far, and the first whose CI does not reach below 0.5. MIDAS also rises (0.69 → 0.71). More clinic-style photos (not atlas images) help at other clinics, even though they are light-skinned.
- The cost lands on Indian images: DermaCon-IN falls to 0.64 and its non-cancer false-alarm rate jumps to 60–64%. The cause is not established. A likely one: Derm7pt's 664 benign photos are almost all moles, so "benign" comes to mean "looks like a mole", and DermaCon-IN's non-cancer images (mostly vitiligo, tinea, scabies, acne) look nothing like moles. Worth checking on the confusion matrix before reporting it as the reason.

## Stronger baselines: Derm Foundation (D15, D20)

Google's Derm Foundation model, frozen; 6144-d embeddings at 448 px. Heads trained on source_train: a linear probe (`e1_fm_lr`, C = 0.001 chosen on source val; deterministic, one run) and an MLP (`e1_fm_mlp`, 3 seeds). `e5_fm_lr` is the linear probe trained on source_train + target_train, C = 0.001 chosen on target val (D20).

| | Source val (macro) | Target val (macro) | Target test (macro) | PAD-UFES-20 IV–VI (macro) | DDI I–II | DDI III–IV | DDI V–VI | DDI V–VI missed (op90) | DermaCon-IN | DermaCon-IN non-cancer called malignant (argmax / op90) | MIDAS I–III |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 (EfficientNet-B0) | 0.894 | 0.916 | 0.888 | 0.828 | 0.645 | 0.658 | 0.526 | 36% | 0.706 | 47% / 43% | 0.685 |
| Derm Foundation, linear | 0.916 | 0.942 | 0.947 | 0.939 | **0.779** | **0.829** | **0.606 [0.52, 0.69]** | 15% | **0.856 [0.71, 0.96]** | 78% / 73% | **0.824** |
| Derm Foundation, MLP | 0.913 | 0.939 | 0.936 | 0.917 | 0.748 | 0.794 | 0.609 [0.53, 0.68] | 18% | 0.820 | 73% / 66% | 0.808 |
| Derm Foundation + target labels (linear) | 0.915 | 0.957 | 0.962 | 0.935 | 0.778 | 0.834 | 0.546 [0.45, 0.65] | 25% | 0.896 [0.78, 0.98] | 57% / 53% | 0.829 |

**Reading:**
- Derm Foundation is the strongest backbone everywhere: +0.13 at DDI I–II, +0.17 at DDI III–IV, +0.15 on DermaCon-IN and +0.14 on MIDAS over E1. Only a linear layer is trained on top.
- DDI V–VI rises to 0.61, with a CI that clears 0.5 for both heads. It misses 15% of dark-skin DDI cancers at op90, against 36% for E1. The skin-tone gradient remains: 0.78 → 0.83 → 0.61 across DDI I–II, III–IV and V–VI.
- Its false alarms are worse: 73% of DermaCon-IN non-cancer images are called malignant at op90. The 4-class model has no "not a tumour" option, so every rash must be filed as one of four lesion classes. Why Derm Foundation files more of them as malignant than E1 is not established; one guess is that its features flag any abnormal skin, which a head trained only on tumours reads as malignant.
- Adding the atlas dark-skin labels repeats E5: target val and target test rise (0.96), and DermaCon-IN improves (0.90) with fewer false alarms (53%), but DDI V–VI drops back to 0.55.
- The source-only probe picked C = 0.001, the low edge of its grid {0.001 … 1}, so a lower C might do slightly better. It was not re-run, since e1_fm_lr has already had its one test-set look. (The + target probe's grid starts at 0.0001, and 0.001 won there too.)
- Caveat (D15): Derm Foundation's pre-training may include public datasets (Fitzpatrick17k, PAD-UFES-20). DDI and MIDAS are access-controlled and DermaCon-IN post-dates the model, so those three comparisons are clean.

## Stronger baselines: E1 at 320 px with flip TTA (D15)

E1 trained at 320 px (resize 366) instead of 224, with test-time averaging over four flips.

| | Source val (macro) | Target val (macro) | Source test (macro) | Target test (macro) | PAD-UFES-20 IV–VI (macro) | DDI I–II | DDI III–IV | DDI V–VI | DDI V–VI missed (op90) | DermaCon-IN | DermaCon-IN non-cancer called malignant (argmax / op90) | MIDAS I–III |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 (224 px) | 0.894 | 0.916 | 0.893 | 0.888 | 0.828 | 0.645 | 0.658 | 0.526 | 36% | 0.706 | 47% / 43% | 0.685 |
| E1 320 px + TTA | 0.902 | 0.915 | 0.910 | 0.892 | 0.841 | 0.663 | 0.683 | 0.525 | 37% | 0.744 | 47% / 42% | 0.710 |

**Reading:**
- Bigger images help a little almost everywhere: source test 0.89 → 0.91, DDI I–II and III–IV +0.02, DermaCon-IN +0.04, MIDAS +0.03. Each gain is inside its CI.
- DDI V–VI does not move (0.53), so the extra resolution helps the lighter-skin outside sets only and widens the skin-tone gap at DDI slightly.
- Target val is unchanged (0.915 vs 0.916), so by the protocol it is not a better baseline, just a slightly stronger one away from home.
