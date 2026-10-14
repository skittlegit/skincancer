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

**MRA-MIDAS (Stanford, biopsy-proven, clinical photos; 1,380 images, 468 lesions; 96% FST I–III):** EfficientNet-B0 binary AUC 0.728 [0.694, 0.761] overall, **0.721 [0.686, 0.757] on FST I–III**; macro 0.674 [0.646, 0.701]; 43.5% of light-skin cancers missed. ResNet50 binary 0.666 on I–III. FST IV holds 42 images (8 malignant): too few to read.

**Two shifts, separated (binary AUC, EfficientNet-B0):**

| Same atlases, light skin (source test) | New institution, light skin (MIDAS I–III / DDI I–II) | New institution, dark skin (DDI V–VI) |
| --- | --- | --- |
| 0.893 | 0.721 / 0.682 | 0.502 |

Changing institution alone (camera, protocol, case mix) costs ~0.17–0.21 AUC, even on light skin; MIDAS and DDI I–II agree. Darker skin costs a further ~0.18, down to chance. So the dark-skin gap in RQ1 sits on top of a sizeable institution gap. The adaptation methods (E5, E7) target only the skin-tone part, since their target data is Fitzpatrick17k IV–VI.

**ResNet50 (same protocol, layer4 unfrozen):** source macro 0.788 [0.745, 0.834], target macro 0.755 [0.641, 0.849], DDI I–II 0.708 [0.636, 0.782] → **DDI V–VI 0.523 [0.422, 0.631]**, missing 62.5% of V–VI cancers. DermaCon-IN 0.668 [0.523, 0.844]; 36.6% of DermaCon-IN non-cancer images called malignant. ResNet50 is weaker in-domain but shows the same collapse on dark skin, so **the gap is not architecture-specific**.

**Reading (RQ1):**
- On biopsy-proven external images (DDI), performance falls with skin tone and reaches **chance on FST V–VI**, where the model misses about half the cancers (51% vs 17% on FST I–II). This matches Daneshjou et al. (2022): 0.72 → 0.57 for a HAM10000 model.
- Within Fitzpatrick17k there is **no gap** (0.864 vs 0.866). Its IV–VI images come from the same textbook atlases as the training images, so the domain shift is mostly skin tone, not source. DDI (another institution, real clinic photos) is the meaningful test.
- On Indian clinic data, almost a third of ordinary non-cancer images are called malignant. That would be a heavy false-alarm load in practice.
- The benign-lookalike class is the weak point everywhere (false-alarm rates of 40–56%). Data v1 has few benign training images (143); data v2 adds about 460.

## E3: image processing ablation, EfficientNet-B0 (14 Oct 2026)

Each variant applies the step to training and every test set alike (`cc` = shades-of-grey colour constancy, `clahe` = contrast on L, `crop` = lesion crop, D12; `all` = the three together).

| Metric | E1 (none) | cc | clahe | crop | all |
| --- | --- | --- | --- | --- | --- |
| **Target val macro AUC (selection metric)** | **0.891** | 0.881 | 0.889 | 0.883 | 0.820 |
| Source val macro AUC | 0.846 | 0.845 | 0.837 | 0.832 | 0.830 |
| Source test macro AUC | 0.864 | 0.867 | 0.859 | 0.853 | 0.848 |
| Fitzpatrick17k IV–VI test macro | 0.866 | 0.837 | 0.847 | 0.832 | 0.777 |
| PAD-UFES-20 IV–VI macro | 0.828 | 0.837 | 0.843 | 0.778 | 0.876 |
| DDI I–II binary | 0.682 | 0.692 | 0.691 | 0.689 | 0.691 |
| DDI III–IV binary | 0.670 | 0.663 | 0.664 | 0.668 | 0.649 |
| **DDI V–VI binary** | 0.502 | 0.520 | 0.527 | 0.531 | 0.533 |
| DDI V–VI missed cancers | 51.4% | 48.6% | 48.6% | 58.3% | 47.9% |
| DermaCon-IN binary | 0.690 | 0.704 | 0.688 | 0.670 | 0.713 |
| DermaCon-IN non-cancer called malignant | 30.8% | 30.6% | 34.9% | 30.4% | 30.5% |
| AK called malignant | 91.6% | 90.2% | 90.5% | 87.1% | 87.0% |

CIs on the external sets are ±0.08–0.10 (DDI V–VI) and ±0.18 (DermaCon-IN); see `results/e3_*/metrics.csv`.

**Reading (RQ2):**
- **No preprocessing variant beats the baseline on target val**, the pre-registered selection metric (D1). CLAHE ties (0.889 vs 0.891); the full pipeline is clearly worse (0.820). So E8 does not carry an image-processing step (decision logged in DECISIONS.md, D14).
- On DDI V–VI every variant moves AUC from 0.50 to 0.52–0.53. That is consistent in direction but far inside the CI (±0.10): not evidence of a fix. Normalising colour and contrast does not remove what the model fails on in dark skin.
- Lesion crop hurts where lesions are small or framed differently (PAD-UFES-20 −0.05, DDI V–VI missed cancers up to 58%). The crop is a pipeline risk; the failure rate by eye on 100 crops is still to be measured.

## Seed ensemble (D15)

Mean of the three seeds' probabilities, against the seed mean:

| | Source val | Target val | Source test | Target test | DDI I–II | DDI V–VI | DermaCon-IN | MIDAS I–III |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 EffB0 (seed mean) | 0.846 | 0.891 | 0.864 | 0.866 | 0.682 | 0.502 | 0.690 | 0.721 |
| E1 EffB0 ensemble | 0.848 | 0.895 | 0.868 | 0.867 | 0.682 | 0.503 | 0.697 | 0.724 |
| E1 ResNet50 ensemble | 0.805 | 0.825 | 0.789 | 0.755 | 0.712 | 0.520 | 0.672 | 0.667 |

Ensembling gains +0.002 to +0.005 AUC: the three seeds already agree closely. It does nothing for dark skin (DDI V–VI 0.503), so seed variance is not the problem.

## E4: Indian-prior weighted loss (14 Oct 2026)

| | Target val macro (selection) | Source test macro | Target test macro | DDI V–VI binary | DermaCon-IN binary | DermaCon-IN non-cancer called malignant |
| --- | --- | --- | --- | --- | --- | --- |
| E1 | 0.891 | 0.864 | 0.866 | 0.502 | 0.690 | 30.8% |
| E4 Indian-prior loss | 0.895 | 0.870 | 0.874 | 0.475 | 0.699 | 31.0% |
| E1 + post-hoc consensus prior (D7) | 0.894 | 0.869 | 0.874 | 0.493 | 0.694 | 30.3% |

**Reading (RQ3):** reweighting towards the Indian case mix (BCC:SCC:melanoma 45:40:15) changes the ranking very little: every AUC moves by under 0.01, well inside the CIs. Training with the prior (E4) and correcting a trained model afterwards (D7) give the same result. So the epidemiological prior mainly moves predictions between the three cancer classes; it does not help separate cancer from benign on dark skin. The four regional mixes (consensus, Malwa-Punjab, Central India, Manipur) agree within 0.01.

## Fixed operating point: 90% sensitivity on source val (D17)

The threshold that catches 90% of cancers on light-skin validation images, applied unchanged everywhere (seed mean [95% CI]):

| | Source test | DDI I–II | **DDI V–VI** | MIDAS I–III | DermaCon-IN | DermaCon-IN non-cancer called malignant |
| --- | --- | --- | --- | --- | --- | --- |
| Missed cancers, E1 EffB0 | 10% [7, 14] | 41% [28, 55] | **66% [53, 78]** | 59% [55, 64] | 49% [16, 76] | 15% |
| Missed cancers, E1 ResNet50 | 12% | 52% | 68% | 58% | 47% | 31% |
| Missed cancers, E3 colour constancy | 8% | 20% | 60% | 48% | 49% | 19% |
| Missed cancers, E3 all steps | 8% | 20% | 54% | — | 49% | 24% |
| Missed cancers, E4 | 10% | 40% | 70% | — | 56% | 15% |

**Reading:**
- A threshold that misses 10% of cancers at home misses **two-thirds of cancers on FST V–VI** at another institution, and about 40–60% on light skin at another institution. This is the clinically meaningful form of the RQ1 gap: the model's scores are not just less discriminative, they are *shifted*, so a threshold set in-domain does not transfer.
- Colour constancy and CLAHE halve the missed rate on DDI I–II (41% → 20%) and cut MIDAS from 59% to 48%. But false alarms rise in step (DDI V–VI 27% → 35–38%) and AUC is unchanged (E3). So they shift the score scale (calibration) rather than separating cancer from benign better. On V–VI the missed rate stays above 50% under every variant.
- These are test-set observations, reported only; nothing was selected on them.
