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

## E5 (first run, superseded by D18)

The first fine-tuning curve is invalid below 100%: the 10–50% subsets got 0–2 gradient steps per epoch (batch 32, drop_last), so they reproduce E1 (target val 0.891 at every fraction). The corrected curve (`e5s_f*`, equal steps per fraction) is queued. The 100% run did train (152 images, 4 steps per epoch): target val 0.891 → 0.930, target test 0.866 → 0.916 [0.85, 0.97], DDI V–VI 0.502 → 0.516, missed cancers on DDI V–VI at the fixed operating point 66% → 57%, but DermaCon-IN non-cancer called malignant 15% → 24%. These numbers will be replaced by `e5s_f100`.

## E6: texture fusion (14 Oct 2026)

Heads on the frozen E1 embedding (1280-d) and 78 hand-crafted texture features (LBP, GLCM, Gabor, colour moments), trained on source_train and selected on source_val:

| | Target val macro (selection) | Source test | Target test | PAD-UFES-20 IV–VI | DDI I–II | **DDI V–VI** | DDI V–VI missed (op90) | DermaCon-IN | DermaCon-IN non-cancer called malignant (op90) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 (fine-tuned CNN) | 0.891 | 0.864 | 0.866 | 0.828 | 0.682 | 0.502 [0.40, 0.60] | 66% | 0.690 | 15% |
| Embedding MLP (control) | 0.882 | 0.841 | 0.871 | 0.791 | 0.656 | 0.495 [0.39, 0.59] | 60% | 0.665 | 23% |
| Embedding + texture MLP | 0.879 | 0.845 | 0.882 | 0.801 | 0.670 | 0.501 [0.40, 0.60] | 58% | 0.658 | 22% |
| Texture only (logistic) | 0.666 | 0.753 | 0.714 | 0.816 | 0.570 | 0.579 [0.49, 0.67] | 15% | 0.658 | 68% |

**Reading (RQ4):**
- **Texture adds nothing on top of the CNN embedding.** Fusion and the embedding-only control are within 0.01 everywhere, with the control's target val slightly higher (0.882 vs 0.879). Whatever the 78 features capture, the embedding already has it.
- Texture alone is much weaker in-domain (target val 0.67). Its DDI V–VI AUC (0.58) is the highest of any model so far and sits near its DDI I–II score (0.57), so it degrades less with skin tone. But the CI includes chance, so this is a hint, not a finding. Its threshold also transfers badly (68% of DermaCon-IN non-cancer images called malignant).
- Exploratory note for data v2: test whether hand-crafted texture is less skin-tone dependent with more DDI-like data. Not pursued now (no selection on test sets).

## E7: Deep CORAL, λ sweep (14 Oct 2026)

Source-trained with a CORAL term that aligns feature covariances with the unlabelled Fitzpatrick17k IV–VI training images. λ is chosen on target val (D1, D13).

| | Target val macro (selection) | Source test | Target test | PAD-UFES-20 IV–VI | DDI I–II | **DDI V–VI** | DDI V–VI missed (op90) | DermaCon-IN | DermaCon-IN non-cancer called malignant (op90) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | 0.891 | 0.864 | 0.866 | 0.828 | 0.682 | 0.502 [0.40, 0.60] | 66% | 0.690 | 15% |
| CORAL λ = 100 | 0.872 | 0.851 | 0.858 | 0.834 | 0.682 | 0.507 [0.40, 0.61] | 69% | 0.720 | 17% |
| **CORAL λ = 1000 (selected)** | **0.883** | 0.859 | 0.877 | 0.846 | 0.690 | 0.522 [0.42, 0.62] | 71% | 0.728 | 14% |
| CORAL λ = 10000 | 0.868 | 0.847 | 0.853 | 0.864 | 0.692 | 0.557 [0.46, 0.66] | 62% | 0.729 | 17% |

**Reading (RQ5):**
- On target val, λ = 1000 is the best CORAL setting, but no λ beats E1 (0.891).
- On the external sets the direction is consistent: stronger alignment raises DDI V–VI (0.50 → 0.52 → 0.56), DermaCon-IN (0.69 → 0.73) and PAD-UFES-20 IV–VI (0.83 → 0.86). Every CI still overlaps E1's, and target val does not see the gain, so it is a trend, not a demonstrated fix. Target val comes from the same atlases as training, so it cannot reward alignment towards other institutions.
- Missed cancers at the fixed operating point stay at 62–71% on DDI V–VI: alignment does not move scores enough for a light-skin threshold to transfer.
- Run note: λ = 100 seed 1 crashed with an out-of-memory error after training. Its predictions were regenerated from the saved checkpoint (same inference; see commit 8630f1b).

## MRA-MIDAS (light-skin institution shift) across methods

Binary and macro AUC on FST I–III (n = 1,322), with rates at the fixed operating point (D17):

| | Binary AUC | Macro AUC | Missed (op90) | False alarm (op90) |
| --- | --- | --- | --- | --- |
| E1 EffB0 | 0.721 [0.69, 0.76] | 0.674 | 59% | 13% |
| E1 ResNet50 | 0.666 | 0.648 | 58% | 20% |
| E3 colour constancy / CLAHE | 0.725 / 0.725 | 0.686 / 0.689 | 48% / 43% | 20% / 24% |
| E4 Indian-prior loss | 0.729 | 0.675 | 57% | 14% |
| E6 fusion | 0.709 | 0.656 | 53% | 20% |
| E7 CORAL λ = 1000 / 10000 | 0.735 / 0.736 | 0.685 / 0.696 | 62% / 63% | 12% / 11% |

No method moves MIDAS beyond the E1 CI (0.69–0.76): the institution gap (≈0.17 AUC from source test) is untouched by every fix aimed at skin tone. Acral lesions (n ≈ 40) are too few to read (CIs ±0.25).

## E5: localized fine-tuning curve (corrected, D18)

Fine-tuning the E1 model on nested fractions of Fitzpatrick17k IV–VI target_train (152 images at 100%), with equal optimizer steps per fraction:

| Fraction of target_train | 0% (E1) | 10% (~17) | 25% (~40) | 50% (~78) | 100% (152) |
| --- | --- | --- | --- | --- | --- |
| Target val macro (selection) | 0.891 | 0.891 | 0.901 | 0.911 | **0.933** |
| Target test macro | 0.866 [0.79, 0.93] | 0.877 | 0.886 | 0.897 | **0.921 [0.85, 0.97]** |
| Source test macro | 0.864 | 0.853 | 0.856 | 0.854 | 0.852 |
| PAD-UFES-20 IV–VI macro | 0.828 | 0.829 | 0.835 | 0.833 | 0.840 |
| DDI V–VI binary | 0.502 [0.40, 0.60] | 0.510 | 0.516 | 0.520 | 0.518 [0.42, 0.62] |
| DDI V–VI missed (op90) | 66% | 64% | 69% | 66% | 57% |
| DermaCon-IN binary | 0.690 | 0.694 | 0.696 | 0.698 | 0.700 |
| DermaCon-IN non-cancer called malignant (op90) | 15% | 17% | 17% | 22% | 25% |
| MIDAS I–III binary | 0.721 | 0.725 | 0.723 | 0.721 | 0.719 |

**Reading (RQ6):**
- **Within the target distribution, dark-skin data closes the gap and keeps paying off:** target test rises steadily from 0.866 to 0.921 as fine-tuning data grows from 0 to 152 images, at a cost of about 0.01 on source test. This is the only method so far with a clear gain on target val.
- **The gain does not transfer to another institution.** DDI V–VI stays near chance (0.50 → 0.52, CI ±0.10) and DermaCon-IN moves by 0.01. Dark-skin images from the *same atlases* teach the model those atlases' dark-skin images, not dark skin in general.
- False alarms on Indian non-cancer images rise (15% → 25% at the fixed operating point) as the model is pushed towards the target class mix.
- With 17–40 images, several seeds' best checkpoint is still the starting model (target val has 51 images, so small gains are within its noise).

## E1 + Derm7pt: more melanoma and benign training data (D16)

954 Derm7pt clinical photos (248 melanoma, 664 benign, 42 BCC) added to source_train only. Every evaluation set is unchanged.

| | Source val (selection) | Target val | Source test | Target test | PAD-UFES-20 IV–VI | DDI I–II | **DDI V–VI** | DDI V–VI missed (op90) | DermaCon-IN | DermaCon-IN non-cancer called malignant (op90) | MIDAS I–III |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | 0.846 | 0.891 | 0.864 | 0.866 | 0.828 | 0.682 | 0.502 [0.40, 0.60] | 66% | 0.690 | 15% | 0.721 |
| E1 + Derm7pt | 0.860 | 0.896 | 0.859 | 0.881 | 0.801 | 0.656 | **0.580 [0.47, 0.68]** | 57% | 0.639 | 29% | 0.729 |

**Reading:**
- Both validation sets improve a little (source val +0.014, target val +0.005).
- DDI V–VI rises to 0.58, the largest gain on dark skin at another institution of any method so far. Missed cancers at the fixed operating point fall from 66% to 57%. The CI (0.47–0.68) still includes chance.
- Other external sets get worse: DDI I–II −0.03, PAD-UFES-20 −0.03, DermaCon-IN −0.05, and twice as many DermaCon-IN non-cancer images are called malignant (29% vs 15%). Benign recall on source test drops (0.51 → 0.38). Derm7pt's benign class is almost all naevi, while the test sets' benign lookalikes are mostly seborrhoeic keratoses, dermatofibromas and other non-naevus lesions.
- So more data helps only where the new data looks like the test lesions. Data v2 (the full Fitzpatrick17k, with benign lookalikes from the same atlases) is the better source of benign images.

## E8: combined method (D19), both backbones

| | Target val (selection) | Source test | Target test | PAD-UFES-20 IV–VI | DDI I–II | DDI V–VI | DDI V–VI missed (op90) | DermaCon-IN | DermaCon-IN non-cancer called malignant (op90) | MIDAS I–III |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 EffB0 | 0.891 | 0.864 | 0.866 | 0.828 | 0.682 | 0.502 | 66% | 0.690 | 15% | 0.721 |
| E5s 100% EffB0 | 0.933 | 0.852 | 0.921 | 0.840 | 0.693 | 0.518 | 57% | 0.700 | 25% | 0.719 |
| **E8 EffB0** | 0.932 | 0.855 | 0.918 [0.85, 0.97] | 0.828 | 0.692 | 0.505 [0.41, 0.61] | 60% | 0.693 | 27% | 0.718 |
| E1 ResNet50 | 0.824 | 0.788 | 0.755 | 0.898 | 0.708 | 0.523 | 68% | 0.668 | 31% | 0.666 |
| **E8 ResNet50** | 0.801 | 0.788 | 0.772 [0.65, 0.87] | 0.909 | 0.712 | 0.535 [0.43, 0.64] | 67% | 0.625 | 29% | 0.671 |

**Reading:** E8 is E5 fine-tuning plus Indian-prior weighting, and it matches E5 alone. The weighting adds nothing, consistent with E4. On ResNet50, fine-tuning does not even raise target val (0.824 → 0.801; 51 images, so noisy). Neither backbone moves DDI V–VI or DermaCon-IN. **The best combination of the planned methods does not close the gap at another institution.**

## Stronger baselines (D15)

| | Source val | Target val | Source test | Target test | PAD-UFES-20 IV–VI | DDI I–II | DDI III–IV | **DDI V–VI** | DermaCon-IN | MIDAS I–III |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 EffB0 (224 px) | 0.846 | 0.891 | 0.864 | 0.866 | 0.828 | 0.682 | 0.670 | 0.502 [0.40, 0.60] | 0.690 | 0.721 [0.69, 0.76] |
| E1 seed ensemble | 0.848 | 0.895 | 0.868 | 0.867 | — | 0.682 | — | 0.503 | 0.697 | 0.724 |
| E1 320 px + flip TTA | 0.855 | 0.905 | 0.879 | 0.839 | 0.824 | 0.700 | 0.718 | 0.508 [0.40, 0.61] | 0.759 | 0.746 [0.71, 0.78] |
| E1 + Derm7pt | 0.860 | 0.896 | 0.859 | 0.881 | 0.801 | 0.656 | 0.694 | 0.580 [0.47, 0.68] | 0.639 | 0.729 |
| **Derm Foundation, linear probe** | **0.893** | **0.924** | **0.904** | **0.903** | 0.935† | **0.743** | **0.795** | **0.602 [0.51, 0.69]** | **0.886** | **0.815 [0.78, 0.85]** |
| Derm Foundation, MLP head | 0.895 | 0.923 | 0.892 | 0.897 | 0.930† | 0.703 | 0.779 | 0.594 [0.52, 0.68] | 0.830 | 0.796 |
| E1 embedding, same MLP head (E6 control) | 0.831 | 0.882 | 0.841 | 0.871 | 0.791 | 0.656 | 0.608 | 0.495 | 0.665 | 0.708 |

† Fitzpatrick17k and PAD-UFES-20 images are public and may be in Derm Foundation's pre-training data ("public image-text pairs from the web" and "additional public images", per its model card), so those columns (source/target val and test, PAD) could be optimistic. DDI and MIDAS are access-controlled, and DermaCon-IN appears to post-date the model, so those columns are the trustworthy ones.

**Reading:**
- **A frozen dermatology foundation model beats every method in the plan on every external set**, with only a linear head trained on 1,255 light-skin images. The like-for-like comparison is the same MLP head on the E1 embedding versus on Derm Foundation: DDI V–VI 0.50 → 0.59, DermaCon-IN 0.67 → 0.83, MIDAS 0.71 → 0.80.
- On DDI V–VI it is the first model whose CI excludes chance (0.51–0.69). The skin-tone gradient remains (DDI I–II 0.74, III–IV 0.80, V–VI 0.60), so pre-training on diverse clinical photos shrinks both the institution gap and the skin-tone gap without removing the latter.
- **Its scores do not transfer as a threshold.** At the source-val operating point (D17) it misses only 21% of DDI V–VI cancers but calls 72% of DermaCon-IN non-cancer images malignant: its score scale shifts upwards on external images. A deployment would need a site-specific threshold.
- 320 px + TTA helps light-skin external sets (MIDAS 0.72 → 0.75, DermaCon-IN 0.69 → 0.76) but not dark skin (DDI V–VI 0.51).
- The linear probe chose C = 0.001, the edge of its grid, so a stronger penalty might do slightly better on validation (not re-run).

**Derm Foundation + same-atlas dark-skin data (`e5_fm_lr`, D20).** Adding Fitzpatrick17k IV–VI target_train to the linear probe lifts target val from 0.924 to 0.977 but target test only from 0.903 to 0.906. DDI V–VI stays at 0.59 and DDI I–II drops from 0.74 to 0.68. This is E5's lesson again, on a much stronger backbone: dark-skin images from the same atlases do not teach dark skin in general. Target val (51 images, same atlases) rewards them; external test sets do not.

**Caveat on false alarms.** Both Derm Foundation heads call 83–86% of DermaCon-IN non-cancer images malignant by argmax (E1: 31%). These are mostly inflammatory and infectious conditions outside the four training classes. The classifier has no "other" class, and the foundation embedding sets them apart from benign tumours, so they land in a cancer class. AUC on cancer vs benign lookalikes (0.89) is excellent, but used on unselected clinic images this would flag most patients. Any deployment needs an out-of-scope gate or an "other skin disease" class trained on such images.
