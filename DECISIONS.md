# Decisions

Settled protocol choices that the semester plan left open or got wrong. Where this file and the plan disagree, this file wins; the code enforces every rule below. Changing a decision means changing it here, in the code, and re-running whatever it touches.

## D1. Two validation sets, with fixed jobs

The plan early-stopped every model on the FST IV–VI validation split. For E1 that leaks target labels into the "source-only" model and understates the gap.

| Split | Size | Used for |
| --- | --- | --- |
| `source_val` | 15% of the FST I–III pool | Early stopping / checkpoint choice for every model trained on source labels only: E1, E3, E4, E6, E7 |
| `target_val` | 20% of the FST IV–VI pool | Early stopping for models trained on target labels (E5 fine-tuning); picking between finished methods (CORAL λ, E3 combination, what goes into E8) |
| test splits | — | Never used to choose anything |

- Source pool (FST I–III) splits 70 / 15 / 15 train / val / test. Target pool (FST IV–VI) splits 60 / 20 / 20, as in the plan.
- **CORAL λ:** chosen on `target_val`, which uses target labels for selection only. So that the "zero labels" claim stays honest, E7 also reports the untuned λ from the Deep CORAL paper's balancing rule (λ ≈ 1000 here, see D13) as the strictly label-free number.
- E7 early-stops on `source_val`, so its checkpoint choice never sees a target label.

## D2. Actinic keratosis is not "benign"

AK is precancerous, and SCC vs AK is a real clinical confusion. Putting it in the benign class would blur the missed-cancer metric.

- AK images (PAD-UFES-20 `ACK`, Fitzpatrick17k actinic keratosis) are never trained on.
- They keep `label_4class = ak` and go to the `ak_side` split. Every model is scored on it, and we report the % of AK images it calls malignant (and which cancer) as a side analysis.
- Related: Bowen's disease (SCC in situ) counts as SCC; PAD-UFES-20 already merges it. Keratoacanthoma maps to SCC but is flagged for review in the label map. Porokeratosis is not AK despite the "actinic" in some of its names, and is excluded.

## D3. FST buckets: I–III / IV / V–VI

The plan's I–II / III–IV / V–VI buckets put training-domain FST III next to target-domain FST IV.

- Fitzpatrick17k, PAD-UFES-20 and DermaCon-IN are reported in **I–III (source) / IV / V–VI**.
- DDI only provides 3 bins (12 / 34 / 56), so it is reported in its native **I–II / III–IV / V–VI**, and the headline DDI number is V–VI only. No other DDI bucket is compared directly with a Fitzpatrick17k bucket.

## D4. Unknown skin type is dropped from training

- Fitzpatrick17k `fitzpatrick_scale = -1` and PAD-UFES-20 rows with no `fitspatrick` value are excluded (`exclude_reason = unknown_fst`) and counted in the data report.
- PAD-UFES-20 rows with FST IV–VI are not added to target training. They form a separate test-only set, `pad_dark_test` (biopsy-confirmed, a different country and different cameras), reported only if it has at least 20 images.
- Fitzpatrick17k rows that the dataset's own QC marked as wrongly labelled are excluded.

## D5. Label map: rules first, then a committed human review

Diagnosis strings are mapped by ordered rules in `src/skincancer/labels.py`. `build_label_map` writes every distinct (dataset, diagnosis) pair to `data/label_map.csv` with the label it got and a `needs_review` flag. The team checks every row (all of them, not only the flagged ones), edits the file, and commits it. After that the file is the source of truth: `build_manifest` refuses to run if any diagnosis is unmapped, and re-running `build_label_map` only adds new rows.

What goes in each class:
- **bcc / scc / mel:** every subtype of each (morpheiform BCC, lentigo maligna, superficial spreading melanoma and so on).
- **benign (lookalikes only):** melanocytic nevi of all kinds (including acral, blue, halo, congenital, dysplastic), seborrheic keratosis and its darker-skin variant dermatosis papulosa nigra, dermatofibroma, solar lentigo, and pyogenic granuloma (an amelanotic melanoma / SCC lookalike; flagged).
- **exclude:** rare cancers (Kaposi sarcoma, DFSP, sebaceous carcinoma, mycosis fungoides, Merkel cell), non-melanocytic "nevi" (sebaceous, epidermal, Becker's, comedonicus, flammeus, depigmentosus, anemicus, lipomatosus; flagged), and everything Fitzpatrick17k marks non-neoplastic.

Status (14 Oct 2026): first-pass review of all 443 rows done (`review_note` column); team confirmation pending. Pyogenic granuloma is about a third of Fitzpatrick17k's benign class, so it gets an evaluation-time sensitivity check (`evaluate --drop-diagnoses`).

## D6. BCC + SCC merge rule

`count_check` recommends the 3-class view (`nmsc / mel / benign`) when `target_test` would hold **fewer than 15 SCC images**: one-vs-rest AUC on fewer positives than that is noise. The decision is made once in week 2 and recorded here. Switching is one config line (`data.label_col: label_3class`).

> **On the partial download (14 Oct 2026, development only):** SCC in target_test = 16 (≥ 15), so `label_col = label_4class`. It was a near miss, so the rule is re-applied on the full set (D21).

**Partial download used during development (superseded by D21):** Fitzpatrick17k 1,919 of 4,491 neoplastic images (Atlas 1,083 + Wayback 836, all MD5-verified), PAD-UFES-20 complete, DDI complete, DermaCon-IN complete. Splits: source 1255 / 271 / 271, target 152 / 51 / 51, DDI 656, DermaCon-IN 65 + 5,379 non-cancer, pad_dark_test 60, ak_side 246. Twelve DermaCon-IN patients appear in both `dermacon_test` and `dermacon_other`. Both are external test sets, so nothing leaks, but don't pool the two sets.

## D7. How class priors work (E4)

- The Indian case series only describe the mix *among cancers*. So the Indian prior **keeps the benign share at its training value** and splits the malignant share 45 : 40 : 15 (BCC : SCC : MEL), or by one of the published series in the sensitivity sweep (Malwa, central India, Manipur; normalised).
- Prior-shift correction divides by the **effective** training prior, not the raw class counts. A model trained with class-balanced loss behaves as if the prior were uniform, so dividing by the raw counts would correct twice. Each run saves its effective prior in `train_prior.json`.
- The weighted-loss variant runs on the consensus mix only (3 seeds). The full sensitivity sweep uses prior-shift correction, which needs no retraining.

## D8. Statistics

- Bootstrap: 1,000 resamples **by group** (`group_id` = patient where known, otherwise near-duplicate cluster), percentile 95% CI.
- Seeds: every resample is scored on all 3 seeds with the same indices and averaged. So the CI covers test-set sampling of the seed-mean, and `std` reports seed-to-seed spread.
- Macro AUC skips any class with no positives in a subgroup and records which classes went in. A subgroup with fewer than 2 scorable classes reports NaN.

## D10. Near-duplicate thresholds (calibrated on real pairs, 14 Oct 2026)

Calibration used ~4,000 real images (PAD-UFES-20, DDI, the Fitzpatrick17k images available then), with sampled pairs checked by eye at each distance.
- **pHash Hamming ≤ 2.** d = 0–2 were always true duplicates. At d = 4–6, unrelated "small brown papule on plain skin" images collide, including a PAD-UFES-20 nevus and a DDI nevus.
- **Embedding cosine ≥ 0.90** (ImageNet EfficientNet-B0). Every pair at or above 0.90 was the same lesion: re-crops, watermarked copies, or a second photo of the same lesion. All of them were within one dataset. Below ~0.89, pairs are merely similar-looking skin.
- A group is the union of: same patient, pHash ≤ 2, and cosine ≥ 0.90.

## D11. Stage-2 learning rate: 3e-5, not 1e-5

Chosen on `source_val` only (pilot split, EfficientNet-B0, seed 0, 14 Oct 2026). Best source_val macro AUC: 1e-5 → 0.858 (epoch 30 of 35, still rising at the cap); 3e-5 → 0.866 (epoch 17); 1e-4 → 0.867 (epoch 13). The AUC differences are within noise. The deciding factor is that 1e-5 doesn't converge within the epoch budget. 3e-5 was preferred to 1e-4 because it is gentler on the small E5 fine-tuning sets. The head learning rate stays at 1e-4. Applies to every config (base, ResNet50, E5).

## D12. Lesion crop: tuned on training images only; failure rate judged by eye

- **First version:** latched onto white rulers, dark clothing or background, and hair in ~10% of crops.
- **Fixes:**
  - mask bright, colourless pixels (rulers, paper) anywhere;
  - mask very dark pixels only where they touch the image border (background, clothing), so a dark central melanoma is never masked;
  - a 9 px opening to remove thin hairs;
  - minimum lesion area 0.2% of the image instead of 1%.
- **Tuning data:** contact sheets sample `source_train` / `target_train` only. Looking at test images to tune would be test-set peeking, and DDI's terms forbid showing its images.
- **After the fixes,** ~5 of 60 training crops (~8%) are off target, mostly wide face shots where the lesion is tiny. The automatic fallback (`crop_ok`) almost never triggers, so it under-reports failures. E3 therefore reports the **by-eye failure rate on 100 training crops**, not `crop_ok`.

## D13. CORAL λ grid: {100, 1000, 10000}, not {0.1, 1, 10}

With 1280-d EfficientNet-B0 features and the paper's 1/(4d²) normalisation, the CORAL loss in the pilot sat at ~3×10⁻⁴, against a cross-entropy of 0.4–1.2. At λ = 1 the alignment term is ~0.1% of the loss, so the plan's grid would have tested a method that does nothing, three times over. Sun & Saenko (2016) choose λ so the two losses are about equal at the end of training, which gives λ ≈ 1000 here (0.37 / 3.5×10⁻⁴). The grid is that value ×0.1, ×1 and ×10. λ = 1000 is the untuned, label-free setting: the rule uses only source labels and unlabelled target images.

## D9. Small protocol details

- **E5 fractions are nested.** For each seed, 10% ⊂ 25% ⊂ 50% ⊂ 100% of `target_train`, sampled by group and stratified by class. The 0% point is E1 itself.
- **E6 needs a fair control.** Besides the plan's fusion MLP and texture-only LR, an embedding-only MLP with the same head is trained, so a gain from fusion can't just be an MLP-vs-linear-head gain. Texture is computed after colour constancy, CLAHE and lesion crop. The plan's specification gives **78 features**, not ~130: LBP 2×10, GLCM 4 props × 4 angles, Gabor 12 filters × 2 stats, colour 6 channels × 3 stats.
- **Lookalike confusion.** The model has a single benign class, so the analysis reads both ways: which *benign diagnoses* get called which cancer, and how many cancers of each type get called benign. It is reported per FST bucket, together with the highest-confidence error examples.
- **Pilot disclosure (14 Oct 2026).** Pilot runs on partial data (`runs/pilot`, `data/pilot`) scored `source_test` and `target_test` to validate the pipeline. Their numbers were seen, but no decision was based on them: D11 used source_val, D12 training images, D13 loss magnitudes. No pilot model was ever scored on DDI, DermaCon-IN or pad_dark_test. From then on, method choices use validation splits only.
- **Test-only sets (DDI, DermaCon-IN, pad_dark_test)** are scored only through `predict --final`, which appends to `runs/final_ledger.csv` and refuses to score the same run on the same set twice.
- **DDI binary scope.** All DDI images are used, with DDI's own `malignant` flag. Every DDI lesion was biopsied, so every benign one is by definition a lookalike. DDI's terms forbid redistributing images, so DDI images never appear in figures; only metrics are reported.
- **DermaCon-IN** (19 cancers, 46 benign lookalikes): `dermacon_test` holds the images mapped to the 4 classes and gives a binary AUC (wide CIs) plus the lookalike false-alarm rate. `dermacon_other` holds the ~5,300 non-cancer clinic images (inflammatory, infectious, pigmentary and so on). For those, only the share called malignant is reported: the false-alarm rate on an everyday Indian dermatology caseload. Images DermaCon-IN marks as insufficient quality are excluded.
- **Acral.** PAD-UFES-20 `HAND` / `FOOT` is flagged `is_acral`, but this is approximate: palm and dorsum are not told apart. Fitzpatrick17k and DDI are flagged when the diagnosis name says acral (DDI has 7 acral lentiginous melanomas). Fewer than 10 acral melanomas means counts only.
- **Fitzpatrick17k images.** DermaAmin's links are dead. The Wayback Machine holds 837 of the 3,408 DermaAmin neoplastic images, and every recovered file is MD5-checked against the dataset's `md5hash`, so it is byte-identical to the original. The authors then sent the full image set (D21), which every result uses.

## D14. E8 drops image processing (14 Oct 2026)

Picked autonomously per the standing instruction (pick the recommended option mid-run). E3 on target_val macro AUC: E1 0.891, cc 0.881, clahe 0.889, crop 0.883, all 0.820. No variant beats the baseline, so E8 is built from E4/E5/E6/E7 components only, again chosen on target_val. The external-test differences (DDI V–VI 0.50 → 0.53) were not used for this choice.

## D15. Stronger-baseline arms (14 Oct 2026)

Added at the team's request after E3, each pre-specified as one arm, compared with E1 on source_val / target_val only and scored once on the test-only sets (ledger):
- **Seed ensemble** (`evaluate --ensemble`, results/<exp>+ens): mean of the three seeds' probabilities. CIs come from the group bootstrap over images only (one prediction). Computed from the already-saved predictions; no test set is predicted again.
- **E1-strong** (`configs/e1_effb0_320.yaml`): the E1 recipe at 320 px (resize 366, same 0.875 crop ratio) with flip TTA (mean softmax over identity, horizontal, vertical and both flips) for saved predictions. Early stopping stays without TTA. Size and TTA are one arm, not ablated on test sets.
- **E1-FM** (`skincancer.foundation`): frozen Google Derm Foundation embeddings (6144-d, 448 px short-side resize + centre crop) with (a) the E6 MLP head, seeds 0–2, directly comparable with e6_embed_mlp; and (b) a class-balanced linear probe with C ∈ {0.001, 0.01, 0.1, 1} picked on source_val macro AUC. Embeddings are computed in a separate TensorFlow venv (.venv-tf) and kept local (gitignored), since DDI embeddings are derivatives.
- HAM10000/ISIC are not added: they are dermoscopy, a different modality from every test set. Derm7pt *clinical* images (CC BY-NC-ND) are added as training-only extra data (D16).
- New arms are also scored on midas_test from the start (FINAL in scripts/run_extras.sh).

## D17. Missed-cancer and false-alarm rates at a fixed operating point (14 Oct 2026)

Argmax rates move with calibration and class weighting (the E1 seed ensemble has the same DDI V–VI AUC as the seed mean but misses 58% instead of 51%), so they don't compare methods fairly. Every experiment now also reports `missed_cancer_rate_op90`, `false_alarm_rate_op90` and, for the no-cancer sets, `pred_malignant_rate_op90`. Per seed, the threshold on p(malignant) is the one that catches 90% of cancers in that seed's source_val predictions, after the same prior shift or ensembling as the set being scored. It is then applied unchanged to every set. A 90% sensitivity target is a common triage operating point; it is fixed here and not tuned. The argmax rates stay in the tables. This is an evaluation-only addition, made under the standing instruction to improve in scope.

## D16. Derm7pt as training-only extra data (14 Oct 2026)

Derm7pt (Kawahara et al., IEEE JBHI 2019; CC BY-NC-ND 4.0; academic use, no redistribution) has one clinical (non-dermoscopic) photo for each of 1,011 cases from an Italian dermoscopy atlas. Only those clinical photos are used. It fills the two thinnest training classes: the partial-download source_train had 128 melanomas and 143 benign. Rules:
- **Outside the frozen manifest.** Rows live in `data/extra/derm7pt.csv` (gitignored; rebuilt by `python -m skincancer.data.extra derm7pt`) with split `source_train_extra`, and `configs/e1_effb0_d7.yaml` opts in with `data.extra_train`. data/manifest.csv, groups.csv and every validation and test split are unchanged, so `e1_effb0_d7` compares directly with `e1_effb0`. Early stopping stays on source_val.
- **Skin type.** Derm7pt has no Fitzpatrick labels. The Italian clinic population is taken as FST I–III (`fst_bucket` = "I-III (assumed)"). This is a deliberate exception to D4, which drops unknown FST, and it is allowed because these images are never evaluated.
- **Labels.** The existing rules apply. "melanoma metastasis" (4) is excluded as not a primary skin lesion; vascular lesion (29), melanosis (16) and miscellaneous (8) map to exclude by default.
- **Leakage.** Any Derm7pt image within pHash ≤ 2 or embedding cosine ≥ 0.90 (D10) of any evaluation image (every split except source_train / target_train) is dropped.
- **No images in figures or the repo** (licence: no redistribution, no derivatives).

## D18. E5 re-run with equal optimizer steps per fraction (14 Oct 2026)

**Bug in the first E5 run.** Batch size 32 with `drop_last=True` gave the 10% subsets (17–18 images) zero gradient steps, and the 25% / 50% subsets 1–2 steps per epoch. Their best checkpoint was therefore the E1 model, and those points of the curve measure nothing. Found from the training logs (loss 0.0, n_train 18), not from any test result.

**Fix.** `train.steps_per_epoch` draws a fixed number of batches per epoch, cycling through the subset with fresh augmentation; a set smaller than one batch becomes one batch. E5 uses 5 steps per epoch for every fraction (≈ one pass over the full 152-image target_train), so the curve compares the amount of target data at equal training compute, the standard learning-curve setup. Everything else (init from E1, one stage, LR, early stopping on target_val) is unchanged.

**Naming and test sets.** The corrected runs are new experiments, `e5s_f10…f100` (`configs/e5_finetune_steps.yaml`), each scored once on the test sets. The old `e5_f*` results stay in results/ and runs/final_ledger.csv for the record, marked as superseded, and are not used for E8 or reported as E5.

## D19. E8 composition, chosen on target val (14 Oct 2026)

Picked autonomously per the standing instruction, using target_val macro AUC only (seed mean). E1 scored 0.891. The candidates:

| | E3 best (CLAHE) | E4 Indian loss | E5s 100% | E6 fusion | E7 CORAL best (λ = 1000) |
| --- | --- | --- | --- | --- | --- |
| Target val macro AUC | 0.889 | 0.895 | **0.933** | 0.879 | 0.883 |

Components that beat E1 are E5 fine-tuning (clearly) and E4 Indian-prior weighting (marginally, +0.004). E8 = the E5 recipe (equal steps, 100% target_train, init from the source-only model of the same backbone and seed) + Indian-prior weighted loss. `configs/e8_combined.yaml` (EfficientNet-B0) and `configs/e8_r50.yaml` (ResNet50, layer4 unfrozen as in e1_r50). Each is scored once on the test sets. Preprocessing (D14), fusion and CORAL are left out because none of them beat E1 on target val. Their external-set trends (E7 on DDI V–VI) were not used for this choice.

## D20. Derm Foundation + target data arm (15 Oct 2026)

Added under the standing instruction to improve in scope: `e5_fm_lr` asks E5's question on the Derm Foundation backbone. It is the linear probe trained on source_train + target_train, with C ∈ {1e-4, 1e-3, 1e-2, 1e-1} picked on target_val, since it is target-trained (D1). The grid extends one step below e1_fm_lr's, because that probe chose its grid edge. It is scored once on the test sets like every other arm. It needs only minutes, since the embeddings are cached.

## D21. One dataset: the full Fitzpatrick17k image set (16 Oct 2026)

The authors sent the complete image set: 16,577 JPEGs, one per CSV row, each named by its MD5. 40 of 40 sampled files hash to their names, and 300 of 300 of our earlier downloads are byte-identical to the authors' copies. So the full set strictly adds images.

- **One version only.** Every reported result comes from the full set. The runs on the partial download (1,919 of 4,491 neoplastic images, 14–15 Oct) were development runs. They shaped the protocol fixes logged in D13–D20, which are kept, but their results are retired and not reported. They remain in git history under the tag `data-v1`. Numbers quoted inside D6–D20 are from those development runs and only explain why each rule was made.
- **Rebuilt from scratch, same rules.** Label map (unchanged: all 114 Fitzpatrick17k diagnoses were already reviewed), manifest, dedup (D10 thresholds), splits (same seed and fractions) and count check (the D6 rule re-applied). The test-set ledger starts empty, so every run is scored once. Derm7pt's training-only list is rebuilt against the new evaluation splits (D16).
- **The protocol fixes are built in from the start.** These are E5 with equal steps (D18; `steps_per_epoch: auto` = one pass over the full target_train), MIDAS in every block's test sets, the operating-point metrics (D17) and the memory fixes.
- **Choices are re-made on the new validation sets.** E3's preprocessing (D14), CORAL's λ (D13) and E8's composition (D19) are re-chosen on target_val by the same rules. The development run's choices are not carried over.

## D22. E8 keeps no image processing, re-chosen on the full set (16 Oct 2026)

D14's rule re-applied (picked autonomously). E3 on target_val macro AUC, 3 seeds each: E1 0.916, colour constancy 0.917, CLAHE 0.896, crop 0.897, all three 0.895. Colour constancy's +0.001 is far inside the CI (0.87–0.96) and below its own seed-to-seed spread, so it does not count as beating the baseline; the simpler pipeline is kept. E8 is again built from E4/E5/E6/E7 components only. External-test results (e.g. CLAHE's DDI V–VI 0.55) were not used.
