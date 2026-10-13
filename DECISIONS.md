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
- **CORAL λ:** chosen on `target_val`, which uses target labels for selection only. So that the "zero labels" claim stays honest, E7 also reports λ = 1 (the Deep CORAL paper's default) untuned, as the strictly label-free number.
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

> Week-2 result: _fill in_ (SCC count in target_test = __; label_col = __)

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

## D9. Small protocol details

- **E5 fractions are nested.** For each seed, 10% ⊂ 25% ⊂ 50% ⊂ 100% of `target_train`, sampled by group and stratified by class. The 0% point is E1 itself.
- **E6 needs a fair control.** Besides the plan's fusion MLP and texture-only LR, an embedding-only MLP with the same head is trained, so a gain from fusion can't just be an MLP-vs-linear-head gain. Texture is computed after colour constancy, CLAHE and lesion crop. The plan's specification gives **78 features**, not ~130: LBP 2×10, GLCM 4 props × 4 angles, Gabor 12 filters × 2 stats, colour 6 channels × 3 stats.
- **Lookalike confusion.** The model has a single benign class, so the analysis reads both ways: which *benign diagnoses* get called which cancer, and how many cancers of each type get called benign. It is reported per FST bucket, together with the highest-confidence error examples.
- **Test-only sets (DDI, DermaCon-IN, pad_dark_test)** are scored only through `predict --final`, which appends to `runs/final_ledger.csv` and refuses to score the same run on the same set twice.
- **DDI binary scope.** All DDI images are used, with DDI's own `malignant` flag. Every DDI lesion was biopsied, so every benign one is by definition a lookalike. DDI's terms forbid redistributing images, so DDI images never appear in figures; only metrics are reported.
- **DermaCon-IN** (19 cancers, 46 benign lookalikes): `dermacon_test` holds the images mapped to the 4 classes and gives a binary AUC (wide CIs) plus the lookalike false-alarm rate. `dermacon_other` holds the ~5,300 non-cancer clinic images (inflammatory, infectious, pigmentary and so on). For those, only the share called malignant is reported: the false-alarm rate on an everyday Indian dermatology caseload. Images DermaCon-IN marks as insufficient quality are excluded.
- **Acral.** PAD-UFES-20 `HAND` / `FOOT` is flagged `is_acral`, but this is approximate: palm and dorsum are not told apart. Fitzpatrick17k and DDI are flagged when the diagnosis name says acral (DDI has 7 acral lentiginous melanomas). Fewer than 10 acral melanomas means counts only.
- **Fitzpatrick17k images.** DermaAmin's links are dead. The Wayback Machine holds 837 of the 3,408 DermaAmin neoplastic images, and every recovered file is MD5-checked against the dataset's `md5hash`, so it is byte-identical to the original. The rest wait on the authors' full-image link (form submitted 14 Oct 2026). Until then, results are **data v1** and are re-run once the full set arrives.
