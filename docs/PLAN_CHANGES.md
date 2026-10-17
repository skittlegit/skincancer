# Changes to make in the semester plan document

The plan (10 Oct 2026) is out of date in these places. The reasons are in [DECISIONS.md](../DECISIONS.md).

| Plan section | Plan says | Change to |
| --- | --- | --- |
| Methods → Baseline | "early stopping on validation macro AUC" | Early stopping on **source_val** (15% of FST I–III) for source-trained models; target_val only for E5 and for choosing between methods (D1) |
| Data plan → Rules | Target split 60/20/20 only | Add: source pool split 70/15/15 train/val/test (D1) |
| Metrics → Primary | FST buckets I–II, III–IV, V–VI | **I–III / IV / V–VI**; DDI in its native bins, V–VI as the headline (D3) |
| Epidemiology → Benign lookalikes | Benign class includes "other pigmented benign lesions" | Benign = nevi, SK, dermatofibroma, solar lentigo, pyogenic granuloma; **actinic keratosis kept out** and scored separately (D2, D5) |
| Methods → Texture | "About 130 features" | **78 features**, as specified (D9) |
| Methods → Texture | Fusion vs texture-only | Add the embedding-only MLP control (D9) |
| Methods → E4 | Prior-shift formula | Divide by the **effective** training prior (uniform under class-balanced loss); the benign share stays fixed (D7) |
| Methods → E7 | "sweep λ ∈ {0.1, 1, 10} on validation" | **λ ∈ {100, 1000, 10000}**: at λ ≤ 10 the CORAL term is ~0.1% of the loss (D13). λ chosen on target_val; λ = 1000 (the paper's balancing rule) also reported untuned as the label-free number (D1) |
| Methods → Baseline | Stage-2 LR ~1e-5 | **3e-5**: 1e-5 doesn't converge in 30 epochs (D11) |
| Methods → Image processing | Lesion crop by colour distance | Rulers and edge-connected dark background masked; failure rate by eye on 100 training crops (D12) |
| Data plan → dedup | pHash + embedding similarity | pHash ≤ 2 OR cosine ≥ 0.90, both calibrated by eye (D10) |
| Data plan → Fitzpatrick17k | "download from URL column" | DermaAmin links (≈3,400 of 4,491 neoplastic images) are dead. The full set (16,577 images) came from the authors via their request form (github.com/mattgroh/fitzpatrick17k), and every reported result uses it. Runs on the partial download are development runs only (D21) |
| Data plan → Test sets | DDI, DermaCon-IN, PAD dark skin | Add **MRA-MIDAS** (Stanford, 1,380 biopsy-proven photos, mostly FST I–III) as a test-only set: it isolates the hospital shift from the skin-tone shift (D21) |
| Data plan → Extra data | — | **Derm7pt** clinical photos (954 after dedup against every evaluation split) as training-only extra data in one arm, outside the manifest (D16) |
| Metrics → Secondary | Missed-cancer and false-alarm rates | Also at a **fixed operating point**: the cut-off that catches 90% of cancers on each seed's source val, applied unchanged to every set (D17) |
| Evaluation → Test sets | Score at the end | Test-only sets are scored **once per method**, recorded in `runs/final_ledger.csv`; the scoring command refuses a second look (D9) |
| Methods → E5 | Fine-tune on 10/25/50/100% | Give every fraction the **same number of optimizer steps** (one pass of the full target_train per epoch); small fractions otherwise got 0–2 steps (D18) |
| Methods → E8 | Combine the best E3–E7 components | Components that beat E1 on target val only. On the full data that is E5 fine-tuning alone, so E8 = the E5 recipe; on EfficientNet-B0 it is the E5 100% run, on ResNet50 it is trained (D19, D22, D23) |
| Methods → Baselines | EfficientNet-B0 and ResNet50 | Add stronger-baseline arms, each scored once: 320 px + flip TTA, + Derm7pt, Derm Foundation embeddings with linear / MLP heads (+ a target-trained probe), and seed ensembles (D15, D16, D20) |
| Deliverables / team table | Per-person ownership | No owners; the README work queue is shared |
