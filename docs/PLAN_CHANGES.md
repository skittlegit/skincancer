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
| Methods → E7 | "sweep λ on validation" | λ chosen on target_val; λ = 1 also reported untuned as the label-free number (D1) |
| Data plan → Fitzpatrick17k | "download from URL column" | DermaAmin links (≈3,400 of 4,491 neoplastic images) are dead. Request the full set with the authors' form (github.com/mattgroh/fitzpatrick17k). Atlas Dermatologico links still work. |
| Deliverables / team table | Per-person ownership | No owners; the README work queue is shared |
