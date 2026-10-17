# Label map review (17 Oct 2026)

Second review of all 460 rows of [data/label_map.csv](../data/label_map.csv) against the D2 / D5 rules, [src/skincancer/labels.py](../src/skincancer/labels.py) and the manifest. Done by Claude for the team; **the team signs off** (README work queue, item 2).

## Verdict

**Fit to sign off as is.** No row changes the training or evaluation data in a way that affects any reported number. The map covers 5 datasets: Fitzpatrick17k 114 rows, DDI 78, DermaCon-IN 245, MIDAS 17, PAD-UFES-20 6. Every (dataset, diagnosis) pair in the manifest has a row, and every manifest label matches the map.

- **Cancer classes are clean.** Every BCC, SCC and melanoma subtype is in its class: morpheiform and solid-cystic BCC, superficial spreading melanoma, lentigo maligna, melanoma in situ, acral lentiginous melanoma, and SCC in situ / Bowen's.
- **AK is handled correctly** in all four datasets. Porokeratosis (321 images) and "actinic dermatitis" (31) are rightly kept out of AK.
- **No inflammatory condition is labelled benign.**

The map stays as committed. It was frozen before any experiment ran (D5), and none of the items below affects a score.

## Definite error (1 image, no effect on any score)

| Dataset | Diagnosis | Images | Now | Should be | Why |
| --- | --- | --- | --- | --- | --- |
| DDI | acral-melanotic-macule | 1 (ddi_test) | exclude | benign | A benign acral pigmented macule: the classic lookalike of acral melanoma, like the solar lentigo D5 counts as benign |

DDI is scored on its own biopsy malignant flag, so this label does not enter any DDI number.

## Bookkeeping only (labels unchanged)

- **DDI `squamous-cell-carcinoma-keratoacanthoma` (8):** `needs_review` is False, whereas D2 flags keratoacanthoma (the DermaCon-IN keratoacanthoma row is flagged). The general SCC rule matches first.
- **The `exclude_rare_cancer` tag is applied unevenly.** These rows are plain `exclude`: DDI metastatic carcinoma (5), leukemia cutis (1), blastic plasmacytoid dendritic cell neoplasm (1); Fitzpatrick17k Langerhans cell histiocytosis (60); MIDAS "malignant- other" (10). It matters only if rare-cancer counts are reported.
- **Spitz wording.** `labels.py` treats Spitz as intermediate (exclude), but "spitz" also appears in the benign nevus rule. MIDAS Spitz stays excluded only because an earlier rule matches first, and DDI `atypical-spindle-cell-nevus-of-reed` (1) is benign.
- **DermaCon-IN "Kyrle's Disease"** has an encoding-garbled string. It still matches the manifest and is excluded.

## For the team to decide (current choice kept unless you change it)

| Rows | Images | Current | Question |
| --- | --- | --- | --- |
| Pyogenic granuloma, in four spellings (Fitzpatrick17k `pyogenic granuloma` 113 + `granuloma pyogenic` 75, DDI 14, DermaCon-IN 4) | ~206 (180 Fitzpatrick17k in splits) | benign | A known D5 call: a vascular lesion, not a typical melanoma lookalike. A sensitivity check with `evaluate --drop-diagnoses` must list all four spellings: `pyogenic granuloma`, `granuloma pyogenic`, `pyogenic-granuloma`, `Granuloma pyogenic` |
| MIDAS "melanocytic lesion, possible re-excision (severe, Spitz, AIMP)" + "melanocytic tumor, …" | 76 | exclude | Severely atypical nevi would be benign under D5. Excluding them is conservative but removes the hardest benign-vs-melanoma cases from midas_test. Recommendation: keep excluded |
| DDI `benign-keratosis` (3), `lichenoid-keratosis` (1) | 4 | exclude | Arguably benign: ISIC's "benign keratosis" covers seborrhoeic and lichen-planus-like keratosis, both real lookalikes. Does not affect DDI scoring |
| Fitzpatrick17k `halo nevus` (82), `congenital nevus` (68) | 147 in splits | benign | Correct per D5, but they dominate Fitzpatrick17k's nevi, and giant congenital and halo nevi look little like melanoma |
| DermaCon-IN `Dermatosis Papulosa Nigra` | 17 (dermacon_test) | benign | Correct per D5, but it is a large share of DermaCon-IN's 46 benign test images. Worth a sentence in the report |
| DermaCon-IN `Bowen's Metastasis` | 1 | scc | Clinically odd name (Bowen's does not metastasise). SCC either way; worth checking the source |
