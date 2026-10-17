# Epidemiology-Guided Image Processing and Texture-Aware Feature Learning for Domain-Adaptive Skin Cancer Classification

Oct 10, 2026 · revised 17 Oct 2026 with the protocol changes in [PLAN_CHANGES.md](PLAN_CHANGES.md) (reasons in [DECISIONS.md](../DECISIONS.md)) · @Deepak

## Proposal at a glance

This plan builds on the proposal signed on 27 Aug 2026 by Dr. Anjli Varghese (supervisor) and Venkatasubramanian K (co-supervisor). The title, the three pillars and the India focus stay exactly as submitted; everything below turns them into experiments that fit one semester.

| Member | HT No | Branch |
| --- | --- | --- |
| Ankita Vemavarapu | SE23UECM007 | ECM |
| Aeleni Deepak Reddy | SE23UARI005 | AI |
| Ranganaini Sai Ruchir | SE23UECM043 | ECM |

> **Abstract as submitted:** This project explores texture-aware domain adaptation for skin cancer classification by tailoring models to population-relevant dermatological data and clinically significant cancer types in India. It combines localized training with texture-based features to improve generalization and reduce confusion between malignant lesions and visually similar benign skin conditions.

**How each phrase of the proposal becomes something we build and measure:**

| Proposal phrase | What we build | Experiments |
| --- | --- | --- |
| Epidemiology-guided | Classes, class priors and subgroup checks taken from Indian case series, not from whatever the datasets contain | E4, acral check |
| Image processing | Colour constancy, contrast enhancement on lightness, lesion-centred crop — chosen for low lesion-to-skin contrast on darker skin | E3 |
| Texture-aware feature learning | LBP, GLCM and Gabor texture + colour statistics fused with CNN embeddings | E6 |
| Domain adaptive, localized training | Fine-tuning on darker-skin images in increasing amounts; Deep CORAL with unlabelled target images | E5, E7 |
| Reduce malignant vs lookalike confusion | A benign-lookalike class, and a confusion rate reported for every model | All |

Why it matters: in a review of public skin cancer image datasets, Fitzpatrick skin type was recorded for only about 2% of images ([Wen et al., 2022](https://pubmed.ncbi.nlm.nih.gov/34772649/)). On Stanford's biopsy-proven DDI set, a HAM10000-trained model fell from 0.72 ROC-AUC on FST I–II to 0.57 on FST V–VI ([Daneshjou et al., 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9374341)). The datasets also don't match Indian practice: Indian case series are dominated by BCC and SCC, while melanoma — the focus of most public data — is a minority and mostly acral.

The deliverable is a credible answer to how big the gap is and what closes it, not a headline accuracy. The system is a research prototype, never a diagnostic tool.

## Research questions

Four questions, one per pillar of the proposal plus the gap they all respond to. Each maps to one experiment block and one figure.

| # | Question | Hypothesis | Answered by |
| --- | --- | --- | --- |
| RQ1 | How much does performance drop when a model trained on FST I–III images is tested on FST IV–VI and Indian images — and which malignant classes get mistaken for which benign lookalikes? | The drop is outside the bootstrap CI of in-domain performance, and is driven mostly by malignant lesions called benign. | E1 vs E2; gap chart + confusion matrices |
| RQ2 | Do epidemiology-guided choices — image processing aimed at darker skin and class priors set to the Indian case mix — improve target performance? | Contrast enhancement and lesion cropping help FST V–VI more than FST I–II; Indian priors raise BCC/SCC recall at little cost elsewhere. | E3, E4 |
| RQ3 | Do texture features fused with CNN embeddings reduce malignant-vs-lookalike confusion and the cross-domain drop? | Fusion lowers the lookalike confusion rate on darker skin; texture-only models are weaker overall but drop less across domains. | E6; ablation table |
| RQ4 | How much localized (darker-skin) labelled data closes the gap, and how much does Deep CORAL close with unlabelled images alone? | \~25% of the target training set recovers most of the gap; CORAL recovers part of it with zero labels. | E5, E7; data-size curve |

On novelty: the DDI authors already showed that fine-tuning on diverse images closes the gap. What's new here is driving the class design and preprocessing from Indian epidemiology, measuring lookalike confusion as a first-class outcome, comparing texture fusion with localized training and CORAL under one protocol, and testing against Indian images.

## Epidemiology: what drives the design

Indian skin cancer is dominated by BCC and SCC, and Indian melanoma is mostly acral — while public datasets are melanoma-heavy. So the class set, class priors and subgroup checks follow Indian hospital series, not the datasets.

| Indian hospital series | BCC | SCC | Melanoma | Note |
| --- | --- | --- | --- | --- |
| [Malwa, Punjab](https://pmc.ncbi.nlm.nih.gov/articles/PMC4963704) | 55% | 37% | 8% | 84 cases |
| [Central India (Supekar et al., IJD 2021)](https://doaj.org/article/58d805c09fda4ebe9764938c00040479) | 41% | 30% | 9% | Acral lentiginous was the most frequent melanoma subtype |
| [Manipur (RIMS, 2004–09)](https://search.bvsalud.org/gim/resource/en/emr-144895) | 33% | 44% | 14% | 92 cases |

Skin cancer is only about 1–2% of cancers diagnosed in India, and several Indian studies report SCC, not BCC, as the most common. By contrast, roughly half or more of HAM10000's malignant images are melanoma (approximate, from its published class counts).

**What this changes in the project**

1. **Classes:** BCC, cutaneous SCC, melanoma and a benign-lookalike class. SCC gets equal weight with BCC even though public datasets under-supply it. Rare cancers (Kaposi sarcoma, sebaceous carcinoma, DFSP) stay out.
2. **Class priors (E4):** set to the Indian case mix, roughly 45 : 40 : 15 for BCC : SCC : melanoma, with a sensitivity sweep because the series disagree.
3. **Acral check:** melanoma recall on palm, sole and nail images wherever datasets record body site. Acral nevi go in the benign class as the matching lookalike.
4. **Image processing (E3):** darker skin lowers lesion-to-skin contrast, which motivates contrast enhancement on lightness only and colour constancy that doesn't wash out pigment.
5. **Benign lookalikes:** melanocytic and acral nevi, seborrheic keratosis, dermatofibroma, solar lentigo and pyogenic granuloma — what a malignant lesion is actually confused with in clinic (D5). Actinic keratosis is a pre-cancer, so it is kept out of both classes and scored separately (D2).

## Scope: what fits in one semester

The proposal is the fixed base: its title, three pillars, India focus and the malignant-vs-lookalike goal. This table is what was trimmed from the longer spec so the proposal can be delivered in one semester.

| Area | Longer spec | This plan | Why |
| --- | --- | --- | --- |
| Epidemiology | Full epidemiology study | Literature-based: Indian hospital series set classes, priors and subgroups | Answers the design questions without new data collection. |
| Source data | HAM10000 / ISIC dermoscopy | Light-skin clinical photos (Fitzpatrick17k FST I–III, PAD-UFES-20) | Dermoscopy → phone photo is a second shift; fixing modality isolates skin tone. |
| Localized (target) data | Indian hospital collection | Public FST IV–VI images + DermaCon-IN as an Indian probe | Hospital data needs ethics approval that won't clear in one semester. |
| Task | Many classes + acral subtype as a class | BCC / SCC / melanoma / benign lookalike; binary view for small test sets; acral as a subgroup check | Keeps the proposal's cancer types without starving any class. |
| Image processing | Hair removal, CLAHE, colour norm, ROI detection | Colour constancy, CLAHE on lightness, lesion-centred crop | The three steps that bear on darker skin; hair removal is a dermoscopy problem. |
| Texture | GLCM, LBP, Gabor, wavelets, shape | LBP, GLCM, Gabor + colour statistics | Gabor stays for ridge and directional patterns; shape needs masks we don't have. |
| Backbones | 3 required, up to 8 | EfficientNet-B0 for all ablations; ResNet50 only for E1, E2, E8 | Halves compute; still shows the gap isn't architecture-specific. |
| Adaptation | Deep CORAL + DANN | Localized fine-tuning + Deep CORAL | DANN moves to stretch. |
| Explainability | Grad-CAM + IoU against masks | Grad-CAM, lookalike confusion analysis, domain probe, UMAP | No masks for clinical photos. |
| Statistics | Bootstrap, McNemar, DeLong | Bootstrap 95% CIs, 3 seeds | Enough to tell real differences from noise. |
| Edge deployment | Future scope | Cut | Distracts from the research question. |

## Data plan

All datasets are clinical photographs, so the shift we measure is skin tone and population, not camera type. Four are test-only (DDI, DermaCon-IN, MRA-MIDAS and PAD-UFES-20's FST IV–VI images) and are never used for training or tuning.

*(Data-roles diagram: see [PROJECT_EXPLAINER.pdf](PROJECT_EXPLAINER.pdf), section "The data".)*

Two validation sets with fixed jobs (D1): **source val** (15% of FST I–III) picks checkpoints for source-trained models; **target val** (20% of FST IV–VI) is used only for target-trained models (E5) and for choosing between methods. Test sets choose nothing.

| Dataset | Role | Size | Skin-tone labels | Label quality |
| --- | --- | --- | --- | --- |
| Fitzpatrick17k | Source (FST I–III) and target train/val/test (FST IV–VI), neoplastic images only | 16,577 images, 114 conditions; the full set from the authors, since many URL links are dead (D21) | FST I–VI per image | Atlas-scraped, not biopsy-verified ([SkinCon notes](https://arxiv.org/pdf/2302.00785)) |
| PAD-UFES-20 | Extra source data (BCC, SCC, melanoma + benign) | \~2.3k smartphone images, Brazil | Partial | Mostly biopsy-confirmed; patient IDs and body region (used for the acral check) |
| [DDI](https://aimi.stanford.edu/datasets/ddi-diverse-dermatology-images) | External test only | 656 images, 570 patients; FST V–VI = 159 benign + 48 malignant | Dermatologist-assigned, 3 bins | Biopsy-proven |
| [DermaCon-IN](https://proceedings.nips.cc/paper_files/paper/2025/hash/a2c548b2ed3141400d86a235eaa6fad0-Abstract-Datasets_and_Benchmarks_Track.html) | Indian probe, test only | 5,450 images, 2,993 patients, South Indian clinics | FST IV–VI included | Dermatologist-annotated, 245 diagnoses |
| MRA-MIDAS (Stanford) | External test only: a new hospital without the skin-tone change (D21) | 1,380 biopsy-proven photos | Mostly FST I–III | Biopsy-proven |
| Derm7pt (Italy) | Training-only extra data, one arm (D16) | 954 clinical photos after dedup against every evaluation split | Assumed FST I–III | Expert-labelled |

DermaCon-IN is mostly infectious and inflammatory conditions, so it may hold few cancers. Count its neoplasm images in week 1. If cancers are scarce, use its benign lesions to measure the false-positive rate on Indian skin — still a useful number.

**Rules that don't bend**

- **Patient-level splits** wherever patient IDs exist (PAD-UFES-20, DDI). Fitzpatrick17k has none and contains near-duplicates, so group images by perceptual hash (≤ 2) or embedding cosine similarity (≥ 0.90), both thresholds calibrated by eye, and split by group (D10).
- **Source split:** FST I–III pool, 70 / 15 / 15 train / val / test (D1).
- **Target split:** FST IV–VI neoplastic subset of Fitzpatrick17k, 60 / 20 / 20 train / val / test. Fine-tuning and CORAL only ever see the train part.
- **Test-only sets** (DDI, DermaCon-IN, MIDAS, PAD dark skin) are run once per finished method, never used to pick hyperparameters. The scoring command records every look in a ledger and refuses a second one (D9).
- **Unknown skin type is dropped** from training and evaluation (D4).
- **Benign class = hard negatives:** nevi, seborrheic keratosis, dermatofibroma, solar lentigo and pyogenic granuloma, not normal skin (D5). The label map (diagnosis → class) is rule-based, then reviewed and committed before any experiment.
- **One manifest CSV** for everything: `image_id, group_id, dataset, diagnosis, label_binary, label_4class, fst, site, is_acral, split`. Splits are saved files, not code run at training time.
- **Week-1 count check:** tally images per class (especially SCC) per FST bucket. If the target test holds fewer than 15 SCC images, merge BCC + SCC into one non-melanoma class and say so (D6). It holds 23, so four classes are kept.

## Methods

Six components, each small enough for one person-week. All share one training script and one evaluation script, so every result is comparable.

1. **Baseline.** EfficientNet-B0 (main) and ResNet50 from `timm`, ImageNet weights, 224 × 224 input. Two stages: train the head (LR \~1e-3), then unfreeze the top blocks (LR 3e-5; 1e-5 does not converge in 30 epochs, D11). AdamW, class-weighted cross-entropy, early stopping on **source val** macro AUC (D1). Light augmentation (flips, rotation, small crops, mild brightness/contrast) and no hue jitter, since hue changes apparent skin tone.
2. **Epidemiology-guided image processing (E3).** Three steps, each switched on and off in turn: (a) Shades-of-Gray colour constancy; (b) CLAHE on the L channel of CIELAB only, which lifts lesion-to-skin contrast without shifting colour; (c) lesion-centred crop by thresholding colour distance from the surrounding skin in LAB, with rulers and edge-connected dark background masked, falling back to a centre crop; tuned on training images only, with 100 training crops checked by eye (D12).
3. **Epidemiology-guided class priors (E4).** Two variants against the baseline: a loss weighted to the Indian case mix, and a no-retraining prior-shift correction at inference, where each class probability is rescaled by the ratio of Indian to *effective* training prevalence (uniform under the class-balanced loss) and renormalised, with the benign share held fixed (D7):

   ```latex
   p_{\mathrm{IN}}(y \mid x) \propto p_{\mathrm{train}}(y \mid x) \cdot \frac{\pi_{\mathrm{IN}}(y)}{\pi_{\mathrm{train}}(y)}
   ```
4. **Localized training curve (E5).** Start from the source model and fine-tune on 0, 10, 25, 50 and 100% of the FST IV–VI training split, 3 seeds each, with the **same number of optimizer steps** at every fraction (otherwise small fractions get almost no updates, D18). Plot target macro AUC against data fraction.
5. **Texture-aware feature learning (E6).** On the processed grayscale lesion: uniform LBP histograms (P = 8, R = 1 and 2), GLCM statistics (contrast, homogeneity, energy, correlation at 4 angles) and a small Gabor bank (4 orientations × 3 frequencies; mean and energy per filter) for ridge and directional structure. Colour: mean, std and skew per channel in HSV and CIELAB. 78 features, standardised, concatenated with the 1280-d EfficientNet embedding, then a small MLP. The same MLP on the embedding alone is the control, and a texture-only logistic regression is the floor (D9).
6. **Deep CORAL (E7).** Loss = cross-entropy on labelled source images + λ × CORAL loss between source and *unlabelled* target batch features; sweep λ ∈ {100, 1000, 10000}, chosen on target val. At λ ≤ 10 the CORAL term is about 0.1% of the loss, and λ = 1000 is the paper's own balancing rule (D13).

**Analysis tools**

- **Lookalike confusion analysis:** for each malignant class, which benign lookalikes it is confused with, per FST bucket, before and after texture fusion — with 10 example pairs in the report.
- **Grad-CAM** on the last conv block: 20 correct and 20 wrong predictions per domain. Look for attention on rulers, markings, borders or background.
- **Domain probe:** logistic regression on frozen embeddings predicting source vs target. High accuracy means the features encode the domain; it should fall after CORAL.
- **UMAP** of embeddings coloured by domain and by class: do points cluster by disease or by dataset?

## Experiments and evaluation

Eight experiments. E1–E2 are the minimum viable project; E3–E7 each test one pillar of the proposal on EfficientNet-B0; E8 combines what worked and is also run on ResNet50. Every run uses 3 seeds.

| ID | Pillar | Change from baseline | Evaluate on | Answers |
| --- | --- | --- | --- | --- |
| E1 | — | Train on source (FST I–III) | Source test | In-domain ceiling |
| E2 | — | Same model | Target test, DDI V–VI, DermaCon-IN | RQ1: the gap and where errors land |
| E3 | Image processing | Colour constancy, CLAHE-on-L, lesion crop, each on/off | Same as E2 | RQ2 |
| E4 | Epidemiology | Indian class priors (weighted loss; prior-shift correction) | Same as E2 | RQ2 |
| E5 | Localized training | Fine-tune on 0–100% of FST IV–VI train | Same as E2 | RQ4: labelled recovery |
| E6 | Texture | LBP + GLCM + Gabor + colour fusion; texture-only floor | Same as E2 | RQ3 |
| E7 | Domain adaptation | Deep CORAL with unlabelled FST IV–VI train | Same as E2 | RQ4: label-free recovery |
| E8 | All | The E3–E7 components that beat E1 on target val, combined, on both backbones (D19, D22, D23) | Same as E2 | Best achievable |
| E1+ | Stronger baselines | 320 px + flip TTA; + Derm7pt; Derm Foundation embeddings; seed ensembles (D15, D16, D20) | Same as E2 | Is the gap a weak-baseline artefact? |

**Metrics**

- **Primary:** macro one-vs-rest ROC-AUC over the four classes, per FST bucket (**I–III, IV, V–VI**; DDI in its native bins, with V–VI as the headline, D3) and per dataset. For DDI and DermaCon-IN, binary malignant-vs-benign AUC (sum of the three malignant probabilities), since their per-class counts are small.
- **Lookalike confusion rate** (the proposal's headline goal): % of malignant images predicted benign (missed cancers) and % of benign lookalikes predicted malignant, per FST bucket. Also reported at a **fixed operating point**: the cut-off that catches 90% of cancers on source val, applied unchanged to every set (D17).
- **Secondary:** per-class recall and precision, balanced accuracy, confusion matrices, melanoma recall on acral-site images where available.
- **Gap** = in-domain AUC (E1) − target AUC, reported for every method.

**Statistics:** 1,000-sample bootstrap 95% CIs on every reported AUC and confusion rate; mean ± std across 3 seeds. With only 48 malignant DDI V–VI images those CIs will be wide — report them honestly rather than over-reading small differences.

**Reproducibility:** fixed seeds, saved split files, one YAML config per run, Weights & Biases logging, and exact library versions in `requirements-lock.txt` (D24).

## Timeline

&#91;embedded content: 14-week plan · 5 phases, 2 reviews\]

Weeks 1–5 are the minimum viable project: if everything after slips, E1–E2 with CIs and a Grad-CAM grid still make a complete report. If fewer than 14 weeks remain, cut in this order: E8 on ResNet50, the 10% and 50% points of E5, then E7. E3, E4 and E6 stay — they are the proposal's pillars.

## Deliverables, stack and team

**Mid-semester review:** dataset manifest and saved splits, epidemiology summary, E1–E3 results with CIs, the gap figure with confusion matrices, first Grad-CAM grid.

**Final submission:**

- GitHub repo with configs, split files and a one-command reproduction of every table (kept private while it holds DDI metadata, which DDI's licence does not allow redistributing)
- Report in IEEE conference format (\~8 pages) following the RQ structure
- Six core figures: gap bar chart with CIs, lookalike confusion matrices before/after, image-processing ablation, localized-training data-size curve, texture ablation table, Grad-CAM before/after grid
- 15-minute presentation

**Stack:** Python, PyTorch + `timm`, `albumentations`, `scikit-image` (LBP, GLCM, Gabor), OpenCV (CLAHE, colour spaces), `scikit-learn`, `pytorch-grad-cam`, `umap-learn`, Weights & Biases. EfficientNet-B0 at 224 px on a few thousand images trains in about 25 minutes per run on a laptop CPU; every experiment ran on one.

**Team:** no per-person ownership; work is taken from the shared README work queue.

## Risks and fallbacks

The biggest risk is data, not modelling — so data access starts on day one.

| Risk | Fallback |
| --- | --- |
| DDI or DermaCon-IN access is slow (both need a request or agreement) | Request in week 1. Without them, Fitzpatrick17k's FST IV–VI test split still answers every RQ. |
| Too few SCC images on darker skin | Merge BCC + SCC into one non-melanoma class for darker-skin tests; keep 4 classes on the source side. |
| Indian case series disagree on the class mix | Treat priors as a sensitivity sweep, not one number; report the range. |
| Lesion-centred crop fails on busy photos | Fall back to centre crop; report crop failure rate. |
| DermaCon-IN has almost no cancers | Report only the benign false-positive rate on Indian skin. |
| Too few acral images | Report counts and examples only; move the acral analysis to future work. |
| Fitzpatrick17k label noise and duplicates | Dedup by hash + embeddings; report label quality as a limitation; DDI (biopsy-proven) is the clean check. |
| A pillar doesn't help (e.g. CORAL or texture) | That is a valid finding. Report it with CIs; a clean negative result beats a tuned-to-the-test positive one. |
| Compute limits | Ablations on EfficientNet-B0 at 224 px only; cache embeddings once for fusion and probe experiments. |

## Stretch goals and future work

Only after E1–E6 are done. Everything cut from the original spec lives here, so none of the idea is lost.

- **If time is left this semester:** DANN as a second adaptation method; wavelet features in the texture ablation; MobileNetV3 as a third backbone; a dermatologist reviewing 30–50 Grad-CAM maps.
- **Modality shift:** add HAM10000 (dermoscopy) as a second source and measure dermoscopy → clinical shift separately from skin-tone shift.
- **Beyond this project:** an Indian hospital collection with institutional ethics approval, used as a true external test set; a dedicated acral melanoma study; edge deployment of a quantised MobileNetV3.

## Key references

- Daneshjou et al. (2022), *Disparities in dermatology AI performance on a diverse, curated clinical image set*, Science Advances — [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9374341), [DDI dataset](https://aimi.stanford.edu/datasets/ddi-diverse-dermatology-images)
- Madarkar et al. (2025), *DermaCon-IN*, NeurIPS Datasets & Benchmarks — [paper](https://proceedings.nips.cc/paper_files/paper/2025/hash/a2c548b2ed3141400d86a235eaa6fad0-Abstract-Datasets_and_Benchmarks_Track.html)
- Wen et al. (2022), *Characteristics of publicly available skin cancer image datasets*, Lancet Digital Health — [abstract](https://pubmed.ncbi.nlm.nih.gov/34772649/)
- *Changing trends of skin cancer: a tertiary care hospital study in Malwa region of Punjab* — [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC4963704)
- Supekar et al. (2021), *Clinical spectrum of cutaneous malignancies in central India*, Indian Journal of Dermatology — [abstract](https://doaj.org/article/58d805c09fda4ebe9764938c00040479)
- *Pattern of skin malignancies in Manipur, India: a 5-year histopathological review* (2010), JPAD — [abstract](https://search.bvsalud.org/gim/resource/en/emr-144895)
- Chavan et al. (2019), *Acral lentiginous melanoma on the palm*, Journal of Skin and Stem Cell — [paper](https://brieflands.com/articles/jssc-101299)
- Daneshjou et al. (2023), *SkinCon* — [paper](https://arxiv.org/pdf/2302.00785)
- Groh et al. (2021), *Evaluating deep neural networks trained on clinical images in dermatology with the Fitzpatrick 17k dataset*, CVPR Workshops
- Pacheco et al. (2020), *PAD-UFES-20*, Data in Brief
- Sun & Saenko (2016), *Deep CORAL: Correlation Alignment for Deep Domain Adaptation*, ECCV Workshops
- Selvaraju et al. (2017), *Grad-CAM*, ICCV
