# Ideas: the full history

Every idea tried or planned in this project, by letter, in the order it was run. An idea that works becomes a checkpoint; one that does not is discarded and its config and trained models are deleted. This file keeps the record of all of them, so the history survives the deleting.

Rules for each idea are written in [DECISIONS.md](../DECISIONS.md) before it is run. Full numbers are in [RESULTS.md](RESULTS.md). The models that were kept are described in [CHECKPOINTS.md](CHECKPOINTS.md).

## At a glance

| Letter | Idea | Date | Outcome |
| --- | --- | --- | --- |
| – | Baseline | 12 Nov 2026 | Starting point (`skinnet`) |
| A | Balanced drawing inside each dataset | 12 Nov | **Kept: checkpoint 1** (`skinnet-c1`) |
| B | Hard lesion types drawn more often | 12 Nov | Discarded |
| C | B plus image-quality augmentation | 12 Nov | Discarded |
| D | Derm7pt photos added to training | 12 Nov | Discarded |
| E | Larger images (320 px) | 12 Nov | Discarded |
| F | Derm7pt and larger images together | 12 Nov | Discarded |
| G | Average the three seeds | 12 Nov | Kept as part of G+H |
| H | Test-time flips | 12 Nov | Kept as part of G+H |
| G+H | Both together | 12 Nov | **Kept: checkpoint 2** (`skinnet-c2`) |
| J | MIDAS photos added to training | 13 Nov | Discarded; led to O |
| O | J with early stopping off | 13 Nov | **Kept: checkpoint 3** (`skinnet-c3`) |
| I | SCIN photos added as not cancer | 13 Nov | Discarded |
| N | Longer schedule (120 epochs) | 14 Nov | Running |
| M | Wider network (four times the parameters) | 14 Nov | Queued after N |
| L | Self-supervised pretraining on our own and other unlabelled photos | – | Not started |
| Q | More biopsy-confirmed tumour photos from new sources | – | Not started: needs a source |
| P | Pretrained (ImageNet) weights | – | Not started: breaks the from-scratch rule, needs a team decision |
| K | ISIC 2019 dermoscopy images added | – | Not started |

Letters were given when an idea was first listed, not when it was run, which is why I comes after O and K is last.

## How to read the scores

- **AUC:** how well the model ranks cancers above non-cancers. 0.5 is chance, 1.0 is perfect.
- **Within-source AUC:** the AUC inside each dataset, then averaged. It cannot be raised by telling datasets apart, so it is the score used on our own validation split.
- **Supported:** the paired comparison's 95% interval for the difference excludes zero.
- **Home:** our own `val` / `test` splits (Fitzpatrick17k and PAD-UFES-20). **Outside:** MIDAS, DDI and DermaCon-IN, photos from other hospitals.

## The ideas

### Baseline (`skinnet`)

- **What:** SkinNet, our own residual CNN (2.8 million parameters), trained from random weights on 12,562 images. Every image once per epoch, class-weighted loss.
- **Result:** test AUC 0.887; MIDAS 0.60, DDI 0.65. It learned "smartphone photo means cancer": 89% false alarms on PAD-UFES-20 non-cancers.
- **Outcome:** the starting point every idea is measured from. Replaced by A.

### A. Balanced drawing inside each dataset

- **What:** during training, cancer and not cancer are drawn equally often inside each dataset, so the kind of photo no longer predicts the class.
- **Compared on:** test and outside sets, 3 seeds.
- **Result:** false alarms on PAD-UFES-20 non-cancers 89% → 63%; cancers against benign lookalikes 0.72 → 0.78. Outside sets unchanged.
- **Outcome:** kept. **Checkpoint 1** (`skinnet-c1`).

### B. Hard lesion types drawn more often

- **What:** A, plus rarer and harder lesion types drawn more often (share in proportion to the square root of their count).
- **Compared on:** validation, 1 seed.
- **Result:** within-source AUC 0.847 against A's 0.869. Fewer false alarms on lookalikes (53% against 64%) but more on everything else.
- **Outcome:** discarded. A trade, not a gain.

### C. Image-quality augmentation

- **What:** B, plus random loss of resolution and JPEG quality in training, to imitate poorer photos.
- **Compared on:** validation, 1 seed.
- **Result:** 0.853 against 0.869. Same trade-off as B.
- **Outcome:** discarded.

### D. Derm7pt photos in training

- **What:** A, plus 958 clinical photos from the Derm7pt atlas (294 cancers, 664 benign) as training-only data.
- **Compared on:** validation, 1 seed.
- **Result:** 0.857 against 0.869.
- **Outcome:** discarded. One more atlas of moles and melanomas added little the network had not seen.

### E. Larger images

- **What:** A at 320 × 320 pixels (not 224 × 224).
- **Compared on:** validation, 1 seed.
- **Result:** 0.856 against 0.869; stopped early with no gain.
- **Outcome:** discarded. Image detail is not what limits this network.

### F. Derm7pt and larger images together

- **What:** D and E combined.
- **Compared on:** validation, 1 seed.
- **Result:** 0.862 against 0.869.
- **Outcome:** discarded.

### G. Average the three seeds

- **What:** no retraining. The three seeds of checkpoint 1 each predict, and their probabilities are averaged.
- **Compared on:** validation.
- **Result:** within-source AUC 0.882 against 0.868.
- **Outcome:** kept as part of G+H.

### H. Test-time flips

- **What:** no retraining. Each model predicts on the image and its three mirror images, and the four are averaged.
- **Compared on:** validation.
- **Result:** 0.879 against 0.868.
- **Outcome:** kept as part of G+H.

### G+H. Seed average and flips together

- **Compared on:** test and outside sets, paired comparison with checkpoint 1.
- **Result:** our test split AUC 0.890 → 0.904 and false alarms 40% → 33%, a supported gain. MIDAS +0.006 and DDI +0.006, neither supported.
- **Outcome:** kept. **Checkpoint 2** (`skinnet-c2`). The rule written beforehand asked for an outside gain, and by that rule it was first discarded; the team restored it because the gain at home is real, free, and nothing is worse elsewhere.

### J. MIDAS photos in training

- **Setup first:** MIDAS was split by patient into a training part (552 images), an outside validation part (414, never trained on) and a test part (414). From here models are also chosen on the outside validation part.
- **What:** checkpoint 1's recipe with the 552 MIDAS training photos added.
- **Compared on:** validation only (ours and the MIDAS validation part), paired comparison with checkpoint 2.
- **Result:** MIDAS validation AUC 0.689 → 0.782 (supported gain), but within-source AUC on our validation split 0.888 → 0.876 (supported loss). Most of the loss was one seed stopped early at epoch 25.
- **Outcome:** discarded at the first step, not scored on test sets. Its diagnosis led to O.

### O. J with early stopping off

- **What:** J, with every seed trained for all 60 epochs.
- **Compared on:** validation, then test and outside sets, paired comparison with checkpoint 2.
- **Result:** DDI 0.629 → 0.657 and MIDAS test 0.645 → 0.698, both supported. Our test split 0.904 → 0.898, a small supported loss.
- **Outcome:** kept. **Checkpoint 3** (`skinnet-c3`). It met the gain conditions and narrowly missed the no-loss one; the team accepted the trade. The first gain shown on a source never seen (DDI).

### I. SCIN photos as not cancer

- **What:** checkpoint 3's recipe with 2,907 SCIN phone photos of everyday conditions (eczema, contact dermatitis, hives, insect bites) added as not cancer, one per case.
- **Compared on:** validation only, paired comparison with checkpoint 3.
- **Result:** our validation split within-source AUC 0.882 → 0.902 (supported gain); MIDAS validation AUC 0.793 → 0.729 (supported loss).
- **Outcome:** discarded at the first step, not scored on test sets. Easy not-cancer photos help at home and hurt on tumours. The photos stay on disk (`data/raw/scin`); a much smaller share of the draws is untested.

### N. Longer schedule

- **What:** checkpoint 3's recipe trained for 120 epochs, not 60 (`configs/skinnet-idea-n.yaml`).
- **Status:** specified in DECISIONS.md, section 14. Stopped on 13 Nov before any seed finished; restarted from seed 0 on 14 Nov (`scripts/run_idea.sh n`).

### M. Wider network

- **What:** checkpoint 3's recipe with the network at twice the width, 11.2 million parameters against 2.8 million (`configs/skinnet-idea-m.yaml`).
- **Status:** specified in DECISIONS.md, section 14. Queued to train after N (14 Nov).

### L. Self-supervised pretraining on our own and other unlabelled photos

- **What:** learn general skin features without labels, then train cancer / not cancer. Widened on 14 Nov (DECISIONS.md, section 16): besides our training photos, also the unlabelled photos on disk that no evaluation set uses (SCIN, Derm7pt clinical). Never any validation or test image. No outside weights.
- **Status:** not started. Needs new training code and a few GPU hours.

### Q. More biopsy-confirmed tumour photos from new sources

- **What:** clinical photos (not dermoscopy) of cancers and benign lookalikes from hospitals not yet in training. Idea O is the only change that moved a score on a source never seen, and it did so with photos of this kind.
- **Status:** not started (DECISIONS.md, section 16). First step is to find such sets and check licences and overlap with our evaluation images.

### P. Pretrained weights

- **What:** start from ImageNet weights and train every layer. The largest known lever for new hospitals.
- **Status:** listed on 14 Nov after the supervisor called the outside scores poor (DECISIONS.md, section 16). Not run: it breaks the from-scratch rule (section 1), and changing that is for the team and the supervisor.

### K. ISIC 2019 in training

- **What:** add about 8,500 cancers and 12,900 nevi from ISIC 2019. These are dermoscopy images, a different kind of picture from ordinary photos.
- **Status:** not started. Large download; uncertain whether it carries over to clinical photos.

## Keeping this file up to date

When an idea is specified, add it here with its status. When it finishes, fill in what it was compared on, the result and the outcome, and update the table at the top. New ideas take the next unused letter (the next is R).
