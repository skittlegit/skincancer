# Skin cancer classification with a CNN trained from scratch

SkinNet is a small residual CNN of our own, trained from random weights to answer one question about a clinical skin photo: is it a skin cancer or not? It trains on about 12,500 images and is tested on held-back images and on three outside datasets, one of them Indian.

No pretrained weights are used in the classifier and no layer is frozen. This is stage one. Telling cancer types apart, and whether darker skin needs its own model, come later.

This is a research prototype, not a diagnostic tool.

**Read [DECISIONS.md](DECISIONS.md) before changing anything.** It holds the rules for labels, splits, the network, the training recipe and the metrics, and the code enforces them.

The project was restarted on 12 Nov 2026. The earlier experiments (fine-tuned pretrained backbones and everything built on them) are in git under the tag `pre-fresh-start`.

Results so far are in [docs/RESULTS.md](docs/RESULTS.md); the models we keep and build on are listed in [docs/CHECKPOINTS.md](docs/CHECKPOINTS.md). What data to add after the baseline is planned in [docs/DATA_PLAN.md](docs/DATA_PLAN.md).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu && pip install -e . --no-deps
pytest                                                 # under a minute on CPU, synthetic data
```

`requirements.txt` pins the exact versions in use. The extra index gives the CPU build of PyTorch; with a GPU, install the matching torch and torchvision builds first.

## Data

The label map, manifest, near-duplicate groups and splits are committed (`data/`). They are frozen: do not re-run the split step.

The images are not in the repository. Put them under `data/raw/` as laid out in [configs/data.yaml](configs/data.yaml):

| Dataset | Used for | Get it from |
| --- | --- | --- |
| Fitzpatrick17k | training, validation, test | github.com/mattgroh/fitzpatrick17k (full image set from the authors) |
| PAD-UFES-20 | training, validation, test | Mendeley Data; merge `imgs_part_*` into `images/` |
| DDI | test only | Stanford AIMI (request access); never shown in figures |
| DermaCon-IN | test only | Harvard Dataverse: `python -m skincancer.data.download_dermacon_in` |
| MRA-MIDAS | test only | Stanford AIMI on Redivis: `python -m skincancer.data.download_midas` |

The pipeline that built the committed files, for reference:

```bash
python -m skincancer.data.build_label_map   # data/label_map.csv
python -m skincancer.data.build_manifest    # data/manifest.csv
python -m skincancer.data.dedup             # data/groups.csv (near-duplicates + patients)
python -m skincancer.data.make_splits       # split column + data/splits/*.csv
python -m skincancer.data.count_check       # split × class and split × type-of-lesion tables
```

## Train and evaluate

```bash
python -m skincancer.train --config configs/skinnet.yaml --seed 0     # one seed
python -m skincancer.predict --run runs/skinnet/s0 --splits ddi_test dermacon_test dermacon_other midas_test --final
python -m skincancer.evaluate --exp skinnet                           # results/skinnet/metrics.csv
python -m skincancer.analysis.gradcam --run runs/skinnet/s0 --split test
python -m skincancer.compare --a skinnet-c1 --b skinnet-c2 --splits val test   # is b better than a?
```

Or all three seeds, the test-only sets and the metrics in one go:

```bash
PY=.venv/Scripts/python bash scripts/run.sh        # Windows venv; PY=python elsewhere
```

`--set key=value` overrides any config field, for example `--set model.width=48 --set name=skinnet_w48`.

## Layout

```
configs/skinnet.yaml         the network and training recipe
configs/data.yaml            dataset paths and split settings
src/skincancer/models.py     SkinNet
src/skincancer/train.py      training loop
src/skincancer/predict.py    scoring a finished run; the test-set ledger
src/skincancer/evaluate.py   metrics with confidence intervals
src/skincancer/data/         label map, manifest, dedup, splits, dataset and augmentation
runs/<name>/s<seed>/         best.pt  config.yaml  history.csv  summary.json  preds/
runs/final_ledger.csv        one row per (run, test-only set) ever scored
results/<name>/metrics.csv   long format, one row per set × subgroup × metric
```

Two files connect everything. `data/manifest.csv` has one row per image (cancer or not, type of lesion, dataset, group, split); [data/splits/SUMMARY.md](data/splits/SUMMARY.md) counts them. `runs/<name>/s<seed>/preds/<split>.csv` has the predicted class probabilities. Evaluation reads only those two.

## Rules the code enforces

- `val` picks the checkpoint. Test splits never choose anything.
- DDI, DermaCon-IN and MIDAS are never scored during training. `predict --final` scores them and refuses to score the same run on the same set twice.
- A near-duplicate group never spans two splits.
- Seeds 0, 1, 2. Results are the seed mean with a group-bootstrap 95% confidence interval.
