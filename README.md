# Domain-adaptive skin cancer classification

Code for *Epidemiology-Guided Image Processing and Texture-Aware Feature Learning for Domain-Adaptive Skin Cancer Classification*: how much a classifier trained on lighter skin (FST I–III) loses on darker skin (FST IV–VI) and on Indian images, and which fixes close that gap. The fixes tested are image processing, Indian class priors, texture fusion, localized fine-tuning and Deep CORAL.

This is a research prototype, not a diagnostic tool.

**Read [DECISIONS.md](DECISIONS.md) before changing anything.** It holds the protocol rules (validation sets, AK handling, FST buckets, priors, statistics), and the code enforces them.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
pytest                                                 # ~1 min on CPU, synthetic data
```

On Kaggle or Colab, torch is already installed. Run `pip install -r requirements.txt && pip install -e . --no-deps`, then set `train.num_workers` to the number of CPUs.

## Data → splits (once, then commit)

Put raw data under `data/raw/` as laid out in [configs/data.yaml](configs/data.yaml):

| Dataset | Get it from | Notes |
| --- | --- | --- |
| Fitzpatrick17k | CSV from github.com/mattgroh/fitzpatrick17k, then `python -m skincancer.data.download_fitzpatrick17k --neoplastic-only` | Dead links → `missing_image` |
| PAD-UFES-20 | Mendeley Data; merge `imgs_part_*` into `images/` | |
| DDI | Stanford AIMI (request access) | Test only |
| DermaCon-IN | Request from the authors | Test only; set its column names in `data.yaml` |

```bash
python -m skincancer.data.build_label_map   # data/label_map.csv: review every row, commit (D5)
python -m skincancer.data.build_manifest    # data/manifest.csv
python -m skincancer.data.dedup             # data/groups.csv (near-duplicates + patients)
python -m skincancer.data.make_splits       # split column + data/splits/*.csv (incl. E5 subsets)
python -m skincancer.data.count_check       # class × FST table; BCC+SCC merge rule (D6)
```

Commit `label_map.csv`, `manifest.csv`, `groups.csv` and `data/splits/`. After that, nobody re-splits.

## Experiments

One config per experiment in [configs/](configs/). `--set key=value` overrides any field, and `{key}` in `name` / `init_from` is filled in from the config.

```bash
python -m skincancer.train --config configs/e1_effb0.yaml --seed 0
python -m skincancer.train --config configs/e5_finetune.yaml --seed 0 --set data.fraction=25
python -m skincancer.train --config configs/e7_coral.yaml --seed 0 --set train.coral_lambda=10000
python -m skincancer.fusion --source-exp e1_effb0 --seeds 0 1 2                     # E6
python -m skincancer.evaluate --exp e1_effb0 [--prior-mix all]                       # E4 prior shift
python -m skincancer.predict --run runs/e1_effb0/s0 --splits ddi_test dermacon_test --final
```

[scripts/run_all.sh](scripts/run_all.sh) has every run in order. Analysis and figures:

```bash
python -m skincancer.report tables --ref e1_effb0          # results/summary.csv (with gap)
python -m skincancer.report gap --exps e1_effb0 e3_all e6_fusion e7_coral_l1000.0
python -m skincancer.report curve --ref e1_effb0           # E5 data-size curve
python -m skincancer.report confusion --exp e1_effb0 --split target_test
python -m skincancer.analysis.lookalike --exp e1_effb0     # which lookalikes get confused
python -m skincancer.analysis.gradcam --run runs/e1_effb0/s0 --split target_test
python -m skincancer.analysis.probe --run runs/e1_effb0/s0 runs/e7_coral_l1000.0/s0
python -m skincancer.analysis.umap_plot --run runs/e1_effb0/s0
```

## How the pieces connect

Every result goes through two files:

- **`data/manifest.csv`** holds one row per image: `image_id, group_id, dataset, path, diagnosis, label_4class, label_3class, label_binary, fst, fst_bucket, site, is_acral, patient_id, exclude_reason, split`.
- **`runs/<exp>/s<seed>/preds/<split>.csv`** holds `image_id, exp, seed, p_<class>…`.

Training writes prediction files. Everything downstream (metrics, CIs, prior shift, figures, lookalike tables) reads only prediction files plus the manifest. To add a method, write prediction files in that format plus a `train_prior.json`, and every evaluation and figure works on it unchanged.

```
runs/<exp>/s<seed>/   best.pt  config.yaml  history.csv  train_prior.json  summary.json  preds/  emb/
runs/final_ledger.csv one row per (run, test-only set) ever scored; committed
results/<exp>/        metrics.csv (long format, with CIs), lookalike_*.csv, errors_*.csv
reports/figures/      report figures
```

## Rules the code enforces

- `source_val` picks checkpoints for source-trained models; `target_val` is only for target-trained models and for choosing between methods. Test splits never choose anything.
- DDI, DermaCon-IN and `pad_dark_test` are never scored during training. `predict --final` scores them and refuses to score the same run on the same set twice.
- A near-duplicate group never spans two splits (`make_splits` asserts this).
- Seeds 0, 1, 2 for every run. Results are reported as the seed mean, with a group-bootstrap 95% CI.

## Work queue

Anyone takes the next unblocked item. Put your name on the issue or PR, not in this file.

1. [ ] Request DDI and DermaCon-IN access; download Fitzpatrick17k and PAD-UFES-20
2. [ ] Build the label map and review every row (D5)
3. [ ] Manifest → dedup (`--embed` on a GPU) → splits → count check; record the D6 decision
4. [ ] `python -m skincancer.preprocess --n 100`: check the crops by eye, record the failure rate
5. [ ] E1 + E2, both backbones → gap figure, confusion matrices, Grad-CAM grid (**minimum viable project**)
6. [ ] E3, E4 (training + prior-shift sweep)
7. [ ] Mid-sem review pack: manifest, splits, E1–E3 with CIs, gap figure, Grad-CAM
8. [ ] E5 curve
9. [ ] E6 fusion and texture ablation
10. [ ] E7 CORAL; pick λ on target_val; domain probe before/after
11. [ ] E8 (choose on target_val), both backbones
12. [ ] Test-only sets via `--final` for each finished method; acral check; UMAP
13. [ ] Report (IEEE), presentation, one-command reproduction check on a clean machine
