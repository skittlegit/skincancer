"""End-to-end run on synthetic data shaped like the real metadata files: label map ->
manifest -> dedup -> splits -> train from random weights -> predict --final -> evaluate -> Grad-CAM."""
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
CLASS_COLOR = {"bcc": (200, 60, 60), "scc": (60, 200, 60), "mel": (30, 30, 30), "benign": (120, 80, 50),
               "ak": (200, 200, 60), "other": (230, 120, 160)}
FITZ = {"bcc": "basal cell carcinoma", "scc": "squamous cell carcinoma", "mel": "melanoma",
        "benign": "seborrheic keratosis", "ak": "actinic keratosis", "other": "psoriasis"}
PAD = {"bcc": "BCC", "scc": "SCC", "mel": "MEL", "benign": "NEV", "ak": "ACK"}


def _img(path: Path, cls: str, rng, skin=(200, 160, 140)):
    img = np.full((96, 96, 3), skin, np.uint8)
    c = (int(rng.integers(35, 60)), int(rng.integers(35, 60)))
    cv2.circle(img, c, int(rng.integers(12, 22)), CLASS_COLOR[cls], -1)
    blotches = cv2.resize(rng.normal(0, 35, (6, 6, 3)), (96, 96), interpolation=cv2.INTER_CUBIC)
    img = np.clip(img + blotches + rng.normal(0, 8, img.shape), 0, 255).astype(np.uint8)  # distinct pHashes
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    root = tmp_path_factory.mktemp("proj")
    rng = np.random.default_rng(0)
    raw = root / "data/raw"
    fz = yaml.safe_load(Path("configs/data.yaml").read_text(encoding="utf-8"))["datasets"]["fitzpatrick17k"]["image_dir"]
    for d in (fz, "pad_ufes_20/images", "ddi/images"):
        (raw / d).mkdir(parents=True)
    rows = []
    for i in range(340):
        cls = ["bcc", "scc", "mel", "benign"][i % 4] if i < 250 else "ak" if i < 260 else "other"
        fst = int(rng.integers(1, 7))
        h = f"h{i:04d}"
        _img(raw / f"{fz}/{h}.jpg", cls, rng, skin=(220 - 25 * fst, 170 - 20 * fst, 150 - 18 * fst))
        rows.append({"md5hash": h, "fitzpatrick_scale": fst, "fitzpatrick_centaur": fst, "label": FITZ[cls],
                     "nine_partition_label": "x", "three_partition_label": {"benign": "benign", "other": "non-neoplastic"}.get(cls, "malignant"),
                     "qc": np.nan, "url": "http://x", "url_alphanum": "x"})
    rows.append(dict(rows[0], md5hash="nofst", fitzpatrick_scale=-1))           # unknown FST -> excluded
    rows.append(dict(rows[0], md5hash="cyst", label="pilar cyst", three_partition_label="benign"))  # tumour outside the classes
    for extra in ("nofst", "cyst"):
        _img(raw / f"{fz}/{extra}.jpg", "benign", rng)
    shutil.copy(raw / f"{fz}/h0001.jpg", raw / f"{fz}/dup1.jpg")
    rows.append(dict(rows[1], md5hash="dup1", fitzpatrick_scale=6 if rows[1]["fitzpatrick_scale"] <= 3 else 1))
    pd.DataFrame(rows).to_csv(raw / "fitzpatrick17k/fitzpatrick17k.csv", index=False)

    pad = []
    for i in range(80):
        cls = ["bcc", "scc", "mel", "benign", "ak"][i % 5]
        name = f"PAT_{i // 2}_{i}.png"
        _img(raw / f"pad_ufes_20/images/{name}", cls, rng)
        pad.append({"patient_id": f"PAT_{i // 2}", "lesion_id": i, "img_id": name, "diagnostic": PAD[cls],
                    "fitspatrick": [1, 2, 3, np.nan][i % 4] if i < 70 else 5,
                    "region": ["HAND", "FACE", "FOOT", "BACK"][i % 4], "biopsed": True})
    pd.DataFrame(pad).to_csv(raw / "pad_ufes_20/metadata.csv", index=False)

    ddi = []
    for i in range(40):
        cls = ["bcc", "mel", "benign", "benign"][i % 4]
        name = f"{i:06d}.png"
        _img(raw / f"ddi/images/{name}", cls, rng, skin=(90, 60, 45))
        ddi.append({"DDI_ID": i, "DDI_file": name, "skin_tone": [12, 34, 56, 56][i % 4],
                    "malignant": cls != "benign",
                    "disease": {"bcc": "basal-cell-carcinoma-nodular", "mel": "melanoma-in-situ",
                                "benign": "verruca-vulgaris"}[cls]})
    pd.DataFrame(ddi).to_csv(raw / "ddi/ddi_metadata.csv", index=False)

    (root / "configs").mkdir()
    for f in (REPO / "configs").glob("*.yaml"):
        shutil.copy(f, root / "configs" / f.name)
    data = yaml.safe_load((root / "configs/data.yaml").read_text())
    data["splits"]["min_pad_dark"] = 5
    (root / "configs/data.yaml").write_text(yaml.safe_dump(data))
    tiny = {"extends": "skinnet.yaml", "name": "tiny", "data": {"img_size": 64, "resize_short": 72},
            "model": {"width": 8, "depths": [1, 1, 1, 1]},
            "train": {"epochs": 3, "warmup_epochs": 1, "patience": 2, "batch_size": 16, "num_workers": 0,
                      "amp": False}}
    (root / "configs/tiny.yaml").write_text(yaml.safe_dump(tiny))
    return root


def test_pipeline(workdir, monkeypatch):
    monkeypatch.chdir(workdir)
    from skincancer import evaluate, predict, train
    from skincancer.data import build_label_map, build_manifest, count_check, dedup, make_splits

    build_label_map.main([])
    lm = pd.read_csv("data/label_map.csv")
    assert lm.set_index("diagnosis").loc["psoriasis", "label_4class"] == "exclude"   # no tumour label ...
    build_manifest.main([])
    dedup.main(["--no-embed"])  # embedding pass needs pretrained weights
    make_splits.main([])
    m = pd.read_csv("data/manifest.csv")
    reasons = m.set_index("image_id")["exclude_reason"]
    by_id = m.set_index("image_id")
    fine = by_id["label_fine"]
    assert fine["f17k_h0300"] == "non_tumour" and by_id.loc["f17k_h0300", "label_cancer"] == "not_cancer"
    assert fine["f17k_cyst"] == "other_benign_tumour" and by_id.loc["f17k_cyst", "label_cancer"] == "not_cancer"
    assert fine["f17k_h0000"] == "bcc" and by_id.loc["f17k_h0000", "label_cancer"] == "cancer"
    assert fine["f17k_h0255"] == "ak" and pd.isna(by_id.loc["f17k_h0255", "label_cancer"])   # precancerous: neither
    assert pd.isna(reasons["f17k_nofst"]) and by_id.loc["f17k_nofst", "split"] in {"train", "val", "test"}
    main = m[m["split"].isin(["train", "val", "test"])]
    assert set(main["label_cancer"]) == {"cancer", "not_cancer"} and set(main["dataset"]) == {"fitzpatrick17k", "pad_ufes_20"}
    assert main["fst"].isna().any() and (main["fst"] >= 4).any()        # skin type plays no part in the split
    assert 0.6 < (main["split"] == "train").mean() < 0.8
    assert by_id.loc["f17k_dup1", "split"] == by_id.loc["f17k_h0001", "split"]   # duplicates stay together
    assert set(m.loc[m["dataset"] == "ddi", "split"]) == {"ddi_test"}
    assert set(m.loc[m["label_4class"] == "ak", "split"]) <= {"ak_side", "excluded"}
    make_splits.check_no_leakage(m)
    assert {p.stem for p in Path("data/splits").glob("*.csv")} == set(m["split"]) - {"excluded"}
    assert "type of lesion" in count_check.report(m)
    make_splits.main([])                                               # re-running changes nothing
    assert pd.read_csv("data/manifest.csv")["split"].equals(m["split"])

    for seed in (0, 1):
        train.main(["--config", "configs/tiny.yaml", "--seed", str(seed)])
    hist = pd.read_csv("runs/tiny/s0/history.csv")
    assert 1 <= len(hist) <= 3 and hist["lr"].iloc[0] == pytest.approx(1e-3)   # 1 warm-up epoch
    summary = yaml.safe_load(Path("runs/tiny/s0/summary.json").read_text())
    assert summary["n_train"] == (m["split"] == "train").sum() and summary["n_params"] > 0
    preds = pd.read_csv("runs/tiny/s0/preds/test.csv")
    assert [c for c in preds if c.startswith("p_")] == ["p_not_cancer", "p_cancer"]
    assert np.allclose(preds[["p_not_cancer", "p_cancer"]].sum(1), 1, atol=1e-5)
    assert not Path("runs/tiny/s0/preds/ddi_test.csv").exists()  # test-only never automatic
    train.main(["--config", "configs/tiny.yaml", "--seed", "0"])   # finished: skipped, not retrained
    assert len(pd.read_csv("runs/tiny/s0/history.csv")) == len(hist)

    with pytest.raises(SystemExit):
        predict.main(["--run", "runs/tiny/s0", "--splits", "ddi_test"])  # needs --final
    for seed in (0, 1):
        predict.main(["--run", f"runs/tiny/s{seed}", "--splits", "ddi_test", "--final"])
    with pytest.raises(SystemExit):
        predict.main(["--run", "runs/tiny/s0", "--splits", "ddi_test", "--final"])  # only once
    ledger = pd.read_csv("runs/final_ledger.csv")
    assert set(ledger["run"]) == {"runs/tiny/s0", "runs/tiny/s1"}

    out = evaluate.evaluate("tiny", n_boot=50)
    assert set(out["eval_set"]) >= {"val", "test", "ddi_test", "ak_side"}
    assert set(out.loc[out["eval_set"] == "ddi_test", "metric"]) == {"binary_auc", "missed_cancer_rate", "false_alarm_rate",
                                                                     "missed_cancer_rate_op90", "false_alarm_rate_op90"}
    test = out[out["eval_set"] == "test"]
    auc = test[(test["metric"] == "binary_auc") & (test["subgroup"] == "all")].iloc[0]
    assert auc["n_seeds"] == 2 and auc["ci_lo"] <= auc["mean"] <= auc["ci_hi"]
    by_type = test[test["subgroup_col"] == "label_fine"]
    assert {"bcc", "non_tumour"} <= set(by_type["subgroup"])              # which cancers are missed, what alarms
    missed = by_type[(by_type["subgroup"] == "bcc") & (by_type["metric"] == "missed_cancer_rate")].iloc[0]
    assert 0 <= missed["mean"] <= 1
    assert set(out.loc[out["eval_set"] == "ak_side", "metric"]) == {"pred_malignant_rate", "pred_malignant_rate_op90"}
    assert Path("results/tiny/metrics.csv").exists()

    assert "source_auc" in set(out.loc[out["eval_set"] == "test", "metric"])
    hard = out[(out["eval_set"] == "test") & (out["subgroup"] == "cancer_vs_lookalike")]
    assert len(hard) and hard["n"].iloc[0] < auc["n"]

    # balanced sampling, checkpoint picked on the within-source AUC
    train.main(["--config", "configs/tiny.yaml", "--seed", "0", "--set", "name=tiny_bal", "--set",
                "train.sampling=source_class", "--set", "train.class_weighting=none", "--set",
                "train.select_metric=source_auc"])
    assert yaml.safe_load(Path("runs/tiny_bal/s0/summary.json").read_text())["select_metric"] == "source_auc"
    val_only = evaluate.evaluate("tiny_bal", n_boot=20, splits=["val"])
    assert set(val_only["eval_set"]) == {"val"} and Path("results/tiny_bal/metrics_val.csv").exists()
    assert not Path("results/tiny_bal/metrics.csv").exists()            # the full table is untouched
    with pytest.raises(ValueError):
        train.main(["--config", "configs/tiny.yaml", "--seed", "0", "--set", "name=bad", "--set",
                    "train.sampling=source_class"])                     # sampling + weighted loss: refused

    # paired comparison of two models on the same images
    from skincancer import compare
    cmp_ = compare.main(["--a", "tiny", "--b", "tiny_bal", "--splits", "val", "test", "--n-boot", "100"])
    assert set(cmp_["split"]) == {"val", "test"} and {"auc", "source_auc"} <= set(cmp_["metric"])
    assert ((cmp_["ci_lo"] <= cmp_["diff"]) & (cmp_["diff"] <= cmp_["ci_hi"])).all()
    same = compare.main(["--a", "tiny", "--b", "tiny", "--splits", "val", "--n-boot", "50"])
    assert (same["diff"] == 0).all() and (same["supported"] == "no").all()

    # derived models (how checkpoint 2 is built): test-time flips, then the seed average
    from skincancer import derive
    derive.main(["tta", "--exp", "tiny", "--name", "tiny_tta"])
    assert yaml.safe_load(Path("runs/tiny_tta/s1/config.yaml").read_text())["eval"]["tta"] is True
    for seed in (0, 1):
        predict.main(["--run", f"runs/tiny_tta/s{seed}", "--splits", "val"])
    derive.main(["ensemble", "--exp", "tiny", "--name", "tiny_ens", "--splits", "val", "test"])
    ens = pd.read_csv("runs/tiny_ens/s0/preds/val.csv").set_index("image_id")
    parts = [pd.read_csv(f"runs/tiny/s{k}/preds/val.csv").set_index("image_id").loc[ens.index] for k in (0, 1)]
    assert np.allclose(ens["p_cancer"], (parts[0]["p_cancer"] + parts[1]["p_cancer"]) / 2)
    with pytest.raises(SystemExit):
        derive.main(["ensemble", "--exp", "tiny", "--name", "tiny_ens", "--splits", "ddi_test"])   # needs --final
    derive.main(["ensemble", "--exp", "tiny", "--name", "tiny_ens", "--splits", "ddi_test", "--final"])
    with pytest.raises(SystemExit):
        derive.main(["ensemble", "--exp", "tiny", "--name", "tiny_ens", "--splits", "ddi_test", "--final"])   # once
    n_rows = len(pd.read_csv("runs/final_ledger.csv"))
    derive.main(["ensemble", "--exp", "tiny", "--name", "tiny_ens", "--splits", "ddi_test", "--final", "--force"])
    predict.main(["--run", "runs/tiny/s0", "--splits", "ddi_test", "--final", "--force"])
    assert len(pd.read_csv("runs/final_ledger.csv")) == n_rows            # reproductions add no ledger rows
    assert "runs/tiny_ens/s0" in set(pd.read_csv("runs/final_ledger.csv")["run"])
    assert set(compare.main(["--a", "tiny", "--b", "tiny_ens", "--splits", "val", "--n-boot", "50"])["b"]) == {"tiny_ens"}

    # training on two splits at once
    train.main(["--config", "configs/tiny.yaml", "--seed", "0", "--set", "name=tiny_both", "--set",
                "data.train_splits=[train, val]", "--set", "train.epochs=1"])
    both = yaml.safe_load(Path("runs/tiny_both/s0/summary.json").read_text())
    assert both["n_train"] == m["split"].isin(["train", "val"]).sum()

    from skincancer.analysis import gradcam
    gradcam.main(["--run", "runs/tiny/s0", "--split", "test", "--n", "3"])
