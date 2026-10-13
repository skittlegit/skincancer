"""End-to-end run on synthetic data shaped like the real metadata files: label map ->
manifest -> dedup -> splits -> train (source / finetune / coral) -> predict --final ->
evaluate (+ prior shift) -> fusion -> report / analysis."""
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
CLASS_COLOR = {"bcc": (200, 60, 60), "scc": (60, 200, 60), "mel": (30, 30, 30), "benign": (120, 80, 50),
               "ak": (200, 200, 60)}
FITZ = {"bcc": "basal cell carcinoma", "scc": "squamous cell carcinoma", "mel": "melanoma",
        "benign": "seborrheic keratosis", "ak": "actinic keratosis"}
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
    for d in ("fitzpatrick17k/images", "pad_ufes_20/images", "ddi/images"):
        (raw / d).mkdir(parents=True)
    rows = []
    for i in range(260):
        cls = ["bcc", "scc", "mel", "benign"][i % 4] if i < 250 else "ak"
        fst = int(rng.integers(1, 7))
        h = f"h{i:04d}"
        _img(raw / f"fitzpatrick17k/images/{h}.jpg", cls, rng, skin=(220 - 25 * fst, 170 - 20 * fst, 150 - 18 * fst))
        rows.append({"md5hash": h, "fitzpatrick_scale": fst, "fitzpatrick_centaur": fst, "label": FITZ[cls],
                     "nine_partition_label": "x", "three_partition_label": "malignant" if cls != "benign" else "benign",
                     "qc": np.nan, "url": "http://x", "url_alphanum": "x"})
    rows.append(dict(rows[0], md5hash="nofst", fitzpatrick_scale=-1))           # unknown FST -> excluded
    rows.append(dict(rows[0], md5hash="psor", label="psoriasis", three_partition_label="non-neoplastic"))
    for extra in ("nofst", "psor"):
        _img(raw / f"fitzpatrick17k/images/{extra}.jpg", "benign", rng)
    shutil.copy(raw / "fitzpatrick17k/images/h0001.jpg", raw / "fitzpatrick17k/images/dup1.jpg")
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
    data["splits"]["e5_fractions"] = [25, 50]
    data["splits"]["e5_seeds"] = [0, 1]
    data["splits"]["min_pad_dark"] = 5
    (root / "configs/data.yaml").write_text(yaml.safe_dump(data))
    tiny = {"extends": "base.yaml", "model": {"backbone": "resnet18", "pretrained": False},
            "data": {"img_size": 64, "resize_short": 72},
            "train": {"batch_size": 16, "num_workers": 0, "amp": False,
                      "stages": [{"epochs": 1, "unfreeze": 0, "lr_head": 1e-3, "lr_backbone": 0.0, "early_stop": False},
                                 {"epochs": 2, "unfreeze": 1, "lr_head": 1e-3, "lr_backbone": 1e-4,
                                  "early_stop": True, "patience": 1}]}}
    (root / "configs/tiny.yaml").write_text(yaml.safe_dump(tiny))
    return root


def test_pipeline(workdir, monkeypatch):
    monkeypatch.chdir(workdir)
    from skincancer import evaluate, fusion, predict, report, train
    from skincancer.analysis import lookalike, probe
    from skincancer.data import build_label_map, build_manifest, count_check, dedup, make_splits

    build_label_map.main([])
    lm = pd.read_csv("data/label_map.csv")
    assert lm.set_index("diagnosis").loc["psoriasis", "label_4class"] == "exclude"
    build_manifest.main([])
    dedup.main(["--no-embed"])  # embedding pass needs pretrained weights
    make_splits.main([])
    m = pd.read_csv("data/manifest.csv")
    reasons = m.set_index("image_id")["exclude_reason"]
    assert reasons["f17k_nofst"] == "unknown_fst"
    assert reasons["f17k_dup1"] == "dup_cross_pool" and reasons["f17k_h0001"] == "dup_cross_pool"
    assert set(m.loc[m["dataset"] == "ddi", "split"]) == {"ddi_test"}
    assert set(m.loc[m["label_4class"] == "ak", "split"]) <= {"ak_side", "excluded"}
    assert (m["split"] == "pad_dark_test").sum() > 0
    make_splits.check_no_leakage(m)
    f25 = set(pd.read_csv("data/splits/e5/target_train_f25_s0.csv")["image_id"])
    f50 = set(pd.read_csv("data/splits/e5/target_train_f50_s0.csv")["image_id"])
    assert f25 < f50  # nested (D9)
    text, _ = count_check.report(m)
    assert "SCC images in target_test" in text

    for seed in (0, 1):
        train.main(["--config", "configs/tiny.yaml", "--seed", str(seed), "--set", "name=e1_tiny"])
        train.main(["--config", "configs/tiny.yaml", "--seed", str(seed), "--set", "name=e5_f50", "--set",
                    "mode=finetune", "--set", "data.fraction=50", "--set", f"init_from=runs/e1_tiny/s{seed}/best.pt"])
    train.main(["--config", "configs/tiny.yaml", "--seed", "0", "--set", "name=e7_tiny", "--set", "mode=coral",
                "--set", "train.coral_lambda=1.0"])
    hist = pd.read_csv("runs/e7_tiny/s0/history.csv")
    assert "target_val_auc" in hist and (hist["coral"] > 0).any()
    assert Path("runs/e1_tiny/s0/preds/target_test.csv").exists()
    assert not Path("runs/e1_tiny/s0/preds/ddi_test.csv").exists()  # test-only never automatic

    with pytest.raises(SystemExit):
        predict.main(["--run", "runs/e1_tiny/s0", "--splits", "ddi_test"])  # needs --final
    for seed in (0, 1):
        predict.main(["--run", f"runs/e1_tiny/s{seed}", "--splits", "ddi_test", "pad_dark_test", "--final"])
    with pytest.raises(SystemExit):
        predict.main(["--run", "runs/e1_tiny/s0", "--splits", "ddi_test", "--final"])  # only once

    out = evaluate.evaluate("e1_tiny", n_boot=50)
    assert set(out["eval_set"]) >= {"source_test", "target_test", "ddi_test", "ak_side"}
    assert "pad_dark_test" not in set(out["eval_set"])  # 6 images < MIN_SUBGROUP: skipped, not reported
    auc = out[(out["eval_set"] == "target_test") & (out["metric"] == "macro_auc") & (out["subgroup"] == "all")].iloc[0]
    assert auc["n_seeds"] == 2 and auc["ci_lo"] <= auc["mean"] <= auc["ci_hi"]
    assert set(out.loc[out["eval_set"] == "ddi_test", "metric"]) == {"binary_auc", "missed_cancer_rate", "false_alarm_rate"}
    shifted = evaluate.evaluate("e1_tiny", prior_mix="consensus", n_boot=20)
    assert shifted["exp"].iloc[0] == "e1_tiny+prior_consensus"
    evaluate.evaluate("e5_f50", n_boot=20)
    no_sk = evaluate.evaluate("e1_tiny", n_boot=20, drop_diagnoses=["seborrheic keratosis", "NEV"], tag="no_benign")
    assert no_sk["exp"].iloc[0] == "e1_tiny+no_benign"
    fa = no_sk[(no_sk["metric"] == "false_alarm_rate") & (no_sk["eval_set"] != "ddi_test")]
    assert fa["mean"].isna().all()  # no benign left outside DDI (its benign is verruca)

    fusion.main(["--source-exp", "e1_tiny", "--seeds", "0", "1", "--workers", "1"])
    assert Path("data/features/texture.csv").exists()
    for name in ("e6_fusion", "e6_embed_mlp", "e6_texture_lr"):
        evaluate.evaluate(name, n_boot=20)

    report.main(["tables", "--ref", "e1_tiny"])
    report.main(["gap", "--ref", "e1_tiny", "--exps", "e1_tiny", "e5_f50", "e6_fusion"])
    report.main(["curve", "--ref", "e1_tiny", "--prefix", "e5_f"])
    report.main(["confusion", "--exp", "e1_tiny", "--split", "target_test"])
    assert Path("reports/figures/gap.png").exists() and Path("results/summary.csv").exists()
    table, _ = lookalike.analyse("e1_tiny", "target_test")
    assert len(table)
    p = probe.probe("runs/e1_tiny/s0")
    assert 0 <= p["acc_mean"] <= 1

    from skincancer.analysis import gradcam
    gradcam.main(["--run", "runs/e1_tiny/s0", "--split", "target_test", "--n", "3"])
