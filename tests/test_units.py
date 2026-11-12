import numpy as np
import pytest
import torch

from skincancer.config import load_config
from skincancer.labels import fst_bucket, map_diagnosis
from skincancer.models import Classifier, SkinNet, target_layer


@pytest.mark.parametrize("diagnosis,label", [
    ("basal cell carcinoma morpheiform", "bcc"),
    ("solid cystic basal cell carcinoma", "bcc"),
    ("BCC", "bcc"),
    ("squamous cell carcinoma", "scc"),
    ("squamous-cell-carcinoma-in-situ", "scc"),
    ("SCC", "scc"),
    ("superficial spreading melanoma ssm", "mel"),
    ("lentigo maligna", "mel"),
    ("melanoma-acral-lentiginous", "mel"),
    ("MEL", "mel"),
    ("ACK", "ak"),
    ("actinic keratosis", "ak"),
    ("seborrheic keratosis", "benign"),
    ("seborrheic-keratosis-irritated", "benign"),
    ("SEK", "benign"),
    ("NEV", "benign"),
    ("melanocytic-nevi", "benign"),
    ("halo nevus", "benign"),
    ("dermatofibroma", "benign"),
    ("solar lentigo", "benign"),
    ("granuloma pyogenic", "benign"),
    ("malignant- sccis", "scc"),
    ("malignant- bcc", "bcc"),
    ("malignant- ak", "ak"),
    ("benign-melanocytic nevus", "benign"),
    ("other- melanocytic lesion, possible re-excision (severe, spitz, aimp)", "exclude"),
    ("melanocytic tumor, possible re-excision (severe, spitz, aimp)", "exclude"),
    ("benign-other", "exclude"),
    ("malignant- other", "exclude"),
    ("Dermatosis Papulosa Nigra", "benign"),
    ("Bowen's Metastasis", "scc"),
    ("Nevus Depigmentosis", "exclude"),
    ("Nevus anemicus", "exclude"),
    ("Pseudolymphoma", "exclude"),
    ("Nevocytic nevus", "benign"),
    ("nevus-lipomatosus-superficialis", "exclude"),
    ("pigmented-spindle-cell-nevus-of-reed", "benign"),
    ("porokeratosis actinic", "exclude"),
    ("disseminated actinic porokeratosis", "exclude"),
    ("mycosis fungoides", "exclude_rare_cancer"),
    ("kaposi sarcoma", "exclude_rare_cancer"),
    ("dermatofibrosarcoma protuberans", "exclude_rare_cancer"),
    ("nevus sebaceous of jadassohn", "exclude"),
    ("becker nevus", "exclude"),
    ("psoriasis", "exclude"),
])
def test_label_rules(diagnosis, label):
    assert map_diagnosis(diagnosis)[0] == label


def test_fst_buckets():
    assert [fst_bucket(v) for v in (1, 3, 4, 5, 6, -1, None, "x", "FST 5", "N/A", "4.0")] == \
        ["I-III", "I-III", "IV", "V-VI", "V-VI", None, None, None, "V-VI", None, "IV"]


def test_config_extends_overrides_and_templates():
    cfg = load_config("configs/skinnet.yaml", ["model.width=48", "seed=2", "name=skinnet_w{model.width}"])
    assert cfg["name"] == "skinnet_w48" and cfg["seed"] == 2
    assert cfg["manifest"] == "data/manifest.csv"  # from data.yaml
    assert cfg["data"]["train_splits"] == ["train"] and cfg["data"]["val_splits"] == ["val"]
    assert cfg["data"]["label_col"] == "label_cancer"
    assert not set(cfg["eval"]["splits"]) & {"ddi_test", "dermacon_test", "dermacon_other", "midas_test"}   # test-only sets are never scored automatically


def test_skinnet():
    m = Classifier(2)
    assert isinstance(m.backbone, SkinNet) and m.dim == 256
    n_params = sum(p.numel() for p in m.parameters())
    assert 2.5e6 < n_params < 3.2e6
    assert all(p.requires_grad for p in m.parameters())          # every layer is trained
    assert sum(isinstance(x, torch.nn.Conv2d) and x.kernel_size == (3, 3) for x in m.modules()) == 19
    logits, f = m(torch.randn(2, 3, 96, 96), return_features=True)
    assert logits.shape == (2, 2) and f.shape == (2, 256)
    assert m.backbone.stages[-1](m.backbone.stages[2](m.backbone.stages[1](m.backbone.stages[0](
        m.backbone.stem(torch.randn(1, 3, 224, 224)))))).shape == (1, 256, 7, 7)
    assert target_layer(m) is m.backbone.stages[-1][-1]
    small = Classifier(3, width=16, depths=[1, 1, 1, 1])
    assert small.dim == 128 and small(torch.randn(2, 3, 64, 64)).shape == (2, 3)
    blk = m.backbone.stages[0][0].eval()                           # blocks start as the identity
    x = torch.rand(1, 32, 8, 8)
    assert torch.allclose(blk(x), x)


def test_skinnet_starts_from_random_weights_per_seed():
    from skincancer import runs
    cfg = load_config("configs/skinnet.yaml")
    def first_conv(seed):
        runs.set_seed(seed)
        return runs.build_model(cfg).backbone.stem[0].weight.detach().clone()
    assert torch.equal(first_conv(0), first_conv(0)) and not torch.equal(first_conv(0), first_conv(1))


def test_one_training_step_updates_every_layer():
    torch.manual_seed(0)
    m = Classifier(4, width=16, depths=[1, 1, 1, 1]).train()
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2, weight_decay=0.0)
    for _ in range(2):   # the second step moves the weights behind the zero-initialised block outputs
        loss = torch.nn.functional.cross_entropy(m(torch.randn(8, 3, 64, 64)), torch.randint(0, 4, (8,)))
        opt.zero_grad()
        loss.backward()
        opt.step()
    frozen = [n for n, p in m.named_parameters() if torch.equal(p, before[n])]
    assert not frozen, frozen


def test_warmup_schedule_class_weights_and_augmentation():
    from skincancer.data.dataset import eval_transform, train_transform
    from skincancer.train import class_weights, make_scheduler
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = make_scheduler(opt, epochs=10, warmup=4)
    lrs = []
    for _ in range(10):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[:4] == pytest.approx([0.25, 0.5, 0.75, 1.0]) and lrs[4] == pytest.approx(1.0)
    assert all(a > b for a, b in zip(lrs[4:], lrs[5:])) and lrs[-1] > 0
    counts = np.array([900.0, 400.0, 340.0, 380.0])
    w = class_weights(counts, "balanced")
    assert w * counts == pytest.approx(np.full(4, counts.sum() / 4))   # each class: equal total weight
    assert (class_weights(counts, "none") == 1).all()
    with pytest.raises(ValueError):
        class_weights(counts, "indian")
    img = np.random.default_rng(0).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    out = train_transform(64)(image=img)["image"]
    assert out.shape == (3, 64, 64) and out.dtype == torch.float32
    assert eval_transform(64, 72)(image=img)["image"].shape == (3, 64, 64)


def test_sampling_weights_balance_classes_inside_each_dataset():
    import pandas as pd
    from skincancer.train import sample_weights
    df = pd.DataFrame({"dataset": ["atlas"] * 100 + ["phone"] * 20,
                       "label_cancer": ["cancer"] * 10 + ["not_cancer"] * 90 + ["cancer"] * 15 + ["not_cancer"] * 5})
    assert sample_weights(df, "none", "label_cancer") is None
    df["w"] = sample_weights(df, "source_class", "label_cancer")
    assert df["w"].mean() == pytest.approx(1)
    cells = df.groupby(["dataset", "label_cancer"])["w"].sum()
    assert cells["atlas"].tolist() == pytest.approx([50, 50]) and cells["phone"].tolist() == pytest.approx([10, 10])
    with pytest.raises(ValueError):
        sample_weights(df, "nope", "label_cancer")


def test_source_auc_ignores_differences_between_datasets():
    from skincancer.metrics import auc, source_auc
    from skincancer.train import selection_score
    rng = np.random.default_rng(0)
    source = np.array(["atlas"] * 200 + ["phone"] * 200)
    y = np.r_[rng.random(200) < 0.1, rng.random(200) < 0.8].astype(float)
    shortcut = (source == "phone") + 0.01 * rng.random(400)     # knows the dataset, nothing about the lesion
    assert auc(y == 1, shortcut) > 0.8                           # looks good overall ...
    assert source_auc(y, shortcut, source) == pytest.approx(0.5, abs=0.1)   # ... and is chance inside each dataset
    assert source_auc(y, y + 0.0, source) == 1.0
    assert selection_score(y.astype(int), shortcut, source, "source_auc") == source_auc(y, shortcut, source)
    assert selection_score(y.astype(int), shortcut, source, "auc") == auc(y == 1, shortcut)


def test_derm7pt_extra_rows_and_loading(tmp_path):
    import pandas as pd
    from skincancer.data import extra
    base = tmp_path / "derm7pt" / "release_v0"
    (base / "meta").mkdir(parents=True)
    (base / "images" / "A").mkdir(parents=True)
    diagnoses = ["clark nevus", "melanoma (in situ)", "basal cell carcinoma", "vascular lesion", "seborrheic keratosis"]
    for k in range(1, 5):                                  # case 5's image is missing on disk
        (base / "images" / "A" / f"a{k}.jpg").write_bytes(b"x")
    pd.DataFrame({"case_num": [1, 2, 3, 4, 5], "clinic": [f"A/a{k}.jpg" for k in range(1, 6)],
                  "diagnosis": diagnoses, "location": ["back", "acral", "chest", "head neck", "back"]}
                 ).to_csv(base / "meta" / "meta.csv", index=False)
    df = extra.build("derm7pt", {"data_root": str(tmp_path)}, check_duplicates=False).set_index("image_id")
    assert df["label_cancer"].tolist()[:3] == ["not_cancer", "cancer", "cancer"]
    assert df.loc["derm7pt_2", "path"] == "derm7pt/release_v0/images/A/a2.jpg" and df.loc["derm7pt_2", "label_fine"] == "mel"
    assert df.loc["derm7pt_4", "exclude_reason"] == "no_cancer_label"      # vascular lesion: no tumour label
    assert df.loc["derm7pt_5", "exclude_reason"] == "missing_image"
    assert df["split"].tolist() == ["train_extra"] * 3 + ["excluded"] * 2
    df.reset_index().to_csv(tmp_path / "derm7pt.csv", index=False)
    loaded = extra.load_extra(["derm7pt"], "label_cancer", ["not_cancer", "cancer"], extra_dir=tmp_path)
    assert loaded["image_id"].tolist() == ["derm7pt_1", "derm7pt_2", "derm7pt_3"] and set(loaded["dataset"]) == {"derm7pt"}


def test_midas_is_split_three_ways_by_patient():
    import pandas as pd
    from skincancer.data.make_splits import check_no_leakage, make
    rng = np.random.default_rng(0)
    rows = []
    for p in range(120):                                   # 120 patients, 1-4 photos each, one lesion type per patient
        fine = ["bcc", "scc", "mel", "benign"][p % 4]
        for k in range(int(rng.integers(1, 5))):
            rows.append({"image_id": f"midas_{p}_{k}", "group_id": f"g{p}", "dataset": "midas", "label_4class": fine,
                         "label_fine": fine, "label_binary": float(fine != "benign"), "exclude_reason": np.nan,
                         "patient_id": f"p{p}", "fst": np.nan, "split": ""})
    df = make(pd.DataFrame(rows), {"splits": {"seed": 1, "fractions": [0.7, 0.15, 0.15], "midas": [0.4, 0.3, 0.3]}})
    assert set(df["split"]) == {"midas_train", "midas_val", "midas_test"}
    assert (df.groupby("group_id")["split"].nunique() == 1).all()          # a patient is never in two parts
    check_no_leakage(df)
    share = df["split"].value_counts(normalize=True)
    assert abs(share["midas_train"] - 0.4) < 0.06 and abs(share["midas_val"] - 0.3) < 0.06
    for part in ("midas_train", "midas_val", "midas_test"):                # every part holds every type
        assert set(df.loc[df["split"] == part, "label_fine"]) == {"bcc", "scc", "mel", "benign"}
    whole = make(pd.DataFrame(rows), {"splits": {"seed": 1, "fractions": [0.7, 0.15, 0.15]}})
    assert set(whole["split"]) == {"midas_test"}                           # without the setting MIDAS stays one test set


def test_leakage_check_allows_external_tests_only():
    import pandas as pd
    from skincancer.data.make_splits import check_no_leakage
    ok = pd.DataFrame({"group_id": ["a", "a", "b"], "split": ["dermacon_test", "dermacon_other", "train"]})
    check_no_leakage(ok)
    bad = pd.DataFrame({"group_id": ["a", "a"], "split": ["train", "ddi_test"]})
    with pytest.raises(AssertionError):
        check_no_leakage(bad)


def test_flip_tta_is_flip_invariant():
    from skincancer.runs import infer

    class Halves(torch.nn.Module):          # logits = mean of left, top and top-left quarter
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(1))

        def forward(self, x, return_features=False):
            h, w = x.shape[2] // 2, x.shape[3] // 2
            z = torch.stack([x[..., :w].mean((1, 2, 3)), x[:, :, :h].mean((1, 2, 3)),
                             x[:, :, :h, :w].mean((1, 2, 3))], 1) * 10 * self.w
            return z, z

    torch.manual_seed(0)
    x = torch.rand(4, 3, 8, 8)
    model = Halves()
    with torch.no_grad():
        p = infer(model, [(x, None)], tta=True)
        p_flip = infer(model, [(torch.flip(x, [3]), None)], tta=True)
        p_plain = infer(model, [(x, None)])
    assert p.shape == (4, 3) and np.allclose(p.sum(1), 1, atol=1e-5)
    assert np.allclose(p, p_flip, atol=1e-5)          # the four flip views are the same set
    assert not np.allclose(p, p_plain, atol=1e-4)


def test_worker_init_limits_threads():
    import cv2
    from skincancer.runs import _seed_worker
    before = torch.get_num_threads(), cv2.getNumThreads()
    try:
        _seed_worker(0, base=1)
        assert cv2.getNumThreads() <= 1
    finally:
        torch.set_num_threads(before[0])
        cv2.setNumThreads(before[1])


def test_make_loader_limits_worker_thread_pools(monkeypatch):
    import os
    import pandas as pd
    from skincancer import runs
    from skincancer.config import load_config
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    cfg = load_config("configs/skinnet.yaml")
    cfg["train"]["num_workers"] = 2
    n = torch.get_num_threads()
    df = pd.DataFrame({"image_id": ["a"], "path": ["x.jpg"], cfg["data"]["label_col"]: ["cancer"]})
    runs.make_loader(df, cfg, train=False, persistent=False)
    assert os.environ["OMP_NUM_THREADS"] == "1" and torch.get_num_threads() == n
