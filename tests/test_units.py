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
    assert cfg["data"]["train_splits"] == ["source_train"] and cfg["data"]["val_splits"] == ["source_val"]
    assert not set(cfg["eval"]["splits"]) & {"ddi_test", "dermacon_test", "dermacon_other", "pad_dark_test",
                                              "midas_test"}   # test-only sets are never scored automatically


def test_skinnet():
    m = Classifier(4)
    assert isinstance(m.backbone, SkinNet) and m.dim == 256
    n_params = sum(p.numel() for p in m.parameters())
    assert 2.5e6 < n_params < 3.2e6
    assert all(p.requires_grad for p in m.parameters())          # every layer is trained
    assert sum(isinstance(x, torch.nn.Conv2d) and x.kernel_size == (3, 3) for x in m.modules()) == 19
    logits, f = m(torch.randn(2, 3, 96, 96), return_features=True)
    assert logits.shape == (2, 4) and f.shape == (2, 256)
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


def test_leakage_check_allows_external_tests_only():
    import pandas as pd
    from skincancer.data.make_splits import check_no_leakage
    ok = pd.DataFrame({"group_id": ["a", "a", "b"], "split": ["dermacon_test", "dermacon_other", "source_train"]})
    check_no_leakage(ok)
    bad = pd.DataFrame({"group_id": ["a", "a"], "split": ["source_train", "ddi_test"]})
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
    df = pd.DataFrame({"image_id": ["a"], "path": ["x.jpg"], cfg["data"]["label_col"]: ["bcc"]})
    runs.make_loader(df, cfg, train=False, persistent=False)
    assert os.environ["OMP_NUM_THREADS"] == "1" and torch.get_num_threads() == n
