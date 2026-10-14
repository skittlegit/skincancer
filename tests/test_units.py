import numpy as np
import pytest
import torch

from skincancer import preprocess
from skincancer.config import load_config
from skincancer.labels import fst_bucket, map_diagnosis
from skincancer.models import Classifier, coral_loss, set_trainable


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
    cfg = load_config("configs/e5_finetune.yaml", ["data.fraction=25", "seed=2"])
    assert cfg["name"] == "e5_f25"
    assert cfg["init_from"] == "runs/e1_effb0/s2/best.pt"
    assert cfg["mode"] == "finetune" and cfg["model"]["backbone"] == "efficientnet_b0"
    assert cfg["manifest"] == "data/manifest.csv"  # from data.yaml via base.yaml
    cfg = load_config("configs/e7_coral.yaml", ["train.coral_lambda=10"])
    assert cfg["name"] == "e7_coral_l10"
    cfg = load_config("configs/e3_crop.yaml")
    assert cfg["data"]["preprocess"] == {"color_constancy": False, "clahe": False, "lesion_crop": True}


def _blob_image(center=(140, 90), r=25):
    import cv2
    img = np.full((200, 280, 3), (190, 150, 130), np.uint8)
    cv2.circle(img, center, r, (70, 40, 30), -1)
    return img


def test_lesion_crop_finds_off_centre_lesion():
    img = _blob_image(center=(190, 120))
    crop, ok = preprocess.lesion_crop(img)
    assert ok
    h, w = crop.shape[:2]
    assert h == w and h < 150
    assert crop[h // 2, w // 2].mean() < 80  # lesion at the centre of the crop


def test_lesion_crop_falls_back_on_blank():
    crop, ok = preprocess.lesion_crop(np.full((100, 100, 3), 128, np.uint8))
    assert not ok and crop.shape[:2] == (80, 80)


def test_colour_constancy_and_clahe_keep_shape_and_ab():
    import cv2
    img = _blob_image()
    tinted = np.clip(img.astype(int) * [1.2, 1.0, 0.8], 0, 255).astype(np.uint8)
    cc = preprocess.shades_of_gray(tinted)
    ch_means = cc.reshape(-1, 3).mean(0)
    assert ch_means.max() - ch_means.min() < np.ptp(tinted.reshape(-1, 3).mean(0))
    out = preprocess.clahe_l(img)
    lab_in, lab_out = cv2.cvtColor(img, cv2.COLOR_RGB2LAB), cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
    assert out.shape == img.shape
    assert np.abs(lab_in[..., 1:].astype(int) - lab_out[..., 1:]).mean() < 3  # colour ~ unchanged
    assert preprocess.tag({"color_constancy": True, "lesion_crop": True}) == "cc-crop"


def test_coral_loss():
    a = torch.randn(64, 16)
    assert coral_loss(a, a).item() == pytest.approx(0, abs=1e-7)
    assert coral_loss(a, 3 * torch.randn(64, 16)).item() > 0


@pytest.mark.parametrize("backbone,unfreeze,expect_trainable", [
    ("efficientnet_b0", 0, []), ("efficientnet_b0", 3, ["blocks.5", "blocks.6", "conv_head", "bn2"]),
    ("resnet50", 1, ["layer4"]),
])
def test_set_trainable(backbone, unfreeze, expect_trainable):
    m = Classifier(backbone, 4, pretrained=False)
    set_trainable(m, unfreeze)
    trainable = {n.split(".")[0] if not n.startswith("blocks") else ".".join(n.split(".")[:2])
                 for n, p in m.backbone.named_parameters() if p.requires_grad}
    assert trainable == {t for t in expect_trainable}
    assert all(p.requires_grad for p in m.head.parameters())


def test_leakage_check_allows_external_tests_only():
    import pandas as pd
    from skincancer.data.make_splits import check_no_leakage
    ok = pd.DataFrame({"group_id": ["a", "a", "b"], "split": ["dermacon_test", "dermacon_other", "source_train"]})
    check_no_leakage(ok)
    bad = pd.DataFrame({"group_id": ["a", "a"], "split": ["source_train", "ddi_test"]})
    with pytest.raises(AssertionError):
        check_no_leakage(bad)


def test_seed_ensemble_averages_probabilities():
    import pandas as pd
    from skincancer.evaluate import ensemble_frame
    classes = ["a", "b"]
    f1 = pd.DataFrame({"image_id": ["x", "y"], "p_a": [0.2, 0.9], "p_b": [0.8, 0.1]})
    f2 = pd.DataFrame({"image_id": ["y", "x"], "p_a": [0.7, 0.4], "p_b": [0.3, 0.6]})   # other order
    out = ensemble_frame([f1, f2], classes).set_index("image_id")
    assert np.allclose(out.loc["x"], [0.3, 0.7]) and np.allclose(out.loc["y"], [0.8, 0.2])


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


def test_derm7pt_rows(tmp_path):
    import pandas as pd
    from skincancer.data.extra import derm7pt_rows
    base = tmp_path / "derm7pt" / "release_v0"
    (base / "meta").mkdir(parents=True)
    pd.DataFrame({"case_num": [1, 2, 3, 4], "clinic": ["A/a1.jpg", "A/a2.jpg", "B/b1.jpg", "B/b2.jpg"],
                  "diagnosis": ["clark nevus", "melanoma (in situ)", "melanoma metastasis", "basal cell carcinoma"],
                  "location": ["back", "acral", "chest", "head neck"]}).to_csv(base / "meta" / "meta.csv", index=False)
    df = derm7pt_rows(tmp_path).set_index("image_id")
    assert df["label_4class"].tolist() == ["benign", "mel", "exclude", "bcc"]
    assert df.loc["derm7pt_2", "path"] == "derm7pt/release_v0/images/A/a2.jpg"
    assert df.loc["derm7pt_2", "is_acral"] and not df.loc["derm7pt_1", "is_acral"]
    assert df.loc["derm7pt_4", "label_3class"] == "nmsc" and df.loc["derm7pt_4", "label_binary"] == 1
