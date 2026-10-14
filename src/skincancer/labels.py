"""Classes, diagnosis-mapping rules, FST buckets and Indian class priors.

The reasoning behind every rule is in DECISIONS.md (D2-D7).
"""
from __future__ import annotations

import math
import re

import numpy as np

CLASSES = {
    "label_4class": ["bcc", "scc", "mel", "benign"],
    "label_3class": ["nmsc", "mel", "benign"],
}
MALIGNANT = {"bcc", "scc", "mel", "nmsc"}
TO_3CLASS = {"bcc": "nmsc", "scc": "nmsc", "mel": "mel", "benign": "benign"}

# Ordered (pattern, label, needs_review). First match wins, so specific rules sit
# above general ones ("lentigo maligna" before "lentigo", subtyped nevi before "nevus").
RULES: list[tuple[str, str, bool]] = [
    (r"actinic keratos|\back\b|\bak\b", "ak", False),
    (r"basal cell carcinoma|\bbcc\b", "bcc", False),
    (r"bowen|squamous cell carcinoma|\bscc\b|\bsccis\b|\bbod\b", "scc", False),
    (r"keratoacanthoma", "scc", True),
    (r"melanoma|lentigo maligna|\bmel\b", "mel", False),
    (r"pseudo ?lymphoma", "exclude", False),
    # intermediate melanocytic tumours (severe atypia, Spitz, AIMP): neither benign nor melanoma
    (r"re ?excision|\baimp\b|intermediate melanocytic", "exclude", True),
    (r"kaposi|sebaceous carcinoma|dermatofibrosarcoma|mycosis fungoides|lymphoma|merkel|angiosarcoma",
     "exclude_rare_cancer", False),
    (r"seborr?h?o?eic keratos|\bsek\b", "benign", False),
    (r"dermatosis papulosa nigra", "benign", True),  # SK variant, common on darker skin
    (r"dermatofibroma(?!sarcoma)", "benign", False),
    # non-melanocytic "nevi": not pigmented-lesion lookalikes
    (r"nevus sebaceous|sebaceous nev|epidermal nev|becker|comedonicus|flammeus|port wine|nevus of ota|"
     r"nevus of ito|depigmentos|anemicus|achromicus|lipomatosus|verrucous epidermal",
     "exclude", True),
    (r"nevocytic|melanocytic|blue nev|halo nev|congenital nev|dysplastic|atypical nev|acral nev|"
     r"compound nev|junctional nev|intradermal nev|spitz|\bnev\b|\bnevi\b|\bnevus\b|\bnaevus\b",
     "benign", False),
    (r"solar lentig|lentigo simplex|\blentigo\b|lentigines", "benign", True),
    (r"pyogenic granuloma|granuloma pyogenic", "benign", True),
]

TRAINABLE = set(CLASSES["label_4class"])


def normalize(diagnosis) -> str:
    s = str(diagnosis).lower()
    s = re.sub(r"[-_/]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def map_diagnosis(diagnosis) -> tuple[str, bool, str]:
    """Return (label_4class, needs_review, matched_rule) for a raw diagnosis string."""
    text = normalize(diagnosis)
    for pattern, label, review in RULES:
        if re.search(pattern, text):
            return label, review, pattern
    return "exclude", False, "default"


def is_malignant(label) -> bool:
    return label in MALIGNANT


# ---- skin type -------------------------------------------------------------------
SOURCE_FST = {1, 2, 3}
TARGET_FST = {4, 5, 6}
DDI_BINS = {12: "I-II", 34: "III-IV", 56: "V-VI"}  # D3: DDI keeps its native bins


def clean_fst(value) -> float:
    """1-6 as float, else nan. Accepts 3, 3.0, "3" and DermaCon-IN's "FST 3"."""
    if isinstance(value, str):
        m = re.fullmatch(r"\s*(?:FST\s*)?(-?\d+(?:\.0)?)\s*", value, flags=re.I)
        value = m.group(1) if m else None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return math.nan
    return v if v in (1, 2, 3, 4, 5, 6) else math.nan


def fst_bucket(fst) -> str | None:
    v = clean_fst(fst)
    if math.isnan(v):
        return None
    if v <= 3:
        return "I-III"
    return "IV" if v == 4 else "V-VI"


# ---- Indian priors (D7) -------------------------------------------------------------
# Share of each cancer among cancers in Indian hospital series (normalised when used).
INDIAN_MIXES = {
    "consensus": {"bcc": 45, "scc": 40, "mel": 15},
    "malwa_punjab": {"bcc": 55, "scc": 37, "mel": 8},
    "central_india": {"bcc": 41, "scc": 30, "mel": 9},
    "manipur": {"bcc": 33, "scc": 44, "mel": 14},
}


def indian_prior(classes: list[str], base_prior, mix: str = "consensus") -> np.ndarray:
    """Keep the benign share of `base_prior`; split the malignant share by the Indian mix."""
    base = np.asarray(base_prior, dtype=float)
    base = base / base.sum()
    m = INDIAN_MIXES[mix]
    if "nmsc" in classes:
        m = {"nmsc": m["bcc"] + m["scc"], "mel": m["mel"]}
    weights = np.array([m.get(c, 0.0) for c in classes], dtype=float)
    weights = weights / weights.sum()
    benign = classes.index("benign")
    out = (1.0 - base[benign]) * weights
    out[benign] = base[benign]
    return out
