"""Assign every manifest row to a split and freeze the result in data/splits/.

    python -m skincancer.data.make_splits

Pools: main = every Fitzpatrick17k and PAD-UFES-20 image labelled cancer or not cancer,
split into train / val / test; ak_side = actinic keratoses; ddi_test / dermacon_test /
dermacon_other = test-only datasets; MIDAS is split by patient into midas_train (may be
added to training), midas_val (never trained on; used to choose between models) and
midas_test. A near-duplicate group that touches a
test-only set loses its trainable members.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_data_config
from ..labels import TUMOUR

TRAINABLE_POOLS = {"main", "ak_side"}
TEST_ONLY_POOLS = {"ddi_test", "dermacon_test", "dermacon_other", "midas_test"}
MADE_HERE = {"ak_patient_in_other_pool", "dup_of_test_only", "dup_cross_pool"}


def assign_pool(row) -> str:
    if isinstance(row["exclude_reason"], str):
        return "excluded"
    ds, label = row["dataset"], row["label_4class"]
    if ds == "ddi":
        return "ddi_test"
    if ds == "midas":
        return "midas_test" if label in TUMOUR else "excluded"   # biopsy-proven tumours only
    if ds == "dermacon_in":
        if label in TUMOUR:
            return "dermacon_test"
        return "dermacon_other" if label == "exclude" else "excluded"
    if label == "ak":
        return "ak_side"
    return "main" if row["label_binary"] in (0.0, 1.0) else "excluded"


def resolve_group_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # ak_side is a side analysis: when a patient also has a lesion in another pool, drop
    # only their AK images rather than losing the training images too.
    pools = df.groupby("group_id")["pool"].agg(set)
    ak_shared = pools[pools.map(lambda s: "ak_side" in s and len(s & (TRAINABLE_POOLS - {"ak_side"})) > 0)].index
    m0 = (df["pool"] == "ak_side") & df["group_id"].isin(ak_shared)
    df.loc[m0, ["pool", "exclude_reason"]] = ["excluded", "ak_patient_in_other_pool"]

    pools = df.groupby("group_id")["pool"].agg(set)
    touches_test = pools[pools.map(lambda s: bool(s & TEST_ONLY_POOLS))].index
    cross = pools[pools.map(lambda s: len(s & TRAINABLE_POOLS) > 1)].index
    trainable = df["pool"].isin(TRAINABLE_POOLS)
    m1 = trainable & df["group_id"].isin(touches_test)
    m2 = trainable & df["group_id"].isin(cross) & ~m1
    df.loc[m1, ["pool", "exclude_reason"]] = ["excluded", "dup_of_test_only"]
    df.loc[m2, ["pool", "exclude_reason"]] = ["excluded", "dup_cross_pool"]
    print(f"group conflicts: {m0.sum()} ak_patient_in_other_pool, {m1.sum()} dup_of_test_only, "
          f"{m2.sum()} dup_cross_pool")
    return df


def grouped_split(df: pd.DataFrame, names: list[str], fractions: list[float], seed: int,
                  strata: pd.Series) -> pd.Series:
    """Group split stratified by `strata` (here the fine label x dataset). Greedy: groups in random order (largest
    first), each to the split furthest below its per-class target. Unlike
    StratifiedGroupKFold it works when a class has only a handful of images."""
    frac = np.asarray(fractions, float)
    assert abs(frac.sum() - 1) < 1e-6, fractions
    rng = np.random.default_rng(seed)
    counts = pd.crosstab(df["group_id"], strata)
    groups = counts.index.to_numpy()[rng.permutation(len(counts))]
    groups = sorted(groups, key=lambda g: -counts.loc[g].sum())  # stable: random among equal sizes
    target = frac[:, None] * counts.sum(axis=0).to_numpy()[None, :]
    current = np.zeros_like(target)
    assign = {}
    for g in groups:
        v = counts.loc[g].to_numpy()
        deficit = (target - current) / np.maximum(target, 1)
        size_deficit = frac - current.sum(1) / max(current.sum(), 1)
        k = int(np.argmax(deficit @ v / v.sum() + 1e-3 * size_deficit))
        current[k] += v
        assign[g] = names[k]
    return df["group_id"].map(assign)


def make(manifest: pd.DataFrame, cfg: dict, groups: pd.DataFrame | None = None):
    df = manifest.copy()
    # start from build_manifest's exclusions: drop the ones an earlier run of this step added
    df["exclude_reason"] = df["exclude_reason"].astype(object)   # all-empty reads back as float
    df.loc[df["exclude_reason"].isin(MADE_HERE), "exclude_reason"] = np.nan
    if groups is not None:
        df = df.drop(columns="group_id").merge(groups[["image_id", "group_id"]], on="image_id", how="left")
        df["group_id"] = df["group_id"].fillna(df["patient_id"]).fillna(df["image_id"])
    df["pool"] = df.apply(assign_pool, axis=1)
    df = resolve_group_conflicts(df)
    sp = cfg["splits"]
    df["split"] = df["pool"]
    main = df["pool"] == "main"
    if main.any():
        strata = df.loc[main, "label_fine"] + " " + df.loc[main, "dataset"]
        df.loc[main, "split"] = grouped_split(df[main], ["train", "val", "test"], sp["fractions"], sp["seed"], strata)
    midas = df["pool"] == "midas_test"
    if midas.any() and sp.get("midas"):   # by patient, after the duplicate rules above treated all of it as a test set
        df.loc[midas, "split"] = grouped_split(df[midas], ["midas_train", "midas_val", "midas_test"], sp["midas"],
                                               sp["seed"], df.loc[midas, "label_fine"])
    return df.drop(columns="pool")


EXTERNAL_TEST = {"ddi_test", "dermacon_test", "dermacon_other", "midas_test"}


def check_no_leakage(df: pd.DataFrame) -> None:
    """No group may span two splits, unless every split it touches is an external test set
    (e.g. a DermaCon-IN patient with a cancer and a non-cancer image): nothing trains or
    selects on those, so sharing a patient between them cannot leak."""
    used = df[df["split"] != "excluded"]
    splits = used.groupby("group_id")["split"].agg(set)
    bad = splits[splits.map(lambda s: len(s) > 1 and not s <= EXTERNAL_TEST)]
    assert bad.empty, f"{len(bad)} groups appear in more than one split: {bad.head().to_dict()}"


def write(df: pd.DataFrame, cfg: dict) -> None:
    out = Path(cfg["splits_dir"])
    out.mkdir(parents=True, exist_ok=True)
    for split, g in df[df["split"] != "excluded"].groupby("split"):
        g[["image_id"]].to_csv(out / f"{split}.csv", index=False)
    for old in out.glob("*.csv"):   # splits of an earlier design
        if old.stem not in set(df["split"]):
            old.unlink()
    by_class = pd.crosstab(df["split"], df["label_cancer"].fillna("no label"), margins=True)
    by_type = pd.crosstab(df["split"], df["label_fine"], margins=True)
    main = df[df["split"].isin(["train", "val", "test"])]
    by_set = pd.crosstab([main["split"], main["dataset"]], main["label_cancer"], margins=True)
    (out / "SUMMARY.md").write_text(
        "# Split summary\n\n```\n" + by_class.to_string() + "\n```\n\n## By type of lesion\n\n```\n"
        + by_type.to_string() + "\n```\n\n## train / val / test by dataset\n\n```\n" + by_set.to_string()
        + "\n```\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-config", default="configs/data.yaml")
    args = ap.parse_args(argv)
    cfg = load_data_config(args.data_config)
    manifest = pd.read_csv(cfg["manifest"])
    groups = pd.read_csv(cfg["groups"]) if Path(cfg["groups"]).exists() else None
    if groups is None:
        print("WARNING: no groups.csv — run dedup first; splitting by patient/image only")
    df = make(manifest, cfg, groups)
    check_no_leakage(df)
    df.to_csv(cfg["manifest"], index=False)
    write(df, cfg)
    print(pd.crosstab(df["split"], df["label_fine"], margins=True).to_string())


if __name__ == "__main__":
    main()
