"""Assign every manifest row to a split and freeze the result in data/splits/ (D1, D4, D9).

    python -m skincancer.data.make_splits

Pools: source = Fitzpatrick17k + PAD-UFES-20 at FST I-III; target = Fitzpatrick17k at
FST IV-VI; pad_dark_test = PAD-UFES-20 at FST IV-VI; ak_side = actinic keratoses;
ddi_test / dermacon_test = test-only datasets. A near-duplicate group that touches a
test-only set, or spans two trainable pools, loses its trainable members.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_data_config
from ..labels import SOURCE_FST, TARGET_FST, TRAINABLE

TRAINABLE_POOLS = {"source", "target", "pad_dark_test", "ak_side"}
TEST_ONLY_POOLS = {"ddi_test", "dermacon_test", "dermacon_other"}


def assign_pool(row) -> str:
    if isinstance(row["exclude_reason"], str):
        return "excluded"
    ds, label = row["dataset"], row["label_4class"]
    if ds == "ddi":
        return "ddi_test"
    if ds == "dermacon_in":
        if label in TRAINABLE:
            return "dermacon_test"
        return "dermacon_other" if label == "exclude" else "excluded"
    if label == "ak":
        return "ak_side"
    if label not in TRAINABLE:
        return "excluded"
    fst = int(row["fst"])
    if fst in SOURCE_FST:
        return "source"
    if fst in TARGET_FST:
        return "target" if ds == "fitzpatrick17k" else "pad_dark_test"
    return "excluded"


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


def grouped_split(df: pd.DataFrame, names: list[str], fractions: list[float], seed: int) -> pd.Series:
    """Group split stratified by label_4class. Greedy: groups in random order (largest
    first), each to the split furthest below its per-class target. Unlike
    StratifiedGroupKFold it works when a class has only a handful of images."""
    frac = np.asarray(fractions, float)
    assert abs(frac.sum() - 1) < 1e-6, fractions
    rng = np.random.default_rng(seed)
    counts = pd.crosstab(df["group_id"], df["label_4class"])
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


def nested_fractions(train: pd.DataFrame, fractions: list[int], seed: int) -> dict[int, list[str]]:
    """Nested subsets (10 ⊂ 25 ⊂ 50) of target_train, by group, stratified by class."""
    rng = np.random.default_rng(seed)
    gl = train.groupby("group_id").agg(label=("label_4class", lambda s: s.mode().iloc[0]),
                                       n=("image_id", "size"))
    order = []
    for _, g in gl.groupby("label"):
        ids = g.index.to_numpy()
        rng.shuffle(ids)
        sizes = g.loc[ids, "n"].to_numpy()
        order.append((ids, np.cumsum(sizes) / sizes.sum()))
    out = {}
    for f in sorted(fractions):
        keep = set()
        for ids, cum in order:
            k = int(np.searchsorted(cum, f / 100.0)) + 1   # at least one group per class
            keep.update(ids[:k])
        out[f] = train.loc[train["group_id"].isin(keep), "image_id"].tolist()
    return out


def make(manifest: pd.DataFrame, cfg: dict, groups: pd.DataFrame | None = None):
    df = manifest.copy()
    if groups is not None:
        df = df.drop(columns="group_id").merge(groups[["image_id", "group_id"]], on="image_id", how="left")
        df["group_id"] = df["group_id"].fillna(df["patient_id"]).fillna(df["image_id"])
    df["pool"] = df.apply(assign_pool, axis=1)
    df = resolve_group_conflicts(df)
    sp = cfg["splits"]
    df["split"] = df["pool"]
    src = df["pool"] == "source"
    tgt = df["pool"] == "target"
    if src.any():
        df.loc[src, "split"] = grouped_split(df[src], ["source_train", "source_val", "source_test"],
                                             sp["source"], sp["seed"])
    if tgt.any():
        df.loc[tgt, "split"] = grouped_split(df[tgt], ["target_train", "target_val", "target_test"],
                                             sp["target"], sp["seed"])
    n_dark = (df["split"] == "pad_dark_test").sum()
    if 0 < n_dark < sp["min_pad_dark"]:
        df.loc[df["split"] == "pad_dark_test", ["split", "exclude_reason"]] = ["excluded", "pad_dark_too_small"]
    fractions = {}
    target_train = df[df["split"] == "target_train"]
    if len(target_train):
        for s in sp["e5_seeds"]:
            fractions[s] = nested_fractions(target_train, sp["e5_fractions"], seed=sp["seed"] + 1000 + s)
    return df.drop(columns="pool"), fractions


EXTERNAL_TEST = {"ddi_test", "dermacon_test", "dermacon_other", "pad_dark_test"}


def check_no_leakage(df: pd.DataFrame) -> None:
    """No group may span two splits, unless every split it touches is an external test set
    (e.g. a DermaCon-IN patient with a cancer and a non-cancer image): nothing trains or
    selects on those, so sharing a patient between them cannot leak."""
    used = df[df["split"] != "excluded"]
    splits = used.groupby("group_id")["split"].agg(set)
    bad = splits[splits.map(lambda s: len(s) > 1 and not s <= EXTERNAL_TEST)]
    assert bad.empty, f"{len(bad)} groups appear in more than one split: {bad.head().to_dict()}"


def write(df: pd.DataFrame, fractions: dict, cfg: dict) -> None:
    out = Path(cfg["splits_dir"])
    (out / "e5").mkdir(parents=True, exist_ok=True)
    for split, g in df[df["split"] != "excluded"].groupby("split"):
        g[["image_id"]].to_csv(out / f"{split}.csv", index=False)
    for seed, fr in fractions.items():
        for f, ids in fr.items():
            pd.DataFrame({"image_id": ids}).to_csv(out / "e5" / f"target_train_f{f}_s{seed}.csv", index=False)
    table = pd.crosstab(df["split"], df["label_4class"], margins=True)
    (out / "SUMMARY.md").write_text("# Split summary\n\n```\n" + table.to_string() + "\n```\n",
                                    encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-config", default="configs/data.yaml")
    args = ap.parse_args(argv)
    cfg = load_data_config(args.data_config)
    manifest = pd.read_csv(cfg["manifest"])
    groups = pd.read_csv(cfg["groups"]) if Path(cfg["groups"]).exists() else None
    if groups is None:
        print("WARNING: no groups.csv — run dedup first; splitting by patient/image only")
    df, fractions = make(manifest, cfg, groups)
    check_no_leakage(df)
    df.to_csv(cfg["manifest"], index=False)
    write(df, fractions, cfg)
    print(pd.crosstab(df["split"], df["label_4class"], margins=True).to_string())


if __name__ == "__main__":
    main()
