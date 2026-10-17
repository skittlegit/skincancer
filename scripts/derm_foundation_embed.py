"""Derm Foundation embeddings (6144-d) for every image in the evaluation splits (E1-FM, D15).

Runs in the separate TensorFlow venv, so TF never touches the PyTorch environment:
    .venv-tf/Scripts/python scripts/derm_foundation_embed.py
Needs Hugging Face access to google/derm-foundation (gated; Health AI Developer Foundations
terms) and a token from `huggingface-cli login`. Writes data/features/derm_foundation/part_*.npz
(float16, gitignored: embeddings of DDI images are derivatives and stay local). Resumable.
"""
from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

# Training and selection splits first, the large non-cancer DermaCon-IN set last.
SPLITS = ["source_train", "source_val", "target_val", "target_train", "source_test", "target_test",
          "ak_side", "pad_dark_test", "ddi_test", "dermacon_test", "midas_test", "dermacon_other"]
SIZE = 448


def load_cfg(path: str) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return cfg


def png_bytes(path: Path) -> bytes:
    """Resize the short side to 448 and centre-crop 448 x 448 (the model's input size)."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = SIZE / min(w, h)
    img = img.resize((max(SIZE, round(w * s)), max(SIZE, round(h * s))), Image.BICUBIC)
    w, h = img.size
    l, t = (w - SIZE) // 2, (h - SIZE) // 2
    buf = io.BytesIO()
    img.crop((l, t, l + SIZE, t + SIZE)).save(buf, "PNG")
    return buf.getvalue()


def done_ids(out: Path) -> set[str]:
    ids = set()
    for f in out.glob("part_*.npz"):
        ids.update(np.load(f, allow_pickle=True)["image_id"].tolist())
    return ids


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--splits", nargs="+", default=SPLITS)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--part-size", type=int, default=512, help="images per output file")
    ap.add_argument("--threads", type=int, default=0, help="TF intra-op threads (0 = all cores)")
    ap.add_argument("--ids", help="CSV with an image_id column: embed these manifest rows instead of --splits (e.g. data/gate/nontumour.csv, D26)")
    args = ap.parse_args(argv)

    import tensorflow as tf
    from huggingface_hub import snapshot_download
    if args.threads:
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    cfg = load_cfg(args.data_config)
    out = Path(cfg["features_dir"]) / "derm_foundation"
    out.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(cfg["manifest"])
    if args.ids:
        todo = manifest[manifest["image_id"].isin(pd.read_csv(args.ids)["image_id"])].sort_values("image_id")
    else:
        order = {s: i for i, s in enumerate(args.splits)}
        todo = manifest[manifest["split"].isin(order)].copy()
        todo["_o"] = todo["split"].map(order)
        todo = todo.sort_values(["_o", "image_id"], kind="stable")
    have = done_ids(out)
    todo = todo[~todo["image_id"].isin(have)]
    print(f"{len(have)} cached, {len(todo)} to embed")
    if todo.empty:
        return

    model = tf.saved_model.load(snapshot_download("google/derm-foundation"))
    infer = model.signatures["serving_default"]
    root = Path(cfg["data_root"])
    part = len(list(out.glob("part_*.npz")))
    ids, embs, t0, n0 = [], [], time.time(), len(have)

    def example(b: bytes) -> bytes:
        feat = {"image/encoded": tf.train.Feature(bytes_list=tf.train.BytesList(value=[b]))}
        return tf.train.Example(features=tf.train.Features(feature=feat)).SerializeToString()

    def flush():
        nonlocal part, ids, embs
        if ids:
            np.savez(out / f"part_{part:04d}.npz", image_id=np.array(ids, dtype=object),
                     embedding=np.vstack(embs).astype(np.float16))
            part, ids, embs = part + 1, [], []

    rows = list(todo[["image_id", "path"]].itertuples(index=False))
    for i in range(0, len(rows), args.batch):
        batch = rows[i:i + args.batch]
        serial = [example(png_bytes(root / r.path)) for r in batch]
        e = infer(inputs=tf.constant(serial))["embedding"].numpy().reshape(len(batch), -1)
        ids += [r.image_id for r in batch]
        embs.append(e)
        if len(ids) >= args.part_size:
            flush()
        done = i + len(batch)
        if done % (args.batch * 20) == 0 or done == len(rows):
            rate = done / (time.time() - t0)
            print(f"{n0 + done}/{n0 + len(rows)}  {rate:.2f} img/s  eta {(len(rows) - done) / rate / 60:.0f} min",
                  flush=True)
    flush()


if __name__ == "__main__":
    main()
