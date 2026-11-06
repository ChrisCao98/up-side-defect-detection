#!/usr/bin/env python3
"""Evaluate a PatchCore ckpt on the 142-defect + 2280-good protocol, using RAW scores.

    python3 eval_zeromiss.py \
        --ckpt result_train_E5DJ/Patchcore/F_my/v0/weights/lightning/model.ckpt \
        --defect-dir datasets_E5DJ/Defect \
        --good-dir test_images_E5DJ \
        --tag base_nosupp

Why raw scores
--------------
`predicate.py` writes `prediction.pred_score`, which is the post-processor's
normalized-and-clamped value. At least 26 of the 142 defects sit exactly at
1.0, so a zero-miss threshold computed on that column degenerates to 1.0
(observed: "threshold 1.0000, FP 1/2280"). Comparing models on that is invalid.

The post-processor is a Lightning *Callback*, so it rewrites the batch before
`predict_step`. Loading the model with `post_processor=False` therefore yields
the engine's exact preprocessing path while returning the un-normalized
distance. Verified to match the inverted normalized score to 0.0 on 3 images.

Metric definitions
------------------
zero-miss threshold = min(raw score over defects). A threshold just below the
    weakest defect lets every defect pass, so this is the highest threshold
    that still misses nothing.
FP at zero-miss      = #good with raw >= that threshold.
97% specificity      = raw threshold at the 97th percentile of good scores.

Outputs `scores/<tag>_defect.csv`, `scores/<tag>_test.csv`
(`name,raw_score,norm_score`) and appends a row to `scores/zeromiss_summary.csv`.
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore

ROOT = Path(__file__).resolve().parent
while not (ROOT / "scores").is_dir() and ROOT != ROOT.parent:   # 宿主机在 scripts/x/，容器里在 /app/
    ROOT = ROOT.parent


def read_norm_constants(ckpt: Path) -> tuple[float, float, float]:
    """Pull the post-processor's normalization constants straight from the ckpt."""
    try:
        obj = torch.load(ckpt, map_location="cpu", weights_only=True)
    except Exception:
        # 非 256 输入的 ckpt 会把 PreProcessor 对象存进 hyper_parameters，
        # weights_only 拒绝反序列化；这些 ckpt 都是本机自己训的，回退安全。
        obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = obj["state_dict"]
    get = lambda k: float(sd[k].reshape(-1)[0]) if torch.is_tensor(sd[k]) else float(sd[k])
    return (get("post_processor._image_threshold"),
            get("post_processor.image_min"),
            get("post_processor.image_max"))


def load_raw_model(ckpt: Path) -> Patchcore:
    """Load with the post-processor disabled so pred_score stays un-normalized."""
    model = Patchcore.load_from_checkpoint(str(ckpt), post_processor=False,
                                           map_location="cpu", strict=False)
    model.eval()
    return model


def score_dir(model: Patchcore, engine: Engine, data: Path, batch_size: int,
              limit: int | None) -> list[tuple[str, float]]:
    ds = PredictDataset(path=data)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=ds.collate_fn)
    preds = engine.predict(dataloaders=loader, model=model, return_predictions=True)
    rows: list[tuple[str, float]] = []
    for p in preds:
        # batch>1 时 image_path 是一串、pred_score 是一个向量；
        # 这里必须逐条配对，否则一个 batch 只记第一张（曾经就这么错过一次）。
        paths = p.image_path
        paths = list(paths) if isinstance(paths, (list, tuple)) else [paths]
        scores = p.pred_score
        scores = (scores.reshape(-1) if torch.is_tensor(scores)
                  else torch.tensor([float(scores)]))
        if len(scores) != len(paths):
            raise RuntimeError(f"batch 内张数不匹配: {len(paths)} vs {len(scores)}")
        for path, sc in zip(paths, scores):
            rows.append((Path(path).name, float(sc)))
            if limit and len(rows) >= limit:
                return rows[:limit]
    return rows


def to_norm(raw: np.ndarray, consts: tuple[float, float, float]) -> np.ndarray:
    thr, lo, hi = consts
    return np.clip((raw - thr) / max(hi - lo, 1e-9) + 0.5, 0.0, 1.0)


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney), average ranks for ties."""
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv), dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    uniq, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(uniq))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--defect-dir", required=True)
    ap.add_argument("--good-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default="scores")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="cap images per set (debug)")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        raise SystemExit(f"✗ ckpt 不存在: {ckpt}")

    consts = read_norm_constants(ckpt)
    print(f"模型: {ckpt}")
    print(f"  归一化常数: threshold={consts[0]:.4f} min={consts[1]:.4f} max={consts[2]:.4f}")

    model = load_raw_model(ckpt)
    engine = Engine(default_root_dir=tempfile.mkdtemp(prefix="eval_zeromiss_"))

    defect = score_dir(model, engine, Path(args.defect_dir), args.batch_size, args.limit)
    print(f"  缺陷 {len(defect)} 张")
    good = score_dir(model, engine, Path(args.good_dir), args.batch_size, args.limit)
    print(f"  良品 {len(good)} 张")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for suffix, rows in (("defect", defect), ("test", good)):
        with (out / f"{args.tag}_{suffix}.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "raw_score", "norm_score"])
            raws = np.array([r for _, r in rows])
            norms = to_norm(raws, consts)
            for (n, r), s in zip(rows, norms):
                w.writerow([n, f"{r:.6f}", f"{s:.6f}"])

    d = np.array([r for _, r in defect])
    g = np.array([r for _, r in good])
    auc = auroc(d, g)
    thr = float(d.min())
    fp = int((g >= thr).sum())
    thr97 = float(np.percentile(g, 97))
    n_clamp = int((to_norm(d, consts) >= 0.999999).sum())

    print("\n" + "=" * 62)
    print(args.tag)
    print("=" * 62)
    print(f"AUROC(raw)                  {auc:.4f}")
    print(f"缺陷 raw  min {d.min():.4f}  中位 {np.median(d):.4f}  max {d.max():.4f}")
    print(f"良品 raw  min {g.min():.4f}  中位 {np.median(g):.4f}  p97 {thr97:.4f}  max {g.max():.4f}")
    print(f"零漏检阈值 (raw)            {thr:.4f}   (归一化后 {to_norm(np.array([thr]), consts)[0]:.4f})")
    print(f"  该阈值误报                {fp:,}/{len(g):,}  ({fp / len(g):.2%})")
    print(f"97% 特异度阈值 (raw)        {thr97:.4f}")
    print(f"缺陷被 clamp 到 1.0 的张数  {n_clamp}/{len(d)}")

    summary = out / "zeromiss_summary.csv"
    new = not summary.exists()
    with summary.open("a", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["tag", "ckpt", "n_defect", "n_good", "auroc",
                        "zeromiss_thr_raw", "fp", "fp_rate", "thr97_raw", "n_clamped"])
        w.writerow([args.tag, str(ckpt), len(d), len(g), f"{auc:.6f}",
                    f"{thr:.6f}", fp, f"{fp / len(g):.6f}",
                    f"{thr97:.6f}", n_clamp])
    print(f"\n✓ {out / f'{args.tag}_defect.csv'}")
    print(f"✓ {out / f'{args.tag}_test.csv'}")
    print(f"✓ {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
