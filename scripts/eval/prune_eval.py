#!/usr/bin/env python3
"""Fast memory-bank pruning evaluation (single pass per model).

    python3 prune_eval.py --ckpt <ckpt> --defect-dir <dir> --good-dir <dir> \
        --defect-names <7 names> --k 20 --tag v2

Prints, for a bank pruned of the k nearest rows at each bottleneck defect's
highest-scoring patch: AUROC, zero-miss FP, and misses at 3%/2%/1% FP, next to
the unpruned baseline.

Why this is fast: `prune_sim.py` cached top-50 per patch, which is O(P*N) per
image and hours for a 113,971-row bank. Here we compute the full distance row
once, take the argmin, and only recompute the masked minimum for the handful of
patches whose nearest neighbour was actually pruned.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
import torch.nn.functional as F

from anomalib.models import Patchcore

_root = Path(__file__).resolve().parent          # 宿主机在 scripts/x/ 下，容器里挂在 /app/
while not (_root / "visualize_layers.py").exists() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))
from visualize_layers import load_image  # noqa: E402


def embedding(model, x):
    feats = model.model.feature_extractor(x)
    pooled = {k: model.model.feature_pooler(v) for k, v in feats.items()}
    emb = None
    for layer in model.model.layers:
        e = pooled[layer]
        emb = e if emb is None else torch.cat(
            [emb, F.interpolate(e, size=emb.shape[-2:], mode="bilinear")], 1)
    return emb.squeeze(0)


def patches(model, p, device):
    x, _ = load_image(p, device)
    with torch.no_grad():
        e = embedding(model, x)
    h = e.shape[1]
    return e.permute(1, 2, 0).reshape(-1, e.shape[0]).float(), h


def image_score(model, bank, p, prune: torch.Tensor, device):
    qf, h = patches(model, p, device)
    with torch.no_grad():
        d = torch.cdist(qf, bank)
        min1, arg1 = d.min(dim=1)
        if prune.numel():
            affected = torch.isin(arg1, prune)
            if affected.any():
                rows = d[affected].clone()
                rows[:, prune] = float("inf")
                min1[affected] = rows.min(dim=1).values
    return float(min1.max())


def auroc(dv, gv):
    dv = np.asarray(dv); gv = np.asarray(gv)
    allv = np.concatenate([dv, gv]); o = allv.argsort()
    rk = np.empty(len(allv)); rk[o] = np.arange(1, len(allv) + 1)
    u, inv, c = np.unique(allv, return_inverse=True, return_counts=True)
    s = np.zeros(len(u)); np.add.at(s, inv, rk); rk = (s / c)[inv]
    return float((rk[:len(dv)].sum() - len(dv) * (len(dv) + 1) / 2) / (len(dv) * len(gv)))


def metrics(dv, gv):
    dv = np.asarray(dv); gv = np.asarray(gv)
    thr = dv.min(); fp = int((gv >= thr).sum()); gs = np.sort(gv)[::-1]
    miss = lambda t: int((dv <= gs[int(np.floor(t * len(gv)))]).sum())
    return fp / len(gv), miss(.03), miss(.02), miss(.01)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--defect-dir", required=True)
    ap.add_argument("--good-dir", required=True)
    ap.add_argument("--defect-names", nargs="+", required=True)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--save-indices", default=None,
                    help="把削除的行号写成 apply_prune.py 能直接吃的 JSON")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Patchcore.load_from_checkpoint(args.ckpt, post_processor=False,
                                           map_location="cpu", strict=False).eval().to(device)
    bank = model.model.memory_bank.float()
    print(f"[{args.tag}] bank {tuple(bank.shape)}")

    by_name = {p.name: p for p in Path(args.defect_dir).rglob("*.jpg")}
    per_defect = {}
    prune = set()
    for n in args.defect_names:
        qf, _ = patches(model, by_name[n], device)
        with torch.no_grad():
            d = torch.cdist(qf, bank)
            br = int(d.min(dim=1).values.argmax())
            idx = [int(v) for v in d[br].topk(args.k, largest=False).indices]
        per_defect[n] = idx
        prune |= set(idx)
    prune_t = torch.tensor(sorted(prune), device=device, dtype=torch.long)
    print(f"  削除 {len(prune)} 行 ({len(prune)/bank.shape[0]:.2%})")

    if args.save_indices:
        json.dump({
            "ckpt": str(args.ckpt),
            "bank_shape": [int(bank.shape[0]), int(bank.shape[1])],
            "k": args.k,
            "n_pruned": len(prune),
            "indices": sorted(int(i) for i in prune),
            "per_defect": per_defect,
        }, open(args.save_indices, "w", encoding="utf-8"), indent=1)
        print(f"  ✓ 行号写入 {args.save_indices}")

    dfiles = sorted(Path(args.defect_dir).rglob("*.jpg"))
    gfiles = sorted(Path(args.good_dir).rglob("*.jpg"))
    dv, gv = [], []
    for i, p in enumerate(dfiles + gfiles, 1):
        s = image_score(model, bank, p, prune_t, device)
        (dv if i <= len(dfiles) else gv).append(s)
        if i % 500 == 0:
            print(f"  {i}/{len(dfiles)+len(gfiles)}", flush=True)

    fr, m3, m2, m1 = metrics(dv, gv)
    print(f"  AUROC {auroc(dv, gv):.4f} | 零漏检FP {fr:.2%} | 3%/2%/1% 漏检 {m3}/{m2}/{m1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
