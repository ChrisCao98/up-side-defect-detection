#!/usr/bin/env python3
"""Write a real ckpt with the pruned rows removed from the memory bank.

    python3 apply_prune.py \
        --src result_train_E5DJ_sweep_l3/Patchcore/F_my/v0/weights/lightning/model.ckpt \
        --indices scores/prune_indices_k20.json \
        --dst result_train_E5DJ_sweep_l3_pruned/Patchcore/F_my/v0/weights/lightning/model.ckpt

Only `state_dict['model.memory_bank']` is modified; every other key (weights,
post-processor normalization constants, hyper-parameters) is copied verbatim,
so the downstream `predicate*.py` path needs no change.

The post-processor constants are NOT recalibrated: they are a global affine map,
so a pruned bank simply shifts raw -> normalized monotonically. The raw
threshold that `predicate.py`'s hard-coded 0.4 corresponds to is printed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--indices", default="scores/prune_indices_k20.json")
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    meta = json.loads(Path(args.indices).read_text(encoding="utf-8"))
    drop = sorted(set(int(i) for i in meta["indices"]))

    ck = torch.load(src, map_location="cpu", weights_only=False)
    mb = ck["state_dict"]["model.memory_bank"]
    n = mb.shape[0]
    if meta.get("bank_shape") and list(meta["bank_shape"]) != [int(n), int(mb.shape[1])]:
        raise SystemExit(f"✗ bank 形状不符: ckpt {tuple(mb.shape)} vs 清单 {meta['bank_shape']}")
    if drop and (drop[-1] >= n or drop[0] < 0):
        raise SystemExit(f"✗ 索引越界: {drop[0]}..{drop[-1]} vs bank {n}")

    keep = torch.tensor([i for i in range(n) if i not in set(drop)], dtype=torch.long)
    ck["state_dict"]["model.memory_bank"] = mb[keep]

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, dst)

    sd = ck["state_dict"]
    T = float(sd["post_processor._image_threshold"].reshape(-1)[0])
    lo = float(sd["post_processor.image_min"].reshape(-1)[0])
    hi = float(sd["post_processor.image_max"].reshape(-1)[0])
    raw04 = (0.4 - 0.5) * (hi - lo) + T
    print(f"✓ {dst}")
    print(f"  bank {n:,} -> {keep.numel():,}  (删除 {len(drop)} 行)")
    print(f"  归一化常数 threshold={T:.4f} min={lo:.4f} max={hi:.4f}")
    print(f"  predicate.py 的 0.4 阈值 ≈ raw {raw04:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
