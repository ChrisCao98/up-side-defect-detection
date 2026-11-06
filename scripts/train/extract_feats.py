#!/usr/bin/env python3
"""抽取每张图的图像级特征（冻结骨干，layer3），用于训练判别式分类头。

    python3 extract_feats.py --ckpt <l3 ckpt> \
        --good-dir test_images_E5DJ --defect-dir eval_defect149 \
        --out scores/feats_e5dj.npz

特征 = layer3 特征图 (1024,H,W) 在空间维上做 mean / max / std 池化后拼接（3072 维）。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

from anomalib.models import Patchcore

_root = Path(__file__).resolve().parent          # 宿主机在 scripts/x/ 下，容器里挂在 /app/
while not (_root / "visualize_layers.py").exists() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))
from visualize_layers import load_image  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--good-dir", required=True)
    ap.add_argument("--defect-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = Patchcore.load_from_checkpoint(args.ckpt, post_processor=False,
                                       map_location="cpu", strict=False).eval().to(dev)

    def run(files, label):
        feats, names = [], []
        for i, p in enumerate(files, 1):
            x, _ = load_image(p, dev)
            with torch.no_grad():
                f = m.model.feature_extractor(x)
                pooled = {k: m.model.feature_pooler(v) for k, v in f.items()}
                e = m.model.generate_embedding(pooled).squeeze(0)   # (C,H,W)
                v = torch.cat([e.mean((1, 2)), e.amax((1, 2)), e.std((1, 2))]).cpu().numpy()
            feats.append(v.astype(np.float32)); names.append(p.name)
            if i % 500 == 0:
                print(f"  {label} {i}/{len(files)}", flush=True)
        return np.stack(feats), np.array(names)

    good = sorted(Path(args.good_dir).rglob("*.jpg"))
    defect = sorted(Path(args.defect_dir).rglob("*.jpg"))
    print(f"良品 {len(good)}  缺陷 {len(defect)}")
    G, gn = run(good, "good")
    D, dn = run(defect, "defect")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, good=G, good_names=gn, defect=D, defect_names=dn)
    print(f"✓ {out}  good {G.shape}  defect {D.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
