#!/usr/bin/env python3
"""在产品内部按曝光再分一层。

    python3 exposure_groups.py                # 测量 + 聚类，写 scores/product_exposure_groups.csv
    python3 exposure_groups.py --tol 0.30     # 放宽到 30%（默认 25%）
    python3 exposure_groups.py --reuse        # 复用已缓存的亮度测量

为什么产品内部还要分
--------------------
产品是按「字母数 + 每字母张数」定的，那是**拍摄结构**；同一个结构下仍可能有
几次拍摄的曝光不一样。实测 KA1Z 的第 5 位 T 组比 H/S 组暗 35%，W64L 40 个
前缀里 16 个亮 30%，按产品训模型会把两套曝光混进同一个记忆库。

怎么分
------
每个前缀量出 A–G 各字母的平均灰度（每格抽 4 张），取对数后在产品内部做
complete-linkage 层次聚类，距离用 chebyshev（任意字母上都不超过 tol）。
complete-linkage 保证同一组里任意两个前缀在每个字母上都差不到 tol，
不会像 single-linkage 那样一路串下去。

输出
----
scores/product_exposure_groups.csv   一行一个 (产品, 曝光组)，列含 prefixes
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
while not (ROOT / "scores").is_dir() and ROOT != ROOT.parent:   # 宿主机在 scripts/x/，容器里在 /app/
    ROOT = ROOT.parent
SRC = ROOT / "test_images_part1" / "class_3"
CACHE = ROOT / "scores/prefix_brightness.csv"


def measure(prefixes: list[str], letters: set[str], per_cell: int, src: Path,
            seed: int = 0) -> dict:
    random.seed(seed)
    out = {}
    for i, p in enumerate(prefixes, 1):
        row = {}
        for L in sorted(letters):
            fs = sorted((src / p / L).glob("*.jpg"))
            if not fs:
                continue
            pick = random.sample(fs, min(per_cell, len(fs)))
            vals = []
            for f in pick:
                a = np.asarray(Image.open(f).convert("L").resize((128, 128)), dtype=np.float32)
                vals.append(float(a.mean()))
            row[L] = float(np.mean(vals))
        out[p] = row
        if i % 200 == 0:
            print(f"  测量 {i}/{len(prefixes)}", flush=True)
    return out


def load_or_measure(prefixes, letters, per_cell, reuse, src: Path) -> dict:
    if reuse and CACHE.exists():
        rows = list(csv.DictReader(open(CACHE, encoding="utf-8-sig")))
        got = {r["prefix"]: {k: float(v) for k, v in r.items()
                             if k not in ("prefix", "") and v} for r in rows}
        if all(p in got for p in prefixes):
            print(f"复用 {CACHE.relative_to(ROOT)}（{len(got)} 个前缀）")
            return got
    print(f"测量 {len(prefixes)} 个前缀的亮度…")
    got = measure(prefixes, letters, per_cell, src)
    with CACHE.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        cols = sorted(letters)
        w.writerow(["prefix"] + cols)
        for p in sorted(got):
            w.writerow([p] + [f"{got[p][L]:.2f}" if L in got[p] else "" for L in cols])
    print(f"✓ {CACHE.relative_to(ROOT)}")
    return got


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="scores/product_groups.csv")
    ap.add_argument("--out", default="scores/product_exposure_groups.csv")
    ap.add_argument("--tol", type=float, default=0.25, help="同一组内允许的相对亮度差")
    ap.add_argument("--per-cell", type=int, default=4)
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--src", help="图像源目录，默认 test_images_part1/class_3")
    args = ap.parse_args()
    src = Path(args.src) if args.src else SRC

    from scipy.cluster.hierarchy import fcluster, linkage

    rows = list(csv.DictReader(open(ROOT / args.groups, encoding="utf-8-sig")))
    counts = {r["prefix"]: int(r["n_images"])
              for r in csv.DictReader(open(ROOT / "scores/product_prefixes.csv",
                                           encoding="utf-8-sig"))}
    letters = set()
    for r in rows:
        letters |= {k for k, _ in (kv.split("=") for kv in r["per_letter"].split())}
    allp = [p for r in rows for p in r["prefixes"].split(";")]
    bright = load_or_measure(allp, letters, args.per_cell, args.reuse, src)

    out_rows = []
    n_multi = 0
    for r in rows:
        prod = r["product"]
        letters_p = [k for k, _ in (kv.split("=") for kv in r["per_letter"].split())]
        prefixes = r["prefixes"].split(";")
        fallback = {L: float(np.median([bright[p][L] for p in prefixes if L in bright[p]]))
                    for L in letters_p}
        X = np.array([[np.log(bright[p].get(L, fallback[L])) for L in letters_p]
                      for p in prefixes], dtype=np.float64)
        if len(prefixes) == 1:
            labels = np.zeros(1, dtype=int)
        else:
            Z = linkage(X, method="complete", metric="chebyshev")
            labels = fcluster(Z, t=np.log(1 + args.tol), criterion="distance")
        groups: dict[int, list[str]] = defaultdict(list)
        for p, lab in zip(prefixes, labels):
            groups[int(lab)].append(p)
        ordered = sorted(groups.values(), key=lambda ps: -len(ps))
        if len(ordered) > 1:
            n_multi += 1
        for gi, ps in enumerate(ordered, 1):
            med = {L: float(np.median([bright[p][L] for p in ps if L in bright[p]]))
                   for L in letters_p}
            out_rows.append({
                "product": prod, "group": f"g{gi}", "n_prefixes": len(ps),
                "n_images": sum(counts[p] for p in ps),
                "per_letter": " ".join(f"{L}={med[L]:.0f}" for L in letters_p),
                "prefixes": ";".join(sorted(ps)),
            })

    out = ROOT / args.out
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["product", "group", "n_prefixes", "n_images",
                                           "per_letter", "prefixes"])
        w.writeheader()
        w.writerows(sorted(out_rows, key=lambda r: (r["product"], r["group"])))

    print(f"\n{len(rows)} 个产品 → {len(out_rows)} 个 (产品, 曝光组)；"
          f"{n_multi} 个产品被拆开")
    for r in sorted(out_rows, key=lambda r: (r["product"], r["group"])):
        if any(x["product"] == r["product"] and x is not r for x in out_rows):
            print(f"  {r['product']:<8}{r['group']}  {r['n_prefixes']:>3} 前缀  "
                  f"{r['n_images']:>5} 张  {r['per_letter']}")
    print(f"\n✓ {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
