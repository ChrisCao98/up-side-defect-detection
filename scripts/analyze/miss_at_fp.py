#!/usr/bin/env python3
"""在**固定误报率**下数漏检张数——不受「最弱缺陷」影响的口径。

    python3 scripts/analyze/miss_at_fp.py --tag ae19b --l2 ae19b_l2_pruned \
        --l3 ae19b_l3_pruned --train-list scores/AE19bright_train_list.csv

对每一种打分方式：把阈值定在未见良品的 (1-fp) 分位，数 149 张缺陷里有几张低于它。
阈值不再由缺陷决定，所以比「零漏检误报」稳定得多。

输出三块：
  1. 单个模型（l2 / l3）
  2. l2+l3 加权融合，扫权重
  3. l3 + 判别头，扫权重
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def load(path: Path) -> dict[str, float]:
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    col = "raw_score" if "raw_score" in rows[0] else "fused"
    return {r["name"]: float(r[col]) for r in rows}


def zscores(defect: dict, good: dict, train_names: set[str]):
    tr = np.array([v for n, v in good.items() if n in train_names])
    mu, sd = tr.mean(), tr.std() + 1e-9
    return ({n: (v - mu) / sd for n, v in defect.items()},
            {n: (v - mu) / sd for n, v in good.items()})


def misses(defect: dict, good: dict, fp: float) -> tuple[int, float]:
    thr = float(np.percentile(list(good.values()), 100 * (1 - fp)))
    return sum(1 for v in defect.values() if v < thr), thr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--l2", required=True, help="l2 分数组名，如 ae19b_l2_pruned")
    ap.add_argument("--l3", required=True)
    ap.add_argument("--train-list", required=True)
    ap.add_argument("--feats", default=None, help="判别头特征；不给就跳过第三块")
    ap.add_argument("--fp", type=float, default=0.01)
    ap.add_argument("--exclude", default=None, help="每行一个文件名，从良品里剔除")
    ap.add_argument("--C", type=float, default=0.2)
    args = ap.parse_args()

    train = {r["name"] for r in csv.DictReader(open(args.train_list, encoding="utf-8-sig"))}
    if args.exclude:
        train |= {l.strip() for l in open(args.exclude).read().splitlines() if l.strip()}
    d2, g2 = load(Path(f"scores/{args.l2}_defect.csv")), load(Path(f"scores/{args.l2}_test.csv"))
    d3, g3 = load(Path(f"scores/{args.l3}_defect.csv")), load(Path(f"scores/{args.l3}_test.csv"))
    # 只用未见良品算分位数
    # z 的均值/标准差取自**训练良品**；分位数取自**未见良品**
    z2d, z2g_all = zscores(d2, g2, train)
    z3d, z3g_all = zscores(d3, g3, train)
    z2g = {n: v for n, v in z2g_all.items() if n not in train}
    z3g = {n: v for n, v in z3g_all.items() if n not in train}

    print(f"误报率固定 {args.fp:.0%}，缺陷 {len(d3)} 张，未见良品 {len(z3g)} 张\n")

    print(f"{'配置':<26}{'阈值':>9}{'漏检':>7}")
    m, t = misses(z3d, z3g, args.fp)   # 必须同为 z 尺度
    print(f"{'l3 单独':<26}{t:>9.2f}{m:>5}/{len(d3)}")
    m, t = misses(z2d, z2g, args.fp)
    print(f"{'l2 单独':<26}{t:>9.2f}{m:>5}/{len(d3)}")

    print(f"\n{'l2+l3 融合 w(l2)':<26}{'阈值':>9}{'漏检':>7}")
    best = None
    for w in np.round(np.arange(0, 1.0001, 0.05), 2):
        df = {n: w * z2d[n] + (1 - w) * z3d[n] for n in z2d}
        gf = {n: w * z2g[n] + (1 - w) * z3g[n] for n in z2g}
        m, t = misses(df, gf, args.fp)
        if best is None or m < best[1]:
            best = (w, m, t)
        print(f"{w:>26.2f}{t:>9.2f}{m:>5}/{len(d3)}")
    print(f"  → 最优 w={best[0]:.2f}，漏 {best[1]}/{len(d3)}")

    if args.feats:
        z = np.load(args.feats, allow_pickle=False)
        G, GN = z["good"], z["good_names"]
        D, DN = z["defect"], z["defect_names"]
        mtr = np.array([n in train for n in GN])
        keep = np.array([n not in train for n in GN])
        sc = StandardScaler().fit(G[mtr])
        clf = LogisticRegression(C=args.C, class_weight="balanced", max_iter=3000)
        clf.fit(np.vstack([sc.transform(D), sc.transform(G[mtr])]),
                np.r_[np.ones(len(D)), np.zeros(int(mtr.sum()))])
        hd = dict(zip(DN, clf.decision_function(sc.transform(D))))
        hg_all = dict(zip(GN, clf.decision_function(sc.transform(G))))
        zhd, zhg_all = zscores(hd, hg_all, train)
        zhg = {n: v for n, v in zhg_all.items() if n not in train}

        print(f"\n{'l3 + 判别头 w(pc)':<26}{'阈值':>9}{'漏检':>7}")
        for w in (0.0, 0.5, 0.6, 0.7, 0.75, 0.85, 0.9, 0.95, 1.0):
            df = {n: w * z3d[n] + (1 - w) * zhd[n] for n in z3d}
            gf = {n: w * z3g[n] + (1 - w) * zhg[n] for n in z3g}
            m, t = misses(df, gf, args.fp)
            print(f"{w:>26.2f}{t:>9.2f}{m:>5}/{len(d3)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
