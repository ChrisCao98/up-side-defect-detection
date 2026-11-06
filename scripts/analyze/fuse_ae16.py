#!/usr/bin/env python3
"""图像级 z 加权融合两个 PatchCore 模型，并扫权重。

    python3 fuse_ae16.py --a ae16_l2 --b ae16_l3
    python3 fuse_ae16.py --a ae16_l2_pruned --b ae16_l3_pruned

score = w·z_a + (1-w)·z_b，z 用**训练良品**（500 张里进了 train split 的那部分）
的均值和标准差。阈值仍取缺陷库 149 张里的最低融合分，误报只统计没进训练集的图。
"""

from __future__ import annotations

import argparse
import csv

import numpy as np

from summarize_ae16 import load, metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--train-list", default="scores/AE16bright_train_list.csv")
    ap.add_argument("--exclude", default=None, help="每行一个文件名，从良品里剔除")
    args = ap.parse_args()

    seen = {r["name"] for r in csv.DictReader(open(args.train_list, encoding="utf-8-sig"))}
    if args.exclude:
        seen |= {l.strip() for l in open(args.exclude).read().splitlines() if l.strip()}

    def zscore(tag: str):
        d = load(__import__("pathlib").Path(f"scores/{tag}_defect.csv"))
        g = load(__import__("pathlib").Path(f"scores/{tag}_test.csv"))
        tr = np.array([v for n, v in g.items() if n in seen])
        mu, sd = tr.mean(), tr.std() + 1e-6
        return ({n: (v - mu) / sd for n, v in d.items()},
                {n: (v - mu) / sd for n, v in g.items()})

    da, ga = zscore(args.a)
    db, gb = zscore(args.b)

    print(f"{'w':>5}{'零漏检FP':>10}{'误报率':>8}{'AUROC':>8}")
    best = None
    for w in np.round(np.arange(0, 1.0001, 0.05), 2):
        d = {n: w * da[n] + (1 - w) * db[n] for n in da}
        g = {n: w * ga[n] + (1 - w) * gb[n] for n in ga}
        m = metrics(d, g, seen)
        print(f"{w:>5.2f}{m['fp']:>10}{m['fp_rate']:>8.2%}{m['auroc']:>8.4f}")
        if best is None or m["fp_rate"] < best[1]:
            best = (w, m["fp_rate"], m)
    w, rate, m = best
    print(f"\n最优 w={w:.2f}（{args.a} 权重）：FP {m['fp']}/{m['n_unseen']} = {rate:.2%}，"
          f"AUROC {m['auroc']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
