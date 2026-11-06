#!/usr/bin/env python3
"""汇总 A-E_16/bright 组的实验结果（含剔除训练样本后的误报）。

    python3 summarize_ae16.py --tags ae16_l2 ae16_l3

口径
----
阈值：缺陷库 149 张里的最低原始分（口径 B），全组共用一条线。
误报：整组 7,920 张里**没进训练集**的那 7,420 张，等于测试集。
     训练过的 500 张单独列出（它们本来就是见过的，不能算泛化）。
AUROC：149 张缺陷 vs 7,420 张未见良品。

融合：可再调 fuse()，把两个 tag 的原始分行级 z 加权。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def load(path: Path) -> dict[str, float]:
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    col = "raw_score" if "raw_score" in rows[0] else "fused"
    return {r["name"]: float(r[col]) for r in rows}


def auroc(pos, neg) -> float:
    allv = np.concatenate([neg, pos])
    order = allv.argsort(kind="mergesort")
    rk = np.empty(len(allv))
    rk[order] = np.arange(1, len(allv) + 1)
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            rk[order[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    return float((rk[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def metrics(defect: dict, good: dict, seen: set[str], z: bool = False,
            exclude: set[str] | None = None) -> dict:
    thr = min(defect.values())
    exclude = exclude or set()
    unseen = {n: v for n, v in good.items() if n not in seen and n not in exclude}
    seenv = {n: v for n, v in good.items() if n in seen}
    fp = sum(1 for v in unseen.values() if v >= thr)
    uv = np.array(list(unseen.values()))
    dv = np.array(list(defect.values()))
    # 与地板无关的指标：固定误报率，看漏几张缺陷
    miss = {p: int((dv < np.percentile(uv, 100 - p)).sum()) for p in (3, 2, 1)}
    return {
        "thr": thr, "n_def": len(defect), "n_unseen": len(unseen),
        "fp": fp, "fp_rate": fp / max(len(unseen), 1),
        "fp_all_rate": sum(1 for v in good.values() if v >= thr) / max(len(good), 1),
        "auroc": auroc(dv, uv),
        "seen_fp": sum(1 for v in seenv.values() if v >= thr),
        "miss3": miss[3], "miss2": miss[2], "miss1": miss[1],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--train-list", default="scores/AE16bright_train_list.csv")
    ap.add_argument("--out", default="scores/ae16_summary.csv")
    ap.add_argument("--exclude", default=None,
                    help="每行一个文件名，统计误报时从良品里剔除（已知缺陷）")
    args = ap.parse_args()

    seen = {r["name"] for r in csv.DictReader(open(args.train_list, encoding="utf-8-sig"))}
    excl = ({l.strip() for l in open(args.exclude).read().splitlines() if l.strip()}
            if args.exclude else set())
    if excl:
        print(f"剔除 {len(excl)} 张已知缺陷：{args.exclude}")
    results = {}
    print(f"{'配置':<20}{'零漏检FP':>9}{'误报率':>8}{'AUROC':>8}"
          f"{'3%漏':>6}{'2%漏':>6}{'1%漏':>6}")
    for tag in args.tags:
        d = load(Path(f"scores/{tag}_defect.csv"))
        g = load(Path(f"scores/{tag}_test.csv"))
        m = metrics(d, g, seen, exclude=excl)
        results[tag] = m
        print(f"{tag:<20}{m['fp']:>9}{m['fp_rate']:>8.2%}{m['auroc']:>8.4f}"
              f"{m['miss3']:>6}{m['miss2']:>6}{m['miss1']:>6}")

    with open(args.out, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tag", "n_defect", "n_unseen", "fp", "fp_rate", "fp_all_rate",
                    "auroc", "seen_fp", "threshold", "miss3", "miss2", "miss1"])
        for tag, m in results.items():
            w.writerow([tag, m["n_def"], m["n_unseen"], m["fp"], f"{m['fp_rate']:.4f}",
                        f"{m['fp_all_rate']:.4f}", f"{m['auroc']:.4f}",
                        m["seen_fp"], f"{m['thr']:.4f}",
                        m["miss3"], m["miss2"], m["miss1"]])
    print(f"✓ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
