#!/usr/bin/env python3
"""留一缺陷家族：检验判别头（以及它和 PatchCore 的融合）能不能抓住没见过的缺陷。

    python3 head_loo_ae16.py --w-pc 0.7

每次留出一个家族的全部缺陷：
  正类 = 其余家族的缺陷
  负类 = 训练集 500 张良品
  阈值 = **其余家族**缺陷里的最低融合分（绝不看被留出的那些）
  然后统计被留出家族的缺陷有几张过线

对照：同一批留出缺陷上，PatchCore（削库后的 l3）用同样口径能抓几张。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def load_scores(path: str) -> dict[str, float]:
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    col = "raw_score" if "raw_score" in rows[0] else "fused"
    return {r["name"]: float(r[col]) for r in rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="scores/feats_ae16.npz")
    ap.add_argument("--pc-defect", default="scores/ae16_l3_pruned_defect.csv")
    ap.add_argument("--pc-test", default="scores/ae16_l3_pruned_test.csv")
    ap.add_argument("--train-list", default="scores/AE16bright_train_list.csv")
    ap.add_argument("--exclude", default=None, help="每行一个文件名，从良品里剔除")
    ap.add_argument("--C", type=float, default=0.2)
    ap.add_argument("--w-pc", type=float, default=0.7)
    args = ap.parse_args()

    z = np.load(args.feats, allow_pickle=False)
    G, GN = z["good"], z["good_names"]
    D, DN = z["defect"], z["defect_names"]
    seen = {r["name"] for r in csv.DictReader(open(args.train_list, encoding="utf-8-sig"))}
    if args.exclude:
        seen |= {l.strip() for l in open(args.exclude).read().splitlines() if l.strip()}
    mtr = np.array([n in seen for n in GN])
    pc_lib = load_scores(args.pc_defect)
    pc_all = load_scores(args.pc_test)

    fam = np.array([n[:4] for n in DN])
    sc = StandardScaler().fit(G[mtr])
    mu_p = float(np.mean([pc_all[n] for n in GN if n in seen]))
    sd_p = float(np.std([pc_all[n] for n in GN if n in seen])) + 1e-6

    rows = []
    for F in sorted(set(fam)):
        te = fam == F
        tr = ~te
        clf = LogisticRegression(C=args.C, class_weight="balanced", max_iter=3000)
        clf.fit(np.vstack([sc.transform(D[tr]), sc.transform(G[mtr])]),
                np.r_[np.ones(int(tr.sum())), np.zeros(int(mtr.sum()))])
        h = dict(zip(DN, clf.decision_function(sc.transform(D))))
        mu_h = float(np.mean([clf.decision_function(sc.transform(G[mtr]))]))
        sd_h = float(np.std(clf.decision_function(sc.transform(G[mtr])))) + 1e-6

        zh = {n: (h[n] - mu_h) / sd_h for n in h}
        zp = {n: (pc_lib[n] - mu_p) / sd_p for n in pc_lib}
        fz = {n: args.w_pc * zp[n] + (1 - args.w_pc) * zh[n] for n in pc_lib}

        tr_names = [n for n in DN if n[:4] != F]
        thr_h = min(zh[n] for n in tr_names)
        thr_p = min(zp[n] for n in tr_names)
        thr_f = min(fz[n] for n in tr_names)

        held = [n for n in DN if n[:4] == F]
        rows.append((F, len(held),
                     sum(1 for n in held if zp[n] >= thr_p),
                     sum(1 for n in held if zh[n] >= thr_h),
                     sum(1 for n in held if fz[n] >= thr_f)))

    print(f"{'留出家族':<8}{'缺陷':>5}{'PatchCore':>11}{'判别头':>9}{'融合':>7}")
    for F, n, a, b, c in rows:
        print(f"{F:<8}{n:>5}{a:>7}/{n:<3}{b:>5}/{n:<3}{c:>4}/{n:<3}")
    tot = sum(r[1] for r in rows)
    print(f"{'合计':<8}{tot:>5}{sum(r[2] for r in rows):>7}/{tot:<3}"
          f"{sum(r[3] for r in rows):>5}/{tot:<3}{sum(r[4] for r in rows):>4}/{tot:<3}")
    print("\n注：阈值只用**其余家族**的缺陷定，被留出的家族完全没进训练、没进阈值。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
