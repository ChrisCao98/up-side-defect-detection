#!/usr/bin/env python3
"""随机留出缺陷，模拟「缺陷库已覆盖所有类型」时的召回。

    python3 head_random_holdout.py --frac 0.2 --repeats 10

与 head_loo_ae16.py 的区别：那边整族留出，模拟「全新的缺陷家族」；这边按家族
分层随机留 20%，模拟「新缺陷能在库里找到同类」。两者给出的是上下界。

每一折：
  正类 = 80% 的缺陷，负类 = 训练集 500 张良品
  阈值 = 这 80% 里的最低分
  统计留出的 20% 有几张过线；同时用同一阈值统计良品误报
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict

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
    ap.add_argument("--frac", type=float, default=0.2)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--exclude", default=None, help="每行一个文件名，从良品里剔除")
    ap.add_argument("--C", type=float, default=0.2)
    ap.add_argument("--w-pc", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
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

    sc = StandardScaler().fit(G[mtr])
    mu_p = float(np.mean([pc_all[n] for n in GN if n in seen]))
    sd_p = float(np.std([pc_all[n] for n in GN if n in seen])) + 1e-6
    zp = {n: (pc_lib[n] - mu_p) / sd_p for n in pc_lib}
    zp_g = {n: (pc_all[n] - mu_p) / sd_p for n in pc_all}
    unseen_g = [n for n in GN if n not in seen]

    by_fam = defaultdict(list)
    for i, n in enumerate(DN):
        by_fam[n[:4]].append(i)

    rng = np.random.default_rng(args.seed)
    tot_held = tot_h = tot_p = tot_f = 0
    fp_h = fp_p = fp_f = 0
    for _ in range(args.repeats):
        held = []
        for fam, idx in by_fam.items():
            k = max(1, round(len(idx) * args.frac))
            held += list(rng.choice(idx, size=k, replace=False))
        held = np.array(sorted(held))
        tr = np.setdiff1d(np.arange(len(DN)), held)

        clf = LogisticRegression(C=args.C, class_weight="balanced", max_iter=3000)
        clf.fit(np.vstack([sc.transform(D[tr]), sc.transform(G[mtr])]),
                np.r_[np.ones(len(tr)), np.zeros(int(mtr.sum()))])
        h = dict(zip(DN, clf.decision_function(sc.transform(D))))
        hm = float(np.mean(clf.decision_function(sc.transform(G[mtr]))))
        hs = float(np.std(clf.decision_function(sc.transform(G[mtr])))) + 1e-6
        zh = {n: (h[n] - hm) / hs for n in h}
        zh_g = dict(zip(GN, (clf.decision_function(sc.transform(G)) - hm) / hs))

        tr_names = [DN[i] for i in tr]
        thr_h = min(zh[n] for n in tr_names)
        thr_p = min(zp[n] for n in tr_names)
        thr_f = min(args.w_pc * zp[n] + (1 - args.w_pc) * zh[n] for n in tr_names)

        hl = [DN[i] for i in held]
        tot_held += len(hl)
        tot_p += sum(1 for n in hl if zp[n] >= thr_p)
        tot_h += sum(1 for n in hl if zh[n] >= thr_h)
        tot_f += sum(1 for n in hl if args.w_pc * zp[n] + (1 - args.w_pc) * zh[n] >= thr_f)

        fp_p += sum(1 for n in unseen_g if zp_g[n] >= thr_p)
        fp_h += sum(1 for n in unseen_g if zh_g[n] >= thr_h)
        fp_f += sum(1 for n in unseen_g
                    if args.w_pc * zp_g[n] + (1 - args.w_pc) * zh_g[n] >= thr_f)

    R = args.repeats
    print(f"随机留出 {args.frac:.0%}，重复 {R} 次（每次阈值只由训练那部分缺陷定）")
    print(f"  PatchCore  召回 {tot_p/R:6.1f}/{tot_held/R:.0f}   "
          f"良品误报 {fp_p/R:6.1f}/{len(unseen_g)} = {fp_p/R/len(unseen_g):.2%}")
    print(f"  判别头     召回 {tot_h/R:6.1f}/{tot_held/R:.0f}   "
          f"良品误报 {fp_h/R:6.1f}/{len(unseen_g)} = {fp_h/R/len(unseen_g):.2%}")
    print(f"  融合 {args.w_pc:.2f}/{1-args.w_pc:.2f} 召回 {tot_f/R:6.1f}/{tot_held/R:.0f}   "
          f"良品误报 {fp_f/R:6.1f}/{len(unseen_g)} = {fp_f/R/len(unseen_g):.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
