#!/usr/bin/env python3
"""A-E_16/bright 组的判别式头：训练、单用、与 PatchCore 融合。

    docker run ... --entrypoint python up_patchcore-v1 /app/head_ae16.py --w-pc 0.7

正类 = 缺陷库 149 张；负类 = 训练集 500 张良品。
评估 = 整组里没进训练集的 7,420 张（阈值仍取缺陷库最低分）。
三个分数各自用自己的阈值，不混用 —— 混用会在别的项目上出错（见 head_sweep 的教训）。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def load_scores(path: Path) -> dict[str, float]:
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    col = "raw_score" if "raw_score" in rows[0] else "fused"
    return {r["name"]: float(r[col]) for r in rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="scores/feats_ae16.npz")
    ap.add_argument("--pc-defect", default="scores/ae16_l3_defect.csv")
    ap.add_argument("--pc-test", default="scores/ae16_l3_test.csv")
    ap.add_argument("--train-list", default="scores/AE16bright_train_list.csv")
    ap.add_argument("--C", type=float, default=0.2)
    ap.add_argument("--w-pc", type=float, default=0.7)
    ap.add_argument("--exclude", default=None, help="每行一个文件名，从良品里剔除")
    ap.add_argument("--out-prefix", default="scores/ae16_head",
                    help="两个输出 CSV 的前缀，多组实验要分开")
    args = ap.parse_args()

    z = np.load(args.feats, allow_pickle=False)
    G, GN = z["good"], z["good_names"]
    D, DN = z["defect"], z["defect_names"]
    seen = {r["name"] for r in csv.DictReader(open(args.train_list, encoding="utf-8-sig"))}
    if args.exclude:
        seen |= {l.strip() for l in open(args.exclude).read().splitlines() if l.strip()}
    mtr = np.array([n in seen for n in GN])
    print(f"特征 良品 {G.shape}  缺陷 {D.shape}  其中训练 {int(mtr.sum())} 张")

    sc = StandardScaler().fit(G[mtr])
    clf = LogisticRegression(C=args.C, class_weight="balanced", max_iter=3000)
    clf.fit(np.vstack([sc.transform(D), sc.transform(G[mtr])]),
            np.r_[np.ones(len(D)), np.zeros(int(mtr.sum()))])
    h_lib = dict(zip(DN, clf.decision_function(sc.transform(D))))
    h_g = dict(zip(GN, clf.decision_function(sc.transform(G))))

    pc_lib = load_scores(Path(args.pc_defect))
    pc_all = load_scores(Path(args.pc_test))

    mu_p = np.mean([pc_all[n] for n in GN if n in seen])
    sd_p = np.std([pc_all[n] for n in GN if n in seen]) + 1e-6
    mu_h = np.mean([h_g[n] for n in GN if n in seen])
    sd_h = np.std([h_g[n] for n in GN if n in seen]) + 1e-6

    thr_h = min((h_lib[n] - mu_h) / sd_h for n in h_lib)
    thr_p = min((pc_lib[n] - mu_p) / sd_p for n in pc_lib)
    fused = lambda n: args.w_pc * (pc_all[n] - mu_p) / sd_p + (1 - args.w_pc) * (h_g[n] - mu_h) / sd_h
    thr_f = min(args.w_pc * (pc_lib[n] - mu_p) / sd_p
                + (1 - args.w_pc) * (h_lib[n] - mu_h) / sd_h for n in pc_lib)

    unseen = [n for n in GN if n not in seen]
    def fp(pred) -> int:
        return sum(1 for n in unseen if pred(n))

    hp = lambda n: (h_g[n] - mu_h) / sd_h >= thr_h
    pp = lambda n: (pc_all[n] - mu_p) / sd_p >= thr_p
    fpp = lambda n: fused(n) >= thr_f

    print(f"\n未见良品 {len(unseen)} 张（阈值来自缺陷库最低分，口径 B）")
    print(f"  PatchCore l3   FP {fp(pp):>5}  {fp(pp)/len(unseen):>7.2%}")
    print(f"  判别式头       FP {fp(hp):>5}  {fp(hp)/len(unseen):>7.2%}")
    print(f"  融合 {args.w_pc:.1f}/{1-args.w_pc:.1f}     FP {fp(fpp):>5}  {fp(fpp)/len(unseen):>7.2%}")

    out = Path(f"{args.out_prefix}_fp.csv")
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "fused", "z_pc", "z_head"])
        for n in sorted(unseen, key=lambda k: -fused(k))[:50]:
            w.writerow([n, f"{fused(n):.3f}", f"{(pc_all[n]-mu_p)/sd_p:.2f}",
                        f"{(h_g[n]-mu_h)/sd_h:.2f}"])
    print(f"✓ {out}（前 50 名）")

    # 全量分数，供 stage_fp_review.py 按阈值挑图
    allp = Path(f"{args.out_prefix}_all.csv")
    with allp.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "fused", "z_pc", "z_head"])
        for n in sorted(unseen, key=lambda k: -fused(k)):
            w.writerow([n, f"{fused(n):.4f}", f"{(pc_all[n]-mu_p)/sd_p:.2f}",
                        f"{(h_g[n]-mu_h)/sd_h:.2f}"])
    print(f"✓ {allp}（{len(unseen)} 张全量）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
