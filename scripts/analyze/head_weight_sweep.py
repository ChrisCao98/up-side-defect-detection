#!/usr/bin/env python3
"""对所有已跑过的组，逐个扫判别头的融合权重，找出每组自己最优点。

    python3 scripts/analyze/head_weight_sweep.py

对每个 (tag, 训练名单)：
  1. 读缓存的 3072 维特征 + 削库后的 PC 分数
  2. 训练判别头（正类 149 张缺陷，负类本组训练良品）
  3. 两个分数各自 z 标准化，扫 w = 0.00 … 1.00（步长 0.05）
  4. 每个 w 下阈值 = 149 张缺陷的最低融合分，数未见良品里的越线数

输出 scores/head_weight_sweep.csv：每组每权重一行。
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

GROUPS = [
    ("A-D_18", "ad18p20", "AD18p20", None),
    ("A-E_12/dark", "ae12d", "AE12dark", None),
    ("A-D_12/dark", "ad12d", "AD12dark", None),
    ("A-G_16/dark", "ag16d", "AG16dark", None),
    ("A-E_16/dark", "ae16d", "AE16dark", None),
    ("A-C_22", "ac22", "AC22", None),
    ("A-E_19/dark", "ae19d", "AE19dark", None),
    ("A-D_12/bright", "ad12b", "AD12bright", None),
    ("A-E_12/bright", "ae12b", "AE12bright", "scores/ae12_exclude.txt"),
    ("A-G_16/bright", "ag16b", "AG16bright", None),
    ("A-E_16/bright", "ae16", "AE16bright", None),
    ("A-E_19/bright", "ae19b5", "AE19bright350", None),
    ("A-D_20", "class1_ad20r", "class1_AD20r", None),
    ("class2 A-E_16", "class2_ae16", "class2_AE16", None),
]


def load(p: Path) -> dict[str, float]:
    rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
    return {r["name"]: float(r["raw_score"]) for r in rows}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--out", default="scores/head_weight_sweep.csv")
    args = ap.parse_args()
    out = Path(args.out)
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "tag", "n_unseen", "weight", "fp", "rate"])
        for label, tag, lst, excl_path in GROUPS:
            feats = Path(f"scores/feats_{tag}.npz")
            if not feats.exists():
                print(f"✗ 缺 {feats}"); continue
            z = np.load(feats, allow_pickle=False)
            G, GN = z["good"], z["good_names"]
            D, DN = z["defect"], z["defect_names"]
            seen = {r["name"] for r in csv.DictReader(
                open(f"scores/{lst}_train_list.csv", encoding="utf-8-sig"))}
            if excl_path:
                seen |= {l.strip() for l in open(excl_path).read().splitlines() if l.strip()}
            mtr = np.array([n in seen for n in GN])

            sc = StandardScaler().fit(G[mtr])
            clf = LogisticRegression(C=0.2, class_weight="balanced", max_iter=3000)
            clf.fit(np.vstack([sc.transform(D), sc.transform(G[mtr])]),
                    np.r_[np.ones(len(D)), np.zeros(int(mtr.sum()))])
            hd = dict(zip(DN, clf.decision_function(sc.transform(D))))
            hg = dict(zip(GN, clf.decision_function(sc.transform(G))))

            pcd = load(Path(f"scores/{tag}_l3_pruned_defect.csv"))
            pcg = load(Path(f"scores/{tag}_l3_pruned_test.csv"))

            trh = np.array([hg[n] for n in GN if n in seen])
            trp = np.array([pcg[n] for n in GN if n in seen])
            mh, sh = trh.mean(), trh.std() + 1e-9
            mp, sp = trp.mean(), trp.std() + 1e-9
            zhd = {n: (v - mh) / sh for n, v in hd.items()}
            zhg = {n: (v - mh) / sh for n, v in hg.items() if n not in seen}
            zpd = {n: (v - mp) / sp for n, v in pcd.items()}
            zpg = {n: (v - mp) / sp for n, v in pcg.items() if n not in seen}

            for wpc in np.round(np.arange(0, 1.0001, args.step), 4):
                fd = {n: wpc * zpd[n] + (1 - wpc) * zhd[n] for n in zpd}
                fg = {n: wpc * zpg[n] + (1 - wpc) * zhg[n] for n in zpg}
                thr = min(fd.values())
                fp = sum(1 for v in fg.values() if v >= thr)
                w.writerow([label, tag, len(zpg), f"{wpc:.2f}", fp, f"{fp/len(zpg):.4f}"])
            print(f"✓ {label}", flush=True)
    print(f"✓ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
