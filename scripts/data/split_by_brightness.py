#!/usr/bin/env python3
"""把一个签名组按明暗拆成两档，输出可直接建目录的分组表。

    python3 split_by_brightness.py --group A-E_16

做法：每个产品取其前缀的平均灰度中位数作为亮度，产品按亮度排序，
在**最大间隙**处切两刀 → bright / dark。写 scores/signature_groups_split.csv，
其余签名组原样保留，只有目标组被替换成两行，product 列写成 "<组>/bright"。

不写任何图片，建目录由 build_product_layout.py 负责。
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
while not (ROOT / "scores").is_dir() and ROOT != ROOT.parent:   # 宿主机在 scripts/x/，容器里在 /app/
    ROOT = ROOT.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="A-E_16")
    ap.add_argument("--groups", default="scores/product_groups.csv")
    ap.add_argument("--signatures", default="scores/signature_groups.csv")
    ap.add_argument("--brightness", default="scores/prefix_brightness.csv")
    ap.add_argument("--out", default="scores/signature_groups_split.csv")
    args = ap.parse_args()

    prod = list(csv.DictReader(open(ROOT / args.groups, encoding="utf-8-sig")))
    sigs = list(csv.DictReader(open(ROOT / args.signatures, encoding="utf-8-sig")))
    bri = {r["prefix"]: {k: float(v) for k, v in r.items()
                         if k not in ("prefix", "") and v}
           for r in csv.DictReader(open(ROOT / args.brightness, encoding="utf-8-sig"))}

    target = next((r for r in sigs if r["product"] == args.group), None)
    if target is None:
        raise SystemExit(f"✗ 签名表里没有 {args.group}")
    members = target["products"].split(";")

    vals = {}
    for r in prod:
        if r["product"] not in members:
            continue
        ps = r["prefixes"].split(";")
        vals[r["product"]] = (statistics.median(
            [statistics.mean(bri[p].values()) for p in ps]), r)
    order = sorted(vals, key=lambda k: vals[k][0])
    scores = [vals[k][0] for k in order]
    gaps = [(scores[i + 1] - scores[i], i) for i in range(len(scores) - 1)]
    gap, cut = max(gaps)
    bright = order[cut + 1:]
    dark = order[:cut + 1]

    print(f"{args.group}: {len(order)} 个产品，最大间隙 "
          f"{scores[cut]:.0f} → {scores[cut+1]:.0f}（差 {gap:.0f}）")
    for name, ps in (("bright", bright), ("dark", dark)):
        n = sum(int(vals[p][1]["n_images"]) for p in ps)
        print(f"  {name:<7}{len(ps)} 个产品  {n:>6} 张  {', '.join(ps)}")

    out_rows = []
    for r in sigs:
        if r["product"] != args.group:
            out_rows.append(dict(r))
            continue
        for name, ps in (("bright", bright), ("dark", dark)):
            prefs = [p for prod_name in ps for p in vals[prod_name][1]["prefixes"].split(";")]
            out_rows.append({
                "product": f"{args.group}/{name}",
                "n_products": len(ps),
                "products": ";".join(ps),
                "n_prefixes": len(prefs),
                "n_images": sum(int(vals[p][1]["n_images"]) for p in ps),
                "per_letter": r["per_letter"],
                "prefixes": ";".join(prefs),
            })

    out = ROOT / args.out
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sigs[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"✓ {out.relative_to(ROOT)}  {len(out_rows)} 行")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
