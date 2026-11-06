#!/usr/bin/env python3
"""按「stripe（前 4 位）+ 各字母图片数」给 class_3 分产品。

    python3 group_products.py                                  # 读 class3_index/manifest.csv
    python3 group_products.py --root <目录> --out-dir <目录>    # 直接扫盘，不写回默认 CSV

产品判定规则
------------
1. 9 位前缀的前 4 位相同 → 同一个 **stripe**（如 E59R、XA1W）。
2. stripe 内部，**字母数相同、每个字母的图片数也相同** → 同一个产品类型。
   XA1W 里 A–E 各 12 张的 34 个前缀和 A–E 各 16 张的 19 个前缀是两个产品。
3. 例外只有一种：某个批次的最后几个字母**拍少了**（如 XA1V 的 G 只有 13 张）。
   判定为「截断」并归回主签名，条件是字母集合完全相同、且每个字母张数都 ≤ 主签名。
   只要字母集合不同（少一个字母）或多出张数，就判为不同产品 —— 那是换了拍摄配方。

输出
----
scores/product_groups.csv    一行一个产品
scores/product_prefixes.csv  一行一个 9 位前缀，附所属产品
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter, defaultdict
from pathlib import Path

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def load_counts(manifest: Path) -> dict[str, Counter]:
    pref: dict[str, Counter] = defaultdict(Counter)
    with manifest.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            pref[r["prefix"]][r["letter"]] += 1
    return pref


def scan_root(root: Path) -> dict[str, Counter]:
    """<root>/<9位前缀>/<字母>/*.jpg 的结果，和 manifest 等价。"""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    pref: dict[str, Counter] = defaultdict(Counter)
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        for ld in d.iterdir():
            if ld.is_dir():
                n = sum(1 for f in ld.iterdir() if f.suffix.lower() in exts)
                if n:
                    pref[d.name][ld.name] += n
    return pref


def sig_of(c: Counter) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(c.items()))


def is_truncated_of(minor: tuple[tuple[str, int], ...],
                    major: tuple[tuple[str, int], ...],
                    tol: float = 0.9) -> bool:
    """minor 只是 major 拍少了最后几个字母（偶尔缺，不拆）。

    三个条件：字母集合相同、每个字母张数都 ≤ 主签名、总量不低于主签名的 tol。
    第三条是必须的 —— 只靠前两条，XA1W0（A–E×12）会被当成 XA1WX（A–E×16）
    的"截断版"并进去，而它是另一套拍法（亮度 25 vs 120）。实测真正的截断
    都在 95% 以上，真拆开的都在 75% 以下，tol=0.9 落在空档里。
    """
    dm, dM = dict(minor), dict(major)
    if set(dm) != set(dM) or any(dm[k] > dM[k] for k in dm):
        return False
    return sum(dm.values()) >= tol * sum(dM.values())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="class3_index/manifest.csv")
    ap.add_argument("--root", help="直接扫这个目录而不是 manifest")
    ap.add_argument("--defect-dir", default="eval_defect149")
    ap.add_argument("--out-dir", default="scores")
    ap.add_argument("--tag", default="", help="输出文件名后缀，如 _molding")
    args = ap.parse_args()

    pref = scan_root(Path(args.root)) if args.root else load_counts(Path(args.manifest))
    if not pref:
        print(f"没有扫到图片：{args.root or args.manifest}")
        return 1
    defect_prefixes: Counter = Counter()
    d = Path(args.defect_dir)
    if d.is_dir():
        for p in d.rglob("*.jpg"):
            defect_prefixes[p.name[:9]] += 1

    # stripe → 签名 → 前缀
    by_stripe: dict[str, dict[tuple, list[str]]] = defaultdict(lambda: defaultdict(list))
    for p, c in pref.items():
        by_stripe[p[:4]][sig_of(c)].append(p)

    products: list[dict] = []
    for stripe in sorted(by_stripe):
        sigs = by_stripe[stripe]
        # 主签名 = 图片总数最多的那个
        major = max(sigs, key=lambda s: sum(n for _, n in s) * len(sigs[s]))
        groups = [(major, list(sigs[major]), [])]
        for s, ps in sigs.items():
            if s == major:
                continue
            if is_truncated_of(s, major):
                groups[0][1].extend(ps)          # 截断批次归回主产品
                groups[0][2].extend(ps)
            else:
                groups.append((s, list(ps), []))
        for g in groups:
            g[1].sort(); g[2].sort()
        groups.sort(key=lambda g: -sum(dict(g[0]).values()) * len(g[1]))

        fives = [{p[4] for p in g[1]} for g in groups]
        named_by_5 = len(groups) > 1 and all(len(f) == 1 for f in fives)
        for i, (s, ps, folded) in enumerate(groups):
            pid = (stripe + next(iter(fives[i]))) if named_by_5 else \
                  (stripe if len(groups) == 1 else f"{stripe}_g{i+1}")
            products.append({
                "product": pid, "stripe": stripe, "sig": s, "prefixes": ps,
                "folded": folded,
                "n_letters": len(s), "n_prefixes": len(ps),
                "n_images": sum(sum(pref[p].values()) for p in ps),
                "n_defects": sum(defect_prefixes.get(p, 0) for p in ps),
                "n_defects_stripe": sum(v for k, v in defect_prefixes.items()
                                        if k.startswith(stripe)),
            })

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"product_groups{args.tag}.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["product", "stripe", "n_prefixes", "n_images", "n_letters",
                    "per_letter", "n_defects", "n_defects_stripe", "n_folded", "prefixes"])
        for r in products:
            w.writerow([r["product"], r["stripe"], r["n_prefixes"], r["n_images"],
                        r["n_letters"],
                        " ".join(f"{k}={v}" for k, v in r["sig"]),
                        r["n_defects"], r["n_defects_stripe"],
                        len(r["folded"]), ";".join(r["prefixes"])])

    with (out / f"product_prefixes{args.tag}.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prefix", "product", "stripe", "n_images", "per_letter"])
        for r in sorted(products, key=lambda r: r["product"]):
            for p in r["prefixes"]:
                w.writerow([p, r["product"], r["stripe"], sum(pref[p].values()),
                            " ".join(f"{k}={v}" for k, v in sorted(pref[p].items()))])

    # 跨 stripe 只看签名：同签名的产品并成一组（E59R 和 E61R 会落在同一组）
    by_sig: dict[tuple, list[dict]] = defaultdict(list)
    for r in products:
        by_sig[r["sig"]].append(r)
    with (out / f"signature_groups{args.tag}.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["product", "n_products", "products", "n_prefixes", "n_images",
                    "per_letter", "prefixes"])
        for sig, rs in sorted(by_sig.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            letters = [k for k, _ in sig]
            label = f"{letters[0]}-{letters[-1]}_{sig[0][1]}" if len(letters) > 1 \
                    else f"{letters[0]}_{sig[0][1]}"
            w.writerow([label, len(rs), ";".join(sorted(r["product"] for r in rs)),
                        sum(r["n_prefixes"] for r in rs), sum(r["n_images"] for r in rs),
                        " ".join(f"{k}={v}" for k, v in sig),
                        ";".join(p for r in rs for p in r["prefixes"])])

    # ---- 汇总 ----
    multi = sorted({r["stripe"] for r in products
                    if sum(1 for x in products if x["stripe"] == r["stripe"]) > 1})
    print(f"前缀 {len(pref)}  张 {sum(sum(c.values()) for c in pref.values())}  "
          f"stripe {len(by_stripe)}  产品 {len(products)}")
    if multi:
        print(f"\n一个 stripe 拆成多个产品的：{len(multi)} 个")
        for s in multi:
            for r in products:
                if r["stripe"] == s:
                    print(f"  {r['product']:<8} {r['n_prefixes']:>3} 前缀  "
                          f"{r['n_images']:>5} 张  "
                          + " ".join(f"{k}={v}" for k, v in r["sig"]))
    folded = [(r["product"], ps) for r in products for ps in [r["folded"]] if ps]
    if folded:
        n = sum(len(ps) for _, ps in folded)
        print(f"\n拍少被并回主产品的批次：{n} 个前缀，在 {len(folded)} 个产品里")
        for pid, ps in folded:
            print(f"  {pid:<8} {len(ps)} 个：{'; '.join(ps)}")
    print(f"\n✓ {out/f'product_groups{args.tag}.csv'}\n✓ {out/f'product_prefixes{args.tag}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
