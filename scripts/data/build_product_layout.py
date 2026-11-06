#!/usr/bin/env python3
"""按产品（以及按曝光）给 class_3 建一份独立的目录树，原目录不动。

    python3 build_product_layout.py                          # 产品层 products/<产品>/<字母>/
    python3 build_product_layout.py --undo                   # 撤销
    python3 build_product_layout.py --by-exposure            # 曝光层 products_by_exposure/<产品>/<组>/<字母>/
    python3 build_product_layout.py --by-exposure --undo     # 撤销曝光层
    python3 build_product_layout.py --by-signature --with-stripe
                                                             # 签名层 products_by_signature/<签名>/<stripe>/<字母>/
    python3 build_product_layout.py --by-signature --with-stripe \
      --src /path/to/class1 --name products_by_signature_class_1 \
      --groups scores/class_1/signature_groups_class1.csv
                                                             # 换一批分类：树名加后缀，源目录指过去

为什么用硬链接
--------------
82,478 张图，拷贝要几十 GB。硬链接同 inode、不占额外空间、不跨设备（class_3 就在
项目里），而且不像软链接那样在 docker 挂载子目录时会断。删掉树只删目录项，原图不受影响。

可复原
------
建之前先把完整对照表 fsync 落盘（products/_index.csv / _index_exposure.csv），
每行 source_rel → target_rel。--undo 按表逐条删，删之前校验目标 inode 与源一致，
不一致就跳过并报告，绝不会删到真图。
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
while not (ROOT / "scores").is_dir() and ROOT != ROOT.parent:   # 宿主机在 scripts/x/，容器里在 /app/
    ROOT = ROOT.parent
SRC = ROOT / "test_images_part1" / "class_3"

# 硬链接只能建在同一设备上。数据不在仓库所在文件系统时，用 --base 指到
# 数据旁边，再在仓库根建一个同名软链接供后续脚本使用。
BASE = ROOT

LAYOUTS = {
    "product": ("products", Path("scores/product_groups.csv"), "_index.csv"),
    "exposure": ("products_by_exposure",
                 Path("scores/product_exposure_groups.csv"), "_index_exposure.csv"),
    "signature": ("products_by_signature",
                  Path("scores/signature_groups.csv"), "_index_signature.csv"),
}


def read_groups(csv_path: Path) -> dict[str, dict[str, str]]:
    """产品 -> {前缀: 目标分组目录名}"""
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8-sig")))
    if not rows:
        raise SystemExit(f"✗ {csv_path} 是空的")
    out: dict[str, dict[str, str]] = {}
    has_group = "group" in rows[0]
    for r in rows:
        target = f"{r['product']}/{r['group']}" if has_group else r["product"]
        for p in r["prefixes"].split(";"):
            out.setdefault(target, {})[p] = target
    return out


def plan(targets: dict[str, dict[str, str]], tree: Path, src: Path,
         insert_stripe: bool = False) -> list[tuple[str, Path, Path]]:
    triples = []
    for target, prefixes in targets.items():
        for p in prefixes:
            mid = Path(p[:4]) if insert_stripe else Path("")   # stripe 溯源层
            for f in sorted((src / p).rglob("*.jpg")):
                rel = f.relative_to(src / p)          # <字母>/<文件名>
                triples.append((target, f, tree / target / mid / rel))
    return triples


def build(targets, tree: Path, index: Path, src: Path,
          insert_stripe: bool = False) -> int:
    triples = plan(targets, tree, src, insert_stripe)
    tree.mkdir(parents=True, exist_ok=True)
    index.parent.mkdir(parents=True, exist_ok=True)

    # 先落盘对照表，再建链接
    with index.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source_rel", "target_rel", "link"])
        for _, s, t in triples:
            w.writerow([s, t, "hard"])
        fh.flush()
        os.fsync(fh.fileno())

    made = skipped = 0
    for _, s, t in triples:
        t.parent.mkdir(parents=True, exist_ok=True)
        if t.exists():
            if t.stat().st_ino == s.stat().st_ino:
                skipped += 1
                continue
            raise SystemExit(f"✗ 目标已存在且不是同一个文件：{t}")
        os.link(s, t)
        made += 1
    print(f"✓ {tree}  新建 {made} 个硬链接，已存在 {skipped} 个")
    print(f"✓ {index}  {len(triples)} 行对照表")
    return 0


def undo(tree: Path, index: Path) -> int:
    if not index.exists():
        raise SystemExit(f"✗ 找不到对照表 {index}，不猜，拒绝删除")
    tree = tree.resolve()          # --base 可能是相对路径；对照表里的目标也按 cwd 解析
    removed = kept = 0
    bad = []
    with index.open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            t = Path(r["target_rel"])
            s = Path(r["source_rel"])
            if not t.exists():
                continue
            if not str(t.resolve()).startswith(str(tree)):
                bad.append(t); continue
            if r["link"] == "hard" and s.exists() and t.stat().st_ino != s.stat().st_ino:
                bad.append(t); continue
            t.unlink(); removed += 1
    if bad:
        print(f"✗ {len(bad)} 个目标未删（inode 不符或越界），例如 {bad[0]}")
        print(f"  对照表保留：{index}")
        return 1
    # 先删对照表再收目录，否则 _index_*.csv 会让 rmdir 失败
    index.unlink(missing_ok=True)
    for d in sorted(tree.rglob("*"), key=lambda p: -len(p.parts)):
        if d.is_dir():
            try:
                d.rmdir()
            except OSError:
                kept += 1
    if tree.exists() and not any(tree.iterdir()):
        tree.rmdir()
    print(f"✓ 删除 {removed} 个链接，保留 {kept} 个非空目录")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--by-exposure", action="store_true", help="建曝光层而不是产品层")
    ap.add_argument("--by-signature", action="store_true", help="按签名跨 stripe 分组")
    ap.add_argument("--with-stripe", action="store_true",
                    help="在分组和字母之间插一层 stripe 便于溯源")
    ap.add_argument("--groups", help="改用这张分组表（列同 product_groups.csv）")
    ap.add_argument("--src", help="图像源目录，默认 test_images_part1/class_3；"
                                  "换分类（如 class1）时必须指过去，否则找不到前缀")
    ap.add_argument("--base", help="树建在哪个目录；必须与 --src 同一文件系统（硬链接）。"
                                   "默认仓库根")
    ap.add_argument("--name", help="树名，默认取 products / products_by_exposure / "
                                   "products_by_signature；换分类时加后缀，如 "
                                   "products_by_signature_class_1")
    ap.add_argument("--undo", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="只统计，不建")
    args = ap.parse_args()

    kind = "exposure" if args.by_exposure else "signature" if args.by_signature else "product"
    name, csv_path, index_name = LAYOUTS[kind]
    if args.groups:
        csv_path = Path(args.groups)
    if args.name:
        name = args.name
    src = Path(args.src) if args.src else SRC
    if not src.is_dir():
        print(f"✗ 源目录不存在：{src}")
        return 1
    base = Path(args.base) if args.base else BASE
    tree = (base / name).resolve()
    index = tree / index_name
    if not csv_path.exists():
        print(f"✗ 缺少 {csv_path}")
        return 1

    targets = read_groups(ROOT / csv_path)
    if args.dry_run:
        triples = plan(targets, tree, src, args.with_stripe)
        per: dict[str, int] = {}
        for t, _, _ in triples:
            per[t] = per.get(t, 0) + 1
        print(f"源 {src}")
        print(f"树 {tree}")
        print(f"目标 {len(targets)} 个目录，{len(triples)} 个链接")
        for t in sorted(targets):
            print(f"  {t:<16} {len(targets[t]):>3} 前缀  {per.get(t, 0):>5} 张")
        return 0
    if args.undo:
        return undo(tree, index)
    return build(targets, tree, index, src, args.with_stripe)


if __name__ == "__main__":
    raise SystemExit(main())
