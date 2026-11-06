#!/usr/bin/env python3
"""从一个分组目录抽 Good 训练集，格式与 datasets_<F> 一致。

    python3 prep_group_dataset.py --src products_by_signature/A-E_16/bright \
        --name AE16bright --per-stripe 100

抽样规则
--------
每个 stripe 抽 --per-stripe 张，平均摊到各字母（100 张 / 5 字母 = 每字母 20 张）。
同一 (stripe, 字母) 内**按帧序号分层取点**：先按 `<前缀>_<字母><序号>_UP.jpg` 的
序号分组，每个序号取一张、前缀轮换；名额还有剩就在序号间轮转。

为什么不用「按文件名字典序等距取点」
------------------------------------
字典序把 A1、A10、A11…A19、A2、A20、A3… 排成一列，等距取点的步长与块长 20
一旦有公因数就会混叠，系统性地跳过某些序号。class1 实测：C65C 每字母 760 张只
取 20 张（步长 38），结果序号 1/2/3/19/20 一张都没进训练集，而那几档正是曝光
斜坡的两端，误报率 45–97%。旧做法还会让 9、10 号被反复取到（AC22 各 45 张、
AE16bright 各 130 张）。

输出
----
datasets_<name>/Good/<name>/*.jpg        拷贝（源可能跨设备，不能硬链）
scores/<name>_train_list.csv             每张的来源、stripe、字母
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

FRAME_RE = re.compile(r"^.{9}_[A-Z](\d+)_")   # <9 位前缀>_<字母><序号>_...

ROOT = Path(__file__).resolve().parent
while not (ROOT / "scores").is_dir() and ROOT != ROOT.parent:   # 宿主机在 scripts/x/，容器里在 /app/
    ROOT = ROOT.parent


def evenly(files: list[Path], k: int) -> list[Path]:
    n = len(files)
    if n <= k:
        return files
    return [files[round(i * (n - 1) / (k - 1))] for i in range(k)]


def stratify(files: list[Path], k: int) -> list[Path]:
    """按帧序号分层：每个序号至少取一张，同序号内前缀轮换。

    文件名不合 `<前缀>_<字母><序号>_` 约定时退回 evenly。
    """
    n = len(files)
    if n <= k:
        return files
    buckets: dict[int, list[Path]] = defaultdict(list)
    for f in files:
        m = FRAME_RE.match(f.name)
        if not m:
            return evenly(files, k)
        buckets[int(m.group(1))].append(f)
    idxs = sorted(buckets)
    if len(idxs) > k:                      # 序号比名额多，只能在序号上再等距
        idxs = [idxs[round(i * (len(idxs) - 1) / (k - 1))] for i in range(k)]
    out: list[Path] = []
    rnd = 0
    while len(out) < k:                    # 一轮覆盖所有序号，同轮内前缀各不相同
        for pos, i in enumerate(idxs):
            if len(out) >= k:
                break
            b = buckets[i]
            out.append(b[(pos + rnd) % len(b)])
        rnd += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="分组目录，下一层是 <stripe>/<字母>/")
    ap.add_argument("--name", required=True, help="数据集名，如 AE16bright")
    ap.add_argument("--per-stripe", type=int, default=100)
    ap.add_argument("--per-stripe-map", default=None,
                    help="按 stripe 单独定额，如 C65C=1000,C57G=240；没列到的用 --per-stripe。"
                         "产品大小差很多时用，让每张图的训练覆盖率接近")
    ap.add_argument("--stripes", nargs="*", default=None,
                    help="只取这些 stripe，用于单产品对照组")
    ap.add_argument("--out", default=None, help="默认 datasets_<name>")
    ap.add_argument("--exclude", default=None,
                    help="每行一个文件名，取样时跳过（用于剔除已知缺陷）")
    ap.add_argument("--defect-dir", default="eval_defect149",
                    help="Folder 必须要有 Defect 目录；沿用缺陷库，空目录不行")
    ap.add_argument("--list", dest="list_csv", default=None,
                    help="默认 scores/<name>_train_list.csv")
    ap.add_argument("--sample", choices=("stratify", "evenly"), default="stratify",
                    help="stratify=按帧序号分层（默认）；evenly=旧的字典序等距，复现旧结果时用")
    args = ap.parse_args()

    src = ROOT / args.src
    if not src.is_dir():
        raise SystemExit(f"✗ 找不到 {src}")

    stripes = sorted(d for d in src.iterdir() if d.is_dir())
    if args.stripes:
        want = set(args.stripes)
        stripes = [s for s in stripes if s.name in want]
        if not stripes:
            raise SystemExit(f"✗ --stripes {args.stripes} 在 {src} 里一个都没匹配上")
    letters = sorted({l.name for s in stripes for l in s.iterdir() if l.is_dir()})
    quota = dict.fromkeys((s.name for s in stripes), args.per_stripe)
    if args.per_stripe_map:
        for kv in args.per_stripe_map.split(","):
            k, _, v = kv.partition("=")
            k, v = k.strip(), v.strip()
            if k not in quota:
                raise SystemExit(f"✗ --per-stripe-map 里的 {k} 不在 {src} 下")
            quota[k] = int(v)
    per_letter = {k: max(1, v // len(letters)) for k, v in quota.items()}

    excl: set[str] = set()
    if args.exclude:
        excl = {l.strip() for l in (ROOT / args.exclude).read_text().splitlines() if l.strip()}
        print(f"排除名单 {len(excl)} 张：{args.exclude}")

    dst = ROOT / (args.out or f"datasets_{args.name}") / "Good" / args.name
    dst.mkdir(parents=True, exist_ok=True)

    rows, taken = [], 0
    print(f"{len(stripes)} 个 stripe × {len(letters)} 字母，每 stripe 配额 "
          + "  ".join(f"{k}={v}(每字母 {per_letter[k]})" for k, v in quota.items()))
    for s in stripes:
        sub = 0
        for L in letters:
            fs = sorted((s / L).glob("*.jpg"))
            if excl:
                fs = [f for f in fs if f.name not in excl]
            if not fs:
                print(f"  ! {s.name}/{L} 没有图片"); continue
            pick = (stratify if args.sample == "stratify" else evenly)(fs, per_letter[s.name])
            for f in pick:
                shutil.copy2(f, dst / f.name)
                rows.append([f.name, s.name, L, f.relative_to(ROOT)
                             if f.is_relative_to(ROOT) else f])
            sub += len(pick)
        taken += sub
        print(f"  {s.name:<6} {sub:>3} 张")

    list_csv = ROOT / (args.list_csv or f"scores/{args.name}_train_list.csv")
    with list_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "stripe", "letter", "source"])
        w.writerows(rows)

    # Defect/：Folder datamodule 强制要求存在，PatchCore 的库只从 Good 建，
    # 所以沿用缺陷库即可（和 datasets_<F> 的约定一致）。
    dsrc = ROOT / args.defect_dir
    ddst = dst.parent.parent / "Defect"
    n_def = 0
    if dsrc.is_dir():
        for p in dsrc.rglob("*.jpg"):
            t = ddst / p.name
            if not t.exists():
                t.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(p, t)
                except OSError:
                    shutil.copy2(p, t)
            n_def += 1

    prefixes = {r[0][:9] for r in rows}
    print(f"\n合计 {taken} 张 → {dst}")
    print(f"覆盖 {len(prefixes)} 个 9 位前缀")
    print(f"Defect/ {n_def} 张（沿用 {args.defect_dir}，只用于 Folder 的验证集）")
    print(f"✓ {list_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
