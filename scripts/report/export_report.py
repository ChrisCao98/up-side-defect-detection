#!/usr/bin/env python3
"""把 report.md 导出成可以直接发给别人的单文件文档。

    python3 export_report.py

生成两个文件（都是单文件、双击即看，不依赖对方装任何东西）：

    report.html   图片已 base64 内嵌，任何浏览器都能打开
    report.pdf    同内容的 PDF，适合微信/邮件/打印

为什么需要这一步：Markdown 没有"内嵌图片"的标准做法。report.md 里图片是
相对路径引用（report_images/xxx.jpg），只发 .md 文件给对方的话，图片全是
红叉。本脚本把图片数据内嵌进 HTML，再交给 libreoffice 转 PDF。

依赖：python-markdown、libreoffice（soffice）。缺 libreoffice 时只出 HTML。
"""

from __future__ import annotations

import base64
import mimetypes
import re
import shutil
import subprocess
import sys
from pathlib import Path

import markdown
from PIL import Image

ROOT = Path(__file__).resolve().parent
while not (ROOT / "scores").is_dir() and ROOT != ROOT.parent:   # 宿主机在 scripts/x/，容器里在 /app/
    ROOT = ROOT.parent
# 默认导 report.md；传参数可导别的，例如: python3 export_report.py report_new.md
SRC = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "report.md")
HTML_OUT = SRC.with_suffix(".html")
PDF_OUT = SRC.with_suffix(".pdf")

# 图片在文档里的目标宽度(像素)。A4 正文宽 190mm ≈ 718px，取 700 留点余量。
# 必须写成显式的 width/height 属性：libreoffice 转 PDF 时不认 CSS 的 max-width，
# 会把图片的像素尺寸当成 pt 用（一张 2550px 宽的图变成 2550pt），超出页宽就被裁掉，
# 而且长宽比也会走样。给了显式尺寸它就老实了。
TARGET_IMG_W = 700

# 中文字体按 Linux/Windows/macOS 依次回退，保证在谁机器上都不出方块字
CSS = """
@page { size: A4; margin: 14mm 10mm; }
body {
  font-family: "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Micro Hei",
               "Microsoft YaHei", "PingFang SC", sans-serif;
  font-size: 11pt; line-height: 1.65; color: #1a1a1a;
  max-width: 960px; margin: 0 auto; padding: 24px 20px;
}
h1 { font-size: 20pt; border-bottom: 2px solid #333; padding-bottom: 8px; }
h2 { font-size: 15pt; margin-top: 1.8em; border-left: 4px solid #4a7;
     padding-left: 10px; }
h3 { font-size: 12.5pt; margin-top: 1.5em; }
table { border-collapse: collapse; width: 100%; margin: 12px 0;
        font-size: 9.5pt; }
th, td { border: 1px solid #ccc; padding: 5px 8px; text-align: left;
         vertical-align: top; word-break: break-word; }
th { background: #f0f4f2; font-weight: 600; }
tr:nth-child(even) td { background: #fafafa; }
img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 3px;
      display: block; margin: 10px 0; }
blockquote { margin: 12px 0; padding: 10px 16px; background: #f6f8f7;
             border-left: 4px solid #4a7; }
code { background: #f2f2f2; padding: 1px 5px; border-radius: 3px;
       font-size: 9.5pt; }
a { color: #2a6; }
"""

HTML_TMPL = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
{body}
</body>
</html>
"""


def inline_images(html: str) -> tuple[str, int, list[str]]:
    """把 <img src="相对路径"> 换成 base64 data URI，让 HTML 变成单文件。

    返回 (新 html, 内嵌成功的张数, 找不到的文件列表)。
    """
    missing: list[str] = []
    count = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal count
        src = match.group(1)
        if src.startswith(("http://", "https://", "data:")):
            return match.group(0)

        path = (ROOT / src).resolve()
        if not path.is_file():
            missing.append(src)
            return match.group(0)

        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        count += 1

        # 按原始长宽比缩到 TARGET_IMG_W 宽，写进属性里
        with Image.open(path) as im:
            w, h = im.size
        out_w = min(w, TARGET_IMG_W)
        out_h = round(h * out_w / w)
        return f'src="data:{mime};base64,{data}" width="{out_w}" height="{out_h}"'

    return re.sub(r'src="([^"]+)"', repl, html), count, missing


def to_pdf(html_path: Path) -> bool:
    """用 libreoffice 把 HTML 转成 PDF。返回是否成功。"""
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        print("! 没找到 libreoffice，只生成 HTML", file=sys.stderr)
        return False

    PDF_OUT.unlink(missing_ok=True)  # 清掉旧文件，否则下一步会误判成功

    cmd = [
        soffice, "--headless", "--norestore",
        # 独立 profile，避免和正在运行的 LibreOffice 抢锁
        "-env:UserInstallation=file:///tmp/lo_export_report",
        "--convert-to", "pdf", "--outdir", str(html_path.parent),
        str(html_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if not PDF_OUT.is_file():
        print(f"! PDF 转换失败：{proc.stdout.strip()} {proc.stderr.strip()}",
              file=sys.stderr)
        return False
    return True


def main() -> int:
    if not SRC.is_file():
        print(f"找不到 {SRC}", file=sys.stderr)
        return 1

    md_text = SRC.read_text(encoding="utf-8")
    body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "attr_list", "sane_lists"],
    )
    body, embedded, missing = inline_images(body)

    # 标题取 markdown 里第一个一级标题，没有就退回文件名
    m = re.search(r"^#\s+(.+)$", md_text, re.M)
    title = m.group(1).strip() if m else SRC.stem

    HTML_OUT.write_text(
        HTML_TMPL.format(css=CSS, body=body, title=title), encoding="utf-8"
    )
    print(f"✓ {HTML_OUT.name}  ({HTML_OUT.stat().st_size / 1024:.0f} KB，"
          f"内嵌 {embedded} 张图)")

    for src in missing:
        print(f"! 图片缺失，已按原路径保留（对方会看到红叉）：{src}",
              file=sys.stderr)

    if to_pdf(HTML_OUT):
        print(f"✓ {PDF_OUT.name}  ({PDF_OUT.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
