"""Render docs/paper/paper.md to a typeset HTML page and a PDF.

    uv run --with markdown python scripts/build_paper.py [--no-pdf]

Maths ($...$ and $$...$$) is set aside before Markdown runs, so underscores and asterisks inside
it survive, and is typeset by KaTeX in the page. The PDF is printed by headless Google Chrome
(macOS path below), which loads KaTeX from the CDN first. Writes docs/paper/paper.html and
docs/paper/paper.pdf; figures are referenced from docs/paper/figures/.
"""

from __future__ import annotations

import argparse
import html
import re
import subprocess
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "docs/paper"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist"

CSS = """
:root { --ink:#1b1b1b; --muted:#555; --rule:#c9c9c9; --accent:#1f4e8c; --bg:#ffffff; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --ink:#e6e6e6; --muted:#a8a8a8; --rule:#444; --accent:#8fb4ea; --bg:#151515; } }
:root[data-theme="dark"] { --ink:#e6e6e6; --muted:#a8a8a8; --rule:#444; --accent:#8fb4ea; --bg:#151515; }
html { background: var(--bg); }
body { font-family: "Source Serif 4", "Charter", "Georgia", serif; color: var(--ink); background: var(--bg);
  max-width: 46rem; margin: 0 auto; padding: 2.5rem 1rem 4rem; font-size: 16px; line-height: 1.55;
  text-align: justify; hyphens: auto; }
h1 { font-size: 1.75rem; line-height: 1.25; text-align: center; margin: 0.5rem 0 0.6rem; font-weight: 600; }
h1 + p { text-align: center; margin: 0; font-size: 1.05rem; }
h1 + p + p { text-align: center; color: var(--muted); font-size: 0.85rem; margin-top: 0.3rem; }
h2 { font-size: 1.2rem; margin: 2.2rem 0 0.6rem; font-weight: 600; }
h3 { font-size: 1.02rem; margin: 1.6rem 0 0.4rem; font-weight: 600; font-style: italic; }
hr { border: 0; border-top: 1px solid var(--rule); margin: 1.8rem 0; }
p { margin: 0.55rem 0; }
a { color: var(--accent); }
code { font-family: "JetBrains Mono", "Menlo", monospace; font-size: 0.82em; }
pre { font-size: 0.74rem; line-height: 1.4; overflow-x: auto; padding: 0.7rem 0.8rem;
  border: 1px solid var(--rule); border-radius: 4px; text-align: left; }
.table-wrap { overflow-x: auto; margin: 1rem 0 0.3rem; }
table { border-collapse: collapse; font-size: 0.84rem; line-height: 1.35; margin: 0 auto; text-align: left; }
thead th { border-top: 1.5px solid var(--ink); border-bottom: 1px solid var(--ink); font-weight: 600; }
tbody tr:last-child td { border-bottom: 1.5px solid var(--ink); }
th, td { padding: 0.28rem 0.6rem; vertical-align: top; }
p > img { display: block; max-width: 100%; margin: 1.4rem auto 0.2rem; background: #fff; border-radius: 2px; }
p.caption { margin-top: 0.2rem; }
p.caption em { display: block; font-size: 0.84rem; line-height: 1.45; color: var(--muted); margin-bottom: 1rem; }
.katex-display { overflow-x: auto; overflow-y: hidden; padding: 0.2rem 0; }
h2#abstract + p { font-size: 0.94rem; }
@media (max-width: 600px) { body { font-size: 15px; text-align: left; } }
@media print {
  body { max-width: none; padding: 0; font-size: 10.5pt; }
  h2 { break-after: avoid; } h3 { break-after: avoid; }
  img, table, .katex-display, pre { break-inside: avoid; }
  p > img { max-width: 100%; }
}
@page { size: A4; margin: 20mm 18mm 20mm 18mm; }
"""


def protect_math(text: str) -> tuple[str, list[str]]:
    keep: list[str] = []

    def stash(m: re.Match) -> str:
        keep.append(m.group(0))
        return f"MATHSLOT{len(keep) - 1}X"

    # Display maths first, then inline. Code blocks are left alone.
    parts = re.split(r"(```.*?```)", text, flags=re.S)
    for i, part in enumerate(parts):
        if part.startswith("```"):
            continue
        part = re.sub(r"\$\$.+?\$\$", stash, part, flags=re.S)
        part = re.sub(r"(?<![\\$])\$(?!\s)[^$\n]+?(?<!\s)\$", stash, part)
        parts[i] = part
    return "".join(parts), keep


def build_html() -> Path:
    src = (PAPER / "paper.md").read_text()
    text, maths = protect_math(src)
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "toc"])
    body = re.sub(r"<table>", '<div class="table-wrap"><table>', body).replace("</table>", "</table></div>")
    body = re.sub(r"<p><em>(Figure|Table) ", r'<p class="caption"><em>\1 ', body)
    for i, m in enumerate(maths):
        body = body.replace(f"MATHSLOT{i}X", html.escape(m, quote=False))
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Measuring Roofs with a LiDAR Teacher</title>
<link rel="stylesheet" href="{KATEX}/katex.min.css">
<script defer src="{KATEX}/katex.min.js"></script>
<script defer src="{KATEX}/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body,{{delimiters:[{{left:'$$',right:'$$',display:true}},{{left:'$',right:'$',display:false}}],throwOnError:false}});document.body.dataset.math='done';"></script>
<style>{CSS}</style></head>
<body>
{body}
</body></html>"""
    out = PAPER / "paper.html"
    out.write_text(page)
    return out


def build_pdf(page: Path) -> Path:
    pdf = PAPER / "paper.pdf"
    subprocess.run(
        [
            CHROME,
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            "--virtual-time-budget=15000",
            f"--print-to-pdf={pdf}",
            page.as_uri(),
        ],
        check=True,
        capture_output=True,
    )
    return pdf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()
    page = build_html()
    print(page)
    if not args.no_pdf:
        print(build_pdf(page))


if __name__ == "__main__":
    main()
