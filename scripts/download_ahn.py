"""Download one AHN5 point-cloud sheet from the official AHN distribution.

    uv run python scripts/download_ahn.py --sheet 32CN1

AHN's dataroom (https://www.ahn.nl/dataroom) publishes a sheet index, bladwijzer.gpkg, with a
download link per sheet; files are served from basisdata.nl. A sheet is 5 x 6.25 km and several
GB, and a single connection is slow (0.5-2 MB/s), so the file is fetched in parallel byte ranges.
Each range is written to its own part file, so an interrupted run resumes where it stopped. The
result is checked against the server's size before the parts are joined.

Writes data/raw/ahn5/2023_C_<sheet>.LAZ (and caches data/raw/ahn5/bladwijzer.gpkg).
AHN is CC0 (https://www.ahn.nl/).
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
INDEX_URL = "https://basisdata.nl/hwh-ahn/AUX/bladwijzer.gpkg"


def sheet_url(index: Path, sheet: str) -> str:
    with sqlite3.connect(index) as c:
        row = c.execute("select AHN5_LAZ from EllipsisDrive_index where AHN = ?", (sheet.upper(),)).fetchone()
    if row is None or not row[0]:
        raise SystemExit(f"sheet {sheet}: no AHN5 point cloud in the index")
    return row[0]


def fetch_range(url: str, part: Path, start: int, end: int, attempts: int = 30) -> None:
    """Bytes start..end (inclusive) into part, resuming from what part already holds.

    Gives up only after `attempts` failures in a row that added no bytes: on a flaky link a
    connection may drop every few minutes and still make steady progress.
    """
    want = end - start + 1
    attempt = 0
    while attempt < attempts:
        have = part.stat().st_size if part.exists() else 0
        if have >= want:
            return
        try:
            headers = {"Range": f"bytes={start + have}-{end}"}
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                if r.status_code != 206:
                    raise RuntimeError(f"expected 206 Partial Content, got {r.status_code}")
                with part.open("ab") as fh:
                    for block in r.iter_content(1 << 20):
                        fh.write(block)
        except (requests.RequestException, RuntimeError) as exc:
            gained = (part.stat().st_size if part.exists() else 0) > have
            attempt = 0 if gained else attempt + 1
            print(
                f"\n  {part.name}: {type(exc).__name__} ({'progress kept' if gained else f'stalled {attempt}'})",
                flush=True,
            )
            time.sleep(min(5 * (attempt + 1), 60))
    if part.stat().st_size != want:
        raise SystemExit(f"{part.name}: incomplete after {attempts} attempts")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sheet", required=True, help="AHN sheet name, e.g. 32CN1")
    ap.add_argument("--parts", type=int, default=8, help="byte ranges; keep it fixed when resuming")
    ap.add_argument("--workers", type=int, default=4, help="ranges fetched at once")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "raw" / "ahn5")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    index = args.out_dir / "bladwijzer.gpkg"
    if not index.exists():
        index.write_bytes(requests.get(INDEX_URL, timeout=120).content)
    url = sheet_url(index, args.sheet)
    # basisdata.nl redirects to object storage; resolve once so every range hits the same file.
    head = requests.head(url, allow_redirects=True, timeout=60)
    head.raise_for_status()
    size = int(head.headers["content-length"])
    url = head.url
    out = args.out_dir / Path(url.split("?")[0]).name
    if out.exists() and out.stat().st_size == size:
        print(f"{out.name} already complete ({size / 1e9:.2f} GB)")
        return
    print(f"{out.name}: {size / 1e9:.2f} GB in {args.parts} parallel ranges")

    bounds = [round(i * size / args.parts) for i in range(args.parts + 1)]
    parts = [out.with_name(f"{out.name}.part{i:02d}") for i in range(args.parts)]
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(fetch_range, url, p, bounds[i], bounds[i + 1] - 1) for i, p in enumerate(parts)]
        while not all(f.done() for f in futures):
            got = sum(p.stat().st_size for p in parts if p.exists())
            rate = got / max(time.time() - t0, 1)
            print(f"\r  {got / size:6.1%}  {got / 1e9:.2f} GB  {rate / 1e6:.1f} MB/s", end="", flush=True)
            time.sleep(10)
        for f in futures:
            f.result()  # re-raise any failure
    print()

    if sum(p.stat().st_size for p in parts) != size:
        raise SystemExit("parts don't add up to the server's size; rerun to resume")
    tmp = out.with_name(out.name + ".joining")
    with tmp.open("wb") as fh:
        for p in parts:
            with p.open("rb") as src:
                while block := src.read(1 << 24):
                    fh.write(block)
    if tmp.stat().st_size != size:
        raise SystemExit("joined file has the wrong size; parts kept, rerun to retry")
    tmp.rename(out)
    for p in parts:
        p.unlink()  # only after the joined file is verified
    print(f"{out.name}: {size / 1e9:.2f} GB, size verified")


if __name__ == "__main__":
    main()
