"""Stage 3: how well does the footprint alone predict roof area? The bar a photo model must beat.

    uv run python scripts/stage3_baseline.py

Houses: those Stage 2 kept. Truth: the LiDAR pipeline's total roof area. Two evaluation sets:
  all     every kept house;
  clean   houses where small or traced facets (approximate area) are at most 10% of the roof,
          so the truth itself is reliable.
Splits, both spatial (whole areas held out, never random houses):
  east    train on central Bilthoven, test on the eastern sub-tiles 10, 15, 20;
  LOSO    leave one sub-tile out, each of the nine in turn: how much the error moves by area.
The 3 Delft houses are a separate out-of-region check (another town, 40 km away).

Writes outputs/stage3/results.json and outputs/stage3/*.png.
"""

from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from roofphoto.baseline import AGE_BANDS, errors, fit  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TEST_TILES = ("32CN1_10", "32CN1_15", "32CN1_20")
CLEAN_MAX_UNCERTAIN = 0.10


def load() -> dict[str, np.ndarray]:
    rows = []
    for f in sorted(glob.glob(str(ROOT / "outputs" / "stage2" / "*" / "manifest.csv"))):
        tile = Path(f).parent.name
        rows += [r | {"tile": tile} for r in csv.DictReader(open(f)) if r["status"] == "kept"]
    return {
        "tile": np.array([r["tile"] for r in rows]),
        "footprint": np.array([float(r["footprint_m2"]) for r in rows]),
        "truth": np.array([float(r["roof_area_m2"]) for r in rows]),
        "built": np.array([int(r["built"]) for r in rows]),
        "uncertain": np.array([float(r["uncertain_area_share"]) for r in rows]),
    }


def evaluate(d: dict, train: np.ndarray, test: np.ndarray) -> dict:
    b = fit(d["footprint"][train], d["truth"][train], d["built"][train])
    out = {"k_all": round(b.k_all, 3), "k_by_age_band": {f"{AGE_BANDS[i]}": round(k, 3) for i, k in b.k_band.items()}}
    for name, subset in (("all", test), ("clean", test & (d["uncertain"] <= CLEAN_MAX_UNCERTAIN))):
        fp, t, y = d["footprint"][subset], d["truth"][subset], d["built"][subset]
        out[name] = {"B0": errors(b.b0(fp), t), "B1": errors(b.b1(fp), t), "B2": errors(b.b2(fp, y), t)}
    return out


def main() -> None:
    d = load()
    out_dir = ROOT / "outputs" / "stage3"
    out_dir.mkdir(parents=True, exist_ok=True)
    bilt = np.char.startswith(d["tile"], "32CN1")
    is_test = np.isin(d["tile"], TEST_TILES)

    results = {
        "houses": {"bilthoven": int(bilt.sum()), "test_east": int(is_test.sum()), "delft": int((~bilt).sum())},
        "east_split": evaluate(d, bilt & ~is_test, is_test),
        "delft_out_of_region": evaluate(d, bilt, ~bilt),
    }
    loso = {}
    for t in sorted(set(d["tile"][bilt])):
        r = evaluate(d, bilt & (d["tile"] != t), d["tile"] == t)
        loso[t] = {"n": r["all"]["B1"]["n"], "B1_median_abs_pct": r["all"]["B1"]["median_abs_pct"]}
        loso[t] |= {"B1_clean_median_abs_pct": r["clean"]["B1"]["median_abs_pct"], "k_all": r["k_all"]}
    results["leave_one_subtile_out"] = loso
    vals = [v["B1_median_abs_pct"] for v in loso.values()]
    results["loso_B1_median_abs_pct_range"] = [min(vals), max(vals)]

    # Pooled LOSO: every Bilthoven house predicted by baselines fitted without its own sub-tile.
    # One number over all 351 houses, not resting on which area happened to be held out.
    est = {"B0": np.zeros(len(d["tile"])), "B1": np.zeros(len(d["tile"])), "B2": np.zeros(len(d["tile"]))}
    for t in sorted(set(d["tile"][bilt])):
        m = d["tile"] == t
        b = fit(d["footprint"][bilt & ~m], d["truth"][bilt & ~m], d["built"][bilt & ~m])
        est["B0"][m], est["B1"][m] = b.b0(d["footprint"][m]), b.b1(d["footprint"][m])
        est["B2"][m] = b.b2(d["footprint"][m], d["built"][m])
    clean = bilt & (d["uncertain"] <= CLEAN_MAX_UNCERTAIN)
    results["loso_pooled"] = {
        name: {k: errors(est[k][subset], d["truth"][subset]) for k in est}
        for name, subset in (("all", bilt), ("clean", clean))
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

    # Figure: B1 estimate against truth on the test area, and relative error against pitch proxy.
    b = fit(d["footprint"][bilt & ~is_test], d["truth"][bilt & ~is_test], d["built"][bilt & ~is_test])
    fp, t = d["footprint"][is_test], d["truth"][is_test]
    clean = d["uncertain"][is_test] <= CLEAN_MAX_UNCERTAIN
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    lim = [0, max(t.max(), b.b1(fp).max()) * 1.05]
    axs[0].scatter(t[clean], b.b1(fp)[clean], s=18, color="#2a78d6", label="clean truth")
    axs[0].scatter(t[~clean], b.b1(fp)[~clean], s=18, color="#bbb", label="truth with >10% uncertain area")
    axs[0].plot(lim, lim, color="#555", lw=0.8)
    axs[0].fill_between(lim, [x * 0.9 for x in lim], [x * 1.1 for x in lim], color="#2a78d6", alpha=0.08, label="±10%")
    axs[0].set_xlim(lim)
    axs[0].set_ylim(lim)
    axs[0].set_xlabel("roof area from LiDAR, m²")
    axs[0].set_ylabel(f"footprint x {b.k_all:.2f}, m²")
    axs[0].set_title("B1 on the held-out east (sub-tiles 10, 15, 20)", fontsize=9)
    axs[0].legend(fontsize=7)
    ratio = d["truth"][bilt] / d["footprint"][bilt]
    axs[1].hist(ratio, bins=40, color="#2a78d6")
    axs[1].axvline(b.k_all, color="#d6572a", label=f"B1 factor {b.k_all:.2f}")
    axs[1].set_xlabel("roof area / footprint (all Bilthoven kept houses)")
    axs[1].set_ylabel("houses")
    axs[1].set_title("what a single factor has to cover", fontsize=9)
    axs[1].legend(fontsize=8)
    fig.savefig(out_dir / "baseline.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
