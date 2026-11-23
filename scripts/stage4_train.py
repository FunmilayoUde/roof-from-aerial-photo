"""Stage 4: train the U-Net on Stage 2 chips, on a spatial split.

    uv run python scripts/stage4_train.py --run east                       # test east: 10, 15, 20
    uv run python scripts/stage4_train.py --run loso_03 --test 32CN1_03 --val 32CN1_13

Training uses every kept Bilthoven house outside the test and validation sub-tiles. The
validation sub-tile picks the best epoch (lowest validation loss); the test area is never looked
at during training. Writes outputs/stage4/<run>/{model.pt, history.json, split.json}.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from roofphoto.models import unet
from roofphoto.models.dataset import ChipDataset, kept_chips

ROOT = Path(__file__).resolve().parents[1]
TILES = [f"32CN1_{k}" for k in ("03", "04", "05", "08", "09", "10", "13", "15", "20")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--test", nargs="+", default=["32CN1_10", "32CN1_15", "32CN1_20"])
    ap.add_argument("--extra-tiles", nargs="*", default=[], help="more tiles to train on (e.g. a second town)")
    ap.add_argument(
        "--orient-loss", choices=["roof", "image"], default="roof", help="image = Stage 4's orientation loss"
    )
    ap.add_argument("--val", nargs="+", default=["32CN1_13"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pitch-weight", type=float, default=1.0)
    ap.add_argument("--edge-weight", type=float, default=0.0, help="> 0 adds an edge-type output (Stage 5)")
    ap.add_argument("--edge-binary", action="store_true", help="edge output learns edge vs none, not the type")
    ap.add_argument(
        "--orient-weight",
        type=float,
        default=1.0,
        help="orientation's share of the loss; the roof-only orientation loss is ~1 vs roof ~0.1-0.2 (Stage 5)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--aug",
        choices=["geom", "colour", "none", "mixed"],
        default="geom",
        help="geom: 90° rotations/flips; colour: jitter only; mixed: see dataset.ChipDataset",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    out = ROOT / "outputs" / "stage4" / args.run
    out.mkdir(parents=True, exist_ok=True)
    train_tiles = [t for t in TILES + args.extra_tiles if t not in args.test + args.val]
    split = {"train": train_tiles, "val": args.val, "test": args.test}
    with_edges = args.edge_weight > 0
    train = ChipDataset(
        kept_chips(ROOT, train_tiles),
        train=True,
        seed=args.seed,
        aug=args.aug,
        edges=with_edges,
        edge_binary=args.edge_binary,
    )
    val = ChipDataset(kept_chips(ROOT, args.val), train=False, edges=with_edges, edge_binary=args.edge_binary)
    split |= {"n_train": len(train), "n_val": len(val)}
    (out / "split.json").write_text(json.dumps(split | {"args": vars(args)}, indent=2))
    print(f"{args.run}: {len(train)} train, {len(val)} val houses, test {args.test}, device {device}")

    model = unet.build(pretrained=True, edges=with_edges).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    train_dl = DataLoader(train, batch_size=args.batch, shuffle=True, drop_last=True)
    val_dl = DataLoader(val, batch_size=args.batch)

    history, best = [], float("inf")
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        tr = []
        for batch in train_dl:
            img, orient, pitch, roof, *rest = (b.to(device) for b in batch)
            losses = unet.loss(
                model(img),
                orient,
                pitch,
                roof,
                args.pitch_weight,
                orient_weight=args.orient_weight,
                edges=rest[0] if rest else None,
                edge_weight=args.edge_weight,
                edge_binary=args.edge_binary,
                orient_loss=args.orient_loss,
            )
            opt.zero_grad()
            losses["total"].backward()
            opt.step()
            tr.append({k: float(v.detach()) for k, v in losses.items()})
        sched.step()
        model.eval()
        va = []
        with torch.no_grad():
            hits = [0, 0]
            for batch in val_dl:
                img, orient, pitch, roof, *rest = (b.to(device) for b in batch)
                out_v = model(img)
                losses_v = unet.loss(
                    out_v,
                    orient,
                    pitch,
                    roof,
                    args.pitch_weight,
                    orient_weight=args.orient_weight,
                    edges=rest[0] if rest else None,
                    edge_weight=args.edge_weight,
                    edge_binary=args.edge_binary,
                    orient_loss=args.orient_loss,
                )
                va.append({k: float(v) for k, v in losses_v.items()})
                c, n = unet.roof_orientation_hits(out_v, orient)
                hits[0] += c
                hits[1] += n
        row = {"epoch": epoch, "seconds": round(time.time() - t0, 1)}
        row |= {f"train_{k}": round(float(np.mean([x[k] for x in tr])), 4) for k in tr[0]}
        row |= {f"val_{k}": round(float(np.mean([x[k] for x in va])), 4) for k in va[0]}
        row["val_roof_orientation_acc"] = round(hits[0] / max(hits[1], 1), 4)
        history.append(row)
        if row["val_total"] < best:
            best = row["val_total"]
            torch.save(model.state_dict(), out / "model.pt")
            row["saved"] = True
        print(json.dumps(row), flush=True)
        (out / "history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
