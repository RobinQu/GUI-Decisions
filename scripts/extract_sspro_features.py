"""Extract frozen final-hidden features at the y slot for supervised joint-region heads."""

import argparse, json, os, time

import torch
from PIL import Image

from slotread.client import agent_user
from slotread.data.evalsets import screenspot_pro
from slotread.local import LocalSlotModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("model")
    ap.add_argument("--part", choices=("calibration", "heldout"), required=True)
    ap.add_argument("--slots", choices=("x", "y"), nargs="+", default=["y"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--name")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.limit < 0 or args.save_every <= 0 or len(set(args.slots)) != len(args.slots):
        raise ValueError("limit must be nonnegative, save-every positive, and slots unique")
    source = json.load(open(args.source))
    if (source.get("which") != "screenspot-pro" or source.get("n") != 1581 or source.get("seed") != 0
            or source.get("model") != args.model or len(source["rows"]) != 1581):
        raise ValueError("Source must be the complete seed-0 ScreenSpot-Pro evaluation with the same model")
    split = int(round(len(source["rows"]) * 0.5))
    selected = source["rows"][:split] if args.part == "calibration" else source["rows"][split:]
    selected = selected[:args.limit] if args.limit else selected
    if not selected:
        raise ValueError("No cases selected")
    name = args.name or f"sspro-features-{args.part}-s0"
    path = f"results/{name}_{len(selected)}.pt"
    os.makedirs("results", exist_ok=True)
    rows = []
    if args.resume and os.path.exists(path):
        previous = torch.load(path, weights_only=True)
        expected = dict(schema=1, source=args.source, model=args.model, part=args.part, slots=args.slots,
                        device=args.device, expected_n=len(selected))
        mismatches = [key for key, value in expected.items() if previous.get(key) != value]
        if mismatches:
            raise ValueError(f"Resume configuration/schema mismatch: {', '.join(mismatches)}")
        rows = previous["rows"]
        if [str(r["id"]) for r in rows] != [str(r["id"]) for r in selected[:len(rows)]]:
            raise ValueError("Resume rows are not a prefix of the selected source rows")
    done = {str(r["id"]) for r in rows}
    if len(rows) == len(selected):
        print(f"Complete: {path}")
        return
    examples = {str(ex["id"]): ex for ex in screenspot_pro()}
    model = LocalSlotModel.from_pretrained(args.model, device_map=args.device)
    started = time.time()
    print(f"{args.part}: {len(selected)} cases, {len(rows)} resumed; slots {args.slots}", flush=True)
    for i, source_row in enumerate(selected):
        if str(source_row["id"]) in done:
            continue
        ex = examples[str(source_row["id"])]
        if ex["box"] != source_row["box"] or list(ex["size"]) != list(source_row["image_size"]):
            raise ValueError(f"Source/data mismatch for {source_row['id']}")
        with Image.open(ex["image"]) as opened:
            image = opened.convert("RGB")
        _, ms, hidden = model.read("agent", image, agent_user(ex["instruction"]), slots=args.slots,
                                   return_hidden=True)
        if hidden.ndim != 2 or hidden.shape[0] != len(args.slots):
            raise ValueError(f"Unexpected hidden shape for {source_row['id']}: {tuple(hidden.shape)}")
        rows.append(dict(id=source_row["id"], group=source_row["group"], box=source_row["box"],
                         area=source_row["area"], image_size=source_row["image_size"],
                         global_default=source_row["global"]["default"],
                         global_kde_grid=source_row["global"]["kde_grid"], ms=ms,
                         feature=hidden.flatten().to(torch.float16)))
        if (i + 1) % args.save_every == 0 or i + 1 == len(selected):
            payload = dict(schema=1, source=args.source, model=args.model, part=args.part, slots=args.slots,
                           device=args.device, expected_n=len(selected), n=len(rows), rows=rows)
            temporary = path + ".tmp"
            torch.save(payload, temporary)
            os.replace(temporary, path)
            print(f"  {i + 1}/{len(selected)} ({time.time() - started:.0f}s)", flush=True)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
