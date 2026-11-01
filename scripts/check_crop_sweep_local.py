"""Sweep crop sizes and local readouts while reusing a completed full-image slot evaluation."""

import argparse, json, os, statistics, time

from PIL import Image

from slotread.client import agent_user
from slotread.data.evalsets import screenspot_pro
from slotread.local import LocalSlotModel
from slotread.zoom import crop_window, distribution_readouts, map_point, point_uncertainty


def hit(point, box):
    return float(box[0] <= point["x"] <= box[2] and box[1] <= point["y"] <= box[3])


def save(path, args, rows):
    payload = dict(schema=2, source=args.source, model=args.model, part=args.part,
                   sizes=args.sizes, readouts=args.readouts, expected_n=args.total_cases, n=len(rows), rows=rows)
    temporary = path + ".tmp"
    with open(temporary, "w") as f:
        json.dump(payload, f)
    os.replace(temporary, path)


def summarize(rows, sizes, readouts):
    for size in sizes:
        values = [r["results"][str(size)] for r in rows]
        for how in readouts:
            pred = [v["readouts"][how]["hit"] for v in values]
            base = [r["global_hit"] for r in rows]
            repaired = sum(not b and p for b, p in zip(base, pred))
            regressed = sum(b and not p for b, p in zip(base, pred))
            print(f"{size:4d} {how:9s}: {statistics.mean(pred):.4f}; "
                  f"wrong->right {repaired}, right->wrong {regressed}", flush=True)
        times = [r["global_ms"] + v["ms"] for r, v in zip(rows, values)]
        print(f"     total latency: median {statistics.median(times):.0f} ms, mean {statistics.mean(times):.0f} ms",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("model")
    ap.add_argument("--part", choices=("calibration", "heldout", "all"), required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[256, 384, 512, 768, 1024])
    ap.add_argument("--readouts", choices=("default", "peak", "kde", "kde_grid"), nargs="+",
                    default=["default", "peak", "kde", "kde_grid"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--name")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if not args.sizes or any(size <= 0 for size in args.sizes) or args.save_every <= 0 or args.limit < 0:
        raise ValueError("Crop sizes and save-every must be positive; limit must be nonnegative")
    if len(set(args.sizes)) != len(args.sizes) or len(set(args.readouts)) != len(args.readouts):
        raise ValueError("Crop sizes and readouts must be unique")
    source = json.load(open(args.source))
    source_rows = source["rows"]
    if (source.get("which") != "screenspot-pro" or source.get("expected_n") != 1581
            or source.get("n") != 1581 or len(source_rows) != 1581 or source.get("model") != args.model):
        raise ValueError("Source must be the complete 1,581-case ScreenSpot-Pro evaluation with the same model")
    split = int(round(len(source_rows) * 0.5))
    selected = dict(calibration=source_rows[:split], heldout=source_rows[split:], all=source_rows)[args.part]
    selected = selected[:args.limit] if args.limit else selected
    args.total_cases = len(selected)
    if not selected:
        raise ValueError("No cases selected")
    name = args.name or f"crop-sweep-{args.part}"
    os.makedirs("results", exist_ok=True)
    path = f"results/{name}_{len(selected)}.json"
    rows = []
    if args.resume and os.path.exists(path):
        previous = json.load(open(path))
        expected = dict(schema=2, source=args.source, model=args.model, part=args.part,
                        sizes=args.sizes, readouts=args.readouts, expected_n=len(selected))
        mismatches = [key for key, value in expected.items() if previous.get(key) != value]
        if mismatches:
            raise ValueError(f"Resume configuration/schema mismatch: {', '.join(mismatches)}")
        rows = previous["rows"]
        if [str(r["id"]) for r in rows] != [str(r["id"]) for r in selected[:len(rows)]]:
            raise ValueError("Resume rows are not a prefix of the selected source rows")
    done = {str(r["id"]) for r in rows}
    examples = {str(ex["id"]): ex for ex in screenspot_pro()}
    if len(rows) == len(selected):
        summarize(rows, args.sizes, args.readouts)
        return
    model = LocalSlotModel.from_pretrained(args.model, device_map=args.device)
    started = time.time()
    print(f"{args.part}: {len(selected)} cases, {len(rows)} resumed; sizes {args.sizes}, "
          f"readouts {args.readouts}", flush=True)
    for i, source_row in enumerate(selected):
        if str(source_row["id"]) in done:
            continue
        ex = examples[str(source_row["id"])]
        if ex["box"] != source_row["box"] or list(ex["size"]) != list(source_row["image_size"]):
            raise ValueError(f"Source/data mismatch for {source_row['id']}")
        with Image.open(ex["image"]) as opened:
            image = opened.convert("RGB")
        user = agent_user(ex["instruction"])
        row = dict(id=source_row["id"], group=source_row["group"], box=source_row["box"],
                   area=source_row["area"], image_size=source_row["image_size"],
                   global_hit=source_row["global"]["default"]["hit"], global_ms=source_row["global_ms"],
                   results={})
        for size in args.sizes:
            crop = crop_window(source_row["global"]["default"], image.size, size)
            local_image = image.crop((crop["left"], crop["top"],
                                      crop["left"] + crop["side"], crop["top"] + crop["side"]))
            lp, ms = model.read("agent", local_image, user)
            px, py, uncertainty = point_uncertainty(lp, model.arm)
            points = {}
            for how in args.readouts:
                if how in ("default", "peak"):
                    point = model.tasks["agent"].decode(lp, model.arm, how)
                else:
                    point = distribution_readouts(px, py)[how]
                points[how] = map_point(point, crop, image.size)
            row["results"][str(size)] = dict(ms=ms, crop=crop, uncertainty=uncertainty,
                                             readouts={how: dict(point, hit=hit(point, ex["box"]))
                                                       for how, point in points.items()})
        rows.append(row)
        if (i + 1) % args.save_every == 0 or i + 1 == len(selected):
            save(path, args, rows)
            print(f"  {i + 1}/{len(selected)} ({time.time() - started:.0f}s)", flush=True)
    summarize(rows, args.sizes, args.readouts)
    save(path, args, rows)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
