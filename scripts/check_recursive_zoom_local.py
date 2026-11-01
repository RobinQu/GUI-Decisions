"""Compare global->768->small sequential crops with global->small direct crops."""

import argparse, json, os, statistics, time

from PIL import Image

from slotread.client import agent_user
from slotread.data.evalsets import screenspot_pro
from slotread.local import LocalSlotModel
from slotread.zoom import crop_window, distribution_readouts, map_point, point_uncertainty


def hit(point, box):
    return float(box[0] <= point["x"] <= box[2] and box[1] <= point["y"] <= box[3])


def complete(path):
    data = json.load(open(path))
    if data.get("expected_n", data.get("n")) != len(data["rows"]):
        raise ValueError(f"Partial result: {path}")
    return data


def save(path, args, rows):
    payload = dict(schema=2, source=args.source, parent=args.parent, model=args.model, part=args.part,
                   sizes=args.sizes, readouts=args.readouts, include_direct=args.include_direct,
                   device=args.device, expected_n=args.total_cases, n=len(rows), rows=rows)
    temporary = path + ".tmp"
    with open(temporary, "w") as f:
        json.dump(payload, f)
    os.replace(temporary, path)


def summarize(rows, sizes, readouts, include_direct):
    for size in sizes:
        for branch in ("sequential", "direct") if include_direct else ("sequential",):
            for how in readouts:
                values = [r["results"][str(size)][branch]["readouts"][how]["hit"] for r in rows]
                print(f"{size:4d} {branch:10s} {how:9s}: {statistics.mean(values):.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="Complete full-image ScreenSpot-Pro result")
    ap.add_argument("parent", help="Completed 768px crop result for the selected split")
    ap.add_argument("model")
    ap.add_argument("--part", choices=("calibration", "heldout"), required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[256, 384, 512])
    ap.add_argument("--readouts", choices=("default", "peak", "kde", "kde_grid"), nargs="+",
                    default=["default", "kde_grid"])
    ap.add_argument("--include-direct", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--name")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if not args.sizes or any(size <= 0 or size >= 768 for size in args.sizes):
        raise ValueError("Recursive crop sizes must be positive and smaller than 768")
    if len(set(args.sizes)) != len(args.sizes) or len(set(args.readouts)) != len(args.readouts):
        raise ValueError("Crop sizes and readouts must be unique")
    if args.save_every <= 0 or args.limit < 0:
        raise ValueError("save-every must be positive and limit nonnegative")
    source = complete(args.source)
    parent = complete(args.parent)
    if (source.get("which") != "screenspot-pro" or source.get("n") != 1581
            or source.get("model") != args.model or parent.get("model") != args.model
            or parent.get("source") != args.source or parent.get("part") != args.part):
        raise ValueError("Source, parent, model, and split must match")
    split = int(round(len(source["rows"]) * 0.5))
    source_selected = source["rows"][:split] if args.part == "calibration" else source["rows"][split:]
    if [str(r["id"]) for r in parent["rows"]] != [str(r["id"]) for r in source_selected]:
        raise ValueError("Parent rows do not match the source split")
    selected = list(zip(source_selected, parent["rows"]))
    selected = selected[:args.limit] if args.limit else selected
    args.total_cases = len(selected)
    if not selected:
        raise ValueError("No cases selected")
    name = args.name or f"recursive-zoom-{args.part}"
    os.makedirs("results", exist_ok=True)
    path = f"results/{name}_{len(selected)}.json"
    rows = []
    if args.resume and os.path.exists(path):
        previous = json.load(open(path))
        expected = dict(schema=2, source=args.source, parent=args.parent, model=args.model, part=args.part,
                        sizes=args.sizes, readouts=args.readouts, include_direct=args.include_direct,
                        device=args.device, expected_n=len(selected))
        mismatches = [key for key, value in expected.items() if previous.get(key) != value]
        if mismatches:
            raise ValueError(f"Resume configuration/schema mismatch: {', '.join(mismatches)}")
        rows = previous["rows"]
        if [str(r["id"]) for r in rows] != [str(r["id"]) for r, _ in selected[:len(rows)]]:
            raise ValueError("Resume rows are not a prefix of the selected source rows")
    done = {str(r["id"]) for r in rows}
    examples = {str(ex["id"]): ex for ex in screenspot_pro()}
    if len(rows) == len(selected):
        summarize(rows, args.sizes, args.readouts, args.include_direct)
        return
    model = LocalSlotModel.from_pretrained(args.model, device_map=args.device)
    started = time.time()
    print(f"{args.part}: {len(selected)} cases, {len(rows)} resumed; sizes {args.sizes}, "
          f"readouts {args.readouts}, direct {args.include_direct}", flush=True)
    for i, (source_row, parent_row) in enumerate(selected):
        if str(source_row["id"]) in done:
            continue
        ex = examples[str(source_row["id"])]
        if ex["box"] != source_row["box"] or list(ex["size"]) != list(source_row["image_size"]):
            raise ValueError(f"Source/data mismatch for {source_row['id']}")
        parent_value = parent_row["results"]["768"]
        parent_point = parent_value["readouts"]["kde_grid"]
        with Image.open(ex["image"]) as opened:
            image = opened.convert("RGB")
        user = agent_user(ex["instruction"])
        row = dict(id=source_row["id"], group=source_row["group"], box=source_row["box"],
                   area=source_row["area"], image_size=source_row["image_size"],
                   global_hit=source_row["global"]["default"]["hit"], global_ms=source_row["global_ms"],
                   parent_hit=parent_point["hit"], parent_ms=parent_value["ms"], results={})
        for size in args.sizes:
            row["results"][str(size)] = {}
            branches = ("sequential", "direct") if args.include_direct else ("sequential",)
            for branch in branches:
                existing = parent_row["results"].get(str(size)) if branch == "direct" else None
                if existing is not None and all(how in existing["readouts"] for how in args.readouts):
                    value = dict(ms=existing["ms"], crop=existing["crop"], uncertainty=existing["uncertainty"],
                                 readouts={how: existing["readouts"][how] for how in args.readouts}, reused=True)
                else:
                    center = parent_point if branch == "sequential" else source_row["global"]["default"]
                    crop = crop_window(center, image.size, size)
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
                    value = dict(ms=ms, crop=crop, uncertainty=uncertainty, reused=False,
                                 readouts={how: dict(point, hit=hit(point, ex["box"]))
                                           for how, point in points.items()})
                row["results"][str(size)][branch] = value
        rows.append(row)
        if (i + 1) % args.save_every == 0 or i + 1 == len(selected):
            save(path, args, rows)
            print(f"  {i + 1}/{len(selected)} ({time.time() - started:.0f}s)", flush=True)
    summarize(rows, args.sizes, args.readouts, args.include_direct)
    save(path, args, rows)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
