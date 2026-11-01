"""Read a second local view at a spatially distinct mode of the global slot distribution."""

import argparse, json, math, os, time

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

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


def alternate_mode(probabilities, reference, min_distance, bandwidth=0.03):
    px, py = np.asarray(probabilities["x"]), np.asarray(probabilities["y"])
    if px.ndim != 1 or py.ndim != 1 or len(px) != len(py) or not len(px):
        raise ValueError("Expected equally sized one-dimensional x/y probabilities")
    density = gaussian_filter(px[:, None] * py[None, :], bandwidth * len(px), mode="constant")
    n = len(px)
    x, y = np.meshgrid((np.arange(n) + 0.5) / n, (np.arange(n) + 0.5) / n, indexing="ij")
    eligible = (x - reference["x"]) ** 2 + (y - reference["y"]) ** 2 >= min_distance ** 2
    if not eligible.any():
        raise ValueError("Minimum distance excludes the entire spatial grid")
    kx, ky = np.unravel_index(np.argmax(np.where(eligible, density, -1)), density.shape)
    x0, x1 = max(0, kx - 2), min(n, kx + 3)
    y0, y1 = max(0, ky - 2), min(n, ky + 3)
    local = density[x0:x1, y0:y1]
    point = dict(x=float((local.sum(axis=1) * x[x0:x1, 0]).sum() / local.sum()),
                 y=float((local.sum(axis=0) * y[0, y0:y1]).sum() / local.sum()))
    if (point["x"] - reference["x"]) ** 2 + (point["y"] - reference["y"]) ** 2 < min_distance ** 2:
        point = dict(x=float(x[kx, ky]), y=float(y[kx, ky]))
    return dict(point, density=float(density[kx, ky]))


def save(path, args, rows):
    payload = dict(schema=2, source=args.source, parent=args.parent, model=args.model, part=args.part,
                   size=args.size, readouts=args.readouts, min_distance=args.min_distance, device=args.device,
                   expected_n=args.total_cases, n=len(rows), rows=rows)
    temporary = path + ".tmp"
    with open(temporary, "w") as f:
        json.dump(payload, f)
    os.replace(temporary, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("parent")
    ap.add_argument("model")
    ap.add_argument("--part", choices=("calibration", "heldout"), required=True)
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--readouts", choices=("default", "peak", "kde", "kde_grid"), nargs="+",
                    default=["default", "kde_grid"])
    ap.add_argument("--min-distance", type=float, default=0.1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--name")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.size <= 0 or not 0 <= args.min_distance < math.sqrt(2) or args.save_every <= 0 or args.limit < 0:
        raise ValueError("Invalid crop size, minimum distance, save interval, or limit")
    if len(set(args.readouts)) != len(args.readouts):
        raise ValueError("Readouts must be unique")
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
    name = args.name or f"multiview-{args.part}-d{args.min_distance:g}"
    os.makedirs("results", exist_ok=True)
    path = f"results/{name}_{len(selected)}.json"
    rows = []
    if args.resume and os.path.exists(path):
        previous = json.load(open(path))
        expected = dict(schema=2, source=args.source, parent=args.parent, model=args.model, part=args.part,
                        size=args.size, readouts=args.readouts, min_distance=args.min_distance,
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
        print(f"Complete: {path}")
        return
    model = LocalSlotModel.from_pretrained(args.model, device_map=args.device)
    started = time.time()
    print(f"{args.part}: {len(selected)} cases, {len(rows)} resumed; second view {args.size}px, "
          f"distance {args.min_distance}", flush=True)
    for i, (source_row, parent_row) in enumerate(selected):
        if str(source_row["id"]) in done:
            continue
        ex = examples[str(source_row["id"])]
        if ex["box"] != source_row["box"] or list(ex["size"]) != list(source_row["image_size"]):
            raise ValueError(f"Source/data mismatch for {source_row['id']}")
        parent_value = parent_row["results"][str(args.size)]
        parent_point = parent_value["readouts"]["kde_grid"]
        alternate = alternate_mode(source_row["probabilities"], source_row["global"]["default"], args.min_distance)
        crop = crop_window(alternate, image_size=ex["size"], side=args.size)
        reused = crop == parent_value["crop"] and all(how in parent_value["readouts"] for how in args.readouts)
        if reused:
            value = dict(ms=0.0, crop=crop, uncertainty=parent_value["uncertainty"], reused=True,
                         readouts={how: parent_value["readouts"][how] for how in args.readouts})
        else:
            with Image.open(ex["image"]) as opened:
                image = opened.convert("RGB")
            local_image = image.crop((crop["left"], crop["top"],
                                      crop["left"] + crop["side"], crop["top"] + crop["side"]))
            lp, ms = model.read("agent", local_image, agent_user(ex["instruction"]))
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
        rows.append(dict(id=source_row["id"], group=source_row["group"], box=source_row["box"],
                         area=source_row["area"], image_size=source_row["image_size"],
                         global_hit=source_row["global"]["default"]["hit"], global_ms=source_row["global_ms"],
                         parent_hit=parent_point["hit"], parent_ms=parent_value["ms"],
                         parent_uncertainty=parent_value["uncertainty"], alternate=alternate, second=value))
        if (i + 1) % args.save_every == 0 or i + 1 == len(selected):
            save(path, args, rows)
            print(f"  {i + 1}/{len(selected)} ({time.time() - started:.0f}s)", flush=True)
    save(path, args, rows)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
