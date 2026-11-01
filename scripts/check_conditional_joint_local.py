"""Evaluate p(x,y)=p(x)p(y|x) by filling candidate x tokens into the slot template."""

import argparse, json, os, statistics, time

import torch
from PIL import Image

from slotread.client import agent_user
from slotread.data.evalsets import screenspot_pro
from slotread.joint import candidate_bins, conditional_joint, joint_readouts
from slotread.local import LocalSlotModel
from slotread.numbers import centers, read_number
from slotread.zoom import distribution_readouts, map_point, point_uncertainty

CANDIDATE_COUNTS = (1, 2, 4, 8, 16, 32)


def hit(point, box):
    return float(box[0] <= point["x"] <= box[2] and box[1] <= point["y"] <= box[3])


def complete(path):
    data = json.load(open(path))
    if data.get("expected_n", data.get("n")) != len(data["rows"]):
        raise ValueError(f"Partial result: {path}")
    return data


def save(path, args, rows):
    payload = dict(schema=1, source=args.source, crop_parent=args.crop_parent, model=args.model, part=args.part,
                   max_candidates=args.max_candidates, separation=args.separation, device=args.device,
                   expected_n=args.total_cases, n=len(rows), rows=rows)
    temporary = path + ".tmp"
    with open(temporary, "w") as f:
        json.dump(payload, f)
    os.replace(temporary, path)


def summarize(rows, max_candidates):
    for k in CANDIDATE_COUNTS:
        if k > max_candidates:
            continue
        for how in ("default", "peak", "expected", "region", "kde", "kde_grid"):
            values = [r["results"][str(k)]["readouts"][how]["hit"] for r in rows]
            print(f"K={k} {how:9s}: {statistics.mean(values):.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("model")
    ap.add_argument("--part", choices=("calibration", "heldout"), required=True)
    ap.add_argument("--crop-parent", help="Optional completed 768px crop result to use as the baseline view")
    ap.add_argument("--max-candidates", type=int, default=8)
    ap.add_argument("--separation", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--name")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.max_candidates not in CANDIDATE_COUNTS or args.separation <= 0 or args.save_every <= 0 or args.limit < 0:
        raise ValueError("Invalid candidate count, separation, save interval, or limit")
    source = complete(args.source)
    if (source.get("which") != "screenspot-pro" or source.get("n") != 1581 or source.get("seed") != 0
            or source.get("model") != args.model):
        raise ValueError("Source must be the complete seed-0 ScreenSpot-Pro evaluation with the same model")
    split = int(round(len(source["rows"]) * 0.5))
    selected = source["rows"][:split] if args.part == "calibration" else source["rows"][split:]
    parent_rows = None
    if args.crop_parent:
        parent = complete(args.crop_parent)
        if (parent.get("source") != args.source or parent.get("model") != args.model
                or parent.get("part") != args.part
                or [str(r["id"]) for r in parent["rows"]] != [str(r["id"]) for r in selected]):
            raise ValueError("Crop parent does not match the source, model, and split")
        parent_rows = {str(r["id"]): r for r in parent["rows"]}
    selected = selected[:args.limit] if args.limit else selected
    args.total_cases = len(selected)
    if not selected:
        raise ValueError("No cases selected")
    name = args.name or f"conditional-joint-{args.part}-k{args.max_candidates}-sep{args.separation}"
    os.makedirs("results", exist_ok=True)
    path = f"results/{name}_{len(selected)}.json"
    rows = []
    if args.resume and os.path.exists(path):
        previous = json.load(open(path))
        expected = dict(schema=1, source=args.source, crop_parent=args.crop_parent, model=args.model, part=args.part,
                        max_candidates=args.max_candidates, separation=args.separation, device=args.device,
                        expected_n=len(selected))
        mismatches = [key for key, value in expected.items() if previous.get(key) != value]
        if mismatches:
            raise ValueError(f"Resume configuration/schema mismatch: {', '.join(mismatches)}")
        rows = previous["rows"]
        if [str(r["id"]) for r in rows] != [str(r["id"]) for r in selected[:len(rows)]]:
            raise ValueError("Resume rows are not a prefix of the selected source rows")
    done = {str(r["id"]) for r in rows}
    examples = {str(ex["id"]): ex for ex in screenspot_pro()}
    if len(rows) == len(selected):
        summarize(rows, args.max_candidates)
        return
    model = LocalSlotModel.from_pretrained(args.model, device_map=args.device)
    started = time.time()
    print(f"{args.part}: {len(selected)} cases, {len(rows)} resumed; max K {args.max_candidates}, "
          f"separation {args.separation}", flush=True)
    for i, source_row in enumerate(selected):
        if str(source_row["id"]) in done:
            continue
        ex = examples[str(source_row["id"])]
        if ex["box"] != source_row["box"] or list(ex["size"]) != list(source_row["image_size"]):
            raise ValueError(f"Source/data mismatch for {source_row['id']}")
        with Image.open(ex["image"]) as opened:
            image = opened.convert("RGB")
        user = agent_user(ex["instruction"])
        global_ms = source_row["global_ms"]
        global_default = source_row["global"]["default"]
        global_kde_grid = source_row["global"]["kde_grid"]
        total_ms = 0.0
        crop = None
        if parent_rows is not None:
            parent_value = parent_rows[str(source_row["id"])]["results"]["768"]
            crop = parent_value["crop"]
            local_image = image.crop((crop["left"], crop["top"],
                                      crop["left"] + crop["side"], crop["top"] + crop["side"]))
            lp, total_ms = model.read("agent", local_image, user)
            px, py, _ = point_uncertainty(lp, model.arm)
            local_default = model.tasks["agent"].decode(lp, model.arm, "default")
            local_kde = distribution_readouts(px, py)["kde_grid"]
            global_default = dict(map_point(local_default, crop, image.size))
            global_default["hit"] = hit(global_default, ex["box"])
            global_kde_grid = dict(map_point(local_kde, crop, image.size))
            global_kde_grid["hit"] = hit(global_kde_grid, ex["box"])
            image = local_image
        else:
            px = torch.tensor(source_row["probabilities"]["x"], dtype=torch.float32)
            py = torch.tensor(source_row["probabilities"]["y"], dtype=torch.float32)
        candidates = candidate_bins(px, args.max_candidates, args.separation)
        candidate_rows, conditional_y, cumulative_ms = [], [], []
        for index in candidates:
            token_id = model.cfg["tasks"]["agent"]["labels"]["x"][index]
            lp, ms = model.read("agent", image, user, condition={"x": token_id}, slots=["y"])
            total_ms += ms
            cumulative_ms.append(total_ms)
            probability = torch.softmax(lp["y"], -1)
            conditional_y.append(lp["y"])
            point = dict(x=float(centers(len(px))[index]), y=read_number(lp, "y", model.arm))
            if crop is not None:
                point = map_point(point, crop, source_row["image_size"])
            candidate_rows.append(dict(bin=index, token_id=token_id, px=float(px[index]),
                                       y=probability.tolist(), tv=float(0.5 * (probability - py).abs().sum()),
                                       y_bin_changed=bool(int(lp["y"].argmax()) != int(py.argmax())),
                                       point=dict(point, hit=hit(point, ex["box"]))))
        results = {}
        for k in CANDIDATE_COUNTS:
            if k > args.max_candidates:
                continue
            joint = conditional_joint(px, candidates[:k], conditional_y[:k])
            points = joint_readouts(joint)
            if crop is not None:
                points = {how: map_point(point, crop, source_row["image_size"]) for how, point in points.items()}
            results[str(k)] = dict(ms=cumulative_ms[k - 1], candidate_mass=float(px[candidates[:k]].sum()),
                                   readouts={how: dict(point, hit=hit(point, ex["box"]))
                                             for how, point in points.items()})
        row = dict(id=source_row["id"], group=source_row["group"], box=source_row["box"],
                   area=source_row["area"], image_size=source_row["image_size"], global_ms=global_ms,
                   global_default=global_default, global_kde_grid=global_kde_grid,
                   candidates=candidate_rows, results=results)
        rows.append(row)
        if (i + 1) % args.save_every == 0 or i + 1 == len(selected):
            save(path, args, rows)
            print(f"  {i + 1}/{len(selected)} ({time.time() - started:.0f}s)", flush=True)
    summarize(rows, args.max_candidates)
    save(path, args, rows)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
