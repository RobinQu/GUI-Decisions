"""Decompose fixed-crop errors by target visibility and available readout candidates."""

import argparse, json

import numpy as np


def complete(path):
    data = json.load(open(path))
    if data.get("expected_n", data.get("n")) != len(data["rows"]):
        raise ValueError(f"Partial result: {path}")
    return data


def coverage(row, size=768):
    crop = row["results"][str(size)]["crop"]
    W, H = row["image_size"]
    left, right = crop["left"] / W, (crop["left"] + crop["side"]) / W
    top, bottom = crop["top"] / H, (crop["top"] + crop["side"]) / H
    x1, y1, x2, y2 = row["box"]
    intersection = max(0.0, min(right, x2) - max(left, x1)) * max(0.0, min(bottom, y2) - max(top, y1))
    center = (x1 + x2) / 2, (y1 + y2) / 2
    center_visible = left <= center[0] <= right and top <= center[1] <= bottom
    full_visible = left <= x1 and x2 <= right and top <= y1 and y2 <= bottom
    return dict(center_visible=center_visible, full_visible=full_visible,
                intersection=intersection > 0, box_fraction=intersection / row["area"])


def rates(rows):
    if not rows:
        return dict(n=0)
    hits = [r["results"]["768"]["readouts"]["kde_grid"]["hit"] for r in rows]
    return dict(n=len(rows), correct=int(sum(hits)), accuracy=float(np.mean(hits)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("calibration")
    ap.add_argument("heldout")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    datasets = {name: complete(path) for name, path in
                (("calibration", args.calibration), ("heldout", args.heldout))}
    source_paths = {data.get("source") for data in datasets.values()}
    models = {data.get("model") for data in datasets.values()}
    if len(source_paths) != 1 or len(models) != 1:
        raise ValueError("Calibration and held-out results must use the same source and model")
    source = complete(source_paths.pop())
    if source.get("which") != "screenspot-pro" or source.get("n") != 1581:
        raise ValueError("Results must reference the complete ScreenSpot-Pro source")
    split = int(round(len(source["rows"]) * 0.5))
    expected = dict(calibration=source["rows"][:split], heldout=source["rows"][split:])
    for name, data in datasets.items():
        if source.get("model") != data.get("model") or [str(r["id"]) for r in data["rows"]] != [
                str(r["id"]) for r in expected[name]]:
            raise ValueError(f"Rows do not match the complete {name} source split")
    result = {}
    for name, data in datasets.items():
        rows = data["rows"]
        if data.get("part") != name or len(rows) not in (790, 791):
            raise ValueError(f"Unexpected split: {name}")
        if any("kde_grid" not in r["results"]["768"]["readouts"] for r in rows):
            raise ValueError(f"Missing 768px KDE-grid readout: {name}")
        for row in rows:
            row["coverage"] = coverage(row)
        groups = {
            "no_intersection": [r for r in rows if not r["coverage"]["intersection"]],
            "partial_center_outside": [r for r in rows if r["coverage"]["intersection"]
                                       and not r["coverage"]["center_visible"]],
            "center_visible": [r for r in rows if r["coverage"]["center_visible"]],
            "fully_visible": [r for r in rows if r["coverage"]["full_visible"]],
        }
        misses = [r for r in rows if not r["results"]["768"]["readouts"]["kde_grid"]["hit"]]
        miss_groups = {
            "no_intersection": [r for r in misses if not r["coverage"]["intersection"]],
            "partial_center_outside": [r for r in misses if r["coverage"]["intersection"]
                                       and not r["coverage"]["center_visible"]],
            "center_visible": [r for r in misses if r["coverage"]["center_visible"]],
        }
        by_type = {}
        for ui_type in ("text", "icon"):
            selected = [r for r in rows if r["group"][2] == ui_type]
            by_type[ui_type] = dict(
                all=rates(selected),
                center_visible=rates([r for r in selected if r["coverage"]["center_visible"]]),
                misses={key: len([r for r in selected if not r["results"]["768"]["readouts"]["kde_grid"]["hit"]
                                  and condition(r)]) for key, condition in (
                    ("no_intersection", lambda r: not r["coverage"]["intersection"]),
                    ("partial_center_outside", lambda r: r["coverage"]["intersection"]
                     and not r["coverage"]["center_visible"]),
                    ("center_visible", lambda r: r["coverage"]["center_visible"]))})
        part = dict(overall=rates(rows), coverage={key: rates(value) for key, value in groups.items()},
                    miss_counts={key: len(value) for key, value in miss_groups.items()},
                    miss_fractions={key: len(value) / len(misses) for key, value in miss_groups.items()},
                    mean_box_fraction=float(np.mean([r["coverage"]["box_fraction"] for r in rows])),
                    by_ui_type=by_type)
        if name == "calibration":
            readouts = ("default", "peak", "kde", "kde_grid")
            candidate_hit = [max(r["results"]["768"]["readouts"][how]["hit"] for how in readouts) for r in rows]
            part["pass_at_4_readouts"] = dict(correct=int(sum(candidate_hit)), accuracy=float(np.mean(candidate_hit)))
            part["candidate_gain_over_kde_grid"] = float(np.mean(candidate_hit) - np.mean(
                [r["results"]["768"]["readouts"]["kde_grid"]["hit"] for r in rows]))
        result[name] = part
    json.dump(result, open(args.out, "w"), indent=2)
    print(json.dumps(result, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
