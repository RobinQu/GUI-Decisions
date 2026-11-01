"""Select a confidence rule for two local views and report Pass@2 and final top-1."""

import argparse, json, statistics

import numpy as np

from analyze_zoom import paired_metrics


def complete(path):
    data = json.load(open(path))
    if data.get("expected_n", data.get("n")) != len(data["rows"]):
        raise ValueError(f"Partial result: {path}")
    return data


def validate_split(data, part, allow_partial=False):
    source = json.load(open(data["source"]))
    if source.get("which") != "screenspot-pro" or source.get("n") != 1581 or len(source["rows"]) != 1581:
        raise ValueError("Result must reference the complete ScreenSpot-Pro source")
    if source.get("model") != data.get("model") or data.get("part") != part:
        raise ValueError("Result source, model, or split does not match")
    split = int(round(len(source["rows"]) * 0.5))
    expected = source["rows"][:split] if part == "calibration" else source["rows"][split:]
    ids = [str(r["id"]) for r in data["rows"]]
    if len(set(ids)) != len(ids) or ids != [str(r["id"]) for r in expected[:len(ids)]]:
        raise ValueError(f"Rows are not an ordered prefix of the {part} source split")
    if not allow_partial and len(ids) != len(expected):
        raise ValueError(f"Rows are not the complete {part} source split")


def selected_hits(rows, how, score):
    out = []
    for row in rows:
        use_second = row["second"]["uncertainty"][score] > row["parent_uncertainty"][score]
        out.append(row["second"]["readouts"][how]["hit"] if use_second else row["parent_hit"])
    return out


def method_metrics(rows, pred, include_second=True):
    base = [r["parent_hit"] for r in rows]
    times = [r["global_ms"] + r["parent_ms"] + (r["second"]["ms"] if include_second else 0) for r in rows]
    reads = [include_second and not r["second"]["reused"] for r in rows]
    return paired_metrics(base, pred, times, reads)


def center_visible(row, crop):
    W, H = row["image_size"]
    x = (row["box"][0] + row["box"][2]) / 2
    y = (row["box"][1] + row["box"][3]) / 2
    return (crop["left"] <= x * W <= crop["left"] + crop["side"]
            and crop["top"] <= y * H <= crop["top"] + crop["side"])


def coverage_metrics(rows, parent_rows, size):
    parent = {str(r["id"]): r["results"][str(size)]["crop"] for r in parent_rows}
    first = [center_visible(r, parent[str(r["id"])]) for r in rows]
    second = [center_visible(r, r["second"]["crop"]) for r in rows]
    return dict(n=len(rows), parent_center_visible=float(np.mean(first)),
                second_center_visible=float(np.mean(second)),
                union_center_visible=float(np.mean([a or b for a, b in zip(first, second)])),
                either_new_center=int(sum(b and not a for a, b in zip(first, second))))


def evaluate(rows, how, score):
    parent = [r["parent_hit"] for r in rows]
    second = [r["second"]["readouts"][how]["hit"] for r in rows]
    selected = selected_hits(rows, how, score)
    pass2 = [max(a, b) for a, b in zip(parent, second)]
    discordant = [i for i, (a, b) in enumerate(zip(parent, second)) if a != b]
    return dict(parent=method_metrics(rows, parent, include_second=False), second=method_metrics(rows, second),
                selected=method_metrics(rows, selected),
                pass_at_2=dict(accuracy=float(np.mean(pass2)), correct=int(sum(pass2))),
                selection_gap=float(np.mean(pass2) - np.mean(selected)),
                discordant_n=len(discordant),
                discordant_selected_correct=int(sum(selected[i] for i in discordant)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("calibration")
    ap.add_argument("--heldout")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    calibration_data = complete(args.calibration)
    validate_split(calibration_data, "calibration", args.allow_partial)
    calibration = calibration_data["rows"]
    readouts = calibration_data["readouts"]
    scores = ("score", "top1", "spatial")
    metrics = {how: {score: evaluate(calibration, how, score) for score in scores} for how in readouts}
    selected_readout, selected_score = max(
        ((how, score) for how in readouts for score in scores),
        key=lambda choice: (metrics[choice[0]][choice[1]]["selected"]["accuracy"],
                            metrics[choice[0]][choice[1]]["selected"]["mean_ms"] * -1))
    result = dict(calibration=args.calibration, selected_readout=selected_readout,
                  selected_score=selected_score, min_distance=calibration_data["min_distance"],
                  metrics=metrics)
    parent_data = complete(calibration_data["parent"])
    result["calibration_coverage"] = coverage_metrics(calibration, parent_data["rows"], calibration_data["size"])
    if args.heldout:
        heldout_data = complete(args.heldout)
        validate_split(heldout_data, "heldout", args.allow_partial)
        expected = ("size", "min_distance")
        mismatches = [key for key in expected if heldout_data.get(key) != calibration_data.get(key)]
        if selected_readout not in heldout_data.get("readouts", []) or mismatches:
            raise ValueError(f"Held-out configuration mismatch: {', '.join(mismatches)}")
        heldout = heldout_data["rows"]
        heldout_result = evaluate(heldout, selected_readout, selected_score)
        heldout_parent = complete(heldout_data["parent"])
        heldout_result["coverage"] = coverage_metrics(heldout, heldout_parent["rows"], heldout_data["size"])
        heldout_result["actual_additional_reads"] = sum(not r["second"]["reused"] for r in heldout)
        heldout_result["median_second_ms"] = statistics.median(r["second"]["ms"] for r in heldout)
        for name, grouper in (("by_ui_type", lambda r: r["group"][2]),
                              ("by_area", lambda r: "area<0.001" if r["area"] < 0.001 else
                               "area0.001-0.01" if r["area"] < 0.01 else "area>=0.01")):
            heldout_result[name] = {}
            for group in sorted({grouper(r) for r in heldout}):
                subset = [r for r in heldout if grouper(r) == group]
                heldout_result[name][group] = evaluate(subset, selected_readout, selected_score)
        result["heldout"] = heldout_result
    json.dump(result, open(args.out, "w"), indent=2)
    print(f"selected {selected_readout} + {selected_score}; wrote {args.out}")
    if args.heldout:
        print(json.dumps(result["heldout"], indent=2))


if __name__ == "__main__":
    main()
