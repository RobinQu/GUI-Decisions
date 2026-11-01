"""Select a recursive zoom configuration and compare it with direct small crops."""

import argparse, json

import numpy as np

from analyze_zoom import paired_metrics


def complete(path):
    data = json.load(open(path))
    if data.get("expected_n", data.get("n")) != len(data["rows"]):
        raise ValueError(f"Partial result: {path}")
    if not data.get("include_direct"):
        raise ValueError(f"Direct-crop comparison is missing: {path}")
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


def prediction(rows, size, how, branch):
    hits = [r["results"][str(size)][branch]["readouts"][how]["hit"] for r in rows]
    if branch == "sequential":
        times = [r["global_ms"] + r["parent_ms"] + r["results"][str(size)][branch]["ms"] for r in rows]
    else:
        times = [r["global_ms"] + r["results"][str(size)][branch]["ms"] for r in rows]
    return hits, times


def comparison(rows, size, how, branch, baseline):
    pred, times = prediction(rows, size, how, branch)
    if baseline == "parent":
        base = [r["parent_hit"] for r in rows]
    elif baseline == "global":
        base = [r["global_hit"] for r in rows]
    else:
        base, _ = prediction(rows, size, how, "direct")
    return paired_metrics(base, pred, times, [branch == "sequential"] * len(rows))


def stratified(rows, size, how, grouper):
    out = {}
    for name in sorted({grouper(r) for r in rows}):
        selected = [r for r in rows if grouper(r) == name]
        out[name] = dict(sequential=comparison(selected, size, how, "sequential", "parent"),
                         sequential_vs_direct=comparison(selected, size, how, "sequential", "direct"))
    return out


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
    sizes = sorted(int(size) for size in calibration[0]["results"])
    readouts = calibration_data["readouts"]
    metrics = {str(size): {how: {
        "direct": comparison(calibration, size, how, "direct", "parent"),
        "sequential": comparison(calibration, size, how, "sequential", "parent"),
        "sequential_vs_direct": comparison(calibration, size, how, "sequential", "direct")}
        for how in readouts} for size in sizes}
    selected_size, selected_readout = max(
        ((size, how) for size in sizes for how in readouts),
        key=lambda choice: (metrics[str(choice[0])][choice[1]]["sequential"]["accuracy"],
                            -metrics[str(choice[0])][choice[1]]["sequential"]["mean_ms"]))
    result = dict(calibration=args.calibration, selected_size=selected_size,
                  selected_readout=selected_readout, metrics=metrics)
    if args.heldout:
        heldout_data = complete(args.heldout)
        validate_split(heldout_data, "heldout", args.allow_partial)
        if heldout_data.get("sizes") != [selected_size] or heldout_data.get("readouts") != [selected_readout]:
            raise ValueError("Held-out run must contain exactly the selected configuration")
        heldout = heldout_data["rows"]
        direct, _ = prediction(heldout, selected_size, selected_readout, "direct")
        sequential, _ = prediction(heldout, selected_size, selected_readout, "sequential")
        parent = [r["parent_hit"] for r in heldout]
        result["heldout"] = dict(
            direct=comparison(heldout, selected_size, selected_readout, "direct", "parent"),
            sequential=comparison(heldout, selected_size, selected_readout, "sequential", "parent"),
            sequential_vs_direct=comparison(heldout, selected_size, selected_readout, "sequential", "direct"),
            pass_at_3=dict(accuracy=float(np.mean([max(p, d, s) for p, d, s in zip(parent, direct, sequential)])),
                           correct=int(sum(max(p, d, s) for p, d, s in zip(parent, direct, sequential)))),
            by_area=stratified(heldout, selected_size, selected_readout,
                               lambda r: "area<0.001" if r["area"] < 0.001 else
                               "area0.001-0.01" if r["area"] < 0.01 else "area>=0.01"),
            by_ui_type=stratified(heldout, selected_size, selected_readout, lambda r: r["group"][2]))
    json.dump(result, open(args.out, "w"), indent=2)
    print(f"selected {selected_size}px + {selected_readout}; wrote {args.out}")
    if args.heldout:
        print(json.dumps(result["heldout"], indent=2))


if __name__ == "__main__":
    main()
