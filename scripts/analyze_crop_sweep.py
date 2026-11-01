"""Select crop size/readout on calibration cases and optionally evaluate it on held-out cases."""

import argparse, json

from analyze_zoom import paired_metrics


def metrics(rows, size, how):
    base = [r["global_hit"] for r in rows]
    values = [r["results"][str(size)] for r in rows]
    pred = [v["readouts"][how]["hit"] for v in values]
    times = [r["global_ms"] + v["ms"] for r, v in zip(rows, values)]
    return paired_metrics(base, pred, times, [True] * len(rows))


def stratified(rows, size, how, grouper):
    out = {}
    for name in sorted({grouper(r) for r in rows}):
        selected = [r for r in rows if grouper(r) == name]
        out[name] = metrics(selected, size, how)
    return out


def complete_data(path):
    data = json.load(open(path))
    if data["n"] != len(data["rows"]):
        raise ValueError(f"Row count does not match metadata: {path}")
    if data.get("expected_n", data["n"]) != data["n"]:
        raise ValueError(f"Partial result: {path}")
    return data


def validate_split(data, source, part, expected_rows, source_path):
    if data.get("source") != source_path or data.get("model") != source["model"] or data.get("part") != part:
        raise ValueError(f"{part} result has a different source, model, or split")
    if len(data["rows"]) != len(expected_rows):
        raise ValueError(f"{part} result is not the complete source split")
    if [str(r["id"]) for r in data["rows"]] != [str(r["id"]) for r in expected_rows]:
        raise ValueError(f"{part} IDs do not match the source split")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("calibration")
    ap.add_argument("--heldout")
    ap.add_argument("--source", required=True)
    ap.add_argument("--reference-size", type=int, default=512)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    source = complete_data(args.source)
    if (source.get("which") != "screenspot-pro" or source.get("n") != 1581
            or source.get("fixed_size") != args.reference_size or source.get("crop_center") != "default"):
        raise ValueError("Source or reference crop configuration is incompatible")
    source_rows = source["rows"]
    split = int(round(len(source_rows) * 0.5))
    calibration_data = complete_data(args.calibration)
    validate_split(calibration_data, source, "calibration", source_rows[:split], args.source)
    calibration = calibration_data["rows"]
    sizes = sorted(int(size) for size in calibration[0]["results"])
    readouts = ("default", "peak", "kde", "kde_grid")
    calibration_metrics = {str(size): {how: metrics(calibration, size, how) for how in readouts} for size in sizes}
    selected_size, selected_readout = max(
        ((size, how) for size in sizes for how in readouts),
        key=lambda choice: (calibration_metrics[str(choice[0])][choice[1]]["accuracy"],
                            -calibration_metrics[str(choice[0])][choice[1]]["mean_ms"]))
    result = dict(calibration=args.calibration, selected_size=selected_size,
                  selected_readout=selected_readout, metrics=calibration_metrics)
    if args.heldout:
        heldout_data = complete_data(args.heldout)
        validate_split(heldout_data, source, "heldout", source_rows[split:], args.source)
        if heldout_data.get("sizes") != [selected_size] or heldout_data.get("readouts") != [selected_readout]:
            raise ValueError("Held-out run must contain exactly the preselected size and readout")
        heldout = heldout_data["rows"]
        reference = {str(r["id"]): r for r in source_rows[split:]}
        selected = metrics(heldout, selected_size, selected_readout)
        baseline = paired_metrics([r["global_hit"] for r in heldout], [r["global_hit"] for r in heldout],
                                  [r["global_ms"] for r in heldout], [False] * len(heldout))
        reference_metrics = paired_metrics(
            [r["global_hit"] for r in heldout], [reference[str(r["id"])]["fixed"]["hit"] for r in heldout],
            [r["global_ms"] + reference[str(r["id"])]["fixed"]["ms"] for r in heldout], [True] * len(heldout))
        selected_values = [r["results"][str(selected_size)] for r in heldout]
        comparison = paired_metrics(
            [reference[str(r["id"])]["fixed"]["hit"] for r in heldout],
            [v["readouts"][selected_readout]["hit"] for v in selected_values],
            [r["global_ms"] + v["ms"] for r, v in zip(heldout, selected_values)], [True] * len(heldout))
        result["heldout"] = dict(global_default=baseline, reference_size=args.reference_size,
                                 reference_default=reference_metrics,
                                 selected=selected, selected_vs_reference=comparison,
                                 by_area=stratified(heldout, selected_size, selected_readout,
                                                    lambda r: "area<0.001" if r["area"] < 0.001 else
                                                    "area0.001-0.01" if r["area"] < 0.01 else "area>=0.01"),
                                 by_ui_type=stratified(heldout, selected_size, selected_readout,
                                                       lambda r: r["group"][2]),
                                 by_category=stratified(heldout, selected_size, selected_readout,
                                                        lambda r: r["group"][0]))
        combined = calibration + heldout
        combined_reference = {str(r["id"]): r for r in source_rows}
        combined_values = [r["results"][str(selected_size)] for r in combined]
        result["combined"] = dict(
            selected=metrics(combined, selected_size, selected_readout),
            reference_default=paired_metrics(
                [r["global_hit"] for r in combined],
                [combined_reference[str(r["id"])]["fixed"]["hit"] for r in combined],
                [r["global_ms"] + combined_reference[str(r["id"])]["fixed"]["ms"] for r in combined],
                [True] * len(combined)),
            selected_vs_reference=paired_metrics(
                [combined_reference[str(r["id"])]["fixed"]["hit"] for r in combined],
                [v["readouts"][selected_readout]["hit"] for v in combined_values],
                [r["global_ms"] + v["ms"] for r, v in zip(combined, combined_values)],
                [True] * len(combined)))
    json.dump(result, open(args.out, "w"), indent=2)
    print(f"selected {selected_size}px + {selected_readout}; wrote {args.out}")
    if args.heldout:
        print(json.dumps(result["heldout"], indent=2))


if __name__ == "__main__":
    main()
