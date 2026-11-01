"""Select conditional-joint candidate count/readout and evaluate it on held-out cases."""

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
    if (source.get("which") != "screenspot-pro" or source.get("n") != 1581 or source.get("seed") != 0
            or len(source["rows"]) != 1581):
        raise ValueError("Result must reference the complete seed-0 ScreenSpot-Pro source")
    if source.get("model") != data.get("model") or data.get("part") != part:
        raise ValueError("Result source, model, or split does not match")
    split = int(round(len(source["rows"]) * 0.5))
    expected = source["rows"][:split] if part == "calibration" else source["rows"][split:]
    ids = [str(r["id"]) for r in data["rows"]]
    if len(set(ids)) != len(ids) or ids != [str(r["id"]) for r in expected[:len(ids)]]:
        raise ValueError(f"Rows are not an ordered prefix of the {part} source split")
    if not allow_partial and len(ids) != len(expected):
        raise ValueError(f"Rows are not the complete {part} source split")


def metrics(rows, k, how, baseline="default"):
    pred = [r["results"][str(k)]["readouts"][how]["hit"] for r in rows]
    base = [r["global_default" if baseline == "default" else "global_kde_grid"]["hit"] for r in rows]
    times = [r["global_ms"] + r["results"][str(k)]["ms"] for r in rows]
    return paired_metrics(base, pred, times, [True] * len(rows), [k] * len(rows))


def distribution_stats(rows, k):
    candidates = [c for r in rows for c in r["candidates"][:k]]
    pass_k = [max(c["point"]["hit"] for c in r["candidates"][:k]) for r in rows]
    masses = [r["results"][str(k)]["candidate_mass"] for r in rows]
    return dict(mean_candidate_mass=float(np.mean(masses)), median_candidate_mass=float(statistics.median(masses)),
                mean_tv=float(np.mean([c["tv"] for c in candidates])),
                y_bin_change_rate=float(np.mean([c["y_bin_changed"] for c in candidates])),
                pass_at_k=dict(accuracy=float(np.mean(pass_k)), correct=int(sum(pass_k))))


def evaluate(rows, k, how):
    return dict(vs_default=metrics(rows, k, how, "default"),
                vs_kde_grid=metrics(rows, k, how, "kde_grid"),
                distribution=distribution_stats(rows, k))


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
    ks = sorted(int(k) for k in calibration[0]["results"])
    readouts = ("default", "peak", "expected", "region", "kde", "kde_grid")
    metrics_by_choice = {str(k): {how: evaluate(calibration, k, how) for how in readouts} for k in ks}
    selected_k, selected_readout = max(
        ((k, how) for k in ks for how in readouts),
        key=lambda choice: (metrics_by_choice[str(choice[0])][choice[1]]["vs_default"]["accuracy"],
                            -metrics_by_choice[str(choice[0])][choice[1]]["vs_default"]["mean_ms"]))
    result = dict(calibration=args.calibration, selected_k=selected_k, selected_readout=selected_readout,
                  separation=calibration_data["separation"], metrics=metrics_by_choice)
    if args.heldout:
        heldout_data = complete(args.heldout)
        validate_split(heldout_data, "heldout", args.allow_partial)
        if (heldout_data.get("separation") != calibration_data.get("separation")
                or heldout_data.get("model") != calibration_data.get("model")
                or str(selected_k) not in heldout_data["rows"][0]["results"]):
            raise ValueError("Held-out result does not contain the selected candidate configuration")
        heldout = heldout_data["rows"]
        selected = evaluate(heldout, selected_k, selected_readout)
        for name, grouper in (("by_ui_type", lambda r: r["group"][2]),
                              ("by_area", lambda r: "area<0.001" if r["area"] < 0.001 else
                               "area0.001-0.01" if r["area"] < 0.01 else "area>=0.01")):
            selected[name] = {}
            for group in sorted({grouper(r) for r in heldout}):
                subset = [r for r in heldout if grouper(r) == group]
                selected[name][group] = evaluate(subset, selected_k, selected_readout)
        result["heldout"] = selected
    json.dump(result, open(args.out, "w"), indent=2)
    print(f"selected K={selected_k} + {selected_readout}; wrote {args.out}")
    if args.heldout:
        print(json.dumps(result["heldout"], indent=2))


if __name__ == "__main__":
    main()
