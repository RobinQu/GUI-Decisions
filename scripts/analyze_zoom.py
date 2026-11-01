"""Select a zoom gate on calibration cases and report paired metrics on held-out cases."""

import argparse, json, statistics

import numpy as np
from scipy.stats import binomtest, rankdata


def outcome(row, mode, threshold=None):
    base = row["global"]["default"]["hit"]
    if mode == "global":
        return base, False
    if mode == "fixed_gated":
        use = row["uncertainty"]["score"] < threshold and not row["fixed"]["skipped"]
        return (row["fixed"]["hit"] if use else base), use
    if mode in ("gated", "accepted", "gated_accept"):
        use = not row["adaptive"]["skipped"]
        if mode != "accepted":
            use = use and row["uncertainty"]["score"] < threshold
        if mode != "gated":
            use = use and row["adaptive"]["uncertainty"]["score"] > row["uncertainty"]["score"]
        return (row["adaptive"]["hit"] if use else base), use
    use = not row[mode]["skipped"]
    return (row[mode]["hit"] if use else base), use


def paired_metrics(base, pred, times, triggered, zoom_reads=None):
    repaired = sum(not b and p for b, p in zip(base, pred))
    regressed = sum(b and not p for b, p in zip(base, pred))
    discordant = repaired + regressed
    pvalue = binomtest(min(repaired, regressed), discordant, 0.5).pvalue if discordant else 1.0
    rng = np.random.default_rng(20261101)
    diffs = np.array(pred) - np.array(base)
    if len(base) > 1:
        boot = np.array([diffs[rng.integers(0, len(base), len(base))].mean() for _ in range(10000)])
        ci = np.quantile(boot, [0.025, 0.975]).tolist()
    else:
        ci = [float(diffs.mean()), float(diffs.mean())]
    zoom_reads = triggered if zoom_reads is None else zoom_reads
    return dict(n=len(base), accuracy=float(np.mean(pred)), correct=int(sum(pred)),
                delta=float(np.mean(diffs)), ci95=ci, wrong_to_right=repaired,
                right_to_wrong=regressed, mcnemar_p=pvalue,
                triggered=sum(triggered), trigger_rate=float(np.mean(triggered)),
                zoom_reads=sum(zoom_reads), read_rate=float(np.mean(zoom_reads)),
                median_ms=float(statistics.median(times)), mean_ms=float(statistics.mean(times)))


def metrics(rows, mode, threshold=None):
    pairs = [outcome(r, mode, threshold) for r in rows]
    base = [r["global"]["default"]["hit"] for r in rows]
    crop = "adaptive" if mode in ("gated", "accepted", "gated_accept") else "fixed" if mode == "fixed_gated" else mode
    reads = []
    for r, (_, use) in zip(rows, pairs):
        if mode == "accepted":
            reads.append(not r["adaptive"]["skipped"])
        elif mode == "gated_accept":
            reads.append(r["uncertainty"]["score"] < threshold and not r["adaptive"]["skipped"])
        else:
            reads.append(use)
    times = [r["global_ms"] + (r[crop]["ms"] if read else 0) for r, read in zip(rows, reads)]
    return paired_metrics(base, [p[0] for p in pairs], times, [p[1] for p in pairs], reads)


def readout_metrics(rows, how):
    base = [r["global"]["default"]["hit"] for r in rows]
    pred = [r["global"][how]["hit"] for r in rows]
    return paired_metrics(base, pred, [r["global_ms"] for r in rows], [False] * len(rows))


def choose_threshold(rows):
    candidates = np.arange(0.0, 2.001, 0.025)
    scored = []
    for threshold in candidates:
        pairs = [outcome(r, "fixed_gated", float(threshold)) for r in rows]
        scored.append((float(np.mean([p[0] for p in pairs])), -sum(p[1] for p in pairs),
                       -threshold, float(threshold)))
    return max(scored)[-1]


def auroc(rows):
    labels = np.array([r["global"]["default"]["hit"] for r in rows])
    scores = np.array([r["uncertainty"]["score"] for r in rows])
    positive, negative = int(labels.sum()), len(labels) - int(labels.sum())
    if not positive or not negative:
        return None
    ranks = rankdata(scores)
    return float((ranks[labels == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))


def area_metrics(rows, threshold):
    bins = [("area<0.001", 0.0, 0.001), ("area0.001-0.01", 0.001, 0.01), ("area>=0.01", 0.01, 2.0)]
    out = {}
    for name, lo, hi in bins:
        selected = [r for r in rows if lo <= r["area"] < hi]
        if selected:
            out[name] = {mode: metrics(selected, mode, threshold) for mode in ("global", "fixed", "fixed_gated")}
    return out


def group_metrics(rows, threshold, index):
    out = {}
    names = {r["group"][index] if len(r["group"]) > index else "all" for r in rows}
    for name in sorted(names):
        selected = [r for r in rows if (r["group"][index] if len(r["group"]) > index else "all") == name]
        out[name] = {mode: metrics(selected, mode, threshold) for mode in ("global", "fixed", "fixed_gated")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--calibration", type=float, default=0.5)
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--out")
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    data = json.load(open(args.path))
    rows = data["rows"]
    if data.get("expected_n", len(rows)) != len(rows) and not args.allow_partial:
        raise ValueError(f"Partial result: {len(rows)}/{data['expected_n']} cases; use --allow-partial to analyze anyway")
    if not 0 < args.calibration < 1:
        raise ValueError("--calibration must be strictly between 0 and 1")
    split = int(round(len(rows) * args.calibration))
    calibration, test = rows[:split], rows[split:]
    if not calibration or not test:
        raise ValueError("Calibration and held-out splits must both contain at least one case")
    threshold = args.threshold if args.threshold is not None else choose_threshold(calibration)
    modes = ["global", "fixed", "fixed_gated", "adaptive", "gated"]
    if any(not r["adaptive"]["skipped"] for r in rows) and all(r["adaptive"]["skipped"] or "uncertainty" in r["adaptive"] for r in rows):
        modes += ["accepted", "gated_accept"]
    result = dict(source=args.path, threshold=threshold, threshold_mode="fixed_gated",
                  calibration_n=len(calibration), test_n=len(test),
                  calibration={mode: metrics(calibration, mode, threshold) for mode in modes},
                  test={mode: metrics(test, mode, threshold) for mode in modes},
                  test_score_auroc=auroc(test), test_by_area=area_metrics(test, threshold),
                  test_by_ui_type=group_metrics(test, threshold, 2),
                  test_by_category=group_metrics(test, threshold, 0))
    test_oracle = [max(r["global"]["default"]["hit"], outcome(r, "adaptive")[0]) for r in test]
    result["test_adaptive_oracle"] = dict(accuracy=float(np.mean(test_oracle)), correct=int(sum(test_oracle)))
    result["test_global_readouts"] = {how: readout_metrics(test, how)
                                       for how in ("peak", "expected", "region", "kde", "kde_grid")
                                       if all(how in r["global"] for r in test)}
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        json.dump(result, open(args.out, "w"), indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
