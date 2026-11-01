"""Paired full-image, fixed-crop, and uncertainty-guided adaptive-crop slot evaluation."""

import argparse, json, os, random, statistics, time

from PIL import Image
from scipy.stats import binomtest

from slotread.client import agent_user
from slotread.data.evalsets import screenspot, screenspot_pro
from slotread.data.refcoco import point_examples
from slotread.local import LocalSlotModel
from slotread.zoom import adaptive_side, crop_window, distribution_readouts, map_point, point_uncertainty


def hit(point, box):
    return float(box[0] <= point["x"] <= box[2] and box[1] <= point["y"] <= box[3])


def prediction(model, lp, ex):
    out = {}
    for how in ("default", "peak", "expected"):
        p = model.tasks["agent"].decode(lp, model.arm, how)
        out[how] = dict(x=p["x"], y=p["y"], hit=hit(p, ex["box"]),
                        action_click=float(p["action"] == "click"))
    return out


def cases_for(which, seed, max_area):
    if which == "refcoco":
        return point_examples(0, seed, max_area)
    cases = screenspot_pro() if which == "screenspot-pro" else screenspot()
    if max_area is not None:
        cases = [ex for ex in cases if (ex["box"][2] - ex["box"][0]) * (ex["box"][3] - ex["box"][1]) < max_area]
    random.Random(seed).shuffle(cases)
    return cases


def save(path, args, rows):
    payload = dict(schema=4, which=args.dataset, model=args.model, seed=args.seed,
                   max_area=args.max_area, expected_n=args.total_cases, gate_score=args.gate_score,
                   fixed_size=args.fixed_size, min_size=args.min_size, gamma=args.gamma, crop_center=args.crop_center,
                   skip_adaptive=args.skip_adaptive,
                   n=len(rows), rows=rows)
    temporary = path + ".tmp"
    with open(temporary, "w") as f:
        json.dump(payload, f)
    os.replace(temporary, path)


def transitions(base, other):
    repaired = sum(not b and o for b, o in zip(base, other))
    regressed = sum(b and not o for b, o in zip(base, other))
    p = binomtest(min(repaired, regressed), repaired + regressed, 0.5).pvalue if repaired + regressed else 1.0
    return repaired, regressed, p


def summarize(rows):
    base = [r["global"]["default"]["hit"] for r in rows]
    print(f"global default: {statistics.mean(base):.4f} ({sum(base)}/{len(base)})")
    for mode in ("fixed", "adaptive"):
        other = [r[mode]["hit"] if not r[mode]["skipped"] else b for r, b in zip(rows, base)]
        repaired, regressed, p = transitions(base, other)
        ms = [r["global_ms"] + r[mode]["ms"] for r in rows]
        print(f"{mode} all: {statistics.mean(other):.4f}; wrong->right {repaired}, "
              f"right->wrong {regressed}, McNemar p={p:.3g}; median {statistics.median(ms):.0f} ms")
    gated, times, triggered = [], [], 0
    for r in rows:
        use = r["gated"]["use_zoom"]
        triggered += use
        gated.append(r["adaptive"]["hit"] if use else r["global"]["default"]["hit"])
        times.append(r["global_ms"] + (r["adaptive"]["ms"] if use else 0))
    repaired, regressed, p = transitions(base, gated)
    print(f"adaptive gated: {statistics.mean(gated):.4f}; triggered {triggered}/{len(rows)}; "
          f"wrong->right {repaired}, right->wrong {regressed}, McNemar p={p:.3g}; "
          f"median {statistics.median(times):.0f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=("screenspot", "screenspot-pro", "refcoco"))
    ap.add_argument("model")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-area", type=float)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--name")
    ap.add_argument("--fixed-size", type=int, default=512)
    ap.add_argument("--min-size", type=int, default=512)
    ap.add_argument("--gamma", type=float, default=4.0)
    ap.add_argument("--gate-score", type=float, default=1.0)
    ap.add_argument("--crop-center", choices=("default", "kde", "kde_grid"), default="default")
    ap.add_argument("--skip-adaptive", action="store_true")
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.n < 0 or args.save_every <= 0 or args.fixed_size <= 0 or args.min_size <= 0 or args.gamma <= 0:
        raise ValueError("n must be nonnegative; save-every, crop sizes, and gamma must be positive")
    if args.max_area is not None and args.max_area <= 0:
        raise ValueError("--max-area must be positive")
    cases = cases_for(args.dataset, args.seed, args.max_area)
    cases = cases[:args.n] if args.n else cases
    if not cases:
        raise ValueError("No evaluation cases selected")
    args.total_cases = len(cases)
    name = args.name or f"zoom-{args.dataset}-s{args.seed}"
    os.makedirs("results", exist_ok=True)
    path = f"results/{name}_{args.dataset}_zoom_{len(cases)}.json"
    rows = []
    if args.resume and os.path.exists(path):
        previous = json.load(open(path))
        expected = dict(schema=4, which=args.dataset, model=args.model, seed=args.seed,
                        max_area=args.max_area, expected_n=len(cases), gate_score=args.gate_score,
                        fixed_size=args.fixed_size, min_size=args.min_size, gamma=args.gamma,
                        crop_center=args.crop_center, skip_adaptive=args.skip_adaptive)
        mismatches = [key for key, value in expected.items() if previous.get(key) != value]
        if mismatches:
            raise ValueError(f"Resume configuration/schema mismatch: {', '.join(mismatches)}")
        rows = previous["rows"]
        expected_ids = [str(ex.get("id", i)) for i, ex in enumerate(cases)]
        if [str(r["id"]) for r in rows] != expected_ids[:len(rows)]:
            raise ValueError("Resume rows are not a prefix of the current case sequence")
    done = {str(r["id"]) for r in rows}
    started = time.time()
    if len(rows) == len(cases):
        summarize(rows)
        print(f"wrote {path}", flush=True)
        return
    model = LocalSlotModel.from_pretrained(args.model, device_map=args.device)
    print(f"{args.dataset}: {len(cases)} cases, {len(rows)} resumed; fixed {args.fixed_size}, "
          f"adaptive min {args.min_size}, gamma {args.gamma}, gate {args.gate_score}", flush=True)
    for i, ex in enumerate(cases):
        if str(ex.get("id", i)) in done:
            continue
        image = ex["image"]
        if isinstance(image, str):
            with Image.open(image) as opened:
                image = opened.convert("RGB")
        else:
            image = image.convert("RGB")
        user = agent_user(ex["instruction"])
        lp, global_ms = model.read("agent", image, user)
        global_pred = prediction(model, lp, ex)
        px, py, uncertainty = point_uncertainty(lp, model.arm)
        for how, point in distribution_readouts(px, py).items():
            global_pred[how] = dict(point, hit=hit(point, ex["box"]), action_click=global_pred["default"]["action_click"])
        row = dict(id=ex.get("id", i), group=ex["group"], image_size=image.size, box=ex["box"],
                   area=(ex["box"][2] - ex["box"][0]) * (ex["box"][3] - ex["box"][1]),
                   global_ms=global_ms, **{"global": global_pred}, uncertainty=uncertainty,
                   probabilities=dict(x=px.tolist(), y=py.tolist()))
        for mode in ("fixed", "adaptive"):
            if mode == "adaptive" and args.skip_adaptive:
                row[mode] = dict(skipped=True, ms=0.0)
                continue
            side = args.fixed_size if mode == "fixed" else adaptive_side(uncertainty, image.size, args.min_size, args.gamma)
            crop = crop_window(global_pred[args.crop_center], image.size, side)
            full = crop["left"] == 0 and crop["top"] == 0 and crop["side"] == min(image.size)
            if full and image.width == image.height:
                row[mode] = dict(skipped=True, ms=0.0, crop=crop)
                continue
            local_image = image.crop((crop["left"], crop["top"],
                                      crop["left"] + crop["side"], crop["top"] + crop["side"]))
            local_lp, ms = model.read("agent", local_image, user)
            local = model.tasks["agent"].decode(local_lp, model.arm)
            _, _, local_uncertainty = point_uncertainty(local_lp, model.arm)
            mapped = map_point(local, crop, image.size)
            row[mode] = dict(skipped=False, ms=ms, crop=crop, local_x=local["x"], local_y=local["y"],
                             x=mapped["x"], y=mapped["y"], hit=hit(mapped, ex["box"]),
                             action_click=float(local["action"] == "click"), uncertainty=local_uncertainty)
        use_zoom = uncertainty["score"] < args.gate_score and not row["adaptive"]["skipped"]
        row["gated"] = dict(use_zoom=use_zoom)
        rows.append(row)
        if (i + 1) % args.save_every == 0 or i + 1 == len(cases):
            save(path, args, rows)
            print(f"  {i + 1}/{len(cases)} ({time.time() - started:.0f}s)", flush=True)
    summarize(rows)
    save(path, args, rows)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
