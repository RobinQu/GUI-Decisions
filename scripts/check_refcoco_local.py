import argparse, json, os, statistics, time

import torch

from check_gui_owl_local import evaluate as vanilla_evaluate
from gui_owl import parse
from slotread.client import agent_user
from slotread.data.refcoco import point_examples
from slotread.local import LocalSlotModel


def hit(point, box):
    return float(box[0] <= point["x"] <= box[2] and box[1] <= point["y"] <= box[3])


def slots_read(model, ex):
    lp, ms = model.read("agent", ex["image"], agent_user(ex["instruction"]))
    out = {}
    for how in ("default", "peak", "expected"):
        pred = model.tasks["agent"].decode(lp, model.arm, how)
        out[how] = dict(hit=hit(pred, ex["box"]), action_click=float(pred["action"] == "click"),
                        x=pred["x"], y=pred["y"], action=pred["action"])
    return out, ms


def save(path, mode, model_path, rows, seed):
    json.dump(dict(which="refcoco_point", mode=mode, model=model_path, seed=seed,
                   n=len(rows), rows=rows), open(path, "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("slots", "vanilla"))
    ap.add_argument("model")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-area", type=float)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--name")
    a = ap.parse_args()
    name = a.name or f"refcoco-{a.mode}"
    cases = point_examples(a.n, a.seed, a.max_area)
    print(f"RefCOCO point: {len(cases)} cases, {a.mode}, seed {a.seed}", flush=True)
    if a.mode == "slots":
        model = LocalSlotModel.from_pretrained(a.model, device_map=a.device)
        processor = None
    else:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16,
                                                            device_map=a.device, attn_implementation="sdpa").eval()
        processor = AutoProcessor.from_pretrained(a.model)
    os.makedirs("results", exist_ok=True)
    path = f"results/{name}_refcoco_point_{len(cases)}.json"
    rows, started = [], time.time()
    for i, ex in enumerate(cases):
        row = dict(id=ex["id"], group=ex["group"])
        if a.mode == "slots":
            row["slots"], row["ms"] = slots_read(model, ex)
        else:
            row["vanilla"] = vanilla_evaluate(model, processor, ex, "capped", torch.device(a.device))
            action, point = parse(row["vanilla"]["text"])
            row["vanilla"].update(action=action, x=point.get("x"), y=point.get("y"))
        rows.append(row)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(cases)} ({time.time() - started:.0f}s)", flush=True)
            save(path, a.mode, a.model, rows, a.seed)
    if a.mode == "slots":
        for how in ("default", "peak", "expected"):
            vals = [r["slots"][how] for r in rows]
            print(f"slots {how}: hit {statistics.mean(v['hit'] for v in vals):.3f}, "
                  f"action_click {statistics.mean(v['action_click'] for v in vals):.3f}", flush=True)
    else:
        vals = [r["vanilla"] for r in rows]
        print(f"vanilla: hit {statistics.mean(v['hit'] for v in vals):.3f}, "
              f"{statistics.median(v['ms'] for v in vals):.0f} ms median", flush=True)
    save(path, a.mode, a.model, rows, a.seed)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
