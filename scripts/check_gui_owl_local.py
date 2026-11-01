import argparse, json, os, statistics, time

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from gui_owl import CAP, SYSTEM, parse, user_text
from slotread.data.evalsets import screenspot
from slotread.images import cap_image


@torch.inference_mode()
def evaluate(model, processor, ex, setting, device):
    image = cap_image(ex["image"].convert("RGB"), CAP if setting == "capped" else 0)
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [{"type": "text", "text": user_text(ex["instruction"])},
                                         {"type": "image", "image": image}]}]
    enc = processor.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors="pt",
                                       add_generation_prompt=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    out = model.generate(**enc, max_new_tokens=160, do_sample=False)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t) * 1000
    generated = out[0, enc["input_ids"].shape[1]:]
    text = processor.decode(generated, skip_special_tokens=True)
    action, point = parse(text)
    b = ex["box"]
    hit = float(action == "click" and "x" in point and
                b[0] <= point["x"] <= b[2] and b[1] <= point["y"] <= b[3])
    return dict(hit=hit, ms=ms, completion_tokens=len(generated), text=text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--settings", nargs="+", choices=("capped", "native"), default=("capped", "native"))
    ap.add_argument("--shard", help="i/k: evaluate every k-th case starting at i")
    ap.add_argument("--name", default="local_gui_owl_vanilla")
    a = ap.parse_args()
    device = torch.device(a.device)
    model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16,
                                                        device_map=a.device, attn_implementation="sdpa").eval()
    processor = AutoProcessor.from_pretrained(a.model)
    indexed = list(enumerate(screenspot(a.n)))
    if a.shard:
        i, k = map(int, a.shard.split("/"))
        indexed = indexed[i::k]
    print(f"click: {len(indexed)} cases, settings {', '.join(a.settings)}", flush=True)
    rows, started = [], time.time()
    os.makedirs("results", exist_ok=True)
    path = f"results/{a.name}_click_{len(indexed)}.json"
    for i, (case_id, ex) in enumerate(indexed):
        row = dict(id=case_id, group=list(ex["group"]))
        for setting in a.settings:
            row[setting] = evaluate(model, processor, ex, setting, device)
        rows.append(row)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(indexed)} ({time.time() - started:.0f}s)", flush=True)
            json.dump(dict(which="click", n=len(rows), model=a.model, rows=rows), open(path, "w"))
    for setting in a.settings:
        rs = [r[setting] for r in rows]
        print(f"vanilla GUI-Owl, {setting} images: hit {statistics.mean(r['hit'] for r in rs):.3f} "
              f"({len(rs)} cases, {statistics.median(r['completion_tokens'] for r in rs):.0f} output tokens, "
              f"{statistics.median(r['ms'] for r in rs):.0f} ms median)", flush=True)
    json.dump(dict(which="click", n=len(rows), model=a.model, rows=rows), open(path, "w"))
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
