"""Evaluate the official GUI-Actor 2D patch-attention model on the same ScreenSpot-Pro cases."""

import argparse, json, os, time

import torch
from PIL import Image

from slotread.data.evalsets import screenspot_pro


def hit(point, box):
    return float(box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3])


def save(path, args, rows):
    payload = dict(schema=1, source=args.source, model=args.model, device=args.device,
                   resize_to_pixels=args.resize_to_pixels, expected_n=args.total_cases, n=len(rows), rows=rows)
    temporary = path + ".tmp"
    with open(temporary, "w") as f:
        json.dump(payload, f)
    os.replace(temporary, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("model")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--name", default="gui-actor-2b-sspro-s0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resize-to-pixels", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.limit < 0 or args.save_every <= 0 or args.resize_to_pixels < 0:
        raise ValueError("limit, resize pixels, and save interval must be nonnegative")
    source = json.load(open(args.source))
    if source.get("which") != "screenspot-pro" or source.get("n") != 1581 or source.get("seed") != 0:
        raise ValueError("Source must be the complete seed-0 ScreenSpot-Pro evaluation")
    selected = source["rows"][:args.limit] if args.limit else source["rows"]
    args.total_cases = len(selected)
    path = f"results/{args.name}_{len(selected)}.json"
    rows = []
    if args.resume and os.path.exists(path):
        previous = json.load(open(path))
        expected = dict(schema=1, source=args.source, model=args.model, device=args.device,
                        resize_to_pixels=args.resize_to_pixels, expected_n=len(selected))
        mismatches = [key for key, value in expected.items() if previous.get(key) != value]
        if mismatches:
            raise ValueError(f"Resume configuration/schema mismatch: {', '.join(mismatches)}")
        rows = previous["rows"]
        if [str(r["id"]) for r in rows] != [str(r["id"]) for r in selected[:len(rows)]]:
            raise ValueError("Resume rows are not a prefix of the selected source rows")
    done = {str(r["id"]) for r in rows}
    if len(rows) == len(selected):
        print(f"Complete: {path}")
        return
    from gui_actor.inference import inference
    from gui_actor.modeling import Qwen2VLForConditionalGenerationWithPointer
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen2VLForConditionalGenerationWithPointer.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device, attn_implementation="sdpa").eval()
    examples = {str(ex["id"]): ex for ex in screenspot_pro()}
    started = time.time()
    print(f"GUI-Actor: {len(selected)} cases, {len(rows)} resumed", flush=True)
    for i, source_row in enumerate(selected):
        if str(source_row["id"]) in done:
            continue
        ex = examples[str(source_row["id"])]
        if ex["box"] != source_row["box"] or list(ex["size"]) != list(source_row["image_size"]):
            raise ValueError(f"Source/data mismatch for {source_row['id']}")
        with Image.open(ex["image"]) as opened:
            image = opened.convert("RGB")
        if args.resize_to_pixels and image.width * image.height != args.resize_to_pixels:
            ratio = (args.resize_to_pixels / (image.width * image.height)) ** 0.5
            image = image.resize((int(image.width * ratio), int(image.height * ratio)))
        system = ("You are a GUI agent. You are given a task and a screenshot of the screen. "
                  "You need to perform a series of pyautogui actions to complete the task.")
        conversation = [{"role": "system", "content": [{"type": "text", "text": system}]},
                        {"role": "user", "content": [{"type": "image", "image": image},
                                                     {"type": "text", "text": ex["instruction"]}]}]
        t0 = time.perf_counter()
        with torch.no_grad():
            prediction = inference(conversation, model, processor.tokenizer, processor,
                                   use_placeholder=True, topk=3)
        ms = (time.perf_counter() - t0) * 1000
        points = prediction["topk_points"]
        if not points:
            raise ValueError(f"GUI-Actor returned no points for {source_row['id']}")
        rows.append(dict(id=source_row["id"], group=source_row["group"], box=source_row["box"],
                         area=source_row["area"], image_size=source_row["image_size"], ms=ms,
                         points=points, top1_hit=hit(points[0], ex["box"]),
                         top3_hit=max(hit(point, ex["box"]) for point in points)))
        if (i + 1) % args.save_every == 0 or i + 1 == len(selected):
            save(path, args, rows)
            accuracy = sum(r["top1_hit"] for r in rows) / len(rows)
            print(f"  {i + 1}/{len(selected)} ({time.time() - started:.0f}s) top1={accuracy:.4f}", flush=True)
    save(path, args, rows)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
