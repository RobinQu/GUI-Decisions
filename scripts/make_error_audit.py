"""Create stratified visual audit sheets for local-crop grounding errors."""

import argparse, json, random

from PIL import Image, ImageDraw

from analyze_crop_errors import complete, coverage
from slotread.data.evalsets import screenspot_pro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--per-type", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20261101)
    ap.add_argument("--out-prefix", default="results/sspro-local-error-audit")
    args = ap.parse_args()
    if args.per_type <= 0:
        raise ValueError("per-type must be positive")
    data = complete(args.result)
    rows = [r for r in data["rows"] if not r["results"]["768"]["readouts"]["kde_grid"]["hit"]
            and coverage(r)["center_visible"]]
    examples = {str(ex["id"]): ex for ex in screenspot_pro()}
    rng = random.Random(args.seed)
    manifest = []
    for ui_type in ("text", "icon"):
        selected = [r for r in rows if r["group"][2] == ui_type]
        rng.shuffle(selected)
        selected = selected[:args.per_type]
        if not selected:
            print(f"no auditable {ui_type} errors")
            continue
        tile_side, label_height, columns = 512, 64, 2
        sheet = Image.new("RGB", (columns * tile_side, ((len(selected) + columns - 1) // columns) *
                                  (tile_side + label_height)), "white")
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(selected):
            ex = examples[str(row["id"])]
            value = row["results"]["768"]
            crop = value["crop"]
            with Image.open(ex["image"]) as opened:
                local = opened.convert("RGB").crop((crop["left"], crop["top"],
                                                    crop["left"] + crop["side"], crop["top"] + crop["side"]))
            W, H = ex["size"]
            box = [(row["box"][0] * W - crop["left"]) / crop["side"],
                   (row["box"][1] * H - crop["top"]) / crop["side"],
                   (row["box"][2] * W - crop["left"]) / crop["side"],
                   (row["box"][3] * H - crop["top"]) / crop["side"]]
            point = value["readouts"]["kde_grid"]
            px = (point["x"] * W - crop["left"]) / crop["side"]
            py = (point["y"] * H - crop["top"]) / crop["side"]
            overlay = ImageDraw.Draw(local)
            overlay.rectangle(tuple(v * crop["side"] for v in box), outline="red", width=4)
            overlay.line((px * crop["side"] - 18, py * crop["side"],
                          px * crop["side"] + 18, py * crop["side"]), fill="blue", width=4)
            overlay.line((px * crop["side"], py * crop["side"] - 18,
                          px * crop["side"], py * crop["side"] + 18), fill="blue", width=4)
            local = local.resize((tile_side, tile_side))
            x = (i % columns) * tile_side
            y = (i // columns) * (tile_side + label_height)
            sheet.paste(local, (x, y))
            draw.text((x + 4, y + tile_side + 3), f"{i}: {row['id']}", fill="black")
            draw.text((x + 4, y + tile_side + 20), ex["instruction"][:90], fill="black")
            draw.text((x + 4, y + tile_side + 38), "red=target box; blue=768px KDE-grid", fill="black")
            manifest.append(dict(index=i, ui_type=ui_type, id=row["id"], instruction=ex["instruction"],
                                 point={key: point[key] for key in ("x", "y")}, box=row["box"], crop=crop))
        path = f"{args.out_prefix}-{ui_type}.jpg"
        sheet.save(path, quality=92)
        print(f"wrote {path}")
    json.dump(manifest, open(f"{args.out_prefix}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
