SCREENSPOT_REPO = "bevaya/ScreenSpot"
SCREENSPOT_REVISION = "0be08781e2e188582f6131625ae1598d443b4d5d"
SCREENSPOT_PRO_REPO = "likaixin/ScreenSpot-Pro"
SCREENSPOT_PRO_REVISION = "210e78d3844251110bff86c95835ebd37a6930fa"


def screenspot(n=0):
    from datasets import load_dataset

    ds = load_dataset(SCREENSPOT_REPO, split="test", revision=SCREENSPOT_REVISION)
    if n:
        ds = ds.select(range(min(n, len(ds))))
    return [dict(image=ex["image"], box=ex["bbox"], instruction=ex["instruction"],
                 group=[ex["data_source"], ex["data_type"]], norm=True) for ex in ds]


def screenspot_pro():
    import json, os
    from huggingface_hub import snapshot_download

    root = snapshot_download(SCREENSPOT_PRO_REPO, repo_type="dataset", revision=SCREENSPOT_PRO_REVISION)
    out = []
    for name in sorted(os.listdir(os.path.join(root, "annotations"))):
        if not name.endswith(".json"):
            continue
        for ex in json.load(open(os.path.join(root, "annotations", name))):
            W, H = ex["img_size"]
            x1, y1, x2, y2 = ex["bbox"]
            x1, x2 = min(W, max(0, x1)), min(W, max(0, x2))
            y1, y2 = min(H, max(0, y1)), min(H, max(0, y2))
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"Invalid clipped box for {ex['id']}: {ex['bbox']}")
            out.append(dict(id=ex["id"], image=os.path.join(root, "images", ex["img_filename"]),
                            size=(W, H), box=[x1 / W, y1 / H, x2 / W, y2 / H],
                            instruction=ex["instruction"], group=[ex["group"], ex["application"], ex["ui_type"]],
                            norm=True))
    return out
