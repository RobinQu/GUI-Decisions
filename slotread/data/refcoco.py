import random

REFCOCO_REPO = "lmms-lab-encoder/RefCOCO"
REFCOCO_REVISION = "a5dff0b3194715fda69b2d6a0d2aaafb41eaa407"


def point_examples(n=0, seed=0, max_area=None):
    from datasets import load_dataset

    ds = load_dataset(REFCOCO_REPO, split="val", revision=REFCOCO_REVISION)
    seen, out = set(), []
    for ex in ds:
        x, y, w, h = ex["bbox"]
        key = (ex["file_name"].rsplit("_", 1)[0], tuple(ex["bbox"]))
        if key in seen or not ex["answer"]:
            continue
        seen.add(key)
        W, H = ex["image"].size
        clip = lambda v: min(1.0, max(0.0, v))
        box = [clip(x / W), clip(y / H), clip((x + w) / W), clip((y + h) / H)]
        if max_area is not None and (box[2] - box[0]) * (box[3] - box[1]) >= max_area:
            continue
        out.append(dict(id=ex["question_id"], image=ex["image"], box=box,
                        instruction="click " + ex["answer"][0], group=["refcoco"], norm=True))
    random.Random(seed).shuffle(out)
    return out[:n] if n else out


def unavailable(*args, **kwargs):
    raise NotImplementedError("This RefCOCO data loader is not included in the upstream repository")


rows = examples = photo_boxes = unavailable
