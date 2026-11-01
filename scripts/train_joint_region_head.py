"""Train independent-axis and joint-region heads on identical frozen GUI slot features."""

import argparse, copy, json, time

import numpy as np
import torch
from torch import nn

from analyze_zoom import paired_metrics


def make_trunk(input_dim, hidden_dim):
    if hidden_dim == 0:
        return nn.Identity()
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1))


class RegionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, joint):
        super().__init__()
        self.joint = joint
        self.trunk = make_trunk(input_dim, hidden_dim)
        output_dim = hidden_dim or input_dim
        self.region = nn.Linear(output_dim, 256 if joint else 32)
        self.offset = nn.Linear(output_dim, 2)

    def forward(self, x):
        hidden = self.trunk(x)
        return self.region(hidden), torch.sigmoid(self.offset(hidden))


class LowRankRegionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, rank=4):
        super().__init__()
        self.trunk = make_trunk(input_dim, hidden_dim)
        output_dim = hidden_dim or input_dim
        self.mixture = nn.Linear(output_dim, rank)
        self.x_regions = nn.Linear(output_dim, rank * 16)
        self.y_regions = nn.Linear(output_dim, rank * 16)
        self.offset = nn.Linear(output_dim, 2)

    def forward(self, x):
        hidden = self.trunk(x)
        parts = (self.mixture(hidden), self.x_regions(hidden), self.y_regions(hidden))
        return parts, torch.sigmoid(self.offset(hidden))


def low_rank_log_probs(parts):
    mixture, x_regions, y_regions = parts
    rank = mixture.shape[1]
    log_mixture = nn.functional.log_softmax(mixture, dim=1)
    log_x = nn.functional.log_softmax(x_regions.reshape(-1, rank, 16), dim=2)
    log_y = nn.functional.log_softmax(y_regions.reshape(-1, rank, 16), dim=2)
    components = log_mixture[:, :, None, None] + log_x[:, :, :, None] + log_y[:, :, None, :]
    return torch.logsumexp(components, dim=1)


def load_complete(path, part, allow_partial=False):
    data = torch.load(path, weights_only=True)
    if data.get("part") != part or (not allow_partial and data.get("expected_n") != len(data["rows"])):
        raise ValueError(f"Partial or mismatched feature result: {path}")
    source = json.load(open(data["source"]))
    if source.get("seed") != 0 or source.get("model") != data.get("model") or source.get("n") != 1581:
        raise ValueError("Features must reference the complete seed-0 source with the same model")
    split = int(round(len(source["rows"]) * 0.5))
    expected = source["rows"][:split] if part == "calibration" else source["rows"][split:]
    ids = [str(r["id"]) for r in data["rows"]]
    if len(set(ids)) != len(ids) or ids != [str(r["id"]) for r in expected[:len(ids)]]:
        raise ValueError(f"Feature IDs are not an ordered prefix of the {part} split")
    if not allow_partial and len(ids) != len(expected):
        raise ValueError(f"Feature rows are not the complete {part} split")
    return data


def arrays(rows):
    features = torch.stack([r["feature"].float() for r in rows])
    boxes = torch.tensor([r["box"] for r in rows], dtype=torch.float32)
    centers = torch.stack(((boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2), dim=1)
    cells = (centers * 16).long().clamp(max=15)
    offsets = centers * 16 - cells
    regions = cells[:, 1] * 16 + cells[:, 0]
    return features, centers, cells, offsets, regions


def predictions(logits, offsets, joint):
    if joint == "low_rank":
        region = low_rank_log_probs(logits).flatten(1).argmax(dim=1)
        xcell, ycell = region // 16, region % 16
    elif joint:
        region = logits.argmax(dim=1)
        xcell, ycell = region % 16, region // 16
    else:
        xcell, ycell = logits[:, :16].argmax(dim=1), logits[:, 16:].argmax(dim=1)
    return (torch.stack((xcell, ycell), dim=1) + offsets) / 16


def hits(points, rows):
    out = []
    for point, row in zip(points.tolist(), rows):
        x, y = point; box = row["box"]
        out.append(float(box[0] <= x <= box[2] and box[1] <= y <= box[3]))
    return out


def metrics(pred, rows, baseline):
    base = [r[baseline]["hit"] for r in rows]
    return paired_metrics(base, pred, [r["ms"] for r in rows], [False] * len(rows))


def evaluate(model, features, rows, joint, baseline="global_default"):
    model.eval()
    with torch.no_grad():
        logits, offsets = model(features)
        points = predictions(logits, offsets, joint)
        centered = predictions(logits, torch.full_like(offsets, 0.5), joint)
    return dict(with_offset=metrics(hits(points, rows), rows, baseline),
                region_center=metrics(hits(centered, rows), rows, baseline))


def train(features, centers, cells, offsets, regions, val_features, rows, joint, hidden_dim, epochs, offset_weight, seed,
          final=False, weight_decay=1e-4):
    torch.manual_seed(seed)
    model = LowRankRegionHead(features.shape[1], hidden_dim) if joint == "low_rank" else RegionHead(
        features.shape[1], hidden_dim, joint)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    best = None
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(features))
        for start in range(0, len(features), 64):
            indices = permutation[start:start + 64]
            logits, predicted_offsets = model(features[indices])
            if joint == "low_rank":
                log_probs = low_rank_log_probs(logits)
                target = cells[indices]
                region_loss = -log_probs[torch.arange(len(target)), target[:, 1], target[:, 0]].mean()
            elif joint:
                region_loss = nn.functional.cross_entropy(logits, regions[indices])
            else:
                region_loss = (nn.functional.cross_entropy(logits[:, :16], cells[indices, 0])
                               + nn.functional.cross_entropy(logits[:, 16:], cells[indices, 1])) / 2
            offset_loss = nn.functional.smooth_l1_loss(predicted_offsets, offsets[indices])
            loss = region_loss + offset_weight * offset_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if final and epoch == epochs:
            best = (0.0, epoch, copy.deepcopy(model.state_dict()))
        elif not final and (epoch % 10 == 0 or epoch == epochs):
            score = evaluate(model, val_features, rows, joint)["with_offset"]["accuracy"]
            if best is None or score > best[0]:
                best = (score, epoch, copy.deepcopy(model.state_dict()))
    model.load_state_dict(best[2])
    return model, best[1], sum(p.numel() for p in model.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("calibration")
    ap.add_argument("--heldout")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-out", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--offset-weight", type=float, default=1.0)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    if args.epochs <= 0 or args.hidden_dim < 0 or args.offset_weight < 0 or args.weight_decay < 0:
        raise ValueError("epochs must be positive; other numeric settings must be nonnegative")
    calibration_data = load_complete(args.calibration, "calibration", args.allow_partial)
    rows = calibration_data["rows"]
    split = int(len(rows) * 0.8)
    train_rows, val_rows = rows[:split], rows[split:]
    if not train_rows or not val_rows:
        raise ValueError("Training and validation splits must both be nonempty")
    train_features, train_centers, train_cells, train_offsets, train_regions = arrays(train_rows)
    mean = train_features.mean(dim=0)
    std = train_features.std(dim=0).clamp_min(1e-6)
    train_features = (train_features - mean) / std
    val_features, _, _, _, _ = arrays(val_rows)
    val_features = (val_features - mean) / std
    result = dict(calibration=args.calibration, train_n=len(train_rows), validation_n=len(val_rows),
                  grid=16, hidden_dim=args.hidden_dim, epochs=args.epochs, weight_decay=args.weight_decay,
                  offset_weight=args.offset_weight, seed=args.seed, heads={})
    started = time.time()
    for joint, name in ((False, "independent"), (True, "joint"), ("low_rank", "low_rank")):
        model, best_epoch, n_params = train(train_features, train_centers, train_cells, train_offsets,
                                           train_regions, val_features, val_rows, joint, args.hidden_dim,
                                           args.epochs, args.offset_weight, args.seed,
                                           weight_decay=args.weight_decay)
        result["heads"][name] = dict(best_epoch=best_epoch, n_params=n_params,
                                     validation=evaluate(model, val_features, val_rows, joint))
        if args.heldout:
            heldout_data = load_complete(args.heldout, "heldout", args.allow_partial)
            if (heldout_data.get("model") != calibration_data.get("model")
                    or heldout_data.get("slots") != calibration_data.get("slots")):
                raise ValueError("Calibration and held-out features use different models or slots")
            all_features, all_centers, all_cells, all_offsets, all_regions = arrays(rows)
            mean = all_features.mean(dim=0)
            std = all_features.std(dim=0).clamp_min(1e-6)
            all_features = (all_features - mean) / std
            model, _, _ = train(all_features, all_centers, all_cells, all_offsets, all_regions,
                                all_features, val_rows, joint, args.hidden_dim, best_epoch,
                                args.offset_weight, args.seed, final=True, weight_decay=args.weight_decay)
            result["heads"][name]["refit_n"] = len(rows)
            heldout_rows = heldout_data["rows"]
            heldout_features, _, _, _, _ = arrays(heldout_rows)
            heldout_features = (heldout_features - mean) / std
            result["heads"][name]["heldout"] = evaluate(model, heldout_features, heldout_rows, joint)
            result["heads"][name]["heldout_vs_kde"] = evaluate(model, heldout_features, heldout_rows, joint,
                                                                baseline="global_kde_grid")
            for group_name, grouper in (("by_ui_type", lambda r: r["group"][2]),
                                        ("by_area", lambda r: "area<0.001" if r["area"] < 0.001 else
                                         "area0.001-0.01" if r["area"] < 0.01 else "area>=0.01")):
                result["heads"][name][group_name] = {}
                for group in sorted({grouper(r) for r in heldout_rows}):
                    indices = [i for i, r in enumerate(heldout_rows) if grouper(r) == group]
                    subset = [heldout_rows[i] for i in indices]
                    result["heads"][name][group_name][group] = evaluate(
                        model, heldout_features[indices], subset, joint)
        result["heads"][name]["state_dict"] = model.state_dict()
    result["training_seconds"] = time.time() - started
    states = {name: result["heads"][name].pop("state_dict") for name in result["heads"]}
    json.dump(result, open(args.out, "w"), indent=2)
    torch.save(dict(grid=16, hidden_dim=args.hidden_dim, mean=mean, std=std, **states), args.model_out)
    print(json.dumps(result, indent=2))
    print(f"wrote {args.out} and {args.model_out}")


if __name__ == "__main__":
    main()
