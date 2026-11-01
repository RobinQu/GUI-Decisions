"""Uncertainty features and square-crop geometry for distribution-guided slot refinement."""

import math

import torch
from scipy.ndimage import gaussian_filter

from .numbers import centers


def axis_distribution(lp, number, arm):
    p = torch.softmax(lp[number].detach().float().cpu(), -1)
    if arm.coarse and number + "c" in lp:
        prior = torch.softmax(lp[number + "c"].detach().float().cpu(), -1)
        p = p * prior[(centers(len(p)) * 10).long().clamp(max=9)]
        p = p / p.sum()
    return p


def point_uncertainty(lp, arm):
    px, py = axis_distribution(lp, "x", arm), axis_distribution(lp, "y", arm)
    c = centers(len(px))
    mx, my = float((c * px).sum()), float((c * py).sum())
    vx = float((c - mx).square().mul(px).sum())
    vy = float((c - my).square().mul(py).sum())
    entropy = 0.5 * sum(float(-(p * p.clamp_min(1e-12).log()).sum()) / math.log(len(p)) for p in (px, py))
    top1 = 0.5 * float(px.max() + py.max())
    spatial = math.exp(-math.sqrt(vx + vy) / 0.1)
    stats = dict(mean_x=mx, mean_y=my, std_x=math.sqrt(vx), std_y=math.sqrt(vy),
                 entropy=entropy, top1=top1, spatial=spatial, score=top1 + spatial)
    return px, py, stats


def distribution_readouts(px, py, regions=16, samples=64, bandwidth=0.03):
    """Readouts from the independent-axis approximation to the joint spatial distribution."""
    c = centers(len(px))
    joint = px[:, None] * py[None, :]
    factor = len(px) // regions
    cells = joint.reshape(regions, factor, regions, factor).sum(dim=(1, 3))
    rx, ry = divmod(int(cells.argmax()), regions)
    mask = torch.zeros_like(joint)
    mask[rx * factor:(rx + 1) * factor, ry * factor:(ry + 1) * factor] = 1
    selected = joint * mask
    region = dict(x=float((selected.sum(dim=1) * c).sum() / selected.sum()),
                  y=float((selected.sum(dim=0) * c).sum() / selected.sum()))
    generator = torch.Generator().manual_seed(0)
    ix = torch.multinomial(px, samples, replacement=True, generator=generator)
    iy = torch.multinomial(py, samples, replacement=True, generator=generator)
    points = torch.stack((c[ix], c[iy]), dim=1)
    weights = torch.exp(-(points[:, None] - points[None, :]).square().sum(dim=2) / (2 * bandwidth ** 2))
    k = int(weights.sum(dim=1).argmax())
    local = weights[k:k + 1, :]
    kde_point = (local @ points / local.sum())[0]
    density = torch.from_numpy(gaussian_filter(joint.numpy(), bandwidth * len(px), mode="constant"))
    kx, ky = divmod(int(density.argmax()), len(px))
    x0, x1 = max(0, kx - 2), min(len(px), kx + 3)
    y0, y1 = max(0, ky - 2), min(len(px), ky + 3)
    local = density[x0:x1, y0:y1]
    grid_x, grid_y = (torch.arange(x0, x1) + 0.5) / len(px), (torch.arange(y0, y1) + 0.5) / len(px)
    grid = dict(x=float((local.sum(dim=1) * grid_x).sum() / local.sum()),
                y=float((local.sum(dim=0) * grid_y).sum() / local.sum()))
    return {"region": region, "kde": dict(x=float(kde_point[0]), y=float(kde_point[1])), "kde_grid": grid}


def crop_window(point, image_size, side):
    W, H = image_size
    side = max(1, min(int(round(side)), W, H))
    left = min(max(int(round(point["x"] * W - side / 2)), 0), W - side)
    top = min(max(int(round(point["y"] * H - side / 2)), 0), H - side)
    return dict(left=left, top=top, side=side)


def adaptive_side(stats, image_size, min_size, gamma):
    W, H = image_size
    spread = max(stats["std_x"] * W, stats["std_y"] * H)
    return max(min_size, 2 * gamma * spread)


def map_point(point, crop, image_size):
    W, H = image_size
    return dict(x=(crop["left"] + point["x"] * crop["side"]) / W,
                y=(crop["top"] + point["y"] * crop["side"]) / H)
