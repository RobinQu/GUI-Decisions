"""Conditional joint spatial distributions and readouts for autoregressive slot templates."""

import torch
from scipy.ndimage import gaussian_filter

from .numbers import centers


def candidate_bins(px, k=4, separation=4):
    if k <= 0 or separation <= 0 or px.ndim != 1 or not len(px):
        raise ValueError("k and separation must be positive, and px must be a one-dimensional distribution")
    selected = []
    for index in torch.argsort(px, descending=True).tolist():
        if all(abs(index - previous) >= separation for previous in selected):
            selected.append(index)
            if len(selected) == k:
                break
    if len(selected) != k:
        raise ValueError("Could not find enough separated x candidates")
    return selected


def conditional_joint(px, candidates, conditional_y):
    if len(candidates) != len(conditional_y) or not candidates:
        raise ValueError("Every x candidate must have one conditional y distribution")
    joint = torch.zeros(len(px), len(px), dtype=torch.float32)
    for index, lp in zip(candidates, conditional_y):
        if lp.ndim != 1 or len(lp) != len(px):
            raise ValueError("x and conditional y bins must have matching sizes")
        joint[index] = px[index] * torch.softmax(lp.detach().float().cpu(), -1)
    total = joint.sum()
    if total <= 0:
        raise ValueError("Conditional joint distribution has zero mass")
    return joint / total


def local_point(density, x, y, radius=2):
    n = density.shape[0]
    x0, x1 = max(0, x - radius), min(n, x + radius + 1)
    y0, y1 = max(0, y - radius), min(n, y + radius + 1)
    local = density[x0:x1, y0:y1]
    grid_x = (torch.arange(x0, x1) + 0.5) / n
    grid_y = (torch.arange(y0, y1) + 0.5) / n
    return dict(x=float((local.sum(dim=1) * grid_x).sum() / local.sum()),
                y=float((local.sum(dim=0) * grid_y).sum() / local.sum()))


def joint_readouts(joint, regions=16, samples=64, bandwidth=0.03):
    if joint.ndim != 2 or joint.shape[0] != joint.shape[1]:
        raise ValueError("Expected a square joint distribution")
    joint = joint.detach().float().cpu()
    joint = joint / joint.sum()
    n = joint.shape[0]
    if regions <= 0 or n % regions or samples <= 0 or bandwidth <= 0:
        raise ValueError("regions must divide the bin count; samples and bandwidth must be positive")
    c = centers(n)
    x, y = divmod(int(joint.argmax()), n)
    default = local_point(joint, x, y)
    peak = dict(x=float(c[x]), y=float(c[y]))
    expected = dict(x=float((joint.sum(dim=1) * c).sum()), y=float((joint.sum(dim=0) * c).sum()))
    factor = n // regions
    cells = joint.reshape(regions, factor, regions, factor).sum(dim=(1, 3))
    rx, ry = divmod(int(cells.argmax()), regions)
    mask = torch.zeros_like(joint)
    mask[rx * factor:(rx + 1) * factor, ry * factor:(ry + 1) * factor] = 1
    selected = joint * mask
    region = dict(x=float((selected.sum(dim=1) * c).sum() / selected.sum()),
                  y=float((selected.sum(dim=0) * c).sum() / selected.sum()))
    generator = torch.Generator().manual_seed(0)
    indices = torch.multinomial(joint.flatten(), samples, replacement=True, generator=generator)
    points = torch.stack((c[indices // n], c[indices % n]), dim=1)
    weights = torch.exp(-(points[:, None] - points[None, :]).square().sum(dim=2) / (2 * bandwidth ** 2))
    k = int(weights.sum(dim=1).argmax())
    local = weights[k:k + 1, :]
    kde_point = (local @ points / local.sum())[0]
    density = torch.from_numpy(gaussian_filter(joint.numpy(), bandwidth * n, mode="constant"))
    kx, ky = divmod(int(density.argmax()), n)
    return {"default": default, "peak": peak, "expected": expected, "region": region,
            "kde": dict(x=float(kde_point[0]), y=float(kde_point[1])),
            "kde_grid": local_point(density, kx, ky)}
