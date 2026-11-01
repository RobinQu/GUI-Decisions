"""Generate figures for the GUI grounding technical report from saved experiment results."""

import json, shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path("report/figures")
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": ["DejaVu Sans", "Droid Sans Fallback"], "axes.unicode_minus": False,
                     "font.size": 10, "figure.dpi": 160})
BLUE, ORANGE, GREEN, GRAY = "#3978b8", "#e38b35", "#359567", "#7b8491"


def load(path):
    return json.load(open(path))


def bars(ax, labels, values, colors, title):
    x = np.arange(len(labels))
    ax.bar(x, values, color=colors, width=0.65)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(values) * 1.25 + 2)
    ax.set_title(title)
    ax.set_ylabel("命中率（%）")
    ax.grid(axis="y", alpha=0.2)
    for i, value in enumerate(values):
        ax.text(i, value + 1, f"{value:.1f}", ha="center", va="bottom", fontsize=9)


def overview():
    vanilla = load("results/gui-owl-vanilla-s0of10_click_128.json")["rows"]
    ref_s = load("results/refcoco-slots-s0_refcoco_point_128.json")["rows"]
    ref_v = load("results/refcoco-vanilla-s0_refcoco_point_128.json")["rows"]
    small_s = load("results/refcoco-slots-small-s0_refcoco_point_128.json")["rows"]
    small_v = load("results/refcoco-vanilla-small-s0_refcoco_point_128.json")["rows"]
    crop = load("results/crop-sweep-final-s0.json")
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    bars(axes[0], ["槽位\n(全量)", "vanilla\ncapped", "vanilla\nnative"],
         [76.49, 100 * np.mean([r["capped"]["hit"] for r in vanilla]),
          100 * np.mean([r["native"]["hit"] for r in vanilla])], [BLUE, GRAY, GRAY], "ScreenSpot")
    bars(axes[1], ["槽位", "vanilla"],
         [100 * np.mean([r["slots"]["default"]["hit"] for r in ref_s]),
          100 * np.mean([r["vanilla"]["hit"] for r in ref_v])], [BLUE, GRAY], "RefCOCO 随机目标")
    bars(axes[2], ["槽位", "vanilla"],
         [100 * np.mean([r["slots"]["default"]["hit"] for r in small_s]),
          100 * np.mean([r["vanilla"]["hit"] for r in small_v])], [BLUE, GRAY], "RefCOCO 小目标")
    bars(axes[3], ["全图", "512px", "768px\nKDE-grid"],
         [13.41, 100 * crop["combined"]["reference_default"]["accuracy"],
          100 * crop["combined"]["selected"]["accuracy"]], [GRAY, ORANGE, GREEN], "ScreenSpot-Pro（全量）")
    fig.tight_layout()
    fig.savefig(OUT / "overview_accuracy.png", bbox_inches="tight")
    plt.close(fig)


def crop_sweep():
    data = load("results/crop-sweep-final-s0.json")
    sizes = sorted(int(size) for size in data["metrics"])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for how, color, marker in [("default", GRAY, "o"), ("peak", BLUE, "s"),
                               ("kde", ORANGE, "^"), ("kde_grid", GREEN, "D")]:
        values = [100 * data["metrics"][str(size)][how]["accuracy"] for size in sizes]
        ax.plot(sizes, values, label=how, color=color, marker=marker, linewidth=2)
    ax.axvline(768, color=GREEN, linestyle="--", alpha=0.35)
    ax.set_xticks(sizes)
    ax.set_xlabel("方形裁剪边长（像素）")
    ax.set_ylabel("校准集命中率（%）")
    ax.set_title("裁剪尺寸与局部读出消融（校准集，n=790）")
    ax.grid(alpha=0.2)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "crop_sweep.png", bbox_inches="tight")
    plt.close(fig)


def breakdown():
    base = load("results/zoom-sspro-full-default-s0_analysis.json")
    crop = load("results/crop-sweep-final-s0.json")["heldout"]
    groups = [("小目标\narea<0.001", "area<0.001", "by_area"),
              ("文本", "text", "by_ui_type"), ("图标", "icon", "by_ui_type")]
    global_values = [100 * base["test_by_area"][key]["global"]["accuracy"] if kind == "by_area"
                     else 100 * base["test_by_ui_type"][key]["global"]["accuracy"] for _, key, kind in groups]
    fixed_values = [100 * base["test_by_area"][key]["fixed"]["accuracy"] if kind == "by_area"
                    else 100 * base["test_by_ui_type"][key]["fixed"]["accuracy"] for _, key, kind in groups]
    selected_values = [100 * crop[kind][key]["accuracy"] for _, key, kind in groups]
    x = np.arange(len(groups)); width = 0.24
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for offset, values, label, color in [(-width, global_values, "全图 default", GRAY),
                                         (0, fixed_values, "512px default", ORANGE),
                                         (width, selected_values, "768px KDE-grid", GREEN)]:
        rects = ax.bar(x + offset, values, width, label=label, color=color)
        ax.bar_label(rects, fmt="%.1f", padding=2, fontsize=8)
    ax.set_xticks(x, [g[0] for g in groups])
    ax.set_ylabel("留出集命中率（%）")
    ax.set_title("ScreenSpot-Pro 分层结果（留出集，n=791）")
    ax.set_ylim(0, 76)
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "sspro_breakdown.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    overview()
    crop_sweep()
    breakdown()
    shutil.copyfile("results/sspro_zoom_examples.jpg", OUT / "sspro_zoom_examples.jpg")
    print(f"Wrote figures to {OUT}")
