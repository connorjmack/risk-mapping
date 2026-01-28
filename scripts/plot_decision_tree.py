#!/usr/bin/env python3
"""Generate a decision tree diagram for the simplified RAI classification logic.

Outputs to output/figures/main/rai_decision_tree.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Class colors for simplified 5-class scheme
CLASS_COLORS = {
    "Talus": "#C8A2C8",
    "Intact": "#4CAF50",
    "Discontinuous": "#2196F3",
    "Steep/Overhang": "#F44336",
    "Structure": "#795548",
}

# Decision node style
DECISION_KW = dict(
    boxstyle="round,pad=0.4",
    facecolor="#F5F5F5",
    edgecolor="black",
    linewidth=1.2,
)

# Leaf node style (colored by class)
LEAF_KW = dict(
    boxstyle="round,pad=0.4",
    edgecolor="black",
    linewidth=1.2,
)


def _draw_edge(ax, x0, y0, x1, y1, label, label_side="left"):
    """Draw an edge between two nodes with a yes/no label."""
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0),
    )
    # Place label near the start of the edge
    lx = x0 + (x1 - x0) * 0.35
    ly = y0 + (y1 - y0) * 0.35
    offset = 0.03 if label_side == "left" else -0.03
    ax.text(
        lx + offset, ly, label,
        fontsize=8, fontweight="bold", ha="center", va="center",
        color="#444444",
    )


def _leaf_color(name):
    """Get face color for a leaf node."""
    return CLASS_COLORS.get(name, "#FFFFFF")


def _text_color(name):
    """Use white text on dark backgrounds."""
    dark = {"Steep/Overhang", "Structure", "Discontinuous"}
    return "white" if name in dark else "black"


def main():
    output_dir = Path("output/figures/main")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 9), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── Layout coordinates (x, y) for each node ──
    #
    # Simplified 5-class decision tree:
    #
    #   slope > 80°?
    #     Y → r_small < 4°?
    #           Y → Structure (5)
    #           N → Steep/Overhang (4)
    #     N → r_small < 6°?
    #           Y → slope < 42°?
    #                 Y → Talus (1)
    #                 N → Intact (2)
    #           N → r_small > 11° OR r_large > 12°?
    #                 Y → Discontinuous (3)
    #                 N → Intact (2)

    # Decision nodes: (x, y, text)
    decisions = {
        "root":       (0.50, 0.90, "slope > 80\u00b0?"),
        "struct":     (0.25, 0.72, "r_small < 4\u00b0?"),
        "smooth":     (0.70, 0.72, "r_small < 6\u00b0?"),
        "talus_chk":  (0.52, 0.52, "slope < 42\u00b0?"),
        "disc_chk":   (0.85, 0.52, "r_small > 11\u00b0\nOR r_large > 12\u00b0?"),
    }

    # Leaf nodes: (x, y, label)
    leaves = {
        "structure":      (0.12, 0.58, "Structure"),
        "steep_overhang": (0.35, 0.58, "Steep/Overhang"),
        "talus":          (0.38, 0.36, "Talus"),
        "intact1":        (0.62, 0.36, "Intact"),
        "discontinuous":  (0.95, 0.36, "Discontinuous"),
        "intact2":        (0.75, 0.36, "Intact"),
    }

    # Draw decision nodes
    for key, (x, y, text) in decisions.items():
        ax.text(
            x, y, text,
            fontsize=10, fontweight="bold", ha="center", va="center",
            bbox=DECISION_KW,
        )

    # Draw leaf nodes
    for key, (x, y, label) in leaves.items():
        ax.text(
            x, y, label,
            fontsize=10, fontweight="bold", ha="center", va="center",
            color=_text_color(label),
            bbox=dict(**LEAF_KW, facecolor=_leaf_color(label)),
        )

    # ── Edges ──
    # root → struct (Y) / smooth (N)
    _draw_edge(ax, 0.50, 0.87, 0.25, 0.76, "Y", "left")
    _draw_edge(ax, 0.50, 0.87, 0.70, 0.76, "N", "right")

    # struct → Structure (Y) / Steep/Overhang (N)
    _draw_edge(ax, 0.25, 0.68, 0.12, 0.62, "Y", "left")
    _draw_edge(ax, 0.25, 0.68, 0.35, 0.62, "N", "right")

    # smooth → talus_chk (Y) / disc_chk (N)
    _draw_edge(ax, 0.70, 0.68, 0.52, 0.56, "Y", "left")
    _draw_edge(ax, 0.70, 0.68, 0.85, 0.56, "N", "right")

    # talus_chk → Talus (Y) / Intact (N)
    _draw_edge(ax, 0.52, 0.48, 0.38, 0.40, "Y", "left")
    _draw_edge(ax, 0.52, 0.48, 0.62, 0.40, "N", "right")

    # disc_chk → Discontinuous (Y) / Intact (N)
    _draw_edge(ax, 0.85, 0.47, 0.95, 0.40, "Y", "right")
    _draw_edge(ax, 0.85, 0.47, 0.75, 0.40, "N", "left")

    # Title
    ax.text(
        0.50, 0.99, "RAI Classification Decision Tree",
        fontsize=14, fontweight="bold", ha="center", va="top",
    )
    ax.text(
        0.50, 0.96, "Simplified 5-class scheme adapted from Markus et al. (2023)",
        fontsize=9, ha="center", va="top", color="#666666",
    )

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor=color, edgecolor="black", label=name)
        for name, color in CLASS_COLORS.items()
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        fontsize=9,
        frameon=True,
        fancybox=True,
        edgecolor="gray",
    )

    plt.tight_layout()
    out_path = output_dir / "rai_decision_tree.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Decision tree saved to {out_path}")


if __name__ == "__main__":
    main()
