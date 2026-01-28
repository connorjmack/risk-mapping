#!/usr/bin/env python3
"""Generate a decision tree diagram for the RAI classification logic.

Outputs to output/figures/main/rai_decision_tree.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Class colors (matching Dunham-style panel colors)
CLASS_COLORS = {
    "Talus": "#C8A2C8",
    "Intact": "#4CAF50",
    "Frag. Disc.": "#81D4FA",
    "Close Disc.": "#2196F3",
    "Wide Disc.": "#1565C0",
    "Steep Cliff": "#FFEB3B",
    "Cant. Ovhg.": "#F44336",
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
        fontsize=7, fontweight="bold", ha="center", va="center",
        color="#444444",
    )


def _leaf_color(name):
    """Get face color for a leaf node, with alpha for dark backgrounds."""
    return CLASS_COLORS.get(name, "#FFFFFF")


def _text_color(name):
    """Use white text on dark backgrounds."""
    dark = {"Wide Disc.", "Cant. Ovhg.", "Structure"}
    return "white" if name in dark else "black"


def main():
    output_dir = Path("output/figures/main")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── Layout coordinates (x, y) for each node ──
    # Level 0: root
    # Level 1: first split
    # etc.

    # Decision nodes: (x, y, text)
    decisions = {
        "root":       (0.50, 0.93, "slope > 150°?"),
        "steep":      (0.38, 0.78, "slope > 80°?"),
        "struct":     (0.22, 0.63, "r_small < 4°?"),
        "smooth":     (0.54, 0.63, "r_small < 6°?"),
        "talus_chk":  (0.42, 0.48, "slope < 42°?"),
        "dw_chk":     (0.68, 0.48, "r_small > 18°?"),
        "dc_chk":     (0.60, 0.33, "r_small > 11°?"),
        "df_chk":     (0.50, 0.18, "r_large > 12°?"),
    }

    # Leaf nodes: (x, y, label)
    leaves = {
        "oc":         (0.68, 0.83, "Cant. Ovhg."),
        "structure":  (0.12, 0.50, "Structure"),
        "sc":         (0.28, 0.50, "Steep Cliff"),
        "talus":      (0.32, 0.35, "Talus"),
        "intact1":    (0.48, 0.38, "Intact"),
        "dw":         (0.80, 0.38, "Wide Disc."),
        "dc":         (0.70, 0.22, "Close Disc."),
        "df":         (0.42, 0.07, "Frag. Disc."),
        "intact2":    (0.58, 0.07, "Intact"),
    }

    # Draw decision nodes
    for key, (x, y, text) in decisions.items():
        ax.text(
            x, y, text,
            fontsize=9, fontweight="bold", ha="center", va="center",
            bbox=DECISION_KW,
        )

    # Draw leaf nodes
    for key, (x, y, label) in leaves.items():
        ax.text(
            x, y, label,
            fontsize=9, fontweight="bold", ha="center", va="center",
            color=_text_color(label),
            bbox=dict(**LEAF_KW, facecolor=_leaf_color(label)),
        )

    # ── Edges ──
    # root → Cant. Ovhg. (yes) / steep (no)
    _draw_edge(ax, 0.50, 0.90, 0.68, 0.86, "Y", "right")
    _draw_edge(ax, 0.50, 0.90, 0.38, 0.81, "N", "left")

    # steep → struct (yes) / smooth (no)
    _draw_edge(ax, 0.38, 0.75, 0.22, 0.66, "Y", "left")
    _draw_edge(ax, 0.38, 0.75, 0.54, 0.66, "N", "right")

    # struct → Structure (yes) / Steep Cliff (no)
    _draw_edge(ax, 0.22, 0.60, 0.12, 0.53, "Y", "left")
    _draw_edge(ax, 0.22, 0.60, 0.28, 0.53, "N", "right")

    # smooth → talus_chk (yes) / dw_chk (no)
    _draw_edge(ax, 0.54, 0.60, 0.42, 0.51, "Y", "left")
    _draw_edge(ax, 0.54, 0.60, 0.68, 0.51, "N", "right")

    # talus_chk → Talus (yes) / Intact (no)
    _draw_edge(ax, 0.42, 0.45, 0.32, 0.38, "Y", "left")
    _draw_edge(ax, 0.42, 0.45, 0.48, 0.41, "N", "right")

    # dw_chk → Wide Disc. (yes) / dc_chk (no)
    _draw_edge(ax, 0.68, 0.45, 0.80, 0.41, "Y", "right")
    _draw_edge(ax, 0.68, 0.45, 0.60, 0.36, "N", "left")

    # dc_chk → Close Disc. (yes) / df_chk (no)
    _draw_edge(ax, 0.60, 0.30, 0.70, 0.25, "Y", "right")
    _draw_edge(ax, 0.60, 0.30, 0.50, 0.21, "N", "left")

    # df_chk → Frag. Disc. (yes) / Intact (no)
    _draw_edge(ax, 0.50, 0.15, 0.42, 0.10, "Y", "left")
    _draw_edge(ax, 0.50, 0.15, 0.58, 0.10, "N", "right")

    # Title
    ax.text(
        0.50, 0.99, "RAI Classification Decision Tree",
        fontsize=14, fontweight="bold", ha="center", va="top",
    )
    ax.text(
        0.50, 0.96, "Adapted from Markus et al. (2023) for coastal bluffs",
        fontsize=9, ha="center", va="top", color="#666666",
    )

    plt.tight_layout()
    out_path = output_dir / "rai_decision_tree.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Decision tree saved to {out_path}")


if __name__ == "__main__":
    main()
