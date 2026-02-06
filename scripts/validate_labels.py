#!/usr/bin/env python3
"""
Validate training data labels by visualizing spatial alignment.

Produces per-location plots showing:
1. Case/control distribution along the cliff (alongshore_m vs zone)
2. Event extents overlaid to verify cases land where events occurred
3. Summary statistics per survey

Usage:
    python scripts/validate_labels.py \
        --training data/training_data.csv \
        --surveys data/test_pre_event_surveys.csv \
        --output output/label_validation/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate training labels with spatial plots"
    )
    parser.add_argument(
        "--training", type=Path, required=True,
        help="Path to training_data.csv",
    )
    parser.add_argument(
        "--surveys", type=Path, required=True,
        help="Path to pre_event_surveys.csv",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=Path("output/label_validation"),
        help="Output directory for plots",
    )
    return parser.parse_args()


def plot_label_map(df, events, location, output_dir):
    """Plot alongshore vs zone heatmap of case density for one location.

    Shows where label=1 polygons are relative to event extents.
    """
    loc_df = df[df["location"] == location]
    loc_events = events[events["location"] == location]

    if len(loc_df) == 0:
        return

    surveys = sorted(loc_df["survey_file"].unique())
    n_surveys = len(surveys)

    fig, axes = plt.subplots(
        n_surveys, 1,
        figsize=(14, 3 * n_surveys + 1),
        squeeze=False,
        sharex=True,
    )
    fig.suptitle(f"{location} — Label Validation", fontsize=14, fontweight="bold")

    zone_order = {"lower": 0, "middle": 1, "upper": 2}

    for i, survey in enumerate(surveys):
        ax = axes[i, 0]
        sdf = loc_df[loc_df["survey_file"] == survey]
        survey_date = sdf["survey_date"].iloc[0]

        # Get events matched to this survey
        survey_date_str = survey[:8]
        matched_events = loc_events[
            loc_events["survey_file"].str.startswith(survey_date_str)
        ]

        # Plot controls as small gray dots
        controls = sdf[sdf["label"] == 0]
        cases = sdf[sdf["label"] == 1]

        ax.scatter(
            controls["alongshore_m"],
            controls["zone"].map(zone_order),
            c="lightgray", s=1, alpha=0.3, rasterized=True,
        )

        # Plot cases colored by event volume
        if len(cases) > 0:
            sc = ax.scatter(
                cases["alongshore_m"],
                cases["zone"].map(zone_order),
                c=cases["event_volume"],
                cmap="YlOrRd", s=4, alpha=0.7,
                vmin=5, vmax=cases["event_volume"].quantile(0.95),
                rasterized=True,
            )

        # Overlay event extents as horizontal bars
        for _, ev in matched_events.iterrows():
            start = ev["event_alongshore_start"]
            end = ev["event_alongshore_end"]
            if pd.isna(start) or pd.isna(end):
                continue
            ax.axvspan(start, end, alpha=0.08, color="blue", zorder=0)

        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["lower", "middle", "upper"])
        ax.set_ylabel("Zone")

        n_cases = len(cases)
        n_events = len(matched_events)
        ax.set_title(
            f"{survey_date}  |  {n_cases} cases, "
            f"{len(controls)} controls, {n_events} events",
            fontsize=9,
        )

    axes[-1, 0].set_xlabel("Alongshore (m)")

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="lightgray", label="Control (label=0)"),
        mpatches.Patch(facecolor="red", alpha=0.7, label="Case (label=1)"),
        mpatches.Patch(facecolor="blue", alpha=0.15, label="Event extent"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center",
        ncol=3, fontsize=9, frameon=False,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    out_path = output_dir / f"{location}_label_validation.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_summary(df, output_dir):
    """Summary plot: case ratio by location and zone."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Cases per location
    ax = axes[0]
    loc_stats = df.groupby("location")["label"].agg(["sum", "count"])
    loc_stats.columns = ["cases", "total"]
    loc_stats = loc_stats.sort_values("cases", ascending=True)
    colors = ["#e74c3c" if c > 0 else "#95a5a6" for c in loc_stats["cases"]]
    ax.barh(loc_stats.index, loc_stats["cases"], color=colors)
    ax.set_xlabel("Case count (label=1)")
    ax.set_title("Cases by Location")

    # 2. Case ratio by zone
    ax = axes[1]
    zone_stats = df.groupby("zone")["label"].mean().reindex(["lower", "middle", "upper"])
    ax.bar(zone_stats.index, zone_stats.values * 100, color=["#3498db", "#2ecc71", "#e67e22"])
    ax.set_ylabel("Case ratio (%)")
    ax.set_title("Case Ratio by Zone")

    # 3. Event volume distribution
    ax = axes[2]
    cases = df[df["label"] == 1]
    volumes = cases.drop_duplicates("event_id")["event_volume"].dropna()
    ax.hist(volumes, bins=50, color="#9b59b6", alpha=0.7, edgecolor="white")
    ax.set_xlabel("Event volume (m³)")
    ax.set_ylabel("Count")
    ax.set_title(f"Event Volume Distribution (n={len(volumes)})")
    ax.axvline(volumes.median(), color="red", linestyle="--", label=f"median={volumes.median():.1f}")
    ax.legend()

    plt.tight_layout()
    out_path = output_dir / "label_summary.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_alongshore_density(df, output_dir):
    """Density of cases vs controls along the cliff for each location."""
    locations = sorted(df["location"].unique())
    n_locs = len(locations)

    fig, axes = plt.subplots(n_locs, 1, figsize=(14, 3 * n_locs), squeeze=False)
    fig.suptitle("Case vs Control Density Along Cliff", fontsize=14, fontweight="bold")

    for i, loc in enumerate(locations):
        ax = axes[i, 0]
        loc_df = df[df["location"] == loc]
        cases = loc_df[loc_df["label"] == 1]["alongshore_m"]
        controls = loc_df[loc_df["label"] == 0]["alongshore_m"]

        bins = np.linspace(
            loc_df["alongshore_m"].min(),
            loc_df["alongshore_m"].max(),
            min(200, int(loc_df["alongshore_m"].max() / 10)),
        )

        ax.hist(controls, bins=bins, alpha=0.5, color="gray", label="Controls", density=True)
        ax.hist(cases, bins=bins, alpha=0.5, color="red", label="Cases", density=True)
        ax.set_ylabel("Density")
        ax.set_title(f"{loc} ({len(cases)} cases, {len(controls)} controls)")
        ax.legend(fontsize=8)

    axes[-1, 0].set_xlabel("Alongshore (m)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = output_dir / "alongshore_density.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    args = parse_args()

    if not args.training.exists():
        print(f"Error: {args.training} not found", file=sys.stderr)
        return 1
    if not args.surveys.exists():
        print(f"Error: {args.surveys} not found", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(args.training)
    surveys = pd.read_csv(args.surveys)

    print(f"Training data: {len(df):,} rows ({(df['label']==1).sum():,} cases)")
    print(f"Events: {len(surveys):,} survey-event pairs")
    print()

    # Per-location label maps
    print("Generating per-location label maps...")
    for loc in sorted(df["location"].unique()):
        plot_label_map(df, surveys, loc, args.output)

    # Summary
    print("\nGenerating summary plots...")
    plot_summary(df, args.output)

    # Alongshore density
    print("\nGenerating alongshore density plots...")
    plot_alongshore_density(df, args.output)

    print(f"\nAll plots saved to: {args.output}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
