#!/usr/bin/env python3
"""
Create ML Pipeline Workflow Diagram

Generates a publication-quality workflow diagram showing the v2.x ML pipeline
structure from input data through model training and evaluation.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
plt.rcParams['font.size'] = 10

# Create figure
fig, ax = plt.subplots(figsize=(14, 16))
ax.set_xlim(0, 10)
ax.set_ylim(0, 20)
ax.axis('off')

# Color scheme
color_input = '#E8F4F8'      # Light blue - inputs
color_process = '#FFF4E6'    # Light orange - processing
color_feature = '#E8F5E9'    # Light green - features
color_model = '#F3E5F5'      # Light purple - modeling
color_output = '#FFF9C4'     # Light yellow - outputs

def draw_box(x, y, width, height, text, color, fontsize=9, bold=False):
    """Draw a rounded box with text."""
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.1",
        facecolor=color,
        edgecolor='black',
        linewidth=1.5
    )
    ax.add_patch(box)

    weight = 'bold' if bold else 'normal'
    ax.text(
        x + width/2, y + height/2, text,
        ha='center', va='center',
        fontsize=fontsize, weight=weight,
        wrap=True
    )

def draw_arrow(x1, y1, x2, y2, label='', style='->'):
    """Draw an arrow between boxes."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        mutation_scale=20,
        linewidth=2,
        color='black'
    )
    ax.add_patch(arrow)

    if label:
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        ax.text(mid_x + 0.3, mid_y, label, fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'))

# Title
ax.text(5, 19.5, 'PC-RAI v2.x ML Pipeline Workflow',
        ha='center', fontsize=16, weight='bold')
ax.text(5, 19, 'Random Forest for Rockfall Prediction from Pre-Failure Cliff Morphology',
        ha='center', fontsize=11, style='italic')

# ============================================================================
# SECTION 1: INPUT DATA (Top)
# ============================================================================
y_start = 17.5

ax.text(5, y_start + 0.5, '1. INPUT DATA', ha='center', fontsize=12, weight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', edgecolor='black'))

# Three input boxes
draw_box(0.5, y_start - 1.5, 2.5, 1.2,
         'LiDAR Point Clouds\n\n959 surveys\n2017-2025\n6 beaches',
         color_input, fontsize=9, bold=True)

draw_box(3.5, y_start - 1.5, 2.5, 1.2,
         'Event Labels\n\n52,365 events\nVolume, location,\nelevation, dates',
         color_input, fontsize=9, bold=True)

draw_box(6.5, y_start - 1.5, 2.5, 1.2,
         'Polygon Shapefiles\n\n1m alongshore bins\n× elevation zones\n(lower/mid/upper)',
         color_input, fontsize=9, bold=True)

# ============================================================================
# SECTION 2: PREPROCESSING
# ============================================================================
y_prep = 14.5

ax.text(5, y_prep + 0.5, '2. PREPROCESSING', ha='center', fontsize=12, weight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', edgecolor='black'))

# Step 1: Survey Selection
draw_box(0.5, y_prep - 1.5, 2.5, 1.2,
         'Step 1:\nSurvey Selection\n\nMatch surveys to\nfuture events\n(≥7 day gap)',
         color_process, fontsize=9)

# Step 2: Subsampling
draw_box(3.5, y_prep - 1.5, 2.5, 1.2,
         'Step 2a:\nSubsample\n\n50cm voxel grid\n~10M → ~400K pts',
         color_process, fontsize=9)

# Step 2b: Normals
draw_box(6.5, y_prep - 1.5, 2.5, 1.2,
         'Step 2b:\nCompute Normals\n\nCloudComPy MST\nWestward oriented',
         color_process, fontsize=9)

# Arrows from inputs to preprocessing
draw_arrow(1.75, y_start - 1.5, 1.75, y_prep - 0.3)
draw_arrow(5, y_start - 1.5, 5, y_prep - 0.3)

# ============================================================================
# SECTION 3: FEATURE EXTRACTION
# ============================================================================
y_feat = 11.5

ax.text(5, y_feat + 0.5, '3. FEATURE EXTRACTION', ha='center', fontsize=12, weight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', edgecolor='black'))

draw_box(1, y_feat - 1.5, 7, 1.2,
         'Step 2c: Extract Point-Level Features\n\n' +
         'Slope • Roughness (small/large) • Height • Eigenvalues (linearity, curvature) • 9 features per point',
         color_feature, fontsize=9, bold=True)

# Arrow from preprocessing to features
draw_arrow(5, y_prep - 1.5, 5, y_feat - 0.3)

# ============================================================================
# SECTION 4: AGGREGATION
# ============================================================================
y_agg = 8.5

ax.text(5, y_agg + 0.8, '4. SPATIAL AGGREGATION', ha='center', fontsize=12, weight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', edgecolor='black'))

draw_box(1, y_agg - 1.2, 3, 1.5,
         'Step 3:\nBin to Polygons\n\n1m alongshore\n× 3 elevation zones\n= polygon-zones',
         color_feature, fontsize=9)

draw_box(5, y_agg - 1.2, 3, 1.5,
         'Aggregate Statistics\n\nmean, std, min, max,\np10, p50, p90\n→ 63 features per zone',
         color_feature, fontsize=9)

# Arrows
draw_arrow(4.5, y_feat - 1.5, 2.5, y_agg + 0.3)
draw_arrow(5.5, y_feat - 1.5, 6.5, y_agg + 0.3)

# ============================================================================
# SECTION 5: TRAINING DATA ASSEMBLY
# ============================================================================
y_train = 5.5

ax.text(5, y_train + 0.8, '5. TRAINING DATA ASSEMBLY', ha='center', fontsize=12, weight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', edgecolor='black'))

# Case-control labeling
draw_box(0.8, y_train - 1.2, 3.5, 1.5,
         'Step 4: Label Polygon-Zones\n\n' +
         'Cases (1): Future rockfall\n' +
         'Controls (0): No failure\n' +
         'Balanced 1:1 ratio',
         color_model, fontsize=9)

draw_box(5.3, y_train - 1.2, 3.5, 1.5,
         'Final Dataset\n\n' +
         '72,782 samples\n' +
         '36,391 cases\n' +
         '42 features',
         color_model, fontsize=9, bold=True)

# Arrows
draw_arrow(2.5, y_agg - 1.2, 2.5, y_train + 0.3)
draw_arrow(6.5, y_agg - 1.2, 7, y_train + 0.3)
draw_arrow(4.3, y_train - 0.3, 5.3, y_train - 0.3)

# Event data arrow
draw_arrow(5, y_start - 1.5, 2.5, y_train + 0.3, style='->', label='')

# ============================================================================
# SECTION 6: MODEL TRAINING
# ============================================================================
y_model = 2.5

ax.text(5, y_model + 0.8, '6. MODEL TRAINING & EVALUATION', ha='center', fontsize=12, weight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', edgecolor='black'))

# Training
draw_box(0.5, y_model - 1.2, 2.8, 1.5,
         'Step 5: Train RF\n\n' +
         'Random Forest\n' +
         'n_estimators=100\n' +
         'class_weight=balanced',
         color_model, fontsize=9)

# Cross-validation
draw_box(3.8, y_model - 1.2, 2.8, 1.5,
         'Cross-Validation\n\n' +
         'Leave-one-year-out\n' +
         'Leave-one-beach-out\n' +
         '5-fold CV',
         color_model, fontsize=9)

# Results
draw_box(7.1, y_model - 1.2, 2.4, 1.5,
         'Performance\n\n' +
         'Temporal CV:\n' +
         'AUC-ROC: 0.701\n' +
         'AUC-PR: 0.667',
         color_output, fontsize=9, bold=True)

# Arrows
draw_arrow(6.5, y_train - 1.2, 2, y_model + 0.3)
draw_arrow(3.3, y_model - 0.3, 3.8, y_model - 0.3)
draw_arrow(6.6, y_model - 0.3, 7.1, y_model - 0.3)

# ============================================================================
# SECTION 7: OUTPUTS
# ============================================================================
y_out = 0

# Outputs
draw_box(1, y_out + 0.2, 2.2, 0.8,
         'Trained Models\nrf_model.joblib',
         color_output, fontsize=8)

draw_box(3.7, y_out + 0.2, 2.2, 0.8,
         'Feature Importance\nAblation Results',
         color_output, fontsize=8)

draw_box(6.4, y_out + 0.2, 2.2, 0.8,
         'Diagnostic Plots\nROC/PR Curves',
         color_output, fontsize=8)

# Arrows to outputs
draw_arrow(2, y_model - 1.2, 2, y_out + 1)
draw_arrow(5, y_model - 1.2, 4.8, y_out + 1)
draw_arrow(8, y_model - 1.2, 7.5, y_out + 1)

# ============================================================================
# LEGEND
# ============================================================================
legend_y = 18.2
legend_elements = [
    mpatches.Patch(facecolor=color_input, edgecolor='black', label='Input Data'),
    mpatches.Patch(facecolor=color_process, edgecolor='black', label='Preprocessing'),
    mpatches.Patch(facecolor=color_feature, edgecolor='black', label='Feature Extraction'),
    mpatches.Patch(facecolor=color_model, edgecolor='black', label='Modeling'),
    mpatches.Patch(facecolor=color_output, edgecolor='black', label='Outputs')
]

ax.legend(handles=legend_elements, loc='upper right',
          bbox_to_anchor=(0.98, 0.98), fontsize=9, frameon=True)

# Add script labels
ax.text(0.2, y_prep - 2, 'scripts/', fontsize=7, style='italic', color='gray')
ax.text(0.2, y_prep - 2.3, '01_identify_surveys.py', fontsize=7, family='monospace', color='gray')
ax.text(0.2, y_feat - 2, '02_extract_features.py', fontsize=7, family='monospace', color='gray')
ax.text(0.2, y_agg - 1.8, '03_aggregate_polygons.py', fontsize=7, family='monospace', color='gray')
ax.text(0.2, y_train - 1.8, '04_assemble_training_data.py', fontsize=7, family='monospace', color='gray')
ax.text(0.2, y_model - 1.8, '05_train_model.py', fontsize=7, family='monospace', color='gray')

plt.tight_layout()
plt.savefig('figures/main/ml_pipeline_workflow.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('figures/main/ml_pipeline_workflow.pdf', bbox_inches='tight', facecolor='white')
print("✓ Saved workflow diagram:")
print("  - figures/main/ml_pipeline_workflow.png")
print("  - figures/main/ml_pipeline_workflow.pdf")
plt.close()
