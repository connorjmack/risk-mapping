# Pipeline Data Access & Visualization Guide

Reference document for accessing, loading, and displaying results from the coastal cliff LiDAR processing pipeline. All patterns are extracted from the existing codebase.

---

## 1. Storage & Path Resolution

All data lives on the shared storage system **"reefbreak"**, mounted differently by OS:

```python
import platform

if platform.system() == "Darwin":  # macOS
    ROOT_LIDAR = "/Volumes/group/LiDAR"
else:  # Linux / HPC
    ROOT_LIDAR = "/project/group/LiDAR"

BASE_DIR = f"{ROOT_LIDAR}/LidarProcessing/LidarProcessingCliffs"
```

**Every script in the repo uses this pattern at module level.** Never hardcode one or the other.

### Key directories under `BASE_DIR`

| Path | Contents |
|------|----------|
| `results/<Location>/erosion/` | Clustered + gridded erosion results |
| `results/<Location>/deposition/` | Clustered + gridded deposition results |
| `results/<Location>/m3c2/` | Raw M3C2 change detection outputs |
| `results/event_lists/` | Per-location event CSVs |
| `results/data_cubes/` | 3D NPZ data cubes |
| `survey_lists/` | Survey inventory CSVs |
| `utilities/` | Models, classifiers, params, shapefiles |
| `figures/` | Generated plots (dashboards, gifs, etc.) |

### Locations

```python
LOCATIONS = ['DelMar', 'Torrey', 'Solana', 'Encinitas', 'SanElijo', 'Blacks']
```

Each location has a MOP (Monitoring and Prediction) range:

```python
MOP_RANGES = {
    "DelMar":    (595, 620),
    "Solana":    (637, 666),
    "Encinitas": (708, 764),
    "SanElijo":  (683, 708),
    "Torrey":    (567, 581),
    "Blacks":    (520, 567),
}
```

---

## 2. Raw Survey Data

### Where surveys live

```
ROOT_LIDAR/
├── MiniRanger_Truck/LiDAR_Processed_Level2/
├── MiniRanger_ATV/LiDAR_Processed_Level2/
├── VMQLZ_Truck/LiDAR_Processed_Level2/
└── VMZ2000_Truck/LiDAR_Processed_Level2/
```

Each instrument folder contains survey subfolders named: `YYYYMMDD_MOP1_MOP2_<description>/`

The target LAS file inside each survey folder is:
```
<survey>/Beach_And_Backshore/<survey>_beach_cliff_ground.las
```

### Survey list CSVs

Location: `BASE_DIR/survey_lists/surveys_<Location>.csv`

| Column | Example |
|--------|---------|
| `path` | Full path to the survey folder |
| `date` | `20171004` |
| `MOP1` | `590` |
| `MOP2` | `708` |
| `beach` | `SanElijo` |
| `method` | `MiniRanger_Truck` |

### Scanning for surveys (Streamlit pattern)

From `code/streamlit/survey_browser.py` — dynamically scans directories with 2/3 MOP overlap matching:

```python
import os

INSTRUMENT_PATHS = {
    "MiniRanger_Truck": os.path.join(ROOT_LIDAR, "MiniRanger_Truck/LiDAR_Processed_Level2"),
    "MiniRanger_ATV":   os.path.join(ROOT_LIDAR, "MiniRanger_ATV/LiDAR_Processed_Level2"),
    "VMQLZ_Truck":      os.path.join(ROOT_LIDAR, "VMQLZ_Truck/LiDAR_Processed_Level2"),
    "VMZ2000_Truck":    os.path.join(ROOT_LIDAR, "VMZ2000_Truck/LiDAR_Processed_Level2"),
}

# For each folder in each instrument path:
# 1. Parse folder name: parts = name.split("_") -> date_str, mop1, mop2
# 2. Check date range
# 3. Check MOP overlap >= floor((target_max - target_min) * 2/3)
# 4. Verify Beach_And_Backshore/<name>_beach_cliff_ground.las exists
```

---

## 3. Pipeline Output Directory Structure

```
results/<Location>/
├── cropped/                           # Step 2: PDAL-cropped LAS
├── nobeach/                           # Step 3: Beach removed LAS
├── noveg/                             # Step 4: Vegetation removed LAS
├── m3c2/                              # Step 5: M3C2 change detection
│   └── pipeline_run_YYYYMMDD/
│       └── YYYYMMDD_to_YYYYMMDD/
│           ├── DATE1.las
│           ├── DATE2.las
│           └── DATE1_to_DATE2_m3c2.las
├── erosion/                           # Steps 6-8: Clustered & gridded
│   └── YYYYMMDD_to_YYYYMMDD/
│       ├── ero_clusters.las           # Step 6: DBSCAN clustered
│       ├── ero_outliers.las           # Step 6: DBSCAN noise
│       ├── 10cm/                      # Step 7-8: Gridded at resolution
│       │   ├── *_ero_grid_10cm.csv
│       │   ├── *_ero_grid_10cm_filled.csv    # <- FINAL PRODUCT
│       │   ├── *_ero_clusters_10cm.csv
│       │   ├── *_ero_clusters_10cm_filled.csv
│       │   └── *_ero_stats_10cm.npz
│       ├── 25cm/                      # Most commonly used resolution
│       └── 1m/
└── deposition/                        # Same structure as erosion with dep_ prefix
```

### Finding the latest pipeline run (Step 6 pattern)

```python
import glob, os

def find_latest_pipeline_run(m3c2_dir):
    """Finds most recent pipeline_run_* by parsing date from folder name."""
    runs = glob.glob(os.path.join(m3c2_dir, "pipeline_run_*"))
    def extract_date(path):
        return int(os.path.basename(path).replace("pipeline_run_", ""))
    runs.sort(key=extract_date, reverse=True)
    return runs[0]
```

---

## 4. Loading Grid CSVs

Grid CSVs are the primary data product used for visualization. The filled versions (`*_filled.csv`) are preferred.

### Grid CSV structure

- **Rows**: `Polygon_ID` (spatial bins along the coast, set as index)
- **Columns**: `M3C2_0.10m`, `M3C2_0.20m`, ... (elevation bins)
- **Values**: Median M3C2 change distance in meters
- **Missing data**: Empty string `""` (not NaN or 0)

### Standard loading pattern

```python
import pandas as pd
import numpy as np

def load_grid(path, resolution_m=0.25):
    """Load a grid CSV and prepare for plotting."""
    df = pd.read_csv(path, index_col=0, na_values=['', 'nan', 'NaN', 'NULL'])

    # Strip letters/underscores from column names: 'M3C2_0.25m' -> '0.25'
    cleaned = df.columns.astype(str).str.replace(r'[a-zA-Z_]', '', regex=True)
    df.columns = cleaned.astype(float)  # Elevation in meters
    df.index = df.index.astype(float)   # Alongshore position

    return df.fillna(0.0)
```

### Alternative: Integer grid indices (for cumulative operations)

Used in `plot_dashboard.py` when accumulating grids across time:

```python
def clean_and_snap_grid(df, resolution_val):
    """Convert to integer grid indices for safe accumulation."""
    cleaned_cols = df.columns.astype(str).str.replace(r'[a-zA-Z_]', '', regex=True)
    col_floats = cleaned_cols.astype(float)
    scale = 1.0 / resolution_val
    df.columns = (col_floats * scale).round().astype(int)
    df.index = df.index.astype(int)
    return df
```

### Finding grid files

```python
import glob, os

def find_grid_file(results_dir, location, event_type, date_folder, resolution):
    """Locate a filled grid CSV. Falls back to unfilled."""
    prefix = 'ero' if event_type == 'erosion' else 'dep'
    grid_dir = os.path.join(results_dir, location, event_type, date_folder, resolution)

    # Prefer filled, fall back to raw
    patterns = [
        f"{date_folder}_{prefix}_grid_{resolution}_filled.csv",
        f"*_{prefix}_grid_{resolution}_filled.csv",
        f"{date_folder}_{prefix}_grid_{resolution}.csv",
        f"*_{prefix}_grid_{resolution}.csv",
    ]
    for pattern in patterns:
        matches = glob.glob(os.path.join(grid_dir, pattern))
        if matches:
            return matches[0]
    return None
```

### Resolutions

```python
RESOLUTION_MAP = {
    '10cm': 0.10,
    '25cm': 0.25,
    '1m':   1.00,
}
# Note: 1m resolution uses '100cm' in some filenames but '1m' in directory names
```

---

## 5. Loading Event Lists

### Event CSV structure

Location: `results/event_lists/erosion/<Location>_events.csv` (also `deposition/`, `combined/`)

| Column | Type | Description |
|--------|------|-------------|
| `mid_date` | date | Midpoint between start and end |
| `start_date` | date | Survey 1 date |
| `end_date` | date | Survey 2 date |
| `volume` | float | Event volume in m^3 |
| `elevation` | float | Volume-weighted elevation centroid (m) |
| `alongshore_centroid_m` | float | Volume-weighted alongshore position (m) |
| `alongshore_start_m` | float | Min alongshore extent |
| `alongshore_end_m` | float | Max alongshore extent |
| `mop_centroid` | float | MOP line at centroid |
| `mop_start` | float | MOP at start |
| `mop_end` | float | MOP at end |
| `width` | float | Alongshore extent (m) |
| `height` | float | Vertical extent (m) |
| `vol_unc` | float | Volume uncertainty (m^3) |
| `month` | int | Month of mid_date |

### Filtered events

Significant events are filtered to separate CSVs:

```
results/event_lists/erosion/<Location>_vol_5_elv_5.csv   # volume > 5 m^3, elevation > 5 m
```

### Loading events

```python
def load_events(csv_path):
    df = pd.read_csv(csv_path)
    for col in ['start_date', 'end_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d')
    return df
```

### Inferring location from filename

```python
import re

def infer_location(csv_path):
    """Extract location name from event CSV filename."""
    name = os.path.basename(csv_path).replace('.csv', '')
    # Strip QC suffix
    name = re.sub(r'_qc_\d{8}_\d{6}.*$', '', name)
    # Strip event suffixes
    for suffix in ['_dep_events_sig', '_events_sig', '_dep_events', '_events']:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    # Strip filter suffix
    name = re.sub(r'_vol_\d+_elv_\d+$', '', name)
    return name
```

---

## 6. Loading 3D Data Cubes (NPZ)

Location: `results/data_cubes/<Location>_cube.npz`

### NPZ contents

```python
data = np.load('path/to/cube.npz', allow_pickle=True)

data['erosion']        # shape: (n_alongshore, n_elevation, n_time) - M3C2 values
data['deposition']     # shape: (n_alongshore, n_elevation, n_time)
data['alongshore_m']   # shape: (n_alongshore,) - physical positions in meters
data['elevation_m']    # shape: (n_elevation,) - elevation values in meters
data['date_strings']   # shape: (n_time,) - folder names like 'YYYYMMDD_to_YYYYMMDD'
data['dates']          # shape: (n_time,) - ordinal dates for plotting
```

### Extracting a 2D slice for one survey interval

```python
def extract_slice(cube_data, start_date, end_date, event_type='erosion'):
    """Extract 2D grid (alongshore x elevation) from cube at a specific time."""
    # Build date folder string
    start = pd.to_datetime(start_date).strftime('%Y%m%d')
    end = pd.to_datetime(end_date).strftime('%Y%m%d')
    date_folder = f"{start}_to_{end}"

    # Find matching time index
    date_strings = [str(s) for s in cube_data['date_strings']]
    time_idx = date_strings.index(date_folder)

    # Extract slice
    cube_3d = cube_data[event_type]  # (alongshore, elevation, time)
    slice_2d = cube_3d[:, :, time_idx]

    # Sort alongshore for imshow
    alongshore = np.array(cube_data['alongshore_m'])
    sort_idx = np.argsort(alongshore)

    df = pd.DataFrame(
        slice_2d[sort_idx, :],
        index=alongshore[sort_idx],
        columns=cube_data['elevation_m']
    ).fillna(0.0)

    return df
```

### Mapping event CSV to NPZ cube

```python
def csv_to_npz(csv_path):
    """Map: results/event_lists/.../SanElijo_events.csv -> results/data_cubes/SanElijo_cube.npz"""
    location = infer_location(csv_path)
    # Walk up from csv_path to find 'results' directory
    current = os.path.dirname(os.path.abspath(csv_path))
    while os.path.basename(current) != 'results' and current != '/':
        current = os.path.dirname(current)
    return os.path.join(current, 'data_cubes', f'{location}_cube.npz')
```

---

## 7. Computing Volumes from Grids

### Per-interval volume with uncertainty

```python
def compute_volume(grid_path, unc_path, resolution_m):
    """Compute total volume and uncertainty bounds from grid CSV."""
    cell_area = resolution_m * resolution_m

    df = pd.read_csv(grid_path, index_col=0)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    distances = df.values

    # Main volume
    vol = distances.sum() * cell_area

    # Mean uncertainty from uncertainty CSV
    mean_unc = 0.0
    if unc_path and os.path.exists(unc_path):
        df_u = pd.read_csv(unc_path)
        unc_vals = df_u.iloc[:, 1:].values.flatten()
        unc_vals = unc_vals[~np.isnan(unc_vals) & (unc_vals > 0)]
        mean_unc = np.mean(unc_vals) if len(unc_vals) > 0 else 0.0

    # Bounds
    erosion_mask = distances > 0.01
    lower = distances.copy()
    lower[erosion_mask] = np.maximum(distances[erosion_mask] - mean_unc, 0)
    upper = distances.copy()
    upper[erosion_mask] = distances[erosion_mask] + mean_unc

    return vol, lower.sum() * cell_area, upper.sum() * cell_area
```

### Per-event volume (from cluster + grid CSVs)

```python
def extract_events(cluster_path, grid_path, resolution_m, mid_date):
    """Extract individual cluster events with volume, elevation, width, height."""
    df_c = pd.read_csv(cluster_path, index_col=0).fillna(0)
    df_g = pd.read_csv(grid_path, index_col=0).fillna(0)

    # Normalize column names: 'ClusterID_0.25m' -> '0.25m' (then parse elevation)
    df_c.columns = [c.split('_')[-1] for c in df_c.columns]
    df_g.columns = [c.split('_')[-1] for c in df_g.columns]

    # Align indices
    common_idx = df_c.index.intersection(df_g.index)
    common_cols = df_c.columns.intersection(df_g.columns)
    df_c = df_c.loc[common_idx, common_cols]
    df_g = df_g.loc[common_idx, common_cols]

    # Parse elevation values from column names
    z_values = np.array([float(re.findall(r"[-+]?\d*\.\d+|\d+", c)[0]) for c in df_c.columns])
    cell_area = resolution_m * resolution_m

    events = []
    for uid in np.unique(df_c.values):
        if uid == 0:
            continue
        mask = (df_c.values == uid)
        rows, cols = np.where(mask)
        dists = df_g.values[mask]

        events.append({
            'date': mid_date,
            'volume': np.sum(np.abs(dists)) * cell_area,
            'elevation': np.average(z_values[cols], weights=np.abs(dists)),
            'width': (rows.max() - rows.min() + 1) * resolution_m,
            'height': (cols.max() - cols.min() + 1) * resolution_m,
        })
    return events
```

---

## 8. Visualization Patterns

### Libraries used

- **matplotlib** + **gridspec**: All static figures (dashboards, heatmaps, GIFs)
- **seaborn**: Statistical plots (violin, histogram, ECDF)
- **PIL/Pillow**: GIF frame assembly
- **pyvista**: 3D point cloud rendering (`vis_ml_classes.py` only)
- **streamlit**: Interactive web tools (survey browser, event QC)
- **laspy**: LAS/LAZ point cloud I/O

### Colormaps

| Use Case | Colormap | Notes |
|----------|----------|-------|
| Erosion heatmaps | `magma_r` | White forced at 0 |
| Deposition heatmaps | `viridis_r` or `Blues` | |
| Combined (ero+dep) | `RdBu_r` | Red=erosion, Blue=deposition, White=0 |
| Event bubbles | `OrRd` | Orange-Red scaled by volume |

### Forcing white at zero (used everywhere)

```python
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm

def get_custom_cmap(cmap_name='magma_r', vmax=6.0):
    base_cmap = cm.get_cmap(cmap_name, 256)
    newcolors = base_cmap(np.linspace(0, 1, 256))
    newcolors[0, :] = np.array([1, 1, 1, 1])  # Pure white at 0
    cmap = LinearSegmentedColormap.from_list(f"White_{cmap_name}", newcolors)
    norm = Normalize(vmin=0, vmax=vmax)
    return cmap, norm
```

### Diverging colormap (QC tool, combined erosion+deposition)

```python
def get_diverging_cmap(vmax=1.0):
    cmap = cm.get_cmap('RdBu_r', 256)
    norm = Normalize(vmin=-vmax, vmax=vmax)
    return cmap, norm
```

### Heatmap (grid) visualization

The standard pattern for plotting grid data as a 2D heatmap:

```python
def plot_grid_heatmap(df, title, cmap_name='magma_r', vmax=6.0):
    """
    df: DataFrame from load_grid() — rows=alongshore, cols=elevation
    """
    matrix = df.values.T  # Transpose to (elevation, alongshore) for imshow

    cmap, norm = get_custom_cmap(cmap_name, vmax)

    fig, ax = plt.subplots(figsize=(14, 6))

    alongshore = df.index.values
    elevation = df.columns.values
    extent = [alongshore.min(), alongshore.max(), elevation.min(), elevation.max()]

    im = ax.imshow(matrix, origin='lower', extent=extent, aspect='auto',
                   interpolation='none', cmap=cmap, norm=norm)

    ax.invert_xaxis()  # Cliff-facing view
    ax.set_xlabel("Alongshore Position (m)")
    ax.set_ylabel("Elevation (m)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Erosion Depth (m)")
    plt.tight_layout()
    return fig
```

**Key conventions:**
- Always `origin='lower'` (elevation increases upward)
- Always `invert_xaxis()` for cliff-facing view
- Transpose the DataFrame: `df.values.T` makes rows=elevation, cols=alongshore

### Cumulative grid (summing across time)

```python
def build_cumulative_grid(base_dir, location, resolution='25cm', res_val=0.25):
    """Accumulate all filled grids over time into one cumulative grid."""
    erosion_dir = os.path.join(base_dir, 'results', location, 'erosion')
    intervals = sorted(os.listdir(erosion_dir))

    cumulative = None
    for interval in intervals:
        grid_path = os.path.join(erosion_dir, interval, resolution,
                                  f"{interval}_ero_grid_{resolution}_filled.csv")
        if not os.path.exists(grid_path):
            continue

        df = pd.read_csv(grid_path, index_col=0)
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        df = clean_and_snap_grid(df, res_val)

        if cumulative is None:
            cumulative = df
        else:
            cumulative = cumulative.add(df, fill_value=0)

    return cumulative
```

### Event QC heatmap (combined erosion + deposition)

From `event_qc.py` — combines erosion (positive, warm) and deposition (negative, cool):

```python
# Combine erosion and deposition into a single diverging matrix
combined = erosion_matrix.copy()
dep_mask = (deposition_matrix > 0) & (erosion_matrix < 0.01)
combined[dep_mask] = -deposition_matrix[dep_mask]

# Dynamic color scale from data (15th-85th percentile)
nonzero = combined[combined != 0]
vmax = np.percentile(np.abs(nonzero), 85) if nonzero.size > 0 else 2.5
cmap, norm = get_diverging_cmap(vmax=vmax)
```

### Zoom extent for event visualization

```python
def get_zoom_extent(event, x_pad_m=10, y_pad_bottom_m=8, y_pad_top_m=3):
    """Calculate viewport bounds centered on an event."""
    return {
        'x_min': event['alongshore_start_m'] - x_pad_m,
        'x_max': event['alongshore_end_m'] + x_pad_m,
        'y_min': max(0, event['elevation'] - event['height']/2 - y_pad_bottom_m),
        'y_max': event['elevation'] + event['height']/2 + y_pad_top_m,
    }
```

---

## 9. Dashboard Patterns

### 5-Panel Master Dashboard (`plot_dashboard.py`)

Panels: (A) Volume bars, (B) Cumulative + rate, (C) Spatiotemporal bubbles, (D) Spatial bubbles, (E) Cumulative heatmap

```python
fig = plt.figure(figsize=(24, 18))
gs = gridspec.GridSpec(6, 1, height_ratios=[1, 1, 1.5, 2.0, 1.2, 0.15], hspace=0.5)

# Data collection loop
for interval in intervals:
    d1, d2 = parse_dates(interval)  # regex: r'(\d{8})_to_(\d{8})'
    mid = d1 + (d2 - d1) / 2
    # Load grid, compute volume, extract events...
```

### Winter shading (Oct 1 - Mar 31)

Applied to all time-series panels:

```python
def add_winter_shading(ax, start_date, end_date):
    for year in range(start_date.year - 1, end_date.year + 1):
        winter_start = datetime(year, 10, 1)
        winter_end = datetime(year + 1, 3, 31)
        if winter_end >= start_date and winter_start <= end_date:
            ax.axvspan(winter_start, winter_end, color='#E9ECEF', alpha=0.5, zorder=0)
```

### Date parsing from folder names

```python
import re
from datetime import datetime

def parse_dates(folder_name):
    match = re.search(r'(\d{8})_to_(\d{8})', folder_name)
    if match:
        d1 = datetime.strptime(match.group(1), '%Y%m%d')
        d2 = datetime.strptime(match.group(2), '%Y%m%d')
        return d1, d2
    return None, None
```

### Bubble/scatter plots (events)

```python
# Spatial distribution of large events
df_large = df_events[df_events['volume'] > 5]

ax.scatter(
    df_large['alongshore_m'], df_large['elevation'],
    s=df_large['volume'] * 3.0,        # Size scales with volume
    c=df_large['volume'],               # Color by volume
    cmap='OrRd',
    norm=Normalize(vmin=0, vmax=50),
    alpha=0.7, edgecolors='black', linewidth=0.3
)
```

### 2x2 Geomorphology Dashboard (`geo_stats.py`)

Panels: (A) Power-law magnitude-frequency, (B) Morphology violin/box, (C) Lorenz curve + Gini, (D) Seasonality rose diagram

Key statistical patterns:
- **Power law**: Hardcoded cutoff at 0.25 m^3, fit log-log tail for beta exponent
- **Gini coefficient**: Lorenz curve inequality of event volumes
- **Seasonality**: Polar bar chart, winter months (Oct-Mar) in dark blue

---

## 10. Publication-Quality Defaults

```python
DPI = 300
FIGURE_SIZES = {
    'dashboard': (24, 18),
    'geomorph':  (24, 18),
    'heatmap':   (14, 6),
    'gif_frame': (14, 10),
}
FONT_SIZES = {
    'title': 18,
    'label': 14,
    'tick':  12,
    'legend': 12,
}

plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
```

---

## 11. Streamlit Apps

### Survey Browser (`code/streamlit/survey_browser.py`)

Interactive survey filtering. Run: `streamlit run code/streamlit/survey_browser.py`

- Sidebar: Beach selector, date range, instrument filter
- Output: Sortable dataframe of matching surveys with CSV download

### Event QC Tool (`code/streamlit/event_qc.py`)

Manual quality control of events. Run: `streamlit run code/streamlit/event_qc.py`

- Loads event CSVs from `results/event_lists/`
- Tries NPZ cube first (fast), falls back to individual grid CSVs
- Displays diverging heatmap with event crosshairs
- QC flags: `unreviewed`, `real`, `construction`, `noise`, `needs_check`
- Keyboard shortcuts: R/N/C/K/U for flags, arrow keys for navigation
- Auto-saves to `results/event_lists_qc/` with timestamped filenames
- Events sorted by volume (largest first)

### Shared utilities (`code/streamlit/utils/grid_loader.py`)

Reusable functions for both Streamlit apps and standalone scripts:

```python
from utils.grid_loader import (
    load_and_prepare_grid,       # CSV -> DataFrame with physical coords
    extract_grid_slice_from_cube, # NPZ cube -> 2D DataFrame at time t
    extract_both_slices_from_cube, # Get erosion + deposition at time t
    find_grid_file,              # Locate grid CSV by location/date/resolution
    get_zoom_extent,             # Calculate viewport around event
    load_event_csv,              # Load + parse event CSV
    csv_path_to_npz_path,        # Map event CSV -> NPZ cube path
    load_npz_cube,               # Load NPZ -> dict
    event_dates_to_folder,       # '2017-10-04', '2018-01-31' -> '20171004_to_20180131'
    infer_location_from_filename, # 'SanElijo_events.csv' -> 'SanElijo'
    infer_event_type_from_filename, # 'SanElijo_dep_events.csv' -> 'deposition'
)
```

---

## 12. Quick Reference: Common Data Access Recipes

### Load and plot a single erosion grid

```python
path = f"{BASE_DIR}/results/SanElijo/erosion/20200101_to_20200301/25cm/20200101_to_20200301_ero_grid_25cm_filled.csv"
df = load_grid(path, resolution_m=0.25)
fig = plot_grid_heatmap(df, "SanElijo Erosion: Jan-Mar 2020")
```

### Load all events for a location

```python
events = pd.read_csv(f"{BASE_DIR}/results/event_lists/erosion/SanElijo_events.csv")
large = events[events['volume'] > 5]
print(f"{len(large)} large events (>5 m^3) out of {len(events)} total")
```

### Build cumulative erosion from NPZ cube

```python
cube = np.load(f"{BASE_DIR}/results/data_cubes/SanElijo_cube.npz", allow_pickle=True)
cumulative = np.nansum(cube['erosion'], axis=2)  # Sum over time axis
```

### Collect time-series data across all intervals

```python
import os, re
from datetime import datetime

erosion_dir = f"{BASE_DIR}/results/SanElijo/erosion"
for folder in sorted(os.listdir(erosion_dir)):
    d1, d2 = parse_dates(folder)
    if not d1:
        continue
    grid_path = os.path.join(erosion_dir, folder, '25cm',
                              f'{folder}_ero_grid_25cm_filled.csv')
    if os.path.exists(grid_path):
        vol, _, _ = compute_volume(grid_path, None, 0.25)
        print(f"{folder}: {vol:.1f} m^3")
```

---

## 13. CloudCompare Coordinate Shifts

Required when running M3C2 or CANUPO steps. Must be consistent between steps 4 and 5.

```python
GLOBAL_SHIFT = {
    "SanElijo":   ("-473000", "-3653000", "0"),
    "Encinitas":  ("-472000", "-3655000", "0"),
    "Solana":     ("-475000", "-3650000", "0"),
    "Torrey":     ("-475000", "-3650000", "0"),
}
```

---

## 14. LAS Point Cloud Fields

### After M3C2 (Step 5)

| Field | Description |
|-------|-------------|
| `X, Y, Z` | Point coordinates (UTM Zone 11N) |
| `M3C2 distance` | Signed change in meters |
| `Distance uncertainty` | Pointwise uncertainty (m) |
| `Significant change` | Binary: 1 = significant, 0 = not |

### After DBSCAN (Step 6)

| Field | Description |
|-------|-------------|
| `M3C2_distance` | From M3C2 step |
| `Distance_uncertainty` | From M3C2 step |
| `ClusterID` | >=0 for clusters, -1 for noise |

Point clouds read with `laspy`:

```python
import laspy

las = laspy.read("path/to/file.las")
xyz = np.vstack([las.x, las.y, las.z]).T
# Scalar fields accessed as: las.point_format.dimension_names
```
