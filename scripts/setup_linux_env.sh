#!/bin/bash
##############################################################################
# setup_linux_env.sh
#
# Sets up the CloudComPy311 conda environment on Linux with additional
# pc-rai / ML pipeline dependencies installed into it.
#
# Follows the official CloudComPy Linux installation docs:
#   https://cloudcompy.github.io/
#
# Prerequisites:
#   - Anaconda or Miniconda installed
#   - CloudComPy Linux binary downloaded from:
#     https://www.simulation.openfields.fr/index.php/cloudcompy-downloads/3-cloudcompy-binaries/4-linux-cloudcompy-binaries
#     (download the latest CloudComPy_Conda311_Linux64_*.tgz)
#   - libomp5 installed: sudo apt-get install libomp5
#
# Usage:
#   chmod +x scripts/setup_linux_env.sh
#   ./scripts/setup_linux_env.sh /path/to/CloudComPy311
#
##############################################################################

set -euo pipefail

# Official CloudComPy environment name — condaCloud.sh expects this
ENV_NAME="CloudComPy311"

# --- Check arguments ---
if [ $# -lt 1 ]; then
    echo "Usage: $0 /path/to/CloudComPy311"
    echo ""
    echo "Steps:"
    echo "  1. Download CloudComPy_Conda311_Linux64_*.tgz from GitHub releases"
    echo "  2. Extract it:  tar xzf CloudComPy_Conda311_Linux64_*.tgz"
    echo "  3. Install libomp5:  sudo apt-get install libomp5"
    echo "  4. Run this script:  $0 /path/to/CloudComPy311"
    exit 1
fi

CLOUDCOMPY_ROOT="$(realpath "$1")"

if [ ! -d "$CLOUDCOMPY_ROOT" ]; then
    echo "ERROR: CloudComPy directory not found: $CLOUDCOMPY_ROOT"
    exit 1
fi

if [ ! -f "$CLOUDCOMPY_ROOT/bin/condaCloud.sh" ]; then
    echo "ERROR: condaCloud.sh not found in $CLOUDCOMPY_ROOT/bin/"
    echo "       Is this the correct CloudComPy install directory?"
    exit 1
fi

echo "============================================"
echo "Setting up CloudComPy311 env + pc-rai deps"
echo "  CloudComPy root: $CLOUDCOMPY_ROOT"
echo "  Conda env name:  $ENV_NAME"
echo "============================================"
echo ""

# --- Detect conda installation ---
CONDA_DIR=""
if [ -n "${CONDA_EXE:-}" ]; then
    CONDA_DIR="$(dirname "$(dirname "$CONDA_EXE")")"
elif [ -d "$HOME/miniconda3" ]; then
    CONDA_DIR="$HOME/miniconda3"
elif [ -d "$HOME/anaconda3" ]; then
    CONDA_DIR="$HOME/anaconda3"
else
    echo "ERROR: Could not find conda installation."
    echo "       Set CONDA_EXE or install Anaconda/Miniconda."
    exit 1
fi

echo "Using conda at: $CONDA_DIR"

# Ensure conda is available in this shell
. "$CONDA_DIR/etc/profile.d/conda.sh"

# --- Step 1: Create conda environment (per official docs) ---
echo ""
echo "[1/4] Creating conda environment '$ENV_NAME' with Python 3.11..."

if conda info --envs | grep -q "^${ENV_NAME} "; then
    echo "  Environment '$ENV_NAME' already exists. Skipping creation."
    echo "  (Delete with: conda env remove -n $ENV_NAME)"
else
    conda create -y --name "$ENV_NAME" python=3.11
fi

# --- Step 2: Install CloudComPy conda dependencies (per official docs) ---
echo ""
echo "[2/4] Installing CloudComPy conda dependencies..."

conda activate "$ENV_NAME"

conda config --add channels conda-forge
conda config --set channel_priority flexible

# Exact package list from official CloudComPy Linux docs
conda install -y \
    "boost=1.84" \
    "cgal=5.6" \
    cmake \
    "draco=1.5" \
    "ffmpeg=6.1" \
    "gdal=3.8" \
    laszip \
    "matplotlib=3.9" \
    "mpir=3.0" \
    "mysql=8" \
    numpy \
    "opencv=4.9" \
    "openmp=8.0" \
    "openssl>=3.1" \
    "pcl=1.14" \
    "pdal=2.6" \
    "psutil=6.0" \
    pybind11 \
    quaternion \
    "qhull=2020.2" \
    "qt=5.15.8" \
    scipy \
    tbb \
    tbb-devel \
    "xerces-c=3.2"

# --- Step 3: Install pc-rai pip dependencies into the same env ---
echo ""
echo "[3/4] Installing pc-rai dependencies via pip..."

# Packages pc-rai needs that CloudComPy doesn't provide
pip install --no-deps \
    "laspy[lazrs]>=2.4" \
    "tqdm>=4.62" \
    "pyyaml>=6.0" \
    "scikit-learn>=1.0" \
    "pyshp>=2.3" \
    "pandas>=1.4" \
    "joblib>=1.1"

# Install pc-rai itself in development mode
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

if [ -f "$REPO_ROOT/pyproject.toml" ]; then
    echo "  Installing pc-rai package in editable mode..."
    pip install -e "$REPO_ROOT" --no-deps
else
    echo "  WARNING: pyproject.toml not found at $REPO_ROOT"
    echo "  Run 'pip install -e .' from the repo root after setup."
fi

# Dev/test dependencies
pip install pytest pytest-cov black ruff

# --- Step 4: Verify and print instructions ---
echo ""
echo "[4/4] Verifying installation..."

# Quick sanity check while env is still active
python -c "import sklearn; print(f'  scikit-learn OK: {sklearn.__version__}')" 2>/dev/null \
    || echo "  WARNING: scikit-learn not found"
python -c "import laspy; print(f'  laspy OK: {laspy.__version__}')" 2>/dev/null \
    || echo "  WARNING: laspy not found"

# Test CloudComPy via the official activation method
conda deactivate
. "$CLOUDCOMPY_ROOT/bin/condaCloud.sh" activate "$ENV_NAME"
python -c "import cloudComPy as cc; print(f'  CloudComPy OK: {cc.GetScalarType(1)}')" 2>/dev/null \
    || echo "  WARNING: CloudComPy import failed — you may need: sudo apt-get install libomp5"

echo ""
echo "============================================"
echo "Setup complete!"
echo ""
echo "To activate the environment (from a new terminal):"
echo "  . $CLOUDCOMPY_ROOT/bin/condaCloud.sh activate $ENV_NAME"
echo ""
echo "Then run the full pipeline:"
echo "  python scripts/02_extract_features.py --subsample-only ..."
echo "  python scripts/compute_normals_mst.py ..."
echo "  python scripts/02_extract_features.py ..."
echo "  python scripts/03_aggregate_polygons.py ..."
echo "  python scripts/05_train_model.py ..."
echo ""
echo "============================================"
