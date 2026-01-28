"""
RAI Energy calculation module.

Implements the Rockfall Activity Index energy calculation from Dunham et al. (2017).
The RAI quantifies rockfall hazard as annual kinetic energy (kJ) based on
morphological class, height above base, and class-specific instability rates.

Reference:
    Dunham, L., Wartman, J., Olsen, M.J., O'Banion, M., & Cunningham, K. (2017).
    Rockfall Activity Index (RAI): A lidar-derived, morphology-based method
    for hazard assessment. Engineering Geology, 221, 184-192.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class RAIEnergyParams:
    """Parameters for RAI energy calculation from Dunham et al. (2017) Table 1.

    Attributes
    ----------
    failure_depth : dict
        Estimated depth of failure mass (m) for each class.
    instability_rate : dict
        Fraction of cells expected to fail per year (%) for each class.
    rock_density : float
        Rock density in kg/m³ (default 2600 kg/m³, ~2.6 specific gravity).
    gravity : float
        Acceleration due to gravity (m/s²).
    cell_area : float
        Cell area for volume calculation (m²). Default 0.0025 m² (5cm × 5cm).
    """

    # Failure depths from Dunham et al. (2017) Table 1
    # Class code -> depth in meters
    failure_depth: Dict[int, float] = None

    # Instability rates from Dunham et al. (2017) Table 1
    # Class code -> annual instability rate as fraction (not percentage)
    instability_rate: Dict[int, float] = None

    # Physical constants
    rock_density: float = 2600.0  # kg/m³ (specific gravity ~2.6)
    gravity: float = 9.8  # m/s²
    cell_area: float = 0.0025  # m² (5cm × 5cm grid cell)

    def __post_init__(self):
        """Initialize default values adapted from Dunham et al. (2017) Table 1.

        Updated for simplified 5-class scheme:
        - Class 3 (Discontinuous) merges Df, Dc, Dw - uses average values
        - Class 4 (Steep/Overhang) merges Os, Oc - uses average values
        - Class 5 (Structure) - engineered surfaces, minimal natural rockfall
        """
        if self.failure_depth is None:
            self.failure_depth = {
                0: 0.0,    # Unclassified - no contribution
                1: 0.025,  # Talus (T) - from Dunham
                2: 0.05,   # Intact (I) - from Dunham
                3: 0.2,    # Discontinuous (D) - avg of Df(0.1), Dc(0.2), Dw(0.3)
                4: 0.625,  # Steep/Overhang (O) - avg of Os(0.75), Oc(0.5)
                5: 0.0,    # Structure (St) - engineered, no natural rockfall
            }

        if self.instability_rate is None:
            # Convert from percentage to fraction
            self.instability_rate = {
                0: 0.0,       # Unclassified - no contribution
                1: 0.0,       # Talus (T) - 0%
                2: 0.001,     # Intact (I) - 0.1%
                3: 0.004,     # Discontinuous (D) - avg of Df, Dc, Dw (~0.4%)
                4: 0.02,      # Steep/Overhang (O) - avg of Os, Oc (~2%)
                5: 0.0,       # Structure (St) - engineered, no natural rockfall
            }


def calculate_point_energy(
    rai_classes: np.ndarray,
    z_coords: np.ndarray,
    base_elevation: Optional[float] = None,
    params: Optional[RAIEnergyParams] = None,
) -> np.ndarray:
    """
    Calculate per-point RAI energy contribution.

    The energy for each point is calculated as:
        E = (1/2) * m * v² * r

    Where:
        m = ρ * V = ρ * A * d  (mass = density × volume)
        v = √(2gh)  (velocity from free fall)
        r = instability rate (fraction per year)

    Substituting:
        E = (1/2) * ρ * A * d * 2gh * r
        E = ρ * A * d * g * h * r

    Parameters
    ----------
    rai_classes : np.ndarray
        (N,) array of RAI class codes (0-7).
    z_coords : np.ndarray
        (N,) array of Z coordinates (elevation) in meters.
    base_elevation : float, optional
        Elevation of slope base (e.g., road surface). If None, uses min(z_coords).
    params : RAIEnergyParams, optional
        Energy calculation parameters. Uses Dunham et al. (2017) defaults if None.

    Returns
    -------
    energy : np.ndarray
        (N,) array of per-point energy in kilojoules (kJ).
        Points with h <= 0 or invalid classes have energy = 0.
    """
    if params is None:
        params = RAIEnergyParams()

    n_points = len(rai_classes)
    energy = np.zeros(n_points, dtype=np.float64)

    # Determine base elevation
    if base_elevation is None:
        base_elevation = np.nanmin(z_coords)

    # Calculate fall height for each point
    fall_height = z_coords - base_elevation
    fall_height = np.maximum(fall_height, 0.0)  # No negative heights

    # Vectorized calculation for each class
    for class_code in range(6):  # 6 classes in simplified scheme
        mask = rai_classes == class_code

        if not np.any(mask):
            continue

        d = params.failure_depth.get(class_code, 0.0)
        r = params.instability_rate.get(class_code, 0.0)

        if d <= 0 or r <= 0:
            continue

        # E = ρ * A * d * g * h * r
        # Convert to kJ by dividing by 1000
        h = fall_height[mask]
        point_energy = (
            params.rock_density
            * params.cell_area
            * d
            * params.gravity
            * h
            * r
            / 1000.0  # Convert J to kJ
        )
        energy[mask] = point_energy

    return energy


def calculate_velocity(
    z_coords: np.ndarray,
    base_elevation: Optional[float] = None,
    gravity: float = 9.8,
) -> np.ndarray:
    """
    Calculate free-fall velocity for each point.

    v = √(2gh)

    Parameters
    ----------
    z_coords : np.ndarray
        (N,) array of Z coordinates (elevation) in meters.
    base_elevation : float, optional
        Elevation of slope base. If None, uses min(z_coords).
    gravity : float
        Acceleration due to gravity (m/s²).

    Returns
    -------
    velocity : np.ndarray
        (N,) array of velocities in m/s.
    """
    if base_elevation is None:
        base_elevation = np.nanmin(z_coords)

    fall_height = z_coords - base_elevation
    fall_height = np.maximum(fall_height, 0.0)

    velocity = np.sqrt(2 * gravity * fall_height)
    return velocity


def calculate_mass(
    rai_classes: np.ndarray,
    params: Optional[RAIEnergyParams] = None,
) -> np.ndarray:
    """
    Calculate per-point mass based on class-specific failure depth.

    m = ρ * V = ρ * A * d

    Parameters
    ----------
    rai_classes : np.ndarray
        (N,) array of RAI class codes (0-7).
    params : RAIEnergyParams, optional
        Energy calculation parameters.

    Returns
    -------
    mass : np.ndarray
        (N,) array of masses in kg.
    """
    if params is None:
        params = RAIEnergyParams()

    n_points = len(rai_classes)
    mass = np.zeros(n_points, dtype=np.float64)

    for class_code in range(6):  # 6 classes in simplified scheme
        mask = rai_classes == class_code
        if not np.any(mask):
            continue

        d = params.failure_depth.get(class_code, 0.0)
        m = params.rock_density * params.cell_area * d
        mass[mask] = m

    return mass


def get_energy_statistics(
    energy: np.ndarray,
    rai_classes: np.ndarray,
) -> Dict:
    """
    Calculate energy statistics by class.

    Parameters
    ----------
    energy : np.ndarray
        (N,) array of per-point energy in kJ.
    rai_classes : np.ndarray
        (N,) array of RAI class codes.

    Returns
    -------
    stats : dict
        Dictionary with energy statistics overall and by class.
    """
    from pc_rai.config import RAI_CLASS_NAMES

    valid_mask = ~np.isnan(energy) & (energy > 0)
    valid_energy = energy[valid_mask]

    stats = {
        "total_energy_kj": float(np.sum(energy)),
        "mean_energy_kj": float(np.mean(valid_energy)) if len(valid_energy) > 0 else 0.0,
        "max_energy_kj": float(np.max(energy)) if len(energy) > 0 else 0.0,
        "n_contributing_points": int(np.sum(energy > 0)),
        "by_class": {},
    }

    for class_code in range(6):  # 6 classes in simplified scheme
        mask = rai_classes == class_code
        class_energy = energy[mask]

        if len(class_energy) == 0:
            continue

        stats["by_class"][class_code] = {
            "name": RAI_CLASS_NAMES.get(class_code, f"Unknown ({class_code})"),
            "total_energy_kj": float(np.sum(class_energy)),
            "mean_energy_kj": float(np.mean(class_energy[class_energy > 0])) if np.any(class_energy > 0) else 0.0,
            "max_energy_kj": float(np.max(class_energy)),
            "n_points": int(np.sum(mask)),
            "contribution_pct": 100.0 * np.sum(class_energy) / np.sum(energy) if np.sum(energy) > 0 else 0.0,
        }

    return stats
