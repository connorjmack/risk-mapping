#!/usr/bin/env python3
"""
Generate synthetic LiDAR point clouds for testing PC-RAI.

This script creates test point clouds with known morphological characteristics
so that classification results can be validated.

Usage:
    python scripts/generate_test_data.py --output tests/test_data/
    python scripts/generate_test_data.py --type cliff --points 10000 --output cliff.las
"""

import argparse
from pathlib import Path
import numpy as np

try:
    import laspy
    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False
    print("Warning: laspy not installed. Will save as NPZ instead of LAS.")


def generate_horizontal_plane(n_points: int = 1000, noise: float = 0.01) -> dict:
    """
    Generate a flat horizontal surface.
    
    Expected classification: Talus (T) - low slope, smooth
    """
    np.random.seed(42)
    
    x = np.random.uniform(0, 10, n_points)
    y = np.random.uniform(0, 10, n_points)
    z = np.zeros(n_points) + np.random.normal(0, noise, n_points)
    
    xyz = np.column_stack([x, y, z]).astype(np.float64)
    
    # Normals pointing straight up
    normals = np.zeros((n_points, 3), dtype=np.float32)
    normals[:, 2] = 1.0
    # Add slight variation
    normals += np.random.normal(0, 0.02, (n_points, 3)).astype(np.float32)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    
    return {
        'xyz': xyz,
        'normals': normals,
        'description': 'Horizontal plane - expect Talus (T)',
        'expected_class': 1,
    }


def generate_vertical_cliff(n_points: int = 1000, noise: float = 0.02) -> dict:
    """
    Generate a vertical cliff face.
    
    Expected classification: Intact (I) - 90° slope, smooth
    """
    np.random.seed(43)
    
    x = np.zeros(n_points) + np.random.normal(0, noise, n_points)
    y = np.random.uniform(0, 10, n_points)
    z = np.random.uniform(0, 10, n_points)
    
    xyz = np.column_stack([x, y, z]).astype(np.float64)
    
    # Normals pointing in +X direction (perpendicular to YZ plane)
    normals = np.zeros((n_points, 3), dtype=np.float32)
    normals[:, 0] = 1.0
    normals += np.random.normal(0, 0.02, (n_points, 3)).astype(np.float32)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    
    return {
        'xyz': xyz,
        'normals': normals,
        'description': 'Vertical cliff - expect Intact (I)',
        'expected_class': 2,
    }


def generate_rough_cliff(n_points: int = 1000, roughness: float = 0.3) -> dict:
    """
    Generate a vertical cliff with rough surface.
    
    Expected classification: Dc or Dw depending on roughness level
    """
    np.random.seed(44)
    
    # Base vertical surface
    x = np.zeros(n_points) + np.random.normal(0, 0.05, n_points)
    y = np.random.uniform(0, 10, n_points)
    z = np.random.uniform(0, 10, n_points)
    
    # Add bumps/roughness to X
    x += 0.3 * np.sin(y * 5) * np.sin(z * 5)
    
    xyz = np.column_stack([x, y, z]).astype(np.float64)
    
    # Varied normals to create roughness
    normals = np.zeros((n_points, 3), dtype=np.float32)
    normals[:, 0] = 1.0
    normals += np.random.normal(0, roughness, (n_points, 3)).astype(np.float32)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    
    return {
        'xyz': xyz,
        'normals': normals,
        'description': f'Rough cliff (roughness={roughness}) - expect Dc or Dw',
        'expected_class': 4 if roughness < 0.4 else 5,
    }


def generate_overhang(n_points: int = 1000, angle_deg: float = 120) -> dict:
    """
    Generate an overhanging surface.
    
    angle_deg: Angle from vertical up (>90 = overhang)
    Expected classification: Os (90-150°) or Oc (>150°)
    """
    np.random.seed(45)
    
    angle_rad = np.radians(angle_deg)
    
    y = np.random.uniform(0, 10, n_points)
    z = np.random.uniform(0, 5, n_points)
    # X position depends on Z to create overhang
    x = z * np.tan(angle_rad - np.pi/2) + np.random.normal(0, 0.02, n_points)
    
    xyz = np.column_stack([x, y, z]).astype(np.float64)
    
    # Normal vector pointing "out" from overhang surface
    nx = np.sin(angle_rad)
    nz = -np.cos(angle_rad)  # Negative because overhanging
    
    normals = np.zeros((n_points, 3), dtype=np.float32)
    normals[:, 0] = nx
    normals[:, 2] = nz
    normals += np.random.normal(0, 0.05, (n_points, 3)).astype(np.float32)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    
    expected = 7 if angle_deg > 150 else 6
    
    return {
        'xyz': xyz,
        'normals': normals,
        'description': f'Overhang ({angle_deg}°) - expect {"Oc" if expected == 7 else "Os"}',
        'expected_class': expected,
    }


def generate_synthetic_cliff(n_points: int = 5000) -> dict:
    """
    Generate a complete synthetic cliff with multiple morphological zones.
    
    Zones (from bottom to top):
    1. Talus apron at base
    2. Smooth vertical cliff
    3. Rough/fractured section
    4. Overhang near top
    """
    np.random.seed(46)
    n_per_zone = n_points // 4
    
    all_xyz = []
    all_normals = []
    all_zones = []
    
    # Zone 1: Talus (Y: 0-2, sloped)
    x1 = np.random.uniform(0, 10, n_per_zone)
    y1 = np.random.uniform(0, 2, n_per_zone)
    z1 = y1 * 0.4 + np.random.normal(0, 0.05, n_per_zone)  # ~22° slope
    n1 = np.tile([0, -0.4, 0.917], (n_per_zone, 1)).astype(np.float32)
    n1 += np.random.normal(0, 0.05, n1.shape).astype(np.float32)
    n1 /= np.linalg.norm(n1, axis=1, keepdims=True)
    
    all_xyz.append(np.column_stack([x1, y1, z1]))
    all_normals.append(n1)
    all_zones.append(np.ones(n_per_zone, dtype=np.int32))
    
    # Zone 2: Smooth vertical cliff (Y: 2-2.2, Z: 0.8-6)
    x2 = np.random.uniform(0, 10, n_per_zone)
    y2 = np.full(n_per_zone, 2.1) + np.random.normal(0, 0.02, n_per_zone)
    z2 = np.random.uniform(0.8, 6, n_per_zone)
    n2 = np.tile([0, 1, 0], (n_per_zone, 1)).astype(np.float32)
    n2 += np.random.normal(0, 0.03, n2.shape).astype(np.float32)
    n2 /= np.linalg.norm(n2, axis=1, keepdims=True)
    
    all_xyz.append(np.column_stack([x2, y2, z2]))
    all_normals.append(n2)
    all_zones.append(np.full(n_per_zone, 2, dtype=np.int32))
    
    # Zone 3: Rough cliff section (Y: 2-2.3, Z: 6-9)
    x3 = np.random.uniform(0, 10, n_per_zone)
    y3 = np.full(n_per_zone, 2.15) + np.random.normal(0, 0.05, n_per_zone)
    z3 = np.random.uniform(6, 9, n_per_zone)
    n3 = np.tile([0, 1, 0], (n_per_zone, 1)).astype(np.float32)
    n3 += np.random.normal(0, 0.25, n3.shape).astype(np.float32)  # High roughness
    n3 /= np.linalg.norm(n3, axis=1, keepdims=True)
    
    all_xyz.append(np.column_stack([x3, y3, z3]))
    all_normals.append(n3)
    all_zones.append(np.full(n_per_zone, 3, dtype=np.int32))
    
    # Zone 4: Overhang (Y: 2-3, Z: 9-10)
    x4 = np.random.uniform(0, 10, n_per_zone)
    y4 = np.random.uniform(2.2, 3.5, n_per_zone)
    z4 = np.random.uniform(9, 10, n_per_zone)
    # Normal pointing down and out (~120° from up)
    n4 = np.tile([0, 0.5, -0.866], (n_per_zone, 1)).astype(np.float32)
    n4 += np.random.normal(0, 0.08, n4.shape).astype(np.float32)
    n4 /= np.linalg.norm(n4, axis=1, keepdims=True)
    
    all_xyz.append(np.column_stack([x4, y4, z4]))
    all_normals.append(n4)
    all_zones.append(np.full(n_per_zone, 4, dtype=np.int32))
    
    xyz = np.vstack(all_xyz).astype(np.float64)
    normals = np.vstack(all_normals).astype(np.float32)
    zones = np.concatenate(all_zones)
    
    return {
        'xyz': xyz,
        'normals': normals,
        'zones': zones,
        'description': 'Synthetic cliff with Talus, Intact, Rough, and Overhang zones',
        'zone_descriptions': {
            1: 'Talus (expect T)',
            2: 'Smooth cliff (expect I)',
            3: 'Rough cliff (expect Dc/Dw)',
            4: 'Overhang (expect Os)',
        }
    }


def save_as_las(data: dict, output_path: Path) -> None:
    """Save point cloud data as LAS file with normals."""
    if not HAS_LASPY:
        # Fallback to NPZ
        npz_path = output_path.with_suffix('.npz')
        np.savez(npz_path, **data)
        print(f"Saved as NPZ: {npz_path}")
        return
    
    xyz = data['xyz']
    normals = data['normals']
    
    las = laspy.create(point_format=0, file_version="1.4")
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]
    
    # Add normals as extra dimensions
    las.add_extra_dim(laspy.ExtraBytesParams(name="NormalX", type=np.float32))
    las.add_extra_dim(laspy.ExtraBytesParams(name="NormalY", type=np.float32))
    las.add_extra_dim(laspy.ExtraBytesParams(name="NormalZ", type=np.float32))
    
    las["NormalX"] = normals[:, 0]
    las["NormalY"] = normals[:, 1]
    las["NormalZ"] = normals[:, 2]
    
    # Add zone info if present
    if 'zones' in data:
        las.add_extra_dim(laspy.ExtraBytesParams(name="zone", type=np.uint8))
        las["zone"] = data['zones'].astype(np.uint8)
    
    las.write(output_path)
    print(f"Saved: {output_path} ({len(xyz):,} points)")


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic point clouds for PC-RAI testing'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('tests/test_data'),
        help='Output directory or file path'
    )
    parser.add_argument(
        '--type', '-t',
        choices=['all', 'cliff', 'horizontal', 'vertical', 'rough', 'overhang'],
        default='all',
        help='Type of test data to generate'
    )
    parser.add_argument(
        '--points', '-n',
        type=int,
        default=5000,
        help='Number of points per surface'
    )
    
    args = parser.parse_args()
    
    # Create output directory if needed
    if args.output.suffix == '':
        args.output.mkdir(parents=True, exist_ok=True)
        output_dir = args.output
        single_file = False
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_dir = args.output.parent
        single_file = True
    
    generators = {
        'horizontal': generate_horizontal_plane,
        'vertical': generate_vertical_cliff,
        'rough': generate_rough_cliff,
        'overhang': lambda n: generate_overhang(n, angle_deg=120),
        'cliff': generate_synthetic_cliff,
    }
    
    if single_file:
        # Generate single type
        gen_type = args.type if args.type != 'all' else 'cliff'
        data = generators[gen_type](args.points)
        save_as_las(data, args.output)
        print(f"\n{data['description']}")
    else:
        # Generate all types
        types_to_generate = list(generators.keys()) if args.type == 'all' else [args.type]
        
        for gen_type in types_to_generate:
            data = generators[gen_type](args.points)
            output_path = output_dir / f"synthetic_{gen_type}.las"
            save_as_las(data, output_path)
            print(f"  {data['description']}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
