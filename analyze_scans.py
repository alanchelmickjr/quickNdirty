#!/usr/bin/env python3
"""
Analyze the collar scans to find the notch shape
"""

import numpy as np
from plyfile import PlyData

# Load first point cloud
ply = PlyData.read('collar_scans/cloud_000.ply')
vertices = ply['vertex']

x = np.array(vertices['x'])
y = np.array(vertices['y'])
z = np.array(vertices['z'])

print(f"Points: {len(x):,}")
print(f"X range: {x.min():.1f} to {x.max():.1f} mm")
print(f"Y range: {y.min():.1f} to {y.max():.1f} mm")
print(f"Z range: {z.min():.1f} to {z.max():.1f} mm")

# Find the center region where the collar should be
# The collar is a void/silhouette, so we look for gaps in depth
center_x = (x.min() + x.max()) / 2
center_y = (y.min() + y.max()) / 2

print(f"\nCenter: ({center_x:.1f}, {center_y:.1f})")

# Look at depth distribution in center region
mask = (np.abs(x - center_x) < 50) & (np.abs(y - center_y) < 50)
center_z = z[mask]
print(f"Points in center region: {len(center_z):,}")
if len(center_z) > 0:
    print(f"Center Z range: {center_z.min():.1f} to {center_z.max():.1f}")

# Histogram of Z values to see depth layers
print("\nZ histogram (mm):")
hist, edges = np.histogram(z, bins=20)
for i, (count, edge) in enumerate(zip(hist, edges[:-1])):
    bar = '#' * (count // 10000)
    print(f"  {edge:7.1f}: {bar} ({count:,})")
