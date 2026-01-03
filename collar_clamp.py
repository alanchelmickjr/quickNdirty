#!/usr/bin/env python3
"""
Collar clamp bracket - clamshell design
Top: 22mm x 3mm (slides over 19mm notch)
Bottom: 18mm x 20mm (slides into mount)
Uniform parabolic curve throughout
"""

import numpy as np
from stl import mesh

# Dimensions (mm)
TOP_WIDTH = 22.0
TOP_HEIGHT = 3.0
BOTTOM_WIDTH = 18.0
BOTTOM_HEIGHT = 20.0
CURVE_DEPTH = 3.0    # parabola depth at edges
THICKNESS = 1.5      # wall thickness

TOTAL_HEIGHT = TOP_HEIGHT + BOTTOM_HEIGHT  # 23mm total

def parabola_z(x, width):
    """Return z offset for parabolic curve. 0 at center, CURVE_DEPTH at edges."""
    half_w = width / 2
    a = CURVE_DEPTH / (half_w ** 2)
    return a * x ** 2

n_points = 50  # points along curve
n_layers = 30  # vertical layers

vertices = []

# Generate vertices layer by layer from bottom to top
for layer in range(n_layers):
    t = layer / (n_layers - 1)  # 0 to 1
    y = t * TOTAL_HEIGHT

    # Width transitions from BOTTOM_WIDTH to TOP_WIDTH at TOP_HEIGHT boundary
    if y <= BOTTOM_HEIGHT:
        width = BOTTOM_WIDTH
    else:
        # Linear transition in top section (could also be instant)
        width = TOP_WIDTH

    half_w = width / 2
    x_vals = np.linspace(-half_w, half_w, n_points)

    for x in x_vals:
        z_outer = -parabola_z(x, width)
        z_inner = z_outer - THICKNESS
        vertices.append([x, y, z_outer])  # outer surface
        vertices.append([x, y, z_inner])  # inner surface

vertices = np.array(vertices)

# Generate faces
faces = []
pts_per_layer = n_points * 2  # outer + inner points per layer

for layer in range(n_layers - 1):
    base = layer * pts_per_layer
    next_base = (layer + 1) * pts_per_layer

    for i in range(n_points - 1):
        # Outer surface
        o0 = base + i * 2
        o1 = base + (i + 1) * 2
        o2 = next_base + i * 2
        o3 = next_base + (i + 1) * 2
        faces.append([o0, o1, o2])
        faces.append([o1, o3, o2])

        # Inner surface (reversed winding)
        i0 = base + i * 2 + 1
        i1 = base + (i + 1) * 2 + 1
        i2 = next_base + i * 2 + 1
        i3 = next_base + (i + 1) * 2 + 1
        faces.append([i0, i2, i1])
        faces.append([i1, i2, i3])

# Top edge (connect outer to inner at top)
top_base = (n_layers - 1) * pts_per_layer
for i in range(n_points - 1):
    to0 = top_base + i * 2
    to1 = top_base + (i + 1) * 2
    ti0 = top_base + i * 2 + 1
    ti1 = top_base + (i + 1) * 2 + 1
    faces.append([to0, to1, ti0])
    faces.append([ti0, to1, ti1])

# Bottom edge (connect outer to inner at bottom)
for i in range(n_points - 1):
    bo0 = i * 2
    bo1 = (i + 1) * 2
    bi0 = i * 2 + 1
    bi1 = (i + 1) * 2 + 1
    faces.append([bo0, bi0, bo1])
    faces.append([bi0, bi1, bo1])

# Left end cap (all layers)
for layer in range(n_layers - 1):
    base = layer * pts_per_layer
    next_base = (layer + 1) * pts_per_layer
    # left edge is index 0 (outer) and 1 (inner)
    faces.append([base, next_base, base + 1])
    faces.append([base + 1, next_base, next_base + 1])

# Right end cap (all layers)
for layer in range(n_layers - 1):
    base = layer * pts_per_layer + (n_points - 1) * 2
    next_base = (layer + 1) * pts_per_layer + (n_points - 1) * 2
    faces.append([base, base + 1, next_base])
    faces.append([base + 1, next_base + 1, next_base])

faces = np.array(faces)

# Create mesh
clamp = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    for j in range(3):
        clamp.vectors[i][j] = vertices[f[j], :]

clamp.save('collar_clamp.stl')
print(f"Saved collar_clamp.stl")
print(f"  Top: {TOP_WIDTH}mm x {TOP_HEIGHT}mm")
print(f"  Bottom: {BOTTOM_WIDTH}mm x {BOTTOM_HEIGHT}mm")
print(f"  Curve depth: {CURVE_DEPTH}mm")
print(f"  Thickness: {THICKNESS}mm")
print(f"  Total height: {TOTAL_HEIGHT}mm")
