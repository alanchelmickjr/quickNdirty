#!/usr/bin/env python3
"""
Generate the collar bracket - parabolic clamshell
19mm wide x 11mm tall, 3mm curve depth at edges
"""

import numpy as np
from stl import mesh

# Dimensions (mm)
WIDTH = 19.0        # total width
HEIGHT = 11.0       # total height
CURVE_DEPTH = 3.0   # how far back the edges are from center
THICKNESS = 1.5     # wall thickness of the bracket

# Parabola: z = a * x^2, where at x=9.5, z=3
# 3 = a * 9.5^2 => a = 3 / 90.25 = 0.0332
A = CURVE_DEPTH / (WIDTH/2)**2

# Generate points along the parabola
n_points = 50
x = np.linspace(-WIDTH/2, WIDTH/2, n_points)
z = A * x**2  # depth (0 at center, 3 at edges)

# Create vertices for front and back surfaces at y=0 and y=HEIGHT
vertices = []
faces = []

# Front surface (outer curve)
for i, (xi, zi) in enumerate(zip(x, z)):
    vertices.append([xi, 0, -zi])           # bottom front
    vertices.append([xi, HEIGHT, -zi])      # top front

# Back surface (inner curve, offset by thickness)
for i, (xi, zi) in enumerate(zip(x, z)):
    vertices.append([xi, 0, -zi - THICKNESS])      # bottom back
    vertices.append([xi, HEIGHT, -zi - THICKNESS]) # top back

vertices = np.array(vertices)

# Generate faces (triangles)
# Front surface quads -> triangles
for i in range(n_points - 1):
    v0 = i * 2
    v1 = i * 2 + 1
    v2 = (i + 1) * 2
    v3 = (i + 1) * 2 + 1
    faces.append([v0, v2, v1])
    faces.append([v1, v2, v3])

# Back surface quads -> triangles
back_offset = n_points * 2
for i in range(n_points - 1):
    v0 = back_offset + i * 2
    v1 = back_offset + i * 2 + 1
    v2 = back_offset + (i + 1) * 2
    v3 = back_offset + (i + 1) * 2 + 1
    faces.append([v0, v1, v2])  # reversed winding
    faces.append([v1, v3, v2])

# Top surface (connect front top to back top)
for i in range(n_points - 1):
    ft0 = i * 2 + 1
    ft1 = (i + 1) * 2 + 1
    bt0 = back_offset + i * 2 + 1
    bt1 = back_offset + (i + 1) * 2 + 1
    faces.append([ft0, ft1, bt0])
    faces.append([bt0, ft1, bt1])

# Bottom surface (connect front bottom to back bottom)
for i in range(n_points - 1):
    fb0 = i * 2
    fb1 = (i + 1) * 2
    bb0 = back_offset + i * 2
    bb1 = back_offset + (i + 1) * 2
    faces.append([fb0, bb0, fb1])
    faces.append([fb1, bb0, bb1])

# Left end cap
faces.append([0, 1, back_offset])
faces.append([1, back_offset + 1, back_offset])

# Right end cap
last = n_points - 1
faces.append([last*2, back_offset + last*2, last*2 + 1])
faces.append([last*2 + 1, back_offset + last*2, back_offset + last*2 + 1])

faces = np.array(faces)

# Create the mesh
collar = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    for j in range(3):
        collar.vectors[i][j] = vertices[f[j], :]

# Save
collar.save('collar_bracket.stl')
print(f"Saved collar_bracket.stl")
print(f"  Width: {WIDTH}mm")
print(f"  Height: {HEIGHT}mm")
print(f"  Curve depth: {CURVE_DEPTH}mm")
print(f"  Thickness: {THICKNESS}mm")
print(f"  Vertices: {len(vertices)}")
print(f"  Faces: {len(faces)}")
