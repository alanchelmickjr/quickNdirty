#!/usr/bin/env python3
"""
RealSense D455 depth test
"""

import pyrealsense2 as rs
import numpy as np
import cv2
from pathlib import Path
from plyfile import PlyData, PlyElement

OUTPUT_DIR = Path("collar_scans_d455")
OUTPUT_DIR.mkdir(exist_ok=True)

# Configure pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start
profile = pipeline.start(config)

# Get depth scale for converting to meters
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"Depth scale: {depth_scale}")

# Enable laser if available
if depth_sensor.supports(rs.option.laser_power):
    depth_sensor.set_option(rs.option.laser_power, 360)  # max power
    print("Laser at max power")

frame_num = 0
print("SPACE=capture, Q=quit")

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Colorize depth
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )

        # Count valid pixels
        valid = np.sum(depth_image > 0)
        cv2.putText(depth_colormap, f"Valid: {valid:,} | Frame: {frame_num}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow('D455 Depth', depth_colormap)
        cv2.imshow('D455 Color', color_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            # Save depth image
            cv2.imwrite(str(OUTPUT_DIR / f"depth_{frame_num:03d}.png"), depth_colormap)
            cv2.imwrite(str(OUTPUT_DIR / f"rgb_{frame_num:03d}.png"), color_image)

            # Generate point cloud
            pc = rs.pointcloud()
            points = pc.calculate(depth_frame)
            vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)

            # Filter out zeros
            mask = vertices[:, 2] > 0
            vertices = vertices[mask]

            if len(vertices) > 0:
                ply_vertices = np.array(
                    [(v[0], v[1], v[2]) for v in vertices],
                    dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
                )
                PlyData([PlyElement.describe(ply_vertices, 'vertex')]).write(
                    str(OUTPUT_DIR / f"cloud_{frame_num:03d}.ply"))
                print(f"Frame {frame_num}: {len(vertices):,} points")
            else:
                print(f"Frame {frame_num}: no points")

            frame_num += 1

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

print(f"Saved to {OUTPUT_DIR.absolute()}")
