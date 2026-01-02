#!/usr/bin/env python3
"""
OAK-D S3 Pro 3D Scanner - using RGBD node (v3 simplified API)
"""

import depthai as dai
import numpy as np
import cv2
from plyfile import PlyData, PlyElement
from pathlib import Path
import time
import argparse
import sys

OUTPUT_DIR = Path("collar_scans")
OUTPUT_DIR.mkdir(exist_ok=True)

IR_LASER_INTENSITY = 0.8


def main():
    parser = argparse.ArgumentParser(description="OAK-D S3 Pro 3D Scanner")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--auto", type=int, metavar="N")
    parser.add_argument("--ir", type=float, default=0.8)
    args = parser.parse_args()

    global IR_LASER_INTENSITY
    IR_LASER_INTENSITY = args.ir

    print("=" * 50)
    print("OAK-D S3 PRO SCANNER (RGBD Node)")
    print("=" * 50)

    # Check devices
    devices = dai.Device.getAllAvailableDevices()
    if not devices:
        print("ERROR: No OAK-D camera detected!")
        sys.exit(1)

    print(f"Found {len(devices)} device(s)")

    # Simple pipeline with RGBD node
    pipeline = dai.Pipeline()

    # Use RGBD node - it auto-creates stereo and color
    rgbd = pipeline.create(dai.node.RGBD).build()

    # Get outputs
    q_depth = rgbd.depth.createOutputQueue()
    q_pcl = rgbd.pcl.createOutputQueue()
    q_rgb = rgbd.color.createOutputQueue()

    print("Starting pipeline...")
    pipeline.start()

    # Enable IR
    try:
        for device in pipeline.getDevices():
            device.setIrLaserDotProjectorIntensity(IR_LASER_INTENSITY)
            print(f"IR laser: {IR_LASER_INTENSITY*100:.0f}%")
    except Exception as e:
        print(f"IR projector: {e}")

    print("Warming up...")
    time.sleep(2)

    frame_num = 0
    total_points = 0

    print("\nREADY - SPACE=capture, Q=quit")

    while pipeline.isRunning():
        depth_frame = q_depth.tryGet()
        if depth_frame is not None:
            depth_img = depth_frame.getCvFrame()
            depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_img, alpha=0.03),
                cv2.COLORMAP_JET
            )
            valid_pixels = np.sum(depth_img > 0)
            cv2.putText(depth_vis, f"Valid: {valid_pixels:,}px | Frame: {frame_num}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imshow("Depth Preview", depth_vis)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            # Capture
            time.sleep(0.2)
            depth_frame = q_depth.get()
            pcl_data = q_pcl.get()
            rgb_frame = q_rgb.get()

            depth_img = depth_frame.getCvFrame()
            cv2.imwrite(str(OUTPUT_DIR / f"depth_{frame_num:03d}.png"),
                       cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET))

            rgb_img = rgb_frame.getCvFrame()
            cv2.imwrite(str(OUTPUT_DIR / f"rgb_{frame_num:03d}.png"), rgb_img)

            points = pcl_data.getPoints().astype(np.float32)
            if len(points) > 0:
                vertices = np.array([(p[0], p[1], p[2]) for p in points],
                                   dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
                PlyData([PlyElement.describe(vertices, 'vertex')]).write(
                    str(OUTPUT_DIR / f"cloud_{frame_num:03d}.ply"))
                print(f"Frame {frame_num}: {len(points):,} points")
                total_points += len(points)
            else:
                print(f"Frame {frame_num}: no points")

            frame_num += 1

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    pipeline.stop()

    print(f"\nCaptured {frame_num} frames, {total_points:,} total points")
    print(f"Output: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
