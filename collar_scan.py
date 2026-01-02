#!/usr/bin/env python3
"""
OAK-D S3 Pro 3D Scanner
Based on luxonis depthai-core v3 RGBD example pattern
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir", type=float, default=0.8)
    args = parser.parse_args()

    global IR_LASER_INTENSITY
    IR_LASER_INTENSITY = args.ir

    print("=" * 50)
    print("OAK-D S3 PRO SCANNER")
    print("=" * 50)

    with dai.Pipeline() as p:
        # Build cameras
        left = p.create(dai.node.Camera)
        right = p.create(dai.node.Camera)
        color = p.create(dai.node.Camera)

        left.build(dai.CameraBoardSocket.CAM_B)
        right.build(dai.CameraBoardSocket.CAM_C)
        color.build(dai.CameraBoardSocket.CAM_A)

        # Stereo depth
        stereo = p.create(dai.node.StereoDepth)
        stereo.setRectifyEdgeFillColor(0)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(True)
        stereo.initialConfig.setConfidenceThreshold(200)

        # RGBD for point cloud
        rgbd = p.create(dai.node.RGBD).build()
        rgbd.setDepthUnits(dai.StereoDepthConfig.AlgorithmControl.DepthUnit.METER)

        # Link cameras to stereo
        left.requestOutput((640, 400)).link(stereo.left)
        right.requestOutput((640, 400)).link(stereo.right)

        # Link stereo to RGBD
        colorOut = color.requestOutput((640, 400), dai.ImgFrame.Type.BGR888i)
        stereo.depth.link(rgbd.inDepth)
        colorOut.link(rgbd.inColor)
        colorOut.link(stereo.inputAlignTo)

        # Output queues
        q_depth = stereo.depth.createOutputQueue()
        q_pcl = rgbd.pcl.createOutputQueue()
        q_rgb = colorOut.createOutputQueue()

        print("Starting pipeline...")
        p.start()

        # Enable IR projector
        try:
            device = p.getDefaultDevice()
            device.setIrLaserDotProjectorIntensity(IR_LASER_INTENSITY)
            print(f"IR laser: {IR_LASER_INTENSITY*100:.0f}%")
        except Exception as e:
            print(f"IR: {e}")

        print("Warming up...")
        time.sleep(2)

        frame_num = 0
        total_points = 0

        print("\nREADY - SPACE=capture, Q=quit")

        while p.isRunning():
            depth_frame = q_depth.tryGet()
            if depth_frame is not None:
                depth_img = depth_frame.getCvFrame()
                depth_vis = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_img, alpha=0.03),
                    cv2.COLORMAP_JET
                )
                valid_pixels = np.sum(depth_img > 0)
                cv2.putText(depth_vis, f"Valid: {valid_pixels:,}px",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.imshow("Depth", depth_vis)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                time.sleep(0.2)
                depth_frame = q_depth.get()
                pcl_data = q_pcl.get()
                rgb_frame = q_rgb.get()

                # Save depth
                depth_img = depth_frame.getCvFrame()
                cv2.imwrite(str(OUTPUT_DIR / f"depth_{frame_num:03d}.png"),
                    cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET))

                # Save RGB
                rgb_img = rgb_frame.getCvFrame()
                cv2.imwrite(str(OUTPUT_DIR / f"rgb_{frame_num:03d}.png"), rgb_img)

                # Save point cloud
                points, colors = pcl_data.getPointsRGB()
                if len(points) > 0:
                    vertices = np.zeros(len(points), dtype=[
                        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
                    ])
                    vertices['x'] = points[:, 0]
                    vertices['y'] = points[:, 1]
                    vertices['z'] = points[:, 2]
                    vertices['red'] = colors[:, 0]
                    vertices['green'] = colors[:, 1]
                    vertices['blue'] = colors[:, 2]
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

    print(f"\nCaptured {frame_num} frames, {total_points:,} points")
    print(f"Output: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
