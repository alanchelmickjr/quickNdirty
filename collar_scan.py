#!/usr/bin/env python3
"""
OAK-D S3 Pro Scanner - with point cloud capture
"""

import cv2
import depthai as dai
import numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement

OUTPUT_DIR = Path("collar_scans")
OUTPUT_DIR.mkdir(exist_ok=True)

pipeline = dai.Pipeline()
monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
stereo = pipeline.create(dai.node.StereoDepth)

# Linking cameras to stereo
monoLeftOut = monoLeft.requestFullResolutionOutput()
monoRightOut = monoRight.requestFullResolutionOutput()
monoLeftOut.link(stereo.left)
monoRightOut.link(stereo.right)

stereo.setRectification(True)
stereo.setExtendedDisparity(True)
stereo.setLeftRightCheck(True)

# Point cloud node
pointcloud = pipeline.create(dai.node.PointCloud)
stereo.depth.link(pointcloud.inputDepth)

# Output queues
disparityQueue = stereo.disparity.createOutputQueue()
depthQueue = stereo.depth.createOutputQueue()
pclQueue = pointcloud.outputPointCloud.createOutputQueue()

colorMap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
colorMap[0] = [0, 0, 0]

with pipeline:
    pipeline.start()

    # Enable IR laser
    try:
        device = pipeline.getDefaultDevice()
        device.setIrLaserDotProjectorIntensity(0.8)
        print("IR laser enabled at 80%")
    except Exception as e:
        print(f"IR: {e}")

    print("SPACE=capture, Q=quit")
    frame_num = 0
    maxDisparity = 1

    while pipeline.isRunning():
        disparity = disparityQueue.get()
        npDisparity = disparity.getFrame()
        maxDisparity = max(maxDisparity, np.max(npDisparity))
        colorizedDisparity = cv2.applyColorMap(((npDisparity / maxDisparity) * 255).astype(np.uint8), colorMap)

        cv2.putText(colorizedDisparity, f"Frame: {frame_num}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imshow("disparity", colorizedDisparity)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            pipeline.stop()
            break
        elif key == ord(' '):
            # Capture
            depth = depthQueue.get()
            pcl = pclQueue.get()

            # Save depth image
            cv2.imwrite(str(OUTPUT_DIR / f"depth_{frame_num:03d}.png"), colorizedDisparity)

            # Save point cloud
            points = pcl.getPoints()
            if len(points) > 0:
                vertices = np.array(
                    [(p[0], p[1], p[2]) for p in points],
                    dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
                )
                PlyData([PlyElement.describe(vertices, 'vertex')]).write(
                    str(OUTPUT_DIR / f"cloud_{frame_num:03d}.ply"))
                print(f"Frame {frame_num}: {len(points):,} points saved")
            else:
                print(f"Frame {frame_num}: no points")

            frame_num += 1

print(f"Saved to {OUTPUT_DIR.absolute()}")
