#!/usr/bin/env python3
"""
Step 2: Stereo depth - stock example from depthai-core/examples/python/StereoDepth/stereo.py
"""

import cv2
import depthai as dai
import numpy as np

pipeline = dai.Pipeline()
monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
stereo = pipeline.create(dai.node.StereoDepth)

# Linking
monoLeftOut = monoLeft.requestFullResolutionOutput()
monoRightOut = monoRight.requestFullResolutionOutput()
monoLeftOut.link(stereo.left)
monoRightOut.link(stereo.right)

stereo.setRectification(True)
stereo.setExtendedDisparity(True)
stereo.setLeftRightCheck(True)

disparityQueue = stereo.disparity.createOutputQueue()

colorMap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
colorMap[0] = [0, 0, 0]  # to make zero-disparity pixels black

with pipeline:
    pipeline.start()

    # Enable IR laser for better stereo matching
    try:
        device = pipeline.getDefaultDevice()
        device.setIrLaserDotProjectorIntensity(0.8)
        print("IR laser enabled at 80%")
    except Exception as e:
        print(f"IR: {e}")

    print("Stereo depth running - press Q to quit")
    maxDisparity = 1
    while pipeline.isRunning():
        disparity = disparityQueue.get()
        assert isinstance(disparity, dai.ImgFrame)
        npDisparity = disparity.getFrame()
        maxDisparity = max(maxDisparity, np.max(npDisparity))
        colorizedDisparity = cv2.applyColorMap(((npDisparity / maxDisparity) * 255).astype(np.uint8), colorMap)
        cv2.imshow("disparity", colorizedDisparity)
        key = cv2.waitKey(1)
        if key == ord('q'):
            pipeline.stop()
            break
