#!/usr/bin/env python3
"""
Step 1: Camera with autofocus
"""

import cv2
import depthai as dai

with dai.Pipeline() as pipeline:
    cam = pipeline.create(dai.node.Camera).build()

    # Set autofocus
    cam.initialControl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)

    videoQueue = cam.requestOutput((640, 400)).createOutputQueue()

    pipeline.start()
    print("Camera running - press Q to quit, F to refocus")

    while pipeline.isRunning():
        videoIn = videoQueue.get()
        frame = videoIn.getCvFrame()
        cv2.imshow("video", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
