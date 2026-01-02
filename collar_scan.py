#!/usr/bin/env python3
"""
OAK-D S3 Pro 3D Scanner for black aluminum collar
DepthAI v3 API with IR dot projector for active stereo

Usage:
  python collar_scan.py              # GUI mode
  python collar_scan.py --headless   # CLI-only mode
  python collar_scan.py --auto 36    # Auto-capture N frames
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

# IR laser intensity for active stereo (0.0 to 1.0)
IR_LASER_INTENSITY = 0.8


def create_pipeline_and_queues():
    """Create pipeline with v3 API - queues from node outputs directly"""
    pipeline = dai.Pipeline()

    # Left mono camera
    cam_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)

    # Right mono camera
    cam_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    # Color camera
    cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

    # Stereo depth - basic config, let it just work
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(False)
    stereo.setSubpixel(True)
    stereo.setRectification(True)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_B)
    stereo.initialConfig.setConfidenceThreshold(200)
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_5x5)

    # Point cloud
    pointcloud = pipeline.create(dai.node.PointCloud)

    # Link cameras to stereo (use 1280x800 - multiple of 16)
    cam_left.requestOutput((1280, 800)).link(stereo.left)
    cam_right.requestOutput((1280, 800)).link(stereo.right)

    # Link depth to point cloud
    stereo.depth.link(pointcloud.inputDepth)

    # Create output queues directly from node outputs (v3 style)
    q_depth = stereo.depth.createOutputQueue()
    q_pcl = pointcloud.outputPointCloud.createOutputQueue()
    q_rgb = cam_rgb.requestOutput((1920, 1080), dai.ImgFrame.Type.BGR888p).createOutputQueue()

    return pipeline, (q_pcl, q_depth, q_rgb)


def enable_ir_projector(pipeline):
    """Enable IR laser dot projector for active stereo"""
    try:
        # v3 API - get device from pipeline's default device
        devices = pipeline.getDevices()
        if devices:
            device = devices[0]
            device.setIrLaserDotProjectorIntensity(IR_LASER_INTENSITY)
            device.setIrFloodLightIntensity(0.0)
            print(f"IR laser dot projector: {IR_LASER_INTENSITY*100:.0f}%")
            return True
    except AttributeError:
        # Try alternative method
        try:
            pipeline.getDefaultDevice().setIrLaserDotProjectorIntensity(IR_LASER_INTENSITY)
            pipeline.getDefaultDevice().setIrFloodLightIntensity(0.0)
            print(f"IR laser dot projector: {IR_LASER_INTENSITY*100:.0f}%")
            return True
        except Exception as e2:
            print(f"IR projector: {e2}")
    except Exception as e:
        print(f"IR projector not available: {e}")
    return False


def capture_frame(queues, frame_num):
    """Capture single frame"""
    q_pcl, q_depth, q_rgb = queues

    time.sleep(0.3)

    depth_frame = q_depth.get()
    pcl_data = q_pcl.get()
    rgb_frame = q_rgb.get()

    # Save depth visualization
    depth_img = depth_frame.getCvFrame()
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_img, alpha=0.03),
        cv2.COLORMAP_JET
    )
    cv2.imwrite(str(OUTPUT_DIR / f"depth_{frame_num:03d}.png"), depth_colormap)

    # Save RGB
    rgb_img = rgb_frame.getCvFrame()
    cv2.imwrite(str(OUTPUT_DIR / f"rgb_{frame_num:03d}.png"), rgb_img)

    valid_pixels = np.sum(depth_img > 0)

    # Save point cloud
    points = pcl_data.getPoints().astype(np.float32)
    if len(points) > 0:
        vertices = np.array(
            [(p[0], p[1], p[2]) for p in points],
            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
        )
        el = PlyElement.describe(vertices, 'vertex')
        PlyData([el]).write(str(OUTPUT_DIR / f"cloud_{frame_num:03d}.ply"))
        print(f"Frame {frame_num}: {len(points):,} points | {valid_pixels:,} valid depth px")
        return len(points)
    else:
        print(f"Frame {frame_num}: NO POINTS ({valid_pixels:,} depth px)")
        return 0


def run_headless(queues, num_frames, delay):
    """Headless CLI mode"""
    print(f"\nCapturing {num_frames} frames")
    print(f"Rotate {360/num_frames:.1f}° between captures\n")

    frame_num = 0
    total_points = 0

    for i in range(num_frames):
        print(f"\n--- Frame {i+1}/{num_frames} ---")
        pts = capture_frame(queues, frame_num)
        total_points += pts
        frame_num += 1

        if i < num_frames - 1:
            print(f"Rotate {360/num_frames:.1f}°, press ENTER...")
            try:
                input()
            except EOFError:
                time.sleep(delay)

    return frame_num, total_points


def run_gui(pipeline, queues):
    """GUI mode with live preview"""
    q_pcl, q_depth, q_rgb = queues
    frame_num = 0
    total_points = 0

    print("\nREADY - SPACE=capture, A=auto-36, Q=quit")

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
            pts = capture_frame(queues, frame_num)
            total_points += pts
            frame_num += 1
        elif key == ord('a'):
            print("\nAuto-capture: 36 frames")
            for i in range(36):
                print(f"--- Frame {i+1}/36 ---")
                pts = capture_frame(queues, frame_num)
                total_points += pts
                frame_num += 1
                if i < 35:
                    print("Rotate 10°, ENTER...")
                    input()
        elif key == ord('q'):
            pipeline.stop()
            break

    cv2.destroyAllWindows()
    return frame_num, total_points


def print_results(frame_num, total_points):
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Frames: {frame_num}")
    print(f"Total points: {total_points:,}")
    if frame_num > 0:
        print(f"Avg points/frame: {total_points//frame_num:,}")
    print(f"\nOutput: {OUTPUT_DIR.absolute()}")


def main():
    parser = argparse.ArgumentParser(description="OAK-D S3 Pro 3D Scanner (v3 API)")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--auto", type=int, metavar="N")
    parser.add_argument("--delay", type=float, default=3.0)
    parser.add_argument("--ir", type=float, default=0.8)
    args = parser.parse_args()

    global IR_LASER_INTENSITY
    IR_LASER_INTENSITY = max(0.0, min(1.0, args.ir))

    headless = args.headless or args.auto is not None
    num_frames = args.auto or 36

    print("=" * 50)
    print("OAK-D S3 PRO SCANNER (DepthAI v3)")
    print("=" * 50)
    print(f"Output: {OUTPUT_DIR.absolute()}")
    print(f"IR Laser: {IR_LASER_INTENSITY*100:.0f}%")

    # Check devices
    devices = dai.Device.getAllAvailableDevices()
    if not devices:
        print("\nERROR: No OAK-D camera detected!")
        sys.exit(1)

    print(f"Found {len(devices)} device(s)")

    pipeline, queues = create_pipeline_and_queues()

    print("Starting pipeline...")
    pipeline.start()

    enable_ir_projector(pipeline)

    print("Warming up...")
    time.sleep(2)

    if headless:
        frame_num, total_points = run_headless(queues, num_frames, args.delay)
        pipeline.stop()
    else:
        frame_num, total_points = run_gui(pipeline, queues)

    print_results(frame_num, total_points)


if __name__ == "__main__":
    main()
