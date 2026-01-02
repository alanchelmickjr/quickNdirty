#!/usr/bin/env python3
"""
OAK-D S3 Pro 3D Scanner for black aluminum collar
With IR dot projector for active stereo on low-texture surfaces

Usage:
  python collar_scan.py              # GUI mode (requires display)
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
# Higher = better for black/textureless surfaces
IR_LASER_INTENSITY = 0.8
IR_FLOOD_INTENSITY = 0.0  # flood not needed for scanning


def create_pipeline():
    pipeline = dai.Pipeline()

    # Left mono camera
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    mono_left.setCamera("left")
    mono_left.setFps(30)

    # Right mono camera
    mono_right = pipeline.create(dai.node.MonoCamera)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    mono_right.setCamera("right")
    mono_right.setFps(30)

    # Color camera for texture reference
    color = pipeline.create(dai.node.ColorCamera)
    color.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    color.setIspScale(1, 2)
    color.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    color.setFps(30)

    # Stereo depth - HIGH_DENSITY for better coverage on detailed objects
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(True)  # closer objects
    stereo.setSubpixel(True)  # sub-pixel accuracy
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setRectifyEdgeFillColor(0)

    # Lower confidence threshold for black surfaces (default ~230)
    stereo.initialConfig.setConfidenceThreshold(150)

    # Point cloud generation
    pointcloud = pipeline.create(dai.node.PointCloud)

    # Sync node for aligned outputs
    sync = pipeline.create(dai.node.Sync)

    # Link cameras to stereo
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    # Link depth to point cloud
    stereo.depth.link(pointcloud.inputDepth)

    # Outputs
    xout_pcl = pipeline.create(dai.node.XLinkOut)
    xout_pcl.setStreamName("pcl")
    pointcloud.outputPointCloud.link(xout_pcl.input)

    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    color.isp.link(xout_rgb.input)

    return pipeline


def enable_ir_projector(device):
    """Enable IR laser dot projector for active stereo (OAK-D Pro/S3 Pro)"""
    try:
        device.setIrLaserDotProjectorIntensity(IR_LASER_INTENSITY)
        device.setIrFloodLightIntensity(IR_FLOOD_INTENSITY)
        print(f"IR laser dot projector: {IR_LASER_INTENSITY*100:.0f}%")
        return True
    except Exception as e:
        print(f"IR projector not available: {e}")
        print("(This device may not have active stereo hardware)")
        return False


def capture_frame(queues, frame_num):
    """Capture single frame: depth + point cloud + rgb"""
    q_pcl, q_depth, q_rgb = queues

    # Let new frames arrive
    time.sleep(0.3)

    pcl_data = q_pcl.get()
    depth_frame = q_depth.get()
    rgb_frame = q_rgb.get()

    # Save depth visualization
    depth_img = depth_frame.getFrame()
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_img, alpha=0.03),
        cv2.COLORMAP_JET
    )
    cv2.imwrite(str(OUTPUT_DIR / f"depth_{frame_num:03d}.png"), depth_colormap)

    # Save RGB
    rgb_img = rgb_frame.getCvFrame()
    cv2.imwrite(str(OUTPUT_DIR / f"rgb_{frame_num:03d}.png"), rgb_img)

    # Get valid depth pixel count
    valid_pixels = np.sum(depth_img > 0)

    # Save point cloud as PLY
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
        print(f"Frame {frame_num}: NO POINTS - black hole detected ({valid_pixels:,} depth px)")
        return 0


def run_headless(queues, num_frames, delay):
    """Headless CLI mode"""
    print(f"\nHeadless mode: capturing {num_frames} frames")
    print(f"Rotate turntable {360/num_frames:.1f}° between each capture\n")

    frame_num = 0
    total_points = 0

    for i in range(num_frames):
        print(f"\n--- Frame {i+1}/{num_frames} ---")
        pts = capture_frame(queues, frame_num)
        total_points += pts
        frame_num += 1

        if i < num_frames - 1:
            print(f"Rotate turntable {360/num_frames:.1f}°, then press ENTER...")
            try:
                input()
            except EOFError:
                print(f"  (waiting {delay}s...)")
                time.sleep(delay)

    return frame_num, total_points


def run_gui(device, queues):
    """GUI mode with live preview"""
    q_pcl, q_depth, q_rgb = queues
    frame_num = 0
    total_points = 0

    print("\nREADY - position collar, press SPACE to capture")

    while True:
        depth_frame = q_depth.tryGet()
        if depth_frame is not None:
            depth_img = depth_frame.getFrame()
            depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_img, alpha=0.03),
                cv2.COLORMAP_JET
            )
            valid_pixels = np.sum(depth_img > 0)
            cv2.putText(depth_vis, f"Valid: {valid_pixels:,}px",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(depth_vis, f"Frame: {frame_num} | Total pts: {total_points:,}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imshow("Depth (SPACE=capture, Q=quit, A=auto-36)", depth_vis)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            pts = capture_frame(queues, frame_num)
            total_points += pts
            frame_num += 1

        elif key == ord('a'):
            print("\nAuto-capture: 36 frames at 10° intervals")
            for i in range(36):
                print(f"\n--- Frame {i+1}/36 ---")
                pts = capture_frame(queues, frame_num)
                total_points += pts
                frame_num += 1
                if i < 35:
                    print("Rotate 10°, press ENTER...")
                    input()
            print("\nAuto-capture complete!")

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    return frame_num, total_points


def print_results(frame_num, total_points):
    """Print scan results"""
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Frames: {frame_num}")
    print(f"Total points: {total_points:,}")
    print(f"Avg points/frame: {total_points//max(frame_num,1):,}")
    print(f"\nOutput: {OUTPUT_DIR.absolute()}")

    if frame_num > 0 and total_points < 10000 * frame_num:
        print("\nLOW POINT COUNT - suggestions:")
        print("  1. Increase IR laser intensity (edit IR_LASER_INTENSITY)")
        print("  2. Apply scanning spray/powder to surface")
        print("  3. Improve lighting (reduce ambient IR)")
        print("  4. Move object closer to camera")


def main():
    parser = argparse.ArgumentParser(description="OAK-D S3 Pro 3D Scanner")
    parser.add_argument("--headless", action="store_true", help="CLI mode (no GUI)")
    parser.add_argument("--auto", type=int, metavar="N", help="Auto-capture N frames")
    parser.add_argument("--delay", type=float, default=3.0, help="Delay between captures")
    parser.add_argument("--ir", type=float, default=0.8, help="IR laser intensity 0.0-1.0")
    args = parser.parse_args()

    global IR_LASER_INTENSITY
    IR_LASER_INTENSITY = max(0.0, min(1.0, args.ir))

    headless = args.headless or args.auto is not None
    num_frames = args.auto or 36

    print("=" * 50)
    print("OAK-D S3 PRO 3D SCANNER")
    print("=" * 50)
    print(f"Output: {OUTPUT_DIR.absolute()}")
    print(f"Mode: {'Headless' if headless else 'GUI'}")
    print(f"IR Laser: {IR_LASER_INTENSITY*100:.0f}%")

    if not headless:
        print("\nControls:")
        print("  SPACE - capture frame")
        print("  A     - auto-capture 36 frames")
        print("  Q     - quit")
    print()

    # Check for devices
    devices = dai.Device.getAllAvailableDevices()
    if not devices:
        print("\nERROR: No OAK-D camera detected!")
        print("\nTroubleshooting:")
        print("  1. Check USB connection (USB3 preferred)")
        print("  2. Try different USB port")
        print("  3. macOS: check System Preferences > Security")
        print("  4. Linux: check udev rules")
        sys.exit(1)

    print(f"Found {len(devices)} device(s)")

    pipeline = create_pipeline()

    with dai.Device(pipeline) as device:
        print(f"Connected: {device.getDeviceName()}")

        # Enable IR projector for active stereo
        enable_ir_projector(device)

        print("Warming up...")
        time.sleep(2)

        # Create output queues
        q_pcl = device.getOutputQueue("pcl", maxSize=4, blocking=False)
        q_depth = device.getOutputQueue("depth", maxSize=4, blocking=False)
        q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
        queues = (q_pcl, q_depth, q_rgb)

        if headless:
            frame_num, total_points = run_headless(queues, num_frames, args.delay)
        else:
            frame_num, total_points = run_gui(device, queues)

        print_results(frame_num, total_points)


if __name__ == "__main__":
    main()
