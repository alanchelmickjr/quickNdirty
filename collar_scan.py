#!/usr/bin/env python3
"""
Quick black aluminum collar scan attempt
OAK-D + turntable + bright lights + prayer

Usage:
  python collar_scan.py              # GUI mode (requires display)
  python collar_scan.py --headless   # CLI-only mode (no display needed)
  python collar_scan.py --auto 36    # Auto-capture N frames
"""

import depthai as dai
import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
import time
import argparse
import sys

OUTPUT_DIR = Path("collar_scans")
OUTPUT_DIR.mkdir(exist_ok=True)

def create_pipeline():
    pipeline = dai.Pipeline()

    # Mono cameras - max res for stereo
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    mono_left.setCamera("left")
    mono_right.setCamera("right")

    # Color for texture reference
    color = pipeline.create(dai.node.ColorCamera)
    color.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    color.setIspScale(1, 2)
    color.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

    # Stereo depth - aggressive settings for black surfaces
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DETAIL)
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(True)  # closer min distance
    stereo.setSubpixel(True)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

    # Lower confidence threshold - accept more questionable depth
    stereo.initialConfig.setConfidenceThreshold(100)  # default 230, lower = more permissive

    # Post-processing
    config = stereo.initialConfig.get()
    config.postProcessing.speckleFilter.enable = True
    config.postProcessing.speckleFilter.speckleRange = 60
    config.postProcessing.temporalFilter.enable = True
    config.postProcessing.spatialFilter.enable = True
    config.postProcessing.spatialFilter.holeFillingRadius = 2
    config.postProcessing.decimationFilter.decimationFactor = 1  # no decimation
    stereo.initialConfig.set(config)

    # Point cloud node
    pointcloud = pipeline.create(dai.node.PointCloud)

    # Links
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)
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

def capture_frame(device, frame_num):
    """Capture single frame: depth image + point cloud + rgb"""
    q_pcl = device.getOutputQueue("pcl", maxSize=4, blocking=False)
    q_depth = device.getOutputQueue("depth", maxSize=4, blocking=False)
    q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)

    # Flush old frames
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

    # Save point cloud
    points = pcl_data.getPoints().astype(np.float64)
    if len(points) > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(str(OUTPUT_DIR / f"cloud_{frame_num:03d}.ply"), pcd)
        print(f"Frame {frame_num}: {len(points):,} points | {valid_pixels:,} valid depth px")
        return len(points)
    else:
        print(f"Frame {frame_num}: NO POINTS - black hole detected ({valid_pixels:,} depth px)")
        return 0

def run_headless(device, num_frames, delay):
    """Headless CLI mode - auto-capture with manual turntable rotation"""
    print(f"\nHeadless mode: capturing {num_frames} frames")
    print(f"Rotate turntable {360/num_frames:.1f}° between each capture\n")

    frame_num = 0
    total_points = 0

    for i in range(num_frames):
        print(f"\n--- Frame {i+1}/{num_frames} ---")
        pts = capture_frame(device, frame_num)
        total_points += pts
        frame_num += 1

        if i < num_frames - 1:
            print(f"Rotate turntable {360/num_frames:.1f}°, then press ENTER...")
            try:
                input()
            except EOFError:
                # If running non-interactively, use delay
                print(f"  (waiting {delay}s for rotation...)")
                time.sleep(delay)

    return frame_num, total_points

def run_gui(device):
    """GUI mode with live preview"""
    frame_num = 0
    total_points = 0

    q_depth = device.getOutputQueue("depth", maxSize=4, blocking=False)

    print("\nREADY - position collar, press SPACE to capture")

    while True:
        # Show live depth preview
        depth_frame = q_depth.tryGet()
        if depth_frame is not None:
            depth_img = depth_frame.getFrame()
            depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_img, alpha=0.03),
                cv2.COLORMAP_JET
            )
            valid_pixels = np.sum(depth_img > 0)
            cv2.putText(depth_vis, f"Valid depth: {valid_pixels:,}px",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imshow("Depth Preview (SPACE=capture, Q=quit, A=auto-36)", depth_vis)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            pts = capture_frame(device, frame_num)
            total_points += pts
            frame_num += 1

        elif key == ord('a'):
            print("\nAuto-capture mode - rotate turntable 10° between captures")
            for i in range(36):
                print(f"\n--- Frame {i+1}/36 ---")
                pts = capture_frame(device, frame_num)
                total_points += pts
                frame_num += 1
                print("Rotate turntable 10°, press ENTER...")
                input()
            print("\nAuto-capture complete!")

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    return frame_num, total_points

def print_results(frame_num, total_points):
    """Print scan results summary"""
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Frames captured: {frame_num}")
    print(f"Total points: {total_points:,}")
    print(f"Avg points/frame: {total_points//max(frame_num,1):,}")
    print(f"\nFiles saved to: {OUTPUT_DIR.absolute()}")

    if frame_num > 0 and total_points < 10000 * frame_num:
        print("\n⚠️  LOW POINT COUNT - black surface eating your photons")
        print("   Options:")
        print("   1. AESUB spray (best)")
        print("   2. Baby powder (ghetto but works)")
        print("   3. Chalk spray")
        print("   4. Adjust lighting angles")

def main():
    parser = argparse.ArgumentParser(description="OAK-D 3D Scanner for black aluminum collar")
    parser.add_argument("--headless", action="store_true", help="Run without GUI (CLI only)")
    parser.add_argument("--auto", type=int, metavar="N", help="Auto-capture N frames (implies --headless)")
    parser.add_argument("--delay", type=float, default=3.0, help="Delay between auto captures (default: 3s)")
    args = parser.parse_args()

    headless = args.headless or args.auto is not None
    num_frames = args.auto or 36

    print("=" * 50)
    print("BLACK ALUMINUM COLLAR SCAN")
    print("=" * 50)
    print(f"Output: {OUTPUT_DIR.absolute()}")
    print(f"Mode: {'Headless CLI' if headless else 'GUI with preview'}")

    if not headless:
        print("\nControls:")
        print("  SPACE - capture frame")
        print("  Q     - quit and show results")
        print("  A     - auto-capture 36 frames (10° intervals)")
    print()

    # Check for available devices first
    devices = dai.Device.getAllAvailableDevices()
    if not devices:
        print("\n❌ ERROR: No OAK-D camera detected!")
        print("\nTroubleshooting:")
        print("  1. Check USB connection")
        print("  2. Try different USB port (USB3 preferred)")
        print("  3. Check udev rules: https://docs.luxonis.com/en/latest/pages/troubleshooting/")
        print("  4. If in Docker, add: -v /dev/bus/usb:/dev/bus/usb --device-cgroup-rule='c 189:* rmw'")
        sys.exit(1)

    pipeline = create_pipeline()

    try:
        with dai.Device(pipeline) as device:
            print(f"Device: {device.getDeviceName()}")
            print("Warming up...")
            time.sleep(2)

            if headless:
                frame_num, total_points = run_headless(device, num_frames, args.delay)
            else:
                frame_num, total_points = run_gui(device)

            print_results(frame_num, total_points)

    except RuntimeError as e:
        if "No DepthAI device found" in str(e) or "No available devices" in str(e):
            print("\n❌ ERROR: No OAK-D camera detected!")
            print("\nTroubleshooting:")
            print("  1. Check USB connection")
            print("  2. Try different USB port (USB3 preferred)")
            print("  3. Check udev rules: https://docs.luxonis.com/en/latest/pages/troubleshooting/")
            print("  4. If in Docker, add: -v /dev/bus/usb:/dev/bus/usb --device-cgroup-rule='c 189:* rmw'")
            sys.exit(1)
        raise

if __name__ == "__main__":
    main()
