#!/usr/bin/env python3
import cv2
import numpy as np
import os
import glob
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import subprocess
import platform

def load_calibration_params(base_dir):
    """Load all calibration parameters (intrinsic and extrinsic)."""
    print("Loading calibration parameters...")
    
    # Load intrinsic parameters
    cam1_matrix = np.loadtxt(os.path.join(base_dir, "calibration_results", "cam1_matrix.txt"))
    dist1 = np.loadtxt(os.path.join(base_dir, "calibration_results", "cam1_distortion.txt"))
    cam2_matrix = np.loadtxt(os.path.join(base_dir, "calibration_results", "cam2_matrix.txt"))
    dist2 = np.loadtxt(os.path.join(base_dir, "calibration_results", "cam2_distortion.txt"))
    
    # Load extrinsic parameters
    R = np.loadtxt(os.path.join(base_dir, "stereo_calibration_results", "stereo_rotation_matrix.txt"))
    T = np.loadtxt(os.path.join(base_dir, "stereo_calibration_results", "stereo_translation_vector.txt")).reshape(3, 1)
    
    print("Successfully loaded calibration parameters")
    return cam1_matrix, dist1, cam2_matrix, dist2, R, T

def extract_frames(video_path, output_dir, camera_id, ruler_id, frame_interval=15, max_frames=5):
    """Extract frames from a validation video at regular intervals."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []
    
    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"\nProcessing {os.path.basename(video_path)}")
    print(f"Total frames: {total_frames}, FPS: {fps}")
    
    # Create output directory
    camera_dir = os.path.join(output_dir, camera_id)
    os.makedirs(camera_dir, exist_ok=True)
    
    extracted_frames = []
    frame_count = 0
    saved_count = 0
    
    while cap.isOpened() and saved_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            # Save frame
            frame_path = os.path.join(camera_dir, f"{camera_id}_{ruler_id}_frame_{saved_count:04d}.png")
            cv2.imwrite(frame_path, frame)
            extracted_frames.append(frame_path)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {saved_count} frames from {camera_id}_{ruler_id}")
    return extracted_frames

def find_matching_frames(cam1_frames, cam2_frames):
    """Find matching frames from both cameras."""
    # Ensure we're not mixing different rulers
    def get_ruler_id(filename):
        parts = os.path.basename(filename).split('_')
        if len(parts) >= 2:
            return parts[1]
        return "ruler1"
    
    def get_frame_num(filename):
        parts = os.path.basename(filename).split('_')
        if len(parts) >= 4:
            return int(parts[-1].split('.')[0])
        return -1
    
    # Group frames by ruler
    cam1_by_ruler = {}
    for frame in cam1_frames:
        ruler_id = get_ruler_id(frame)
        if ruler_id not in cam1_by_ruler:
            cam1_by_ruler[ruler_id] = []
        cam1_by_ruler[ruler_id].append(frame)
    
    cam2_by_ruler = {}
    for frame in cam2_frames:
        ruler_id = get_ruler_id(frame)
        if ruler_id not in cam2_by_ruler:
            cam2_by_ruler[ruler_id] = []
        cam2_by_ruler[ruler_id].append(frame)
    
    # Match frames by ruler_id and frame number
    pairs = []
    
    for ruler_id in cam1_by_ruler:
        if ruler_id not in cam2_by_ruler:
            continue
        
        cam1_ruler_frames = sorted(cam1_by_ruler[ruler_id])
        cam2_ruler_frames = sorted(cam2_by_ruler[ruler_id])
        
        # Create a dictionary of cam2 frames by frame number
        cam2_by_frame_num = {get_frame_num(frame): frame for frame in cam2_ruler_frames}
        
        # Match frames with the same frame number
        for cam1_frame in cam1_ruler_frames:
            frame_num = get_frame_num(cam1_frame)
            if frame_num in cam2_by_frame_num:
                pairs.append((cam1_frame, cam2_by_frame_num[frame_num]))
    
    print(f"Found {len(pairs)} matching frame pairs")
    return pairs

def manual_enter_ruler_points():
    """Have the user manually enter coordinates for ruler endpoints."""
    print("\nEnter the pixel coordinates for the ruler endpoints.")
    print("Format: x,y (e.g., 100,200)")
    
    points1 = []
    points2 = []
    
    for i in range(2):
        valid_input = False
        while not valid_input:
            try:
                cam1_point = input(f"Camera 1 point {i+1} (x,y): ")
                x1, y1 = map(int, cam1_point.strip().split(','))
                points1.append((x1, y1))
                valid_input = True
            except ValueError:
                print("Invalid format. Please use x,y (e.g., 100,200)")
    
    for i in range(2):
        valid_input = False
        while not valid_input:
            try:
                cam2_point = input(f"Camera 2 point {i+1} (x,y): ")
                x2, y2 = map(int, cam2_point.strip().split(','))
                points2.append((x2, y2))
                valid_input = True
            except ValueError:
                print("Invalid format. Please use x,y (e.g., 100,200)")
    
    return np.array(points1, dtype=np.float32), np.array(points2, dtype=np.float32)

def measure_object_distance(cam1_img_path, cam2_img_path, cam1_matrix, dist1, cam2_matrix, dist2, R, T, debug_dir, known_length=None, actual_distance=None):
    """Measure the 3D position and distance of an object visible in both cameras."""
    # Read images
    img1 = cv2.imread(cam1_img_path)
    img2 = cv2.imread(cam2_img_path)
    
    if img1 is None or img2 is None:
        print(f"Error reading images: {cam1_img_path} or {cam2_img_path}")
        return None, None, None
    
    # Undistort images for display
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Calculate optimal new camera matrix
    newcam1, roi1 = cv2.getOptimalNewCameraMatrix(cam1_matrix, dist1, (w1, h1), 1, (w1, h1))
    newcam2, roi2 = cv2.getOptimalNewCameraMatrix(cam2_matrix, dist2, (w2, h2), 1, (w2, h2))
    
    # Undistort
    undistorted1 = cv2.undistort(img1, cam1_matrix, dist1, None, newcam1)
    undistorted2 = cv2.undistort(img2, cam2_matrix, dist2, None, newcam2)
    
    # Save the images directly for reference
    base_name = os.path.basename(cam1_img_path).split('_')[1]
    frame_num = os.path.basename(cam1_img_path).split('_')[-1].split('.')[0]
    
    ref_path1 = os.path.join(debug_dir, f"reference_cam1_{base_name}_{frame_num}.png")
    ref_path2 = os.path.join(debug_dir, f"reference_cam2_{base_name}_{frame_num}.png")
    
    cv2.imwrite(ref_path1, undistorted1)
    cv2.imwrite(ref_path2, undistorted2)
    
    # Try to open images with system's default viewer
    try:
        if platform.system() == 'Darwin':  # macOS
            subprocess.run(['open', ref_path1])
            subprocess.run(['open', ref_path2])
        elif platform.system() == 'Windows':
            os.startfile(ref_path1)
            os.startfile(ref_path2)
        else:  # Linux
            subprocess.run(['xdg-open', ref_path1])
            subprocess.run(['xdg-open', ref_path2])
    except Exception as e:
        print(f"Could not automatically open images: {e}")
    
    print(f"\nReference images saved to:")
    print(f"  - {ref_path1}")
    print(f"  - {ref_path2}")
    print("Please look at these images to locate the ruler endpoints.")
    
    # Create projection matrices
    P1 = newcam1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = newcam2 @ np.hstack((R, T))
    
    # Have the user manually enter coordinates
    points1, points2 = manual_enter_ruler_points()
    
    # Undistort points
    points1_t = points1.reshape(-1, 1, 2)
    points2_t = points2.reshape(-1, 1, 2)
    
    points1_undist = cv2.undistortPoints(points1_t, cam1_matrix, dist1, None, newcam1)
    points2_undist = cv2.undistortPoints(points2_t, cam2_matrix, dist2, None, newcam2)
    
    # Reshape for triangulation
    points1_undist = points1_undist.reshape(-1, 2).T
    points2_undist = points2_undist.reshape(-1, 2).T
    
    # Triangulate
    points4D = cv2.triangulatePoints(P1, P2, points1_undist, points2_undist)
    
    # Convert to 3D points
    points3D = (points4D[:3] / points4D[3]).T
    
    # Calculate distance between endpoints
    measured_length = np.linalg.norm(points3D[1] - points3D[0])
    
    print(f"\nMeasured length: {measured_length:.2f} mm")
    
    if known_length is not None:
        error_percentage = 100 * abs(measured_length - known_length) / known_length
        print(f"Known length: {known_length:.2f} mm")
        print(f"Error: {error_percentage:.2f}%")
    else:
        error_percentage = None
        
    # Calculate distance to object (midpoint between endpoints)
    midpoint = (points3D[0] + points3D[1]) / 2
    distance_to_object = np.linalg.norm(midpoint)
    print(f"Distance to object (from Camera 1): {distance_to_object:.2f} mm")
    
    # Calculate error in distance if actual distance is provided
    if actual_distance is not None:
        distance_error = 100 * abs(distance_to_object - actual_distance) / actual_distance
        print(f"Actual distance to object: {actual_distance:.2f} mm")
        print(f"Distance error: {distance_error:.2f}%")
    
    # Save debug images with drawn lines
    display_img1 = undistorted1.copy()
    display_img2 = undistorted2.copy()
    
    # Draw lines on the images
    cv2.line(display_img1, tuple(map(int, points1[0])), tuple(map(int, points1[1])), (0, 0, 255), 2)
    cv2.line(display_img2, tuple(map(int, points2[0])), tuple(map(int, points2[1])), (0, 0, 255), 2)
    
    debug_path1 = os.path.join(debug_dir, f"cam1_{base_name}_points_{frame_num}.png")
    debug_path2 = os.path.join(debug_dir, f"cam2_{base_name}_points_{frame_num}.png")
    
    cv2.imwrite(debug_path1, display_img1)
    cv2.imwrite(debug_path2, display_img2)
    
    # Create and save a 3D visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot camera positions
    ax.scatter([0], [0], [0], c='r', marker='o', s=100, label='Camera 1')
    ax.scatter([T[0][0]], [T[1][0]], [T[2][0]], c='b', marker='o', s=100, label='Camera 2')
    
    # Plot measured points
    ax.scatter(points3D[:, 0], points3D[:, 1], points3D[:, 2], c='g', marker='o', s=50)
    
    # Draw line between endpoints
    ax.plot([points3D[0, 0], points3D[1, 0]], 
            [points3D[0, 1], points3D[1, 1]], 
            [points3D[0, 2], points3D[1, 2]], 'g-', linewidth=2)
    
    # Label the plot
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'3D Measurement - Length: {measured_length:.2f} mm')
    
    if known_length is not None:
        ax.text2D(0.05, 0.95, f"Error: {error_percentage:.2f}%", transform=ax.transAxes)
    
    ax.legend()
    
    # Save the plot
    plot_path = os.path.join(debug_dir, f"3d_{base_name}_measurement_{frame_num}.png")
    plt.savefig(plot_path)
    plt.close()
    
    return points3D, measured_length, error_percentage