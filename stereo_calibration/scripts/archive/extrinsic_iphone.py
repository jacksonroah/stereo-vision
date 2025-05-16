#!/usr/bin/env python3
import cv2
import numpy as np
import os
import glob
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import time
import pickle
import shutil

def create_directories(test_dir):
    """Create necessary directories and clear previous data."""
    # Clear temp directories
    temp_dir = os.path.join(test_dir, "temp")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    os.makedirs(os.path.join(temp_dir, "left"), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, "right"), exist_ok=True)
    
    # Clear previous results
    debug_dir = os.path.join(test_dir, "results", "debug_images")
    if os.path.exists(debug_dir):
        for file in glob.glob(os.path.join(debug_dir, "*corners*")):
            os.remove(file)
    
    extrinsic_dir = os.path.join(test_dir, "results", "extrinsic_params")
    if os.path.exists(extrinsic_dir):
        for file in glob.glob(os.path.join(extrinsic_dir, "*")):
            if os.path.isfile(file):
                os.remove(file)
    
    # Create directories
    os.makedirs(os.path.join(test_dir, "results", "extrinsic_params"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "results", "debug_images"), exist_ok=True)
    
    return temp_dir

def load_intrinsic_params(test_dir, camera_id):
    """Load intrinsic camera parameters from calibration files."""
    intrinsic_dir = os.path.join(test_dir, "results", "intrinsic_params")
    
    # First try to load from pickle files
    pickle_file = os.path.join(intrinsic_dir, f"{camera_id}_intrinsics.pkl")
    
    if os.path.exists(pickle_file):
        print(f"Loading intrinsic parameters from {pickle_file}")
        with open(pickle_file, 'rb') as f:
            camera_matrix, dist_coeffs = pickle.load(f)
    else:
        # Fall back to text files
        matrix_file = os.path.join(intrinsic_dir, f"{camera_id}_matrix.txt")
        dist_file = os.path.join(intrinsic_dir, f"{camera_id}_distortion.txt")
        
        if not os.path.exists(matrix_file) or not os.path.exists(dist_file):
            print(f"Error: Calibration files for camera {camera_id} not found")
            return None, None
        
        camera_matrix = np.loadtxt(matrix_file)
        dist_coeffs = np.loadtxt(dist_file)
    
    # Quick sanity check
    focal_length_avg = (camera_matrix[0, 0] + camera_matrix[1, 1]) / 2
    if focal_length_avg > 10000 or focal_length_avg < 100:
        print(f"WARNING: Camera {camera_id} has unusual focal length: {focal_length_avg:.1f}")
    
    print(f"Successfully loaded intrinsic calibration for camera {camera_id}")
    return camera_matrix, dist_coeffs

def find_extrinsic_videos(camera_dir):
    """Find all extrinsic calibration videos in a camera directory."""
    # First check for an extrinsic_videos folder
    extrinsic_folder = os.path.join(camera_dir, "extrinsic_videos")
    
    if os.path.exists(extrinsic_folder) and os.path.isdir(extrinsic_folder):
        # Find all video files in this folder
        video_paths = []
        for ext in ['.mp4', '.mov', '.avi', '.MP4', '.MOV', '.AVI']:
            video_paths.extend(glob.glob(os.path.join(extrinsic_folder, f"*{ext}")))
        
        if video_paths:
            video_paths.sort()
            return video_paths
    
    # Fall back to videos in the main directory
    video_paths = []
    for ext in ['.mp4', '.mov', '.avi', '.MP4', '.MOV', '.AVI']:
        video_paths.extend(glob.glob(os.path.join(camera_dir, f"*{ext}")))
    
    # Filter out any videos that might be intrinsic
    video_paths = [v for v in video_paths if "intrinsic" not in os.path.basename(v).lower()]
    
    if video_paths:
        video_paths.sort()
    
    return video_paths

def match_video_pairs(left_videos, right_videos):
    """Match videos from left and right cameras based on filename or position."""
    pairs = []
    
    # If both lists have the same number of videos, match them by position
    if len(left_videos) == len(right_videos):
        for i in range(len(left_videos)):
            pairs.append((left_videos[i], right_videos[i]))
        return pairs
    
    # Otherwise, use positional matching for the minimum number of videos
    min_count = min(len(left_videos), len(right_videos))
    for i in range(min_count):
        pairs.append((left_videos[i], right_videos[i]))
    
    return pairs

def extract_frames(video_path, output_dir, camera_id, video_id, 
                   frame_interval=30, max_frames=3, start_frame=60, quality_check=True):
    """
    Extract frames from a video with quality checking.
    
    Args:
        quality_check: If True, will verify frames have good contrast and sharpness
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []
    
    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"\nProcessing {os.path.basename(video_path)}")
    print(f"Total frames: {total_frames}, FPS: {fps}")
    
    # Skip initial frames
    if start_frame > 0 and start_frame < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Calculate how many frames to skip
    adjusted_interval = max(1, min(frame_interval, (total_frames - start_frame) // max_frames))
    
    extracted_frames = []
    frame_count = 0
    saved_count = 0
    
    pbar = tqdm(total=min(max_frames, (total_frames - start_frame) // adjusted_interval + 1))
    
    while cap.isOpened() and saved_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % adjusted_interval == 0:
            # Quality check - verify the frame has good contrast and isn't blurry
            if quality_check:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Check contrast
                std_dev = np.std(gray)
                if std_dev < 30:  # Low contrast
                    frame_count += 1
                    continue
                
                # Check sharpness using Laplacian
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                if laplacian_var < 100:  # Blurry image
                    frame_count += 1
                    continue
            
            # Save high-quality frame
            frame_path = os.path.join(output_dir, camera_id, f"{camera_id}_static_{video_id}_{saved_count:04d}.png")
            os.makedirs(os.path.dirname(frame_path), exist_ok=True)
            cv2.imwrite(frame_path, frame)
            
            extracted_frames.append(frame_path)
            saved_count += 1
            pbar.update(1)
        
        frame_count += 1
    
    pbar.close()
    cap.release()
    
    print(f"Extracted {saved_count} frames from {camera_id}")
    return extracted_frames

def ensure_corner_ordering(corners, pattern_size):
    """Ensure consistent ordering of checkerboard corners."""
    width, height = pattern_size
    
    # Check first and last corner in first row
    first_corner_idx = 0
    last_corner_idx = width - 1
    
    first_x = corners[first_corner_idx][0][0]
    last_x = corners[last_corner_idx][0][0]
    
    # If corners are in reverse order, flip them
    if first_x > last_x:
        new_corners = np.zeros_like(corners)
        for i in range(height):
            for j in range(width):
                new_corners[i * width + j] = corners[i * width + (width - 1 - j)]
        return new_corners, True
    
    return corners, False

def find_matching_frames(left_frames, right_frames, checkerboard_size, debug_dir, max_rms_error=1.0):
    """
    Find matching frames with good checkerboard detection and low error.
    
    Args:
        max_rms_error: Maximum allowed reprojection error for checkerboard corners
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Find all valid left frames with checkerboard
    valid_left_frames = []
    valid_left_corners = []
    
    print("\nFinding frames with checkerboard in left camera...")
    for frame_path in tqdm(left_frames):
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, 
                                                cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        if not ret:
            # Try with additional filters
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            ret, corners = cv2.findChessboardCorners(blurred, checkerboard_size, 
                                                   cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                   cv2.CALIB_CB_NORMALIZE_IMAGE + 
                                                   cv2.CALIB_CB_FILTER_QUADS)
        
        if ret:
            # Refine corners for better accuracy
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Ensure consistent corner ordering
            corners2, was_flipped = ensure_corner_ordering(corners2, checkerboard_size)
            if was_flipped:
                print(f"Correcting corner order for left frame: {os.path.basename(frame_path)}")
            
            # Calculate corner detection quality
            # Higher stddev = better defined corners with better contrast
            patch_size = 11
            patch_quality = []
            for corner in corners2:
                x, y = corner[0]
                x, y = int(x), int(y)
                # Extract patch around corner
                patch = gray[max(0, y-patch_size//2):min(gray.shape[0], y+patch_size//2+1),
                           max(0, x-patch_size//2):min(gray.shape[1], x+patch_size//2+1)]
                if patch.size > 0:
                    patch_quality.append(np.std(patch))
            
            # Skip if corner patches have low contrast
            if np.mean(patch_quality) < 20:
                continue
            
            valid_left_frames.append(frame_path)
            valid_left_corners.append(corners2)
            
            # Save debug image
            debug_frame = frame.copy()
            cv2.drawChessboardCorners(debug_frame, checkerboard_size, corners2, ret)
            
            frame_parts = os.path.basename(frame_path).split('_')
            video_id = "unknown"
            frame_num = "0000"
            
            if len(frame_parts) >= 4:
                video_id = frame_parts[2]
                frame_num = frame_parts[3].split('.')[0]
            
            debug_path = os.path.join(debug_dir, f"left_corners_{video_id}_{frame_num}.png")
            cv2.imwrite(debug_path, debug_frame)
    
    # Find valid right frames with checkerboard
    valid_right_frames = []
    valid_right_corners = []
    
    print("\nFinding frames with checkerboard in right camera...")
    for frame_path in tqdm(right_frames):
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, 
                                                cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        if not ret:
            # Try with additional filters
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            ret, corners = cv2.findChessboardCorners(blurred, checkerboard_size, 
                                                   cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                   cv2.CALIB_CB_NORMALIZE_IMAGE + 
                                                   cv2.CALIB_CB_FILTER_QUADS)
        
        if ret:
            # Refine corners
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Ensure consistent corner ordering
            corners2, was_flipped = ensure_corner_ordering(corners2, checkerboard_size)
            if was_flipped:
                print(f"Correcting corner order for right frame: {os.path.basename(frame_path)}")
            
            # Calculate corner detection quality
            patch_size = 11
            patch_quality = []
            for corner in corners2:
                x, y = corner[0]
                x, y = int(x), int(y)
                # Extract patch around corner
                patch = gray[max(0, y-patch_size//2):min(gray.shape[0], y+patch_size//2+1),
                           max(0, x-patch_size//2):min(gray.shape[1], x+patch_size//2+1)]
                if patch.size > 0:
                    patch_quality.append(np.std(patch))
            
            # Skip if corner patches have low contrast
            if np.mean(patch_quality) < 20:
                continue
            
            valid_right_frames.append(frame_path)
            valid_right_corners.append(corners2)
            
            # Save debug image
            debug_frame = frame.copy()
            cv2.drawChessboardCorners(debug_frame, checkerboard_size, corners2, ret)
            
            frame_parts = os.path.basename(frame_path).split('_')
            video_id = "unknown"
            frame_num = "0000"
            
            if len(frame_parts) >= 4:
                video_id = frame_parts[2]
                frame_num = frame_parts[3].split('.')[0]
            
            debug_path = os.path.join(debug_dir, f"right_corners_{video_id}_{frame_num}.png")
            cv2.imwrite(debug_path, debug_frame)
    
    # Match frames from both cameras
    matched_left_frames = []
    matched_left_corners = []
    matched_right_frames = []
    matched_right_corners = []
    
    # Extract video_id and frame_number for each frame
    left_frame_info = {}
    for i, frame_path in enumerate(valid_left_frames):
        frame_parts = os.path.basename(frame_path).split('_')
        if len(frame_parts) >= 4:
            video_id = frame_parts[2]
            frame_num = frame_parts[3].split('.')[0]
            key = f"{video_id}_{frame_num}"
            left_frame_info[key] = (i, frame_path, valid_left_corners[i])
    
    right_frame_info = {}
    for i, frame_path in enumerate(valid_right_frames):
        frame_parts = os.path.basename(frame_path).split('_')
        if len(frame_parts) >= 4:
            video_id = frame_parts[2]
            frame_num = frame_parts[3].split('.')[0]
            key = f"{video_id}_{frame_num}"
            right_frame_info[key] = (i, frame_path, valid_right_corners[i])
    
    # Find matching frame pairs
    exact_matches = set(left_frame_info.keys()) & set(right_frame_info.keys())
    for key in exact_matches:
        left_idx, left_frame, left_corner = left_frame_info[key]
        right_idx, right_frame, right_corner = right_frame_info[key]
        
        matched_left_frames.append(left_frame)
        matched_left_corners.append(left_corner)
        matched_right_frames.append(right_frame)
        matched_right_corners.append(right_corner)
    
    print(f"Found {len(matched_left_frames)} matching frame pairs with checkerboard")
    
    return matched_left_frames, matched_left_corners, matched_right_frames, matched_right_corners

def validate_frame_pairs(left_frames, left_corners, right_frames, right_corners,
                        left_matrix, left_dist, right_matrix, right_dist, 
                        checkerboard_size, square_size, max_rms_error=2.0):
    """Filter out frame pairs with high reprojection error."""
    print("\nValidating frame pairs and removing high-error frames...")
    
    if len(left_frames) < 5:
        print("Warning: Too few frame pairs to perform validation.")
        return left_frames, left_corners, right_frames, right_corners
    
    # Prepare object points
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size
    
    # Prepare object points for each frame pair
    objpoints = [objp] * len(left_frames)
    
    # Get image size from first frame
    img1 = cv2.imread(left_frames[0])
    img_shape = img1.shape[:2][::-1]  # (width, height)
    
    # Calculate reprojection error for each frame pair
    frame_errors = []
    
    for i in range(len(left_frames)):
        # Try to calibrate using just this frame pair
        flags = cv2.CALIB_FIX_INTRINSIC
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        
        try:
            ret, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
                [objpoints[i]], [left_corners[i]], [right_corners[i]],
                left_matrix, left_dist, right_matrix, right_dist,
                img_shape, criteria=criteria, flags=flags)
            
            frame_errors.append((ret, i))
        except cv2.error as e:
            print(f"Error calibrating frame pair {i}: {e}")
            frame_errors.append((100.0, i))  # Assign a high error
    
    # Sort frames by error
    frame_errors.sort()
    
    # Select frames with lowest error
    good_indices = [idx for error, idx in frame_errors if error < max_rms_error]
    
    if len(good_indices) < 5:
        print(f"Warning: Found only {len(good_indices)} frames with error < {max_rms_error}.")
        print("Using all frames to ensure we have enough data.")
        return left_frames, left_corners, right_frames, right_corners
    
    # Create new lists with only the good frames
    print(f"Keeping {len(good_indices)} frames with reprojection error < {max_rms_error}")
    
    filtered_left_frames = [left_frames[i] for i in good_indices]
    filtered_left_corners = [left_corners[i] for i in good_indices]
    filtered_right_frames = [right_frames[i] for i in good_indices]
    filtered_right_corners = [right_corners[i] for i in good_indices]
    
    return filtered_left_frames, filtered_left_corners, filtered_right_frames, filtered_right_corners

def calibrate_stereo(left_frames, left_corners, right_frames, right_corners, 
                    left_matrix, left_dist, right_matrix, right_dist,
                    checkerboard_size, square_size, output_dir, debug_dir, actual_distance=None):
    """Perform stereo calibration to find extrinsic parameters."""
    if not left_frames or len(left_frames) < 5:
        print("Error: Need at least 5 valid frame pairs for stereo calibration")
        return None, None, None, None
    
    # Prepare object points
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size
    
    # Prepare object points for each frame pair
    objpoints = [objp] * len(left_frames)
    
    # Get image size from first frame
    img1 = cv2.imread(left_frames[0])
    img_shape = img1.shape[:2][::-1]  # (width, height)
    
    print(f"\nPerforming stereo calibration with {len(objpoints)} frame pairs...")
    
    # Stability check for intrinsic parameters
    l_focal = (left_matrix[0, 0] + left_matrix[1, 1]) / 2
    r_focal = (right_matrix[0, 0] + right_matrix[1, 1]) / 2
    
    focal_ratio = max(l_focal, r_focal) / min(l_focal, r_focal)
    if focal_ratio > 1.5:
        print("\nWARNING: Large difference in camera focal lengths detected!")
        print(f"Left focal length: {l_focal:.1f}")
        print(f"Right focal length: {r_focal:.1f}")
        print(f"Ratio: {focal_ratio:.2f}")
        print("This may cause scale errors in your calibration.")
    
    # Set calibration flags
    flags = cv2.CALIB_FIX_INTRINSIC  # Use fixed intrinsic parameters
    
    # Stereo calibration criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    
    # Perform stereo calibration
    ret, cam1_matrix, dist1, cam2_matrix, dist2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, left_corners, right_corners,
        left_matrix, left_dist, right_matrix, right_dist,
        img_shape, criteria=criteria, flags=flags)
    
    print(f"Stereo calibration complete with RMS error: {ret}")
    
    # Calculate camera separation distance (baseline)
    camera_distance = np.linalg.norm(T)
    print(f"Camera separation distance: {camera_distance:.2f} mm")
    
    # Perform sanity checks on the calibration result
    scale_plausible = True
    
    # Check if actual distance provided
    if actual_distance is not None:
        distance_error = 100 * abs(camera_distance - actual_distance) / actual_distance
        print(f"Actual distance: {actual_distance:.2f} mm")
        print(f"Distance error: {distance_error:.2f}%")
        
        if distance_error > 50:  # More than 50% error
            scale_plausible = False
            print("\nWARNING: Large distance error detected!")
    
    # Convert rotation matrix to Euler angles for readability
    r_vec, _ = cv2.Rodrigues(R)
    euler_angles = r_vec * 180.0 / np.pi
    
    print("\nCamera 2 orientation relative to Camera 1 (Euler angles in degrees):")
    print(f"Rotation around X: {euler_angles[0][0]:.2f}°")
    print(f"Rotation around Y: {euler_angles[1][0]:.2f}°")
    print(f"Rotation around Z: {euler_angles[2][0]:.2f}°")
    
    if not scale_plausible:
        print("\n========== CALIBRATION WARNING ==========")
        print("The scale of your calibration appears to be incorrect!")
        print("Possible causes:")
        print("1. Incorrect intrinsic calibration")
        print("2. Wrong checkerboard square size")
        print("3. Insufficient or bad quality calibration frames")
        print("\nRecommendations:")
        print("- Verify your checkerboard measurements")
        print("- Try different checkerboard positions")
        print("- Make sure lighting is good and checkerboard is flat")
        print("=========================================")
    
    # Save results
    np.savetxt(os.path.join(output_dir, 'stereo_rotation_matrix.txt'), R)
    np.savetxt(os.path.join(output_dir, 'stereo_translation_vector.txt'), T)
    np.savetxt(os.path.join(output_dir, 'essential_matrix.txt'), E)
    np.savetxt(os.path.join(output_dir, 'fundamental_matrix.txt'), F)
    
    # Save as pickle for better preservation
    with open(os.path.join(output_dir, 'extrinsic_params.pkl'), 'wb') as f:
        pickle.dump({
            'R': R,
            'T': T,
            'E': E,
            'F': F,
            'rms_error': ret,
            'camera_distance': camera_distance
        }, f)
    
    # Save human-readable results
    with open(os.path.join(output_dir, 'stereo_calibration_info.txt'), 'w') as f:
        f.write(f"Stereo Calibration\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"RMS Reprojection Error: {ret}\n")
        f.write(f"Number of frame pairs used: {len(objpoints)}\n\n")
        
        f.write("Rotation Matrix (Camera 2 relative to Camera 1):\n")
        for row in R:
            f.write(f"{row[0]:.8f} {row[1]:.8f} {row[2]:.8f}\n")
        
        f.write("\nTranslation Vector (mm):\n")
        f.write(f"{T[0][0]:.8f} {T[1][0]:.8f} {T[2][0]:.8f}\n")
        
        f.write(f"\nCamera Separation Distance: {camera_distance:.2f} mm\n")
        
        if actual_distance is not None:
            f.write(f"Actual camera distance: {actual_distance:.2f} mm\n")
            f.write(f"Distance error: {distance_error:.2f}%\n")
        
        if not scale_plausible:
            f.write("\nWARNING: Calibration scale appears incorrect!\n")
        
        f.write("\nEuler Angles (degrees):\n")
        f.write(f"Rotation around X: {euler_angles[0][0]:.2f}°\n")
        f.write(f"Rotation around Y: {euler_angles[1][0]:.2f}°\n")
        f.write(f"Rotation around Z: {euler_angles[2][0]:.2f}°\n")
        
        # List which video IDs were used
        f.write("\nVideos used in calibration:\n")
        video_ids = set()
        for frame in left_frames:
            parts = os.path.basename(frame).split('_')
            if len(parts) >= 3:
                video_ids.add(parts[2])
        
        for video_id in sorted(video_ids):
            f.write(f"- extrinsic video {video_id}\n")
    
    # Visualize camera positions
    visualize_camera_positions(R, T, output_dir, scale_plausible)
    
    return R, T, E, F

def visualize_camera_positions(R, T, output_dir, scale_warning=False):
    """Visualize the relative positions of two cameras in 3D space."""
    try:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Camera 1 is at the origin
        cam1_pos = np.array([0, 0, 0])
        
        # Camera 2 position is determined by the translation vector
        cam2_pos = T.flatten()
        
        # Draw camera positions
        ax.scatter([cam1_pos[0]], [cam1_pos[1]], [cam1_pos[2]], c='r', marker='o', s=100, label='Left Camera')
        ax.scatter([cam2_pos[0]], [cam2_pos[1]], [cam2_pos[2]], c='b', marker='o', s=100, label='Right Camera')
        
        # Draw line connecting cameras
        ax.plot([cam1_pos[0], cam2_pos[0]], 
                [cam1_pos[1], cam2_pos[1]], 
                [cam1_pos[2], cam2_pos[2]], 'k-')
        
        # Draw camera orientations
        # For Camera 1, the Z axis points along positive Z
        arrow_length = min(100, np.linalg.norm(T) / 5)  # Scale arrows to be visible
        
        ax.quiver(cam1_pos[0], cam1_pos[1], cam1_pos[2], 
                  0, 0, arrow_length, color='r', arrow_length_ratio=0.1)
        
        # For Camera 2, apply rotation to the Z axis
        cam2_direction = R @ np.array([0, 0, arrow_length])
        ax.quiver(cam2_pos[0], cam2_pos[1], cam2_pos[2], 
                  cam2_direction[0], cam2_direction[1], cam2_direction[2], 
                  color='b', arrow_length_ratio=0.1)
        
        # Set axis labels
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        
        title = 'Relative Camera Positions'
        if scale_warning:
            title += ' (WARNING: Scale may be incorrect)'
        ax.set_title(title)
        
        # Try to make axis scales equal
        # Calculate a reasonable range based on camera distance
        max_range = max([
            np.abs(T[0][0]), np.abs(T[1][0]), np.abs(T[2][0])
        ]) * 1.5
        
        # If max_range is very large, use a more reasonable value
        if max_range > 5000:
            ax.text(0, 0, 0, "WARNING: Scale appears incorrect!", 
                   color='red', fontsize=12, fontweight='bold')
        
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range/2, max_range)
        
        ax.legend()
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, 'camera_positions.png'))
        print(f"Camera position visualization saved to {os.path.join(output_dir, 'camera_positions.png')}")
        
        plt.close()
    except Exception as e:
        print(f"Error visualizing camera positions: {e}")

def main():
    """Main function to run extrinsic calibration."""
    parser = argparse.ArgumentParser(description='High-Quality Stereo Camera Extrinsic Calibration')
    parser.add_argument('--test_dir', required=True, 
                      help='Test directory name (e.g., test_001)')
    parser.add_argument('--base_dir', default='.', 
                      help='Base directory for the project (default: current directory)')
    parser.add_argument('--checkerboard_size', default='9,7', 
                      help='Size of checkerboard as width,height inner corners (default: 9,7)')
    parser.add_argument('--square_size', type=float, default=25.0,
                      help='Size of checkerboard square in mm (default: 25.0)')
    parser.add_argument('--frame_interval', type=int, default=30,
                      help='Extract every Nth frame from video (default: 30)')
    parser.add_argument('--max_frames', type=int, default=3,
                      help='Maximum number of frames to extract per video (default: 3)')
    parser.add_argument('--start_frame', type=int, default=60,
                      help='Skip this many frames at the start (default: 60)')
    parser.add_argument('--actual_distance', type=float, default=None,
                      help='Actual distance between cameras in mm (if known)')
    parser.add_argument('--no_validation', action='store_true',
                      help='Skip frame pair validation step')
    parser.add_argument('--max_rms_error', type=float, default=2.0,
                      help='Maximum allowed RMS error for valid frame pairs (default: 2.0)')
    
    args = parser.parse_args()
    
    # Parse checkerboard size
    cb_width, cb_height = map(int, args.checkerboard_size.split(','))
    checkerboard_size = (cb_width, cb_height)
    
    # Set up paths
    base_dir = args.base_dir
    test_path = os.path.join(base_dir, "data", args.test_dir)
    
    # Create needed directories with consistent naming
    temp_dir = create_directories(test_path)
    results_dir = os.path.join(test_path, "results")
    extrinsic_dir = os.path.join(results_dir, "extrinsic_params")
    debug_dir = os.path.join(results_dir, "debug_images")
    
    # Load intrinsic parameters
    left_matrix, left_dist = load_intrinsic_params(test_path, "left")
    right_matrix, right_dist = load_intrinsic_params(test_path, "right")
    
    if left_matrix is None or right_matrix is None:
        print("Error: Could not load intrinsic calibration parameters")
        return
    
    # Define camera directories
    left_camera_dir = os.path.join(test_path, "left_camera")
    right_camera_dir = os.path.join(test_path, "right_camera")
    
    # Find all extrinsic calibration videos
    left_videos = find_extrinsic_videos(left_camera_dir)
    right_videos = find_extrinsic_videos(right_camera_dir)
    
    if not left_videos or not right_videos:
        print("Error: Could not find extrinsic calibration videos")
        print("Place videos in each camera directory")
        return
    
    # Match videos between cameras
    video_pairs = match_video_pairs(left_videos, right_videos)
    
    if not video_pairs:
        print("Error: Could not match any video pairs between cameras")
        return
    
    print(f"Found {len(video_pairs)} matching video pairs for extrinsic calibration")
    
    # Process each video pair to extract frames
    all_left_frames = []
    all_right_frames = []
    
    for i, (left_video, right_video) in enumerate(video_pairs):
        video_id = f"{i+1:03d}"
        
        print(f"\nProcessing video pair {i+1}/{len(video_pairs)}")
        print(f"Left video: {os.path.basename(left_video)}")
        print(f"Right video: {os.path.basename(right_video)}")
        
        # Extract frames from videos
        left_frames = extract_frames(
            left_video, temp_dir, "left", video_id, 
            args.frame_interval, args.max_frames, args.start_frame, quality_check=True
        )
        
        right_frames = extract_frames(
            right_video, temp_dir, "right", video_id, 
            args.frame_interval, args.max_frames, args.start_frame, quality_check=True
        )
        
        if not left_frames or not right_frames:
            print(f"Warning: Could not extract quality frames from video pair {i+1}")
            continue
        
        all_left_frames.extend(left_frames)
        all_right_frames.extend(right_frames)
    
    if not all_left_frames or not all_right_frames:
        print("Error: No frames extracted from any video pair")
        return
    
    # Find matching frames with checkerboard
    matched_left_frames, matched_left_corners, matched_right_frames, matched_right_corners = find_matching_frames(
        all_left_frames, all_right_frames, checkerboard_size, debug_dir
    )
    
    if not matched_left_frames:
        print("Error: No matching frame pairs found")
        return
    
    # Validate and filter frame pairs if requested
    if not args.no_validation:
        matched_left_frames, matched_left_corners, matched_right_frames, matched_right_corners = validate_frame_pairs(
            matched_left_frames, matched_left_corners, matched_right_frames, matched_right_corners,
            left_matrix, left_dist, right_matrix, right_dist,
            checkerboard_size, args.square_size, args.max_rms_error
        )
    
    # Perform stereo calibration
    R, T, E, F = calibrate_stereo(
        matched_left_frames, matched_left_corners, 
        matched_right_frames, matched_right_corners,
        left_matrix, left_dist, right_matrix, right_dist,
        checkerboard_size, args.square_size, 
        extrinsic_dir, debug_dir, args.actual_distance
    )
    
    if R is not None:
        print("\nExtrinsic calibration complete!")
        print(f"Results saved to {extrinsic_dir}")
        print("\nNext steps:")
        print("1. Check camera_positions.png to verify the calibration looks reasonable")
        print("2. Run the validator to test distance measurements")
        print(f"   python scripts/validate.py --test_dir {args.test_dir} --base_dir {args.base_dir}")

if __name__ == "__main__":
    main()