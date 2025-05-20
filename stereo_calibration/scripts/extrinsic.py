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
from checkerboard_distance import validate_distance_measurement, load_validation_videos

def create_directories(test_dir):
    """Create necessary directories for output with consistent naming."""
    # Create temp directories - both variants for compatibility
    os.makedirs(os.path.join(test_dir, "temp", "left"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "temp", "right"), exist_ok=True)
    
    # Create results directories
    os.makedirs(os.path.join(test_dir, "results", "extrinsic_params"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "results", "debug_images"), exist_ok=True)
    
    return os.path.join(test_dir, "temp")

def load_intrinsic_params(test_dir, camera_id):
    """
    Load intrinsic camera parameters from calibration files.
    
    Args:
        test_dir (str): Test directory path
        camera_id (str): Camera identifier (left or right)
        
    Returns:
        tuple: (camera_matrix, distortion_coefficients)
    """
    intrinsic_dir = os.path.join(test_dir, "results", "intrinsic_params")
    matrix_file = os.path.join(intrinsic_dir, f"{camera_id}_matrix.txt")
    dist_file = os.path.join(intrinsic_dir, f"{camera_id}_distortion.txt")
    
    if not os.path.exists(matrix_file) or not os.path.exists(dist_file):
        print(f"Error: Calibration files for camera {camera_id} not found at:")
        print(f"  - {matrix_file}")
        print(f"  - {dist_file}")
        return None, None
    
    camera_matrix = np.loadtxt(matrix_file)
    dist_coeffs = np.loadtxt(dist_file)
    
    print(f"Successfully loaded intrinsic calibration for camera {camera_id}")
    return camera_matrix, dist_coeffs

def cleanup_temp_files(temp_dir, keep_temp=False):
    """
    Clean up temporary files after successful calibration.
    
    Args:
        temp_dir (str): Path to the temporary directory
        keep_temp (bool): Whether to keep temporary files
    """
    if keep_temp:
        print(f"Keeping temporary files in {temp_dir} as requested")
        return
        
    print(f"Cleaning up temporary files in {temp_dir}")
    try:
        # Remove the temporary frame directories
        for subdir in ['left', 'right']:
            dir_path = os.path.join(temp_dir, subdir)
            if os.path.exists(dir_path):
                for file in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        
        print("Temporary files cleaned up successfully")
    except Exception as e:
        print(f"Warning: Error while cleaning up temporary files: {e}")

def find_extrinsic_videos(camera_dir):
    """
    Find all extrinsic calibration videos in a camera directory.
    More flexible naming support.
    """
    video_paths = []
    
    # First check for an extrinsic_videos subfolder
    extrinsic_folder = os.path.join(camera_dir, "extrinsic_videos")
    
    if os.path.exists(extrinsic_folder) and os.path.isdir(extrinsic_folder):
        # Find all video files in this folder
        for ext in ['.mp4', '.mov', '.avi', '.MP4', '.MOV']:
            video_paths.extend(glob.glob(os.path.join(extrinsic_folder, f"*{ext}")))
        
        if video_paths:
            video_paths.sort()  # Sort by filename to ensure consistent ordering
            return video_paths
    
    # Fall back to the traditional method - check for various naming patterns
    patterns = [
        "extrinsic_video_*.*",  # Traditional naming
        "extrinsic*.*",         # Any name starting with extrinsic
        "x*.*"         # Any name containing extrinsic
    ]
    
    for pattern in patterns:
        matches = glob.glob(os.path.join(camera_dir, pattern))
        if matches:
            for match in matches:
                if match.lower().endswith(('.mp4', '.mov', '.avi')):
                    video_paths.append(match)
    
    return sorted(list(set(video_paths)))  # Return unique, sorted paths

def match_video_pairs(left_videos, right_videos):
    """
    Match videos from left and right cameras.
    
    Args:
        left_videos (list): List of paths to left camera videos
        right_videos (list): List of paths to right camera videos
        
    Returns:
        list: List of tuples (left_video, right_video)
    """
    pairs = []
    
    # Extract video IDs from filenames
    # Extract video IDs from filenames
    def get_video_id(path):
        filename = os.path.basename(path)
        
        # Try to match extrinsic_video_001.mp4 format
        parts = filename.split('_')
        if len(parts) >= 3 and parts[0] == "extrinsic" and parts[1] == "video":
            return parts[2].split('.')[0]  # extrinsic_video_001.mp4 -> 001
        
        # Try to match extrinsic1.mp4 format
        if filename.startswith("x"):
            # Extract numeric part from the name
            import re
            match = re.search(r'x(\d+)', filename)
            if match:
                return match.group(1)  # extrinsic1.mp4 -> 1
        
        return "default"  # For single video without number
    
    # Group videos by ID
    left_dict = {}
    for video in left_videos:
        video_id = get_video_id(video)
        left_dict[video_id] = video
    
    # Match with right videos
    for right_video in right_videos:
        right_id = get_video_id(right_video)
        if right_id in left_dict:
            pairs.append((left_dict[right_id], right_video))
    
    # If no matches found but we have videos, try a simple match
    if not pairs and left_videos and right_videos:
        # Sort both lists to ensure we match corresponding videos
        left_videos.sort()
        right_videos.sort()
        
        # Match videos by position in sorted list
        for i in range(min(len(left_videos), len(right_videos))):
            pairs.append((left_videos[i], right_videos[i]))
    
    return pairs

def extract_frames(video_path, output_dir, camera_id, video_id, frame_interval=15, max_frames=20):
    """
    Extract frames from a video at regular intervals.
    
    Args:
        video_path (str): Path to the video file
        output_dir (str): Directory to save extracted frames
        camera_id (str): Camera identifier (left or right)
        video_id (str): Identifier for the video (e.g., "001")
        frame_interval (int): Extract every nth frame
        max_frames (int): Maximum number of frames to extract
        
    Returns:
        list: Paths to extracted frames
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
    
    extracted_frames = []
    frame_count = 0
    saved_count = 0
    
    # Setup progress bar
    pbar = tqdm(total=min(max_frames, total_frames // frame_interval + 1))
    
    while cap.isOpened() and saved_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            # Save frame in both directories for compatibility
            frame_path1 = os.path.join(output_dir, camera_id, f"{camera_id}_static_{video_id}_{saved_count:04d}.png")
            
            # Ensure directories exist
            os.makedirs(os.path.dirname(frame_path1), exist_ok=True)
            
            cv2.imwrite(frame_path1, frame)
            
            extracted_frames.append(frame_path1)
            saved_count += 1
            pbar.update(1)
        
        frame_count += 1
    
    pbar.close()
    cap.release()
    
    print(f"Extracted {saved_count} frames from {camera_id}")
    return extracted_frames

def find_matching_frames(left_frames, right_frames, checkerboard_size, debug_dir):
    """
    Find matching frames from both cameras that contain the checkerboard.
    Includes corner order correction to ensure consistent orientation.
    
    Args:
        left_frames (list): Paths to frames from left camera
        right_frames (list): Paths to frames from right camera
        checkerboard_size (tuple): Size of the checkerboard (internal corners)
        debug_dir (str): Directory to save debug images
        
    Returns:
        tuple: (matched_left_frames, matched_left_corners, matched_right_frames, matched_right_corners)
    """
    # Group frames by video_id
    def get_video_id_from_path(path):
        filename = os.path.basename(path)
        parts = filename.split('_')
        if len(parts) >= 4:
            return parts[2]  # left_static_001_0000.png -> 001
        return "default"
    
    # Find all valid left frames with checkerboard
    valid_left_frames = []
    valid_left_corners = []
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Get checkerboard dimensions
    numx, numy = checkerboard_size
    
    print("\nFinding frames with checkerboard in left camera...")
    for frame_path in tqdm(left_frames):
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Could not read frame {frame_path}")
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, 
                                                cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        if ret:
            # CORNER ORDER CORRECTION: Check if corners are detected in reverse order
            if numx > 1:  # Only check if we have multiple columns
                diff = corners[0][0][0] - corners[numx-1][0][0]
                if diff > 0:
                    print(f"Correcting corner order for left frame: {os.path.basename(frame_path)}")
                    # Make a copy to maintain the numpy array structure
                    corners_reversed = np.copy(corners[::-1])
                    corners = corners_reversed
            
            # Refine corners
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            valid_left_frames.append(frame_path)
            valid_left_corners.append(corners)
            
            # Optional: Save debug image
            debug_frame = frame.copy()
            cv2.drawChessboardCorners(debug_frame, checkerboard_size, corners, ret)
            
            # Extract meaningful parts from filename
            frame_parts = os.path.basename(frame_path).split('_')
            if len(frame_parts) >= 4:
                video_id = frame_parts[2]
                frame_num = frame_parts[3].split('.')[0]
                debug_path = os.path.join(debug_dir, f"left_corners_{video_id}_{frame_num}.png")
                cv2.imwrite(debug_path, debug_frame)
    
    # Find all valid right frames with checkerboard
    valid_right_frames = []
    valid_right_corners = []
    
    print("\nFinding frames with checkerboard in right camera...")
    for frame_path in tqdm(right_frames):
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Could not read frame {frame_path}")
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, 
                                                cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        if ret:
            # CORNER ORDER CORRECTION: Check if corners are detected in reverse order
            if numx > 1:  # Only check if we have multiple columns
                diff = corners[0][0][0] - corners[numx-1][0][0]
                if diff > 0:
                    print(f"Correcting corner order for right frame: {os.path.basename(frame_path)}")
                    # Make a copy to maintain the numpy array structure
                    corners_reversed = np.copy(corners[::-1])
                    corners = corners_reversed
            
            # Refine corner positions
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            valid_right_frames.append(frame_path)
            valid_right_corners.append(corners)
            
            # Optional: Save debug image
            debug_frame = frame.copy()
            cv2.drawChessboardCorners(debug_frame, checkerboard_size, corners, ret)
            
            # Extract meaningful parts from filename
            frame_parts = os.path.basename(frame_path).split('_')
            if len(frame_parts) >= 4:
                video_id = frame_parts[2]
                frame_num = frame_parts[3].split('.')[0]
                debug_path = os.path.join(debug_dir, f"right_corners_{video_id}_{frame_num}.png")
                cv2.imwrite(debug_path, debug_frame)
    
    # Match frames from the same video with the same frame number
    matched_left_frames = []
    matched_left_corners = []
    matched_right_frames = []
    matched_right_corners = []
    
    # Create a dictionary for faster matching
    right_frames_dict = {}
    for i, frame_path in enumerate(valid_right_frames):
        # Extract video_id and frame_number
        frame_parts = os.path.basename(frame_path).split('_')
        if len(frame_parts) >= 4:
            video_id = frame_parts[2]
            frame_num = frame_parts[3].split('.')[0]
            key = f"{video_id}_{frame_num}"
            right_frames_dict[key] = (frame_path, valid_right_corners[i])
    
    # Match with left frames
    for i, frame_path in enumerate(valid_left_frames):
        frame_parts = os.path.basename(frame_path).split('_')
        if len(frame_parts) >= 4:
            video_id = frame_parts[2]
            frame_num = frame_parts[3].split('.')[0]
            key = f"{video_id}_{frame_num}"
            
            if key in right_frames_dict:
                right_frame, right_corner = right_frames_dict[key]
                matched_left_frames.append(frame_path)
                matched_left_corners.append(valid_left_corners[i])
                matched_right_frames.append(right_frame)
                matched_right_corners.append(right_corner)
    
    print(f"Found {len(matched_left_frames)} matching frame pairs with checkerboard")
    return matched_left_frames, matched_left_corners, matched_right_frames, matched_right_corners

def calibrate_stereo(left_frames, left_corners, right_frames, right_corners, 
                    left_matrix, left_dist, right_matrix, right_dist,
                    checkerboard_size, square_size, output_dir, debug_dir, actual_distance=None):
    """
    Perform stereo calibration to find extrinsic parameters.
    
    Args:
        left_frames, left_corners: Left camera frames and corners
        right_frames, right_corners: Right camera frames and corners
        left_matrix, left_dist: Left camera intrinsic parameters
        right_matrix, right_dist: Right camera intrinsic parameters
        checkerboard_size (tuple): Size of the checkerboard (internal corners)
        square_size (float): Size of each checkerboard square in mm
        output_dir (str): Directory to save results
        debug_dir (str): Directory to save debug information
        actual_distance (float, optional): Actual distance between cameras in mm
    """
    if not left_frames or not right_frames:
        print("Error: No matching frames provided for stereo calibration")
        return None, None, None, None
    
    # Prepare object points
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size  # Convert to actual size
    
    # Prepare object points for each frame pair
    objpoints = [objp] * len(left_frames)
    
    # Check if we have any valid frames
    if len(objpoints) < 5:
        print(f"Error: Not enough valid frame pairs ({len(objpoints)}) for stereo calibration")
        return None, None, None, None
    
    # Get image size from first frame
    img1 = cv2.imread(left_frames[0])
    if img1 is None:
        print(f"Error: Could not read frame {left_frames[0]}")
        return None, None, None, None
        
    img_shape = img1.shape[:2][::-1]  # (width, height)
    
    print(f"\nPerforming stereo calibration with {len(objpoints)} frame pairs...")
    
    # Set calibration flags - use fixed intrinsic parameters
    flags = cv2.CALIB_FIX_INTRINSIC
    
    # Stereo calibration criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    
    # Perform stereo calibration
    ret, cam1_matrix, dist1, cam2_matrix, dist2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, left_corners, right_corners,
        left_matrix, left_dist, right_matrix, right_dist,
        img_shape, criteria=criteria, flags=flags)
    
    print(f"Stereo calibration complete with RMS error: {ret}")
    
    # Calculate camera separation distance (baseline)
    camera_distance = np.linalg.norm(T)
    print(f"Camera separation distance: {camera_distance:.2f} mm")
    
    # Calculate error percentage if actual distance is provided
    if actual_distance is not None:
        distance_error = 100 * abs(camera_distance - actual_distance) / actual_distance
        print(f"Actual distance: {actual_distance:.2f} mm")
        print(f"Distance error: {distance_error:.2f}%")
    
    # Convert rotation matrix to Euler angles
    r_vec, _ = cv2.Rodrigues(R)
    euler_angles = r_vec * 180.0 / np.pi
    
    print("\nCamera 2 orientation relative to Camera 1 (Euler angles in degrees):")
    print(f"Rotation around X: {euler_angles[0][0]:.2f}°")
    print(f"Rotation around Y: {euler_angles[1][0]:.2f}°")
    print(f"Rotation around Z: {euler_angles[2][0]:.2f}°")
    
    # Save results
    np.savetxt(os.path.join(output_dir, 'stereo_rotation_matrix.txt'), R)
    np.savetxt(os.path.join(output_dir, 'stereo_translation_vector.txt'), T)
    np.savetxt(os.path.join(output_dir, 'essential_matrix.txt'), E)
    np.savetxt(os.path.join(output_dir, 'fundamental_matrix.txt'), F)
    
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
        
        f.write("\nEuler Angles (degrees):\n")
        f.write(f"Rotation around X: {euler_angles[0][0]:.2f}°\n")
        f.write(f"Rotation around Y: {euler_angles[1][0]:.2f}°\n")
        f.write(f"Rotation around Z: {euler_angles[2][0]:.2f}°\n")
        
        # List which video IDs were used
        f.write("\nVideos used in calibration:\n")
        video_ids = set()
        for frame in left_frames:
            parts = os.path.basename(frame).split('_')
            if len(parts) >= 4:
                video_ids.add(parts[2])
        
        for video_id in sorted(video_ids):
            f.write(f"- extrinsic_video_{video_id}\n")
    
    # Visualize camera positions
    visualize_camera_positions(R, T, output_dir)
    
    return R, T, E, F

def visualize_camera_positions(R, T, output_dir):
    """
    Visualize the relative positions of two cameras in 3D space.
    
    Args:
        R (ndarray): Rotation matrix (3x3)
        T (ndarray): Translation vector (3x1)
        output_dir (str): Directory to save the visualization
    """
    try:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Camera 1 is at the origin
        cam1_pos = np.array([0, 0, 0])
        
        # Camera 2 position is determined by the translation vector in camera 1's coordinate system
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
        ax.quiver(cam1_pos[0], cam1_pos[1], cam1_pos[2], 
                  0, 0, 100, color='r', arrow_length_ratio=0.1)
        
        # For Camera 2, apply rotation to the Z axis
        cam2_direction = R @ np.array([0, 0, 100])
        ax.quiver(cam2_pos[0], cam2_pos[1], cam2_pos[2], 
                  cam2_direction[0], cam2_direction[1], cam2_direction[2], 
                  color='b', arrow_length_ratio=0.1)
        
        # Set axis labels
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title('Relative Camera Positions')
        
        # Try to make axis scales equal
        max_range = max([
            np.abs(T[0][0]), np.abs(T[1][0]), np.abs(T[2][0])
        ]) * 1.5
        
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
    parser = argparse.ArgumentParser(description='Stereo Camera Extrinsic Calibration')
    parser.add_argument('--test_dir', required=True, 
                      help='Test directory name (e.g., test_001)')
    parser.add_argument('--base_dir', default='.', 
                      help='Base directory for the project (default: current directory)')
    parser.add_argument('--checkerboard_size', default='7,4', 
                      help='Size of checkerboard as width,height inner corners (default: 7,4)')
    parser.add_argument('--square_size', type=float, default=100.0,
                      help='Size of checkerboard square in mm (default: 100.0)')
    parser.add_argument('--frame_interval', type=int, default=15,
                      help='Extract every Nth frame from video (default: 15)')
    parser.add_argument('--max_frames', type=int, default=20,
                      help='Maximum number of frames to extract per video (default: 20)')
    parser.add_argument('--actual_distance', type=float, default=None,
                      help='Actual distance between cameras in mm (if known)')
    parser.add_argument('--keep_temp', action='store_true',
                  help='Keep temporary files after calibration (default: False)')
    parser.add_argument('--measure', type=float, default=None)
    parser.add_argument('--validation_pattern', default='validation')

    
    
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
    left_camera_dir = os.path.join(test_path, "left")
    right_camera_dir = os.path.join(test_path, "right")
    
    # Find all extrinsic calibration videos
    left_videos = find_extrinsic_videos(left_camera_dir)
    right_videos = find_extrinsic_videos(right_camera_dir)
    
    if not left_videos or not right_videos:
        print("Error: Could not find extrinsic calibration videos")
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
        # Generate a video ID from the filename or use index
        video_id = f"{i+1:03d}"
        
        # Try to extract ID from filename
        filename = os.path.basename(left_video)
        parts = filename.split('_')
        if len(parts) >= 3 and parts[0] == "extrinsic" and parts[1] == "video":
            video_id = parts[2].split('.')[0]
        
        print(f"\nProcessing video pair {i+1}/{len(video_pairs)}")
        print(f"Left video: {os.path.basename(left_video)}")
        print(f"Right video: {os.path.basename(right_video)}")
        
        # Extract frames from videos
        left_frames = extract_frames(
            left_video, temp_dir, "left", video_id, 
            args.frame_interval, args.max_frames
        )
        
        right_frames = extract_frames(
            right_video, temp_dir, "right", video_id, 
            args.frame_interval, args.max_frames
        )
        
        if not left_frames or not right_frames:
            print(f"Warning: Could not extract frames from video pair {i+1}")
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

        cleanup_temp_files(temp_dir, args.keep_temp)

        # Run distance validation if requested
    if args.measure is not None:
        print("\n" + "="*70)
        print("DISTANCE VALIDATION")
        print("="*70)
        
        # Load calibration data
        if R is None:
            print("Error: Extrinsic calibration must be performed first")
            return
            
        # Find validation videos
        left_video, right_video = load_validation_videos(
            test_path, args.validation_pattern)
        
        if not left_video or not right_video:
            print("Error: Could not find validation videos")
            return
            
        print("\nExtracting frames from validation videos...")
        
        # Extract frames from validation videos
        video_id = "validation"
        left_frames = extract_frames(
            left_video, temp_dir, "left", video_id, 
            args.frame_interval, args.max_frames
        )
        
        right_frames = extract_frames(
            right_video, temp_dir, "right", video_id, 
            args.frame_interval, args.max_frames
        )
        
        # Find matching frames with checkerboard
        matched_left_frames, matched_left_corners, matched_right_frames, matched_right_corners = find_matching_frames(
            left_frames, right_frames, checkerboard_size, debug_dir
        )
        
        if not matched_left_frames:
            print("Error: No matching frame pairs found in validation videos")
            return
            
        # Create validation directory
        validation_dir = os.path.join(results_dir, "validation_results")
        os.makedirs(validation_dir, exist_ok=True)
        
        # Validate distance measurement
        measured_distance = validate_distance_measurement(
            matched_left_frames, matched_right_frames, 
            matched_left_corners, matched_right_corners,
            left_matrix, left_dist, right_matrix, right_dist,
            R, T, checkerboard_size, args.square_size, 
            validation_dir, args.measure
        )
        
        print("\nDistance validation complete!")
        if measured_distance is not None:
            if args.measure is not None:
                error_percent = 100 * abs(measured_distance - args.measure) / args.measure
                print(f"Measured distance: {measured_distance:.2f} mm (actual: {args.measure:.2f} mm)")
                print(f"Error: {error_percent:.2f}%")
            else:
                print(f"Measured distance: {measured_distance:.2f} mm")

    

if __name__ == "__main__":
    main()