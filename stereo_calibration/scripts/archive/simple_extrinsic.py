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
    
    # Create results directories
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
    
    print(f"Successfully loaded intrinsic calibration for camera {camera_id}")
    return camera_matrix, dist_coeffs

def find_matching_videos(left_camera_dir, right_camera_dir):
    """Find matching video files in left and right camera directories."""
    # Simple approach: Match by filename only, ignoring extensions
    left_videos = {}
    right_videos = {}
    
    # Get all video files
    video_extensions = ['.mp4', '.mov', '.MOV', '.MP4', '.avi', '.AVI']
    
    # Find all left videos (excluding intrinsic)
    for ext in video_extensions:
        for video_path in glob.glob(os.path.join(left_camera_dir, f"*{ext}")):
            if "intrinsic" not in os.path.basename(video_path).lower():
                basename = os.path.splitext(os.path.basename(video_path))[0]
                left_videos[basename] = video_path
    
    # Find all right videos (excluding intrinsic)
    for ext in video_extensions:
        for video_path in glob.glob(os.path.join(right_camera_dir, f"*{ext}")):
            if "intrinsic" not in os.path.basename(video_path).lower():
                basename = os.path.splitext(os.path.basename(video_path))[0]
                right_videos[basename] = video_path
    
    # Find common video names
    common_names = set(left_videos.keys()) & set(right_videos.keys())
    
    if not common_names:
        print("No matching video filenames found. Trying to match by number...")
        
        # Try matching by number in filename
        left_by_num = {}
        right_by_num = {}
        
        for name, path in left_videos.items():
            # Extract numbers from filename
            num = ''.join(filter(str.isdigit, name))
            if num:
                left_by_num[num] = path
        
        for name, path in right_videos.items():
            # Extract numbers from filename
            num = ''.join(filter(str.isdigit, name))
            if num:
                right_by_num[num] = path
        
        common_nums = set(left_by_num.keys()) & set(right_by_num.keys())
        
        if common_nums:
            print(f"Found {len(common_nums)} video pairs matching by number.")
            return [(left_by_num[num], right_by_num[num]) for num in sorted(common_nums)]
    
    # If we found matching names
    if common_names:
        print(f"Found {len(common_names)} matching video pairs.")
        return [(left_videos[name], right_videos[name]) for name in sorted(common_names)]
    
    # Fallback: Just pair videos by position in sorted lists
    print("No matching filenames found. Pairing videos by order...")
    left_paths = sorted(left_videos.values())
    right_paths = sorted(right_videos.values())
    
    if not left_paths or not right_paths:
        return []
    
    # Take the minimum length
    count = min(len(left_paths), len(right_paths))
    return [(left_paths[i], right_paths[i]) for i in range(count)]

def extract_frames(video_path, output_dir, camera_id, video_id, 
                  frame_interval=15, max_frames=5, start_frame=30):
    """Extract frames from video at regular intervals."""
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
    
    # Adjust frame interval if needed
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
            # Save frame
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

def find_checkerboard_corners(frame_path, checkerboard_size, debug_dir=None, camera_id=None):
    """Find checkerboard corners in a single frame."""
    frame = cv2.imread(frame_path)
    if frame is None:
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Find checkerboard
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, 
                                           cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                           cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    if not ret:
        # Try with more aggressive preprocessing
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray_blur)
        
        ret, corners = cv2.findChessboardCorners(gray_eq, checkerboard_size, 
                                               cv2.CALIB_CB_ADAPTIVE_THRESH +
                                               cv2.CALIB_CB_NORMALIZE_IMAGE + 
                                               cv2.CALIB_CB_FILTER_QUADS)
    
    if ret:
        # Refine corners
        refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # Ensure consistent corner ordering
        width, height = checkerboard_size
        first_x = refined_corners[0][0][0]
        last_x = refined_corners[width-1][0][0]
        
        # If corners are in reverse order, flip them
        if first_x > last_x:
            print(f"Correcting corner order for {os.path.basename(frame_path)}")
            new_corners = np.zeros_like(refined_corners)
            for i in range(height):
                for j in range(width):
                    new_corners[i * width + j] = refined_corners[i * width + (width - 1 - j)]
            refined_corners = new_corners
        
        # Save debug image if requested
        if debug_dir is not None and camera_id is not None:
            debug_img = frame.copy()
            cv2.drawChessboardCorners(debug_img, checkerboard_size, refined_corners, ret)
            
            # Create a filename from the original frame path
            base_filename = os.path.basename(frame_path)
            debug_filename = f"{camera_id}_debug_{base_filename}"
            debug_path = os.path.join(debug_dir, debug_filename)
            
            cv2.imwrite(debug_path, debug_img)
        
        return refined_corners
    
    return None

def find_matching_corners(left_frames, right_frames, checkerboard_size, debug_dir):
    """Find matching frames that both contain good checkerboard corners."""
    print("\nFinding checkerboard corners in all frames...")
    
    # Process all frames
    left_results = {}
    right_results = {}
    
    print("Processing left camera frames...")
    for frame_path in tqdm(left_frames):
        # Extract frame ID from filename
        frame_id = '_'.join(os.path.basename(frame_path).split('_')[2:])
        corners = find_checkerboard_corners(frame_path, checkerboard_size, debug_dir, "left")
        if corners is not None:
            left_results[frame_id] = (frame_path, corners)
    
    print("Processing right camera frames...")
    for frame_path in tqdm(right_frames):
        # Extract frame ID from filename
        frame_id = '_'.join(os.path.basename(frame_path).split('_')[2:])
        corners = find_checkerboard_corners(frame_path, checkerboard_size, debug_dir, "right")
        if corners is not None:
            right_results[frame_id] = (frame_path, corners)
    
    # Find matching frames
    matching_frames = set(left_results.keys()) & set(right_results.keys())
    
    if not matching_frames:
        print("No matching frames with checkerboard corners found.")
        return [], [], [], []
    
    print(f"Found {len(matching_frames)} matching frame pairs with checkerboard corners.")
    
    # Create ordered lists of frames and corners
    matched_left_frames = []
    matched_left_corners = []
    matched_right_frames = []
    matched_right_corners = []
    
    for frame_id in sorted(matching_frames):
        left_frame, left_corner = left_results[frame_id]
        right_frame, right_corner = right_results[frame_id]
        
        matched_left_frames.append(left_frame)
        matched_left_corners.append(left_corner)
        matched_right_frames.append(right_frame)
        matched_right_corners.append(right_corner)
    
    return matched_left_frames, matched_left_corners, matched_right_frames, matched_right_corners

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
    
    # Check error if actual distance provided
    if actual_distance is not None:
        distance_error = 100 * abs(camera_distance - actual_distance) / actual_distance
        print(f"Actual distance: {actual_distance:.2f} mm")
        print(f"Distance error: {distance_error:.2f}%")
    
    # Get Euler angles
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
        
        f.write("\nEuler Angles (degrees):\n")
        f.write(f"Rotation around X: {euler_angles[0][0]:.2f}°\n")
        f.write(f"Rotation around Y: {euler_angles[1][0]:.2f}°\n")
        f.write(f"Rotation around Z: {euler_angles[2][0]:.2f}°\n")
        
        # List which videos were used
        f.write("\nVideos used in calibration:\n")
        video_ids = set()
        for frame in left_frames:
            parts = os.path.basename(frame).split('_')
            if len(parts) >= 3:
                video_ids.add(parts[2])
        
        for video_id in sorted(video_ids):
            f.write(f"- extrinsic video {video_id}\n")
    
    # Visualize camera positions
    visualize_camera_positions(R, T, output_dir)
    
    return R, T, E, F

def visualize_camera_positions(R, T, output_dir):
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
        ax.set_title('Relative Camera Positions')
        
        # Make axis scales equal
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
    """Main function for simplified stereo camera extrinsic calibration."""
    parser = argparse.ArgumentParser(description='Simplified Stereo Camera Extrinsic Calibration')
    parser.add_argument('--test_dir', required=True, 
                      help='Test directory name (e.g., test_001)')
    parser.add_argument('--base_dir', default='.', 
                      help='Base directory for the project (default: current directory)')
    parser.add_argument('--checkerboard_size', default='9,7', 
                      help='Size of checkerboard as width,height inner corners (default: 9,7)')
    parser.add_argument('--square_size', type=float, default=25.0,
                      help='Size of checkerboard square in mm (default: 25.0)')
    parser.add_argument('--frame_interval', type=int, default=15,
                      help='Extract every Nth frame from video (default: 15)')
    parser.add_argument('--max_frames', type=int, default=5,
                      help='Maximum number of frames to extract per video (default: 5)')
    parser.add_argument('--start_frame', type=int, default=30,
                      help='Skip this many frames at the start (default: 30)')
    parser.add_argument('--actual_distance', type=float, default=None,
                      help='Actual distance between cameras in mm (if known)')
    
    args = parser.parse_args()
    
    # Parse checkerboard size
    cb_width, cb_height = map(int, args.checkerboard_size.split(','))
    checkerboard_size = (cb_width, cb_height)
    
    # Set up paths
    base_dir = args.base_dir
    test_path = os.path.join(base_dir, "data", args.test_dir)
    
    # Create needed directories
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
    
    # Define camera directories and find matching videos
    left_camera_dir = os.path.join(test_path, "left_camera")
    right_camera_dir = os.path.join(test_path, "right_camera")
    
    video_pairs = find_matching_videos(left_camera_dir, right_camera_dir)
    
    if not video_pairs:
        print("Error: No matching video pairs found")
        return
    
    print(f"Found {len(video_pairs)} video pairs")
    
    # Process each video pair to extract frames
    all_left_frames = []
    all_right_frames = []
    
    for i, (left_video, right_video) in enumerate(video_pairs):
        # Use simple video ID based on position
        video_id = f"{i+1:03d}"
        
        print(f"\nProcessing video pair {i+1}/{len(video_pairs)}")
        print(f"Left: {os.path.basename(left_video)}")
        print(f"Right: {os.path.basename(right_video)}")
        
        # Extract frames
        left_frames = extract_frames(
            left_video, temp_dir, "left", video_id, 
            args.frame_interval, args.max_frames, args.start_frame
        )
        
        right_frames = extract_frames(
            right_video, temp_dir, "right", video_id, 
            args.frame_interval, args.max_frames, args.start_frame
        )
        
        all_left_frames.extend(left_frames)
        all_right_frames.extend(right_frames)
    
    # Find matching frames with checkerboard corners
    matched_left_frames, matched_left_corners, matched_right_frames, matched_right_corners = find_matching_corners(
        all_left_frames, all_right_frames, checkerboard_size, debug_dir
    )
    
    if len(matched_left_frames) < 5:
        print(f"Error: Found only {len(matched_left_frames)} matching frame pairs with checkerboard")
        print("Need at least 5 good frame pairs for calibration.")
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

if __name__ == "__main__":
    main()