#!/usr/bin/env python3
# The intrinsic_iphone.py renamed
import cv2
import numpy as np
import os
import glob
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import pickle
import shutil

def create_directories(test_dir):
    """Create necessary directories for output with consistent naming."""
    # Main directories
    os.makedirs(os.path.join(test_dir, "left"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "right"), exist_ok=True)
    
    # Temp directories
    temp_dir = os.path.join(test_dir, "temp")
    os.makedirs(os.path.join(temp_dir, "left"), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, "right"), exist_ok=True)
    
    # Results directories
    os.makedirs(os.path.join(test_dir, "results", "intrinsic_params"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "results", "debug_images"), exist_ok=True)
    
    return temp_dir

def extract_frames(video_path, output_dir, camera_id, frame_interval=15, max_frames=20, start_frame=30):
    """
    Extract frames from a video at regular intervals, optimized for iPhone videos.
    
    Args:
        video_path (str): Path to the video file
        output_dir (str): Directory to save extracted frames
        camera_id (str): Camera identifier (left or right)
        frame_interval (int): Extract every nth frame
        max_frames (int): Maximum number of frames to extract
        start_frame (int): Skip initial frames which might have movement
        
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
    
    # Skip initial frames (camera might be adjusting or moving)
    if start_frame > 0 and start_frame < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        print(f"Skipping first {start_frame} frames")
    
    # Adjust frame_interval based on video length
    adjusted_interval = frame_interval
    estimated_frames = (total_frames - start_frame) // adjusted_interval
    
    if estimated_frames < max_frames:
        # If we'd get too few frames, reduce the interval
        adjusted_interval = max(1, (total_frames - start_frame) // (max_frames + 5))
        print(f"Adjusting frame interval to {adjusted_interval} to get enough frames")
    
    extracted_frames = []
    frame_count = 0
    saved_count = 0
    
    # Setup progress bar
    pbar = tqdm(total=min(max_frames, (total_frames - start_frame) // adjusted_interval + 1))
    
    while cap.isOpened() and saved_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % adjusted_interval == 0:
            # Save frame to directory
            frame_path = os.path.join(output_dir, camera_id, f"{camera_id}_frame_{saved_count:04d}.png")
            os.makedirs(os.path.dirname(frame_path), exist_ok=True)
            cv2.imwrite(frame_path, frame)
            
            # Create a copy in the alternate location for compatibility
            alt_path = os.path.join(output_dir, f"{camera_id}_camera", f"{camera_id}_frame_{saved_count:04d}.png")
            os.makedirs(os.path.dirname(alt_path), exist_ok=True)
            shutil.copy2(frame_path, alt_path)
            
            extracted_frames.append(frame_path)
            saved_count += 1
            pbar.update(1)
        
        frame_count += 1
    
    pbar.close()
    cap.release()
    
    print(f"Extracted {saved_count} frames from {camera_id}")
    return extracted_frames

def find_good_checkerboard_frames(frame_paths, checkerboard_size, debug_dir):
    """
    Find frames with good checkerboard corners, with enhanced corner detection.
    
    Args:
        frame_paths (list): Paths to the extracted frames
        checkerboard_size (tuple): Size of the checkerboard (internal corners)
        debug_dir (str): Directory to save debug images
        
    Returns:
        list: Paths to frames with good checkerboard detection
        list: Corresponding corners for each good frame
    """
    good_frames = []
    good_corners = []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    print("\nFinding frames with good checkerboard corners...")
    for frame_path in tqdm(frame_paths):
        image = cv2.imread(frame_path)
        if image is None:
            print(f"Warning: Could not read image {frame_path}")
            # Try the alternate path version
            alt_path = frame_path.replace("/left/", "/left_camera/").replace("/right/", "/right_camera/")
            if "/left_camera/" in frame_path:
                alt_path = frame_path.replace("/left_camera/", "/left/")
            elif "/right_camera/" in frame_path:
                alt_path = frame_path.replace("/right_camera/", "/right/")
                
            image = cv2.imread(alt_path)
            if image is None:
                print(f"Warning: Could not read alternate image {alt_path} either. Skipping.")
                continue
            
            # Update frame_path to the working version
            frame_path = alt_path
        
        # Try multiple detection methods for better corner detection on iPhones
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Standard detection
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, 
                                                cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        # Method 2: Try with additional filters if standard fails
        if not ret:
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            ret, corners = cv2.findChessboardCorners(blurred, checkerboard_size, 
                                                   cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                   cv2.CALIB_CB_NORMALIZE_IMAGE + 
                                                   cv2.CALIB_CB_FILTER_QUADS)
        
        if ret:
            # Refine corner positions
            refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Ensure consistent corner ordering (important for stereo calibration)
            refined_corners = ensure_corner_ordering(refined_corners, checkerboard_size)
            
            # Save a debug image with drawn corners
            camera_id = os.path.basename(frame_path).split('_')[0]
            frame_num = os.path.basename(frame_path).split('_')[-1]
            
            # Draw corners on a copy
            debug_img = image.copy()
            cv2.drawChessboardCorners(debug_img, checkerboard_size, refined_corners, ret)
            
            debug_path = os.path.join(debug_dir, f"{camera_id}_corners_{frame_num}")
            cv2.imwrite(debug_path, debug_img)
            
            good_frames.append(frame_path)
            good_corners.append(refined_corners)
    
    print(f"Found {len(good_frames)} frames with clear checkerboard pattern")
    return good_frames, good_corners

def ensure_corner_ordering(corners, pattern_size):
    """
    Ensure consistent ordering of checkerboard corners.
    
    This is important for stereo calibration to work correctly.
    The function checks if the corners are ordered from left-to-right 
    and fixes the order if needed.
    
    Args:
        corners: Detected corners from cv2.findChessboardCorners
        pattern_size: Size of the checkerboard (width, height)
        
    Returns:
        Corners with consistent ordering
    """
    width, height = pattern_size
    
    # Check if we need to flip the corner ordering
    # We do this by checking if first and last corners in the first row 
    # are ordered from left to right
    first_corner_idx = 0
    last_corner_idx = width - 1
    
    # Get x-coordinates of the first and last corner in the first row
    first_x = corners[first_corner_idx][0][0]
    last_x = corners[last_corner_idx][0][0]
    
    # If corners are not ordered left-to-right in the first row, flip the order
    if first_x > last_x:
        # Create a new array with flipped ordering
        new_corners = np.zeros_like(corners)
        for i in range(height):
            for j in range(width):
                # Flip the order in each row
                new_corners[i * width + j] = corners[i * width + (width - 1 - j)]
        return new_corners
    
    return corners

def calculate_frame_errors(objpoints, imgpoints, camera_matrix, dist_coeffs, rvecs, tvecs):
    """Calculate reprojection error for each frame."""
    frame_errors = []
    
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        frame_errors.append((error, i))
    
    return frame_errors

def calibrate_camera(frame_paths, corners, checkerboard_size, square_size, camera_id, debug_dir):
    """
    Calibrate camera using frames with checkerboard, optimized for iPhone cameras.
    
    Args:
        frame_paths (list): Paths to good frames with checkerboard
        corners (list): Detected corners for each frame
        checkerboard_size (tuple): Size of the checkerboard (internal corners)
        square_size (float): Size of each checkerboard square in mm
        camera_id (str): Camera identifier (left or right)
        debug_dir (str): Directory to save debug information
    
    Returns:
        tuple: (camera_matrix, dist_coeffs, rms_error)
    """
    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ... (8,6,0)
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size  # Convert to actual size
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    image_size = None
    
    print(f"\nCalibrating camera {camera_id}...")
    for i, frame_path in enumerate(tqdm(frame_paths)):
        image = cv2.imread(frame_path)
        if image is None:
            print(f"Warning: Could not read image {frame_path}. Skipping.")
            continue
            
        if image_size is None:
            image_size = image.shape[:2][::-1]  # (width, height)
        
        objpoints.append(objp)
        imgpoints.append(corners[i])
    
    if not objpoints:
        print(f"Error: No usable frames found for camera {camera_id}")
        return None, None, None
    
    print(f"Using {len(objpoints)} frames for calibration")
    
    # iPhone-optimized calibration settings
    # We fix higher-order distortion coefficients to prevent overfitting
    flags = cv2.CALIB_RATIONAL_MODEL  # Use rational model for iPhone wide lenses
    flags |= cv2.CALIB_FIX_K4
    flags |= cv2.CALIB_FIX_K5
    flags |= cv2.CALIB_FIX_K6
    
    # Initial calibration
    print("Performing initial calibration...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None, flags=flags)
    
    # Calculate per-frame errors
    frame_errors = calculate_frame_errors(objpoints, imgpoints, mtx, dist, rvecs, tvecs)
    mean_error = sum(error for error, _ in frame_errors) / len(frame_errors)
    
    print(f"Initial calibration RMS error: {mean_error}")
    
    # Filter out frames with high reprojection error (iterative refinement)
    if len(objpoints) > 12:  # Only filter if we have enough frames
        # Two rounds of filtering for better results
        for round_num in range(2):
            if round_num == 0:
                error_threshold = 1.0  # First round: remove very bad frames
            else:
                error_threshold = mean_error * 1.5  # Second round: remove frames > 1.5x average error
            
            # Sort frames by error
            frame_errors.sort(reverse=True)
            
            # Remove frames with high error (up to 20% of frames)
            frames_to_remove = int(len(frame_errors) * 0.2)
            frames_to_remove = min(frames_to_remove, 
                                  len(frame_errors) - 10)  # Keep at least 10 frames
            
            # Only remove frames above threshold
            remove_indices = []
            for error, idx in frame_errors[:frames_to_remove]:
                if error > error_threshold:
                    remove_indices.append(idx)
            
            if not remove_indices:
                print(f"No frames exceed error threshold ({error_threshold:.4f})")
                break
            
            print(f"Removing {len(remove_indices)} frames with error > {error_threshold:.4f}")
            
            # Sort indices in descending order to avoid index shifting during removal
            remove_indices.sort(reverse=True)
            
            # Remove frames with high error
            for idx in remove_indices:
                objpoints.pop(idx)
                imgpoints.pop(idx)
                frame_paths.pop(idx)
            
            print(f"Recalibrating with {len(objpoints)} frames...")
            
            # Recalibrate
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, image_size, None, None, flags=flags)
            
            # Recalculate errors
            frame_errors = calculate_frame_errors(objpoints, imgpoints, mtx, dist, rvecs, tvecs)
            mean_error = sum(error for error, _ in frame_errors) / len(frame_errors)
            
            print(f"Calibration RMS error after filtering: {mean_error}")
    
    # Visualize the reprojection error distribution
    plt.figure(figsize=(10, 6))
    error_vals = [error for error, _ in frame_errors]
    
    plt.bar(range(len(error_vals)), error_vals)
    plt.axhline(y=mean_error, color='r', linestyle='-', label=f'Mean Error: {mean_error:.4f}')
    plt.xlabel('Frame Index')
    plt.ylabel('Reprojection Error')
    plt.title(f'Reprojection Error Distribution - {camera_id}')
    plt.legend()
    plt.savefig(os.path.join(debug_dir, f'{camera_id}_reprojection_error.png'))
    plt.close()
    
    # Visualize the camera calibration with a grid distortion map
    visualize_distortion(mtx, dist, image_size, camera_id, debug_dir)
    
    return mtx, dist, mean_error, frame_paths

def visualize_distortion(camera_matrix, dist_coeffs, image_size, camera_id, debug_dir):
    """Create a visual representation of lens distortion."""
    width, height = image_size
    
    # Create a grid of points
    grid_size = 20
    x_step = width // grid_size
    y_step = height // grid_size
    
    grid_points = []
    for y in range(0, height, y_step):
        for x in range(0, width, x_step):
            grid_points.append((x, y))
    
    grid_points = np.array(grid_points, dtype=np.float32).reshape(-1, 1, 2)
    
    # Get undistorted points
    undistorted = cv2.undistortPoints(grid_points, camera_matrix, dist_coeffs, None, camera_matrix)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot original grid points
    grid_x = grid_points[:, 0, 0].reshape(-1, grid_size)
    grid_y = grid_points[:, 0, 1].reshape(-1, grid_size)
    plt.plot(grid_x, grid_y, 'b.', markersize=1)
    
    # Plot undistorted points
    undist_x = undistorted[:, 0, 0].reshape(-1, grid_size)
    undist_y = undistorted[:, 0, 1].reshape(-1, grid_size)
    plt.plot(undist_x, undist_y, 'r.', markersize=3)
    
    # Draw lines between original and undistorted points
    for i in range(len(grid_points)):
        orig = grid_points[i, 0]
        undist = undistorted[i, 0]
        plt.plot([orig[0], undist[0]], [orig[1], undist[1]], 'k-', linewidth=0.5, alpha=0.3)
    
    plt.xlim(0, width)
    plt.ylim(height, 0)  # Invert Y axis to match image coordinates
    plt.title(f'Lens Distortion Map - {camera_id}')
    plt.savefig(os.path.join(debug_dir, f'{camera_id}_distortion_map.png'))
    plt.close()

def save_calibration_results(camera_matrix, dist_coeffs, error, frame_paths, test_dir, camera_id):
    """Save calibration results to files."""
    # Create directory for results if it doesn't exist
    results_dir = os.path.join(test_dir, "results", "intrinsic_params")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save camera matrix and distortion coefficients - text format for compatibility
    np.savetxt(os.path.join(results_dir, f"{camera_id}_matrix.txt"), camera_matrix)
    np.savetxt(os.path.join(results_dir, f"{camera_id}_distortion.txt"), dist_coeffs)
    
    # Save as pickle for future use with full precision
    with open(os.path.join(results_dir, f"{camera_id}_intrinsics.pkl"), 'wb') as f:
        pickle.dump((camera_matrix, dist_coeffs), f)
    
    # Save frames used for calibration
    with open(os.path.join(results_dir, f"{camera_id}_frames_used.txt"), 'w') as f:
        for frame in frame_paths:
            f.write(f"{os.path.basename(frame)}\n")
    
    # Save calibration info
    with open(os.path.join(results_dir, f"{camera_id}_info.txt"), 'w') as f:
        f.write(f"Camera: {camera_id}\n")
        f.write(f"Calibration Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"RMS Reprojection Error: {error}\n\n")
        
        f.write("Camera Matrix:\n")
        for row in camera_matrix:
            f.write(f"{row[0]:.8f} {row[1]:.8f} {row[2]:.8f}\n")
        
        f.write("\nDistortion Coefficients (k1, k2, p1, p2, k3):\n")
        f.write(" ".join([f"{d:.8f}" for d in dist_coeffs.flatten()]))
        
        f.write(f"\n\nCalibration used {len(frame_paths)} frames\n")
    
    print(f"\nCalibration results for camera {camera_id} saved to {results_dir}")
    print(f"Focal length (pixels): fx={camera_matrix[0,0]:.1f}, fy={camera_matrix[1,1]:.1f}")
    print(f"Principal point: cx={camera_matrix[0,2]:.1f}, cy={camera_matrix[1,2]:.1f}")
    print(f"Distortion: k1={dist_coeffs[0,0]:.4f}, k2={dist_coeffs[0,1]:.4f}, k3={dist_coeffs[0,4]:.4f}")
    print(f"Tangential distortion: p1={dist_coeffs[0,2]:.4f}, p2={dist_coeffs[0,3]:.4f}")

def main():
    """Main function to run intrinsic calibration for iPhone cameras."""
    parser = argparse.ArgumentParser(description='iPhone Camera Intrinsic Calibration')
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
    parser.add_argument('--start_frame', type=int, default=30,
                      help='Skip this many frames at the start of the video (default: 30)')
    
    args = parser.parse_args()
    
    # Parse checkerboard size
    cb_width, cb_height = map(int, args.checkerboard_size.split(','))
    checkerboard_size = (cb_width, cb_height)
    
    # Set up paths
    base_dir = args.base_dir
    test_dir = os.path.join(base_dir, "data", args.test_dir)
    
    # Create needed directories - with consistent naming
    temp_dir = create_directories(test_dir)
    debug_dir = os.path.join(test_dir, "results", "debug_images")
    
    # Process each camera
    for camera_id in ['left', 'right']:
        # Define camera directory name and look for videos
        camera_dir = f"{camera_id}"
        video_path = os.path.join(test_dir, camera_dir, "intrinsic.mov")
        
        # Try common video formats if the default isn't found
        if not os.path.exists(video_path):
            for ext in ['.mov', '.avi', '.MP4', '.MOV']:
                alt_path = os.path.join(test_dir, camera_dir, f"intrinsic{ext}")
                if os.path.exists(alt_path):
                    video_path = alt_path
                    print(f"Found alternative video file: {alt_path}")
                    break
            
        if not os.path.exists(video_path):
            print(f"Error: No video file found for {camera_id}")
            continue
        
        # Extract frames from video
        frame_paths = extract_frames(
            video_path, 
            temp_dir, 
            camera_id, 
            frame_interval=args.frame_interval,
            max_frames=args.max_frames,
            start_frame=args.start_frame
        )
        
        if not frame_paths:
            print(f"Error: No frames extracted for {camera_id}")
            continue
        
        # Find frames with good checkerboard corners
        good_frames, good_corners = find_good_checkerboard_frames(frame_paths, checkerboard_size, debug_dir)
        
        if not good_frames:
            print(f"Error: No usable checkerboard frames found for {camera_id}")
            continue
        
        # Calibrate camera
        camera_matrix, dist_coeffs, error, used_frames = calibrate_camera(
            good_frames, good_corners, checkerboard_size, args.square_size, camera_id, debug_dir
        )
        
        if camera_matrix is not None:
            # Save calibration results
            save_calibration_results(camera_matrix, dist_coeffs, error, used_frames, test_dir, camera_id)
    
    print("\nIntrinsic calibration complete! Results saved to results/intrinsic_params/")
    print("\nNext steps:")
    print("1. Run the extrinsic calibration with: ")
    print(f"   python scripts/extrinsic.py --test_dir {args.test_dir} --actual_distance 'your distance'")
    print("2. Validate the calibration with:")
    print(f"   python scripts/simple_validator_updated.py --test_dir {args.test_dir} --base_dir {args.base_dir}")

if __name__ == "__main__":
    main()