#!/usr/bin/env python3
import cv2
import numpy as np
import os
import glob
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

def create_directories(base_dir):
    """Create necessary directories for output."""
    os.makedirs(os.path.join(base_dir, "extracted_frames", "cam1"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "extracted_frames", "cam2"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "calibration_results"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "debug_images"), exist_ok=True)
    
    return os.path.join(base_dir, "extracted_frames")

def extract_frames(video_path, output_dir, frame_interval=10, max_frames=50):
    """
    Extract frames from a video at regular intervals.
    
    Args:
        video_path (str): Path to the video file
        output_dir (str): Directory to save extracted frames
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
    camera_id = os.path.basename(video_path).split('.')[0]  # Extract cam1 or cam2
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
            # Save frame
            frame_path = os.path.join(output_dir, camera_id, f"{camera_id}_frame_{saved_count:04d}.png")
            cv2.imwrite(frame_path, frame)
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
    Find frames with good checkerboard corners.
    
    Args:
        frame_paths (list): Paths to the extracted frames
        checkerboard_size (tuple): Size of the checkerboard (internal corners)
        debug_dir (str): Directory to save debug images
        
    Returns:
        list: Paths to frames with good checkerboard detection
    """
    good_frames = []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    print("\nFinding frames with good checkerboard corners...")
    for frame_path in tqdm(frame_paths):
        image = cv2.imread(frame_path)
        if image is None:
            continue
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, 
                                                cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        if ret:
            # Refine corner positions
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Save a debug image with drawn corners
            camera_id = os.path.basename(frame_path).split('_')[0]
            frame_num = os.path.basename(frame_path).split('_')[-1]
            
            # Draw corners on a copy
            debug_img = image.copy()
            cv2.drawChessboardCorners(debug_img, checkerboard_size, corners2, ret)
            
            debug_path = os.path.join(debug_dir, f"{camera_id}_corners_{frame_num}")
            cv2.imwrite(debug_path, debug_img)
            
            good_frames.append(frame_path)
    
    print(f"Found {len(good_frames)} frames with clear checkerboard pattern")
    return good_frames

def calibrate_camera(frame_paths, checkerboard_size, square_size, camera_id, debug_dir):
    """
    Calibrate camera using frames with checkerboard.
    
    Args:
        frame_paths (list): Paths to good frames with checkerboard
        checkerboard_size (tuple): Size of the checkerboard (internal corners)
        square_size (float): Size of each checkerboard square in mm
        camera_id (str): Camera identifier (cam1 or cam2)
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
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    image_size = None
    
    print(f"\nCalibrating camera {camera_id}...")
    for frame_path in tqdm(frame_paths):
        image = cv2.imread(frame_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if image_size is None:
            image_size = gray.shape[::-1]  # (width, height)
        
        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, 
                                               cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                               cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        if ret:
            # Refine corner positions
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            objpoints.append(objp)
            imgpoints.append(corners2)
    
    if not objpoints:
        print(f"Error: No usable frames found for camera {camera_id}")
        return None, None, None
    
    print(f"Using {len(objpoints)} frames for calibration")
    
    # Calibration with stronger constraints to prevent extreme values
    flags = cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None, flags=flags)
    
    # Calculate reprojection error
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    
    mean_error = total_error / len(objpoints)
    
    # Filter out frames with high reprojection error and recalibrate if necessary
    if mean_error > 1.0 and len(objpoints) > 10:
        print(f"Initial calibration RMS error: {mean_error}. Filtering frames with high error...")
        
        # Get per-frame errors
        frame_errors = []
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            frame_errors.append((error, i))
        
        # Sort by error
        frame_errors.sort(reverse=True)
        
        # Remove top 20% of high-error frames
        frames_to_remove = int(len(frame_errors) * 0.2)
        frames_to_remove = max(frames_to_remove, 1)  # Remove at least one
        
        remove_indices = [idx for _, idx in frame_errors[:frames_to_remove]]
        remove_indices.sort(reverse=True)
        
        # Remove frames with high error
        for idx in remove_indices:
            objpoints.pop(idx)
            imgpoints.pop(idx)
        
        print(f"Recalibrating with {len(objpoints)} frames after filtering...")
        
        # Recalibrate
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, image_size, None, None, flags=flags)
        
        # Recalculate error
        total_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        
        mean_error = total_error / len(objpoints)
    
    print(f"Final calibration RMS error: {mean_error}")
    
    # Visualize the reprojection error distribution
    plt.figure(figsize=(10, 6))
    frame_errors = []
    
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        frame_errors.append(error)
    
    plt.bar(range(len(frame_errors)), frame_errors)
    plt.xlabel('Frame Index')
    plt.ylabel('Reprojection Error')
    plt.title(f'Reprojection Error Distribution - Camera {camera_id}')
    plt.savefig(os.path.join(debug_dir, f'{camera_id}_reprojection_error.png'))
    
    return mtx, dist, mean_error

def save_calibration_results(camera_matrix, dist_coeffs, error, base_dir, camera_id):
    """Save calibration results to files."""
    # Create directory for results if it doesn't exist
    results_dir = os.path.join(base_dir, "calibration_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save camera matrix and distortion coefficients
    np.savetxt(os.path.join(results_dir, f"{camera_id}_matrix.txt"), camera_matrix)
    np.savetxt(os.path.join(results_dir, f"{camera_id}_distortion.txt"), dist_coeffs)
    
    # Save calibration info
    with open(os.path.join(results_dir, f"{camera_id}_info.txt"), 'w') as f:
        f.write(f"Camera: {camera_id}\n")
        f.write(f"Calibration Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"RMS Reprojection Error: {error}\n\n")
        
        f.write("Camera Matrix:\n")
        for row in camera_matrix:
            f.write(f"{row[0]:.8f} {row[1]:.8f} {row[2]:.8f}\n")
        
        f.write("\nDistortion Coefficients:\n")
        f.write(" ".join([f"{d:.8f}" for d in dist_coeffs.flatten()]))
    
    print(f"\nCalibration results for camera {camera_id} saved to {results_dir}")

def main():
    """Main function to run intrinsic calibration."""
    parser = argparse.ArgumentParser(description='Camera Intrinsic Calibration from Videos')
    parser.add_argument('--base_dir', default='.', 
                      help='Base directory for the project (default: current directory)')
    parser.add_argument('--vids_dir', default='intrinsic_calibration_vids',
                      help='Directory containing calibration videos (default: intrinsic_calibration_vids)')
    parser.add_argument('--checkerboard_size', default='9,7', 
                      help='Size of checkerboard as width,height inner corners (default: 9,7)')
    parser.add_argument('--square_size', type=float, default=25.0,
                      help='Size of checkerboard square in mm (default: 25.0)')
    parser.add_argument('--frame_interval', type=int, default=10,
                      help='Extract every Nth frame from video (default: 10)')
    parser.add_argument('--max_frames', type=int, default=50,
                      help='Maximum number of frames to extract per video (default: 50)')
    
    args = parser.parse_args()
    
    # Parse checkerboard size
    cb_width, cb_height = map(int, args.checkerboard_size.split(','))
    checkerboard_size = (cb_width, cb_height)
    
    # Set up paths
    base_dir = args.base_dir
    vids_dir = os.path.join(base_dir, args.vids_dir)
    
    # Create needed directories
    extracted_frames_dir = create_directories(base_dir)
    debug_dir = os.path.join(base_dir, "debug_images")
    
    # Process each camera
    for camera_id in ['cam1', 'cam2']:
        # Look for the video file
        video_path = os.path.join(vids_dir, f"{camera_id}.mov")
        
        if not os.path.exists(video_path):
            print(f"Warning: Video file for {camera_id} not found at {video_path}")
            continue
        
        # Extract frames from video
        frame_paths = extract_frames(
            video_path, 
            extracted_frames_dir, 
            frame_interval=args.frame_interval,
            max_frames=args.max_frames
        )
        
        if not frame_paths:
            print(f"Error: No frames extracted for {camera_id}")
            continue
        
        # Find frames with good checkerboard corners
        good_frames = find_good_checkerboard_frames(frame_paths, checkerboard_size, debug_dir)
        
        if not good_frames:
            print(f"Error: No usable checkerboard frames found for {camera_id}")
            continue
        
        # Calibrate camera
        camera_matrix, dist_coeffs, error = calibrate_camera(
            good_frames, checkerboard_size, args.square_size, camera_id, debug_dir
        )
        
        if camera_matrix is not None:
            # Save calibration results
            save_calibration_results(camera_matrix, dist_coeffs, error, base_dir, camera_id)
            
            print(f"\nCamera Matrix for {camera_id}:")
            print(camera_matrix)
            print(f"\nDistortion Coefficients for {camera_id}:")
            print(dist_coeffs)
            print(f"RMS Reprojection Error: {error}")
    
    print("\nIntrinsic calibration complete! Results saved to calibration_results/")

if __name__ == "__main__":
    main()