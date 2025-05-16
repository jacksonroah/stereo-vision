#!/usr/bin/env python3
# I TRIED TO SIMPLIFY THE INTRISIC.PY SCRIPT AND IT FAILED MISERABLY. DONT USE. 
import cv2
import numpy as np
import os
import glob
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

def setup_directories(test_dir):
    """Create only necessary directories for output."""
    # Ensure camera directories exist
    os.makedirs(os.path.join(test_dir, "left_camera"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "right_camera"), exist_ok=True)
    
    # Create results directories
    os.makedirs(os.path.join(test_dir, "results", "intrinsic_params"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "results", "debug_images"), exist_ok=True)
    
    return os.path.join(test_dir, "results")

def find_calibration_video(camera_dir):
    """Find calibration video with simplified naming."""
    # Try common video extensions
    for ext in ['.mp4', '.mov', '.avi', '.MP4', '.MOV', '.AVI']:
        # First check for simple "intrinsic" name
        simple_path = os.path.join(camera_dir, f"intrinsic{ext}")
        if os.path.exists(simple_path):
            return simple_path
            
        # Fall back to any file with "intrinsic" in the name
        matches = glob.glob(os.path.join(camera_dir, f"*intrinsic*{ext}"))
        if matches:
            return matches[0]
    
    return None

def extract_frames(video_path, camera_id, frame_interval=15, max_frames=30):
    """Extract frames from video directly into memory without saving temporary files."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []
    
    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"\nProcessing {os.path.basename(video_path)}")
    print(f"Total frames: {total_frames}, FPS: {fps}")
    
    frames = []
    frame_count = 0
    
    # Setup progress bar
    pbar = tqdm(total=min(max_frames, total_frames // frame_interval + 1))
    
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frames.append(frame)
            pbar.update(1)
        
        frame_count += 1
    
    pbar.close()
    cap.release()
    
    print(f"Extracted {len(frames)} frames from {camera_id}")
    return frames

def ensure_corner_ordering(corners, pattern_size):
    """Ensure consistent ordering of checkerboard corners."""
    width, height = pattern_size
    
    # Check if we need to flip the corner ordering
    first_corner_idx = 0
    last_corner_idx = width - 1
    
    # Get x-coordinates of the first and last corner in the first row
    first_x = corners[first_corner_idx][0][0]
    last_x = corners[last_corner_idx][0][0]
    
    # If corners are not ordered left-to-right in the first row, flip the order
    if first_x > last_x:
        print("Correcting corner order (flipping horizontally)")
        new_corners = np.zeros_like(corners)
        for i in range(height):
            for j in range(width):
                new_corners[i * width + j] = corners[i * width + (width - 1 - j)]
        return new_corners
    
    return corners

def find_checkerboard_corners(frames, checkerboard_size, camera_id, debug_dir):
    """Find checkerboard corners in extracted frames."""
    good_frames = []
    good_corners = []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    print("\nFinding frames with good checkerboard corners...")
    for i, frame in enumerate(tqdm(frames)):
        if frame is None:
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Standard detection
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, 
                                              cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                              cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        if ret:
            # Refine corner positions
            refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Ensure consistent corner ordering
            refined_corners = ensure_corner_ordering(refined_corners, checkerboard_size)
            
            # Save a debug image with drawn corners
            debug_img = frame.copy()
            cv2.drawChessboardCorners(debug_img, checkerboard_size, refined_corners, ret)
            
            debug_path = os.path.join(debug_dir, f"{camera_id}_corners_{i:03d}.png")
            cv2.imwrite(debug_path, debug_img)
            
            good_frames.append(frame)
            good_corners.append(refined_corners)
    
    print(f"Found {len(good_frames)} frames with clear checkerboard pattern")
    return good_frames, good_corners

def calibrate_camera(frames, corners, checkerboard_size, square_size, camera_id):
    """Calibrate camera using frames with checkerboard."""
    # Prepare object points
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size  # Convert to actual size
    
    # Arrays to store object points and image points
    objpoints = [objp] * len(frames)
    imgpoints = corners
    
    # Get image size from first frame
    image_size = frames[0].shape[:2][::-1]  # (width, height)
    
    print(f"\nCalibrating camera {camera_id} with {len(frames)} frames...")
    
    # Calibration settings - use rational model for better results with phone cameras
    flags = cv2.CALIB_RATIONAL_MODEL
    
    # Perform calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None, flags=flags)
    
    # Calculate reprojection error
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    
    mean_error = total_error / len(objpoints)
    print(f"Calibration RMS error: {mean_error}")
    
    return camera_matrix, dist_coeffs, mean_error

def save_calibration_results(camera_matrix, dist_coeffs, error, test_dir, camera_id):
    """Save calibration results to files."""
    results_dir = os.path.join(test_dir, "results", "intrinsic_params")
    
    # Save camera matrix and distortion coefficients
    np.savetxt(os.path.join(results_dir, f"{camera_id}_matrix.txt"), camera_matrix)
    np.savetxt(os.path.join(results_dir, f"{camera_id}_distortion.txt"), dist_coeffs)
    
    # Save as pickle for future use with full precision
    with open(os.path.join(results_dir, f"{camera_id}_intrinsics.pkl"), 'wb') as f:
        pickle.dump((camera_matrix, dist_coeffs), f)
    
    # Save calibration info
    with open(os.path.join(results_dir, f"{camera_id}_info.txt"), 'w') as f:
        f.write(f"Camera: {camera_id}\n")
        f.write(f"RMS Reprojection Error: {error}\n\n")
        
        f.write("Camera Matrix:\n")
        for row in camera_matrix:
            f.write(f"{row[0]:.8f} {row[1]:.8f} {row[2]:.8f}\n")
        
        f.write("\nDistortion Coefficients (k1, k2, p1, p2, k3):\n")
        f.write(" ".join([f"{d:.8f}" for d in dist_coeffs.flatten()]))
    
    print(f"\nCalibration results for camera {camera_id} saved to {results_dir}")
    print(f"Focal length (pixels): fx={camera_matrix[0,0]:.1f}, fy={camera_matrix[1,1]:.1f}")
    print(f"Principal point: cx={camera_matrix[0,2]:.1f}, cy={camera_matrix[1,2]:.1f}")
    print(f"Distortion coefficients: {dist_coeffs.flatten()}")

def main():
    """Main function to run intrinsic calibration."""
    parser = argparse.ArgumentParser(description='Stereo Camera Intrinsic Calibration')
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
    parser.add_argument('--max_frames', type=int, default=30,
                      help='Maximum number of frames to extract per video (default: 30)')
    
    args = parser.parse_args()
    
    # Parse checkerboard size
    cb_width, cb_height = map(int, args.checkerboard_size.split(','))
    checkerboard_size = (cb_width, cb_height)
    
    # Set up paths
    base_dir = args.base_dir
    test_dir = os.path.join(base_dir, "data", args.test_dir)
    
    # Setup directories
    results_dir = setup_directories(test_dir)
    debug_dir = os.path.join(test_dir, "results", "debug_images")
    
    # Process each camera
    for camera_id in ['left', 'right']:
        camera_dir = os.path.join(test_dir, f"{camera_id}_camera")
        
        # Find calibration video
        video_path = find_calibration_video(camera_dir)
        
        if not video_path:
            print(f"Error: No calibration video found for {camera_id}")
            continue
        
        # Extract frames directly into memory
        frames = extract_frames(
            video_path, 
            camera_id, 
            frame_interval=args.frame_interval,
            max_frames=args.max_frames
        )
        
        if not frames:
            print(f"Error: No frames extracted for {camera_id}")
            continue
        
        # Find checkerboard corners
        good_frames, good_corners = find_checkerboard_corners(
            frames, checkerboard_size, camera_id, debug_dir
        )
        
        if not good_frames:
            print(f"Error: No usable checkerboard frames found for {camera_id}")
            continue
        
        # Calibrate camera
        camera_matrix, dist_coeffs, error = calibrate_camera(
            good_frames, good_corners, checkerboard_size, args.square_size, camera_id
        )
        
        # Save calibration results
        save_calibration_results(camera_matrix, dist_coeffs, error, test_dir, camera_id)
    
    print("\nIntrinsic calibration complete! Results saved to results/intrinsic_params/")
    print("\nNext steps:")
    print("1. Run the extrinsic calibration with: ")
    print(f"   python scripts/extrinsic_calibrator.py --test_dir {args.test_dir} --base_dir {args.base_dir}")

if __name__ == "__main__":
    main()