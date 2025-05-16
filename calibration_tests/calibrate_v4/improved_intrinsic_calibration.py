#!/usr/bin/env python3
"""
Simple Intrinsic Camera Calibration
----------------------------------
Takes a video of a moving checkerboard, extracts optimal frames,
and calculates intrinsic camera parameters.
"""

import cv2
import numpy as np
import os
import argparse
import glob

def extract_checkerboard_frames(video_path, output_dir, checkerboard_size=(9, 7), 
                               max_frames=20, frame_interval=15):
    """
    Extract frames from a video where checkerboard is visible and well-positioned
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []
    
    # Video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Processing video with {total_frames} frames at {fps} FPS")
    
    # Skip frames to get diversity
    frame_indices = []
    saved_frames = []
    frame_idx = 0
    
    while len(saved_frames) < max_frames and frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            break
            
        # Check if checkerboard is visible
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
        
        if ret:
            # Save frame
            output_path = os.path.join(output_dir, f"frame_{len(saved_frames):04d}.png")
            cv2.imwrite(output_path, frame)
            saved_frames.append(output_path)
            frame_indices.append(frame_idx)
            print(f"Saved frame {len(saved_frames)} (video frame {frame_idx})")
            
            # Draw corners and save visualization
            cv2.drawChessboardCorners(frame, checkerboard_size, corners, ret)
            viz_path = os.path.join(output_dir, f"frame_{len(saved_frames)-1:04d}_viz.png")
            cv2.imwrite(viz_path, frame)
        
        # Move to next frame by interval
        frame_idx += frame_interval
    
    cap.release()
    print(f"Extracted {len(saved_frames)} frames with visible checkerboard")
    return saved_frames

def calibrate_camera(image_paths, checkerboard_size=(9, 7), square_size=25.0):
    """Calibrate camera using checkerboard images"""
    
    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ...
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size  # convert to real world units
    
    # Arrays to store object points and image points
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane
    
    # Process each image
    successful_images = 0
    img_shape = None
    
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue
            
        if img_shape is None:
            img_shape = img.shape[:2]
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
        
        if ret:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            objpoints.append(objp)
            imgpoints.append(corners2)
            successful_images += 1
            print(f"Processed image {successful_images}: {os.path.basename(img_path)}")
    
    if successful_images < 10:
        print(f"Warning: Only {successful_images} successful images. Calibration may not be accurate.")
    
    print(f"Performing calibration using {successful_images} images...")
    
    # Use rational model and fix higher-order distortions to prevent extreme values
    flags = cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_FIX_K3
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape[::-1], None, None, flags=flags)
    
    # Calculate reprojection error
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    
    avg_error = total_error / len(objpoints) if objpoints else float('inf')
    print(f"Calibration complete. Average reprojection error: {avg_error:.6f} pixels")
    
    return mtx, dist, avg_error

def main():
    parser = argparse.ArgumentParser(description='Simple camera calibration from video')
    parser.add_argument('--video', type=str, required=True,
                      help='Path to calibration video')
    parser.add_argument('--output-dir', type=str, required=True,
                      help='Directory to save calibration results')
    parser.add_argument('--camera-id', type=int, required=True,
                      help='Camera identifier (1 or 2)')
    parser.add_argument('--checkerboard-size', type=str, default='9,7',
                      help='Size of checkerboard as width,height of internal corners')
    parser.add_argument('--square-size', type=float, default=25.0,
                      help='Size of checkerboard square in mm')
    parser.add_argument('--max-frames', type=int, default=20,
                      help='Maximum number of frames to extract')
    
    args = parser.parse_args()
    
    # Parse checkerboard size
    checkerboard_size = tuple(map(int, args.checkerboard_size.split(',')))
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    frames_dir = os.path.join(args.output_dir, f"camera{args.camera_id}_frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Extract frames with visible checkerboard
    print(f"Extracting frames from {args.video}...")
    frames = extract_checkerboard_frames(
        args.video, frames_dir, 
        checkerboard_size=checkerboard_size, 
        max_frames=args.max_frames
    )
    
    if not frames:
        print("Error: No frames with visible checkerboard were extracted")
        return
    
    # Perform calibration
    print("\nPerforming camera calibration...")
    camera_matrix, dist_coeffs, error = calibrate_camera(
        frames, checkerboard_size=checkerboard_size, square_size=args.square_size
    )
    
    # Save calibration results
    matrix_path = os.path.join(args.output_dir, f"camera_{args.camera_id}_matrix.txt")
    dist_path = os.path.join(args.output_dir, f"camera_{args.camera_id}_distortion.txt")
    
    np.savetxt(matrix_path, camera_matrix)
    np.savetxt(dist_path, dist_coeffs)
    
    # Save calibration info
    info_path = os.path.join(args.output_dir, f"camera_{args.camera_id}_info.txt")
    with open(info_path, 'w') as f:
        f.write(f"Camera {args.camera_id} Calibration\n")
        f.write(f"=======================\n\n")
        f.write(f"Reprojection error: {error:.6f} pixels\n\n")
        f.write(f"Camera Matrix:\n{camera_matrix}\n\n")
        f.write(f"Distortion Coefficients:\n{dist_coeffs}\n")
    
    print(f"\nCalibration results saved to {args.output_dir}")
    print(f"Camera Matrix: {matrix_path}")
    print(f"Distortion Coefficients: {dist_path}")
    print(f"Reprojection Error: {error:.6f} pixels")

if __name__ == "__main__":
    main()