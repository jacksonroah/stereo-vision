#!/usr/bin/env python3
"""
Optimal Frame Extractor for Camera Calibration
---------------------------------------------
Extracts frames from calibration videos at optimal intervals, selecting only
frames where the checkerboard is clearly visible and in different orientations.
"""

import cv2
import numpy as np
import os
import glob
import argparse
from scipy.spatial import distance

def create_output_directories(base_dir):
    """Create output directories for calibration images"""
    cam1_dir = os.path.join(base_dir, "calibrate_v4/videos/cam1/calib1.mov")
    cam2_dir = os.path.join(base_dir, "calibrate_v4/videos/cam/calib1.mov")
    
    os.makedirs(cam1_dir, exist_ok=True)
    os.makedirs(cam2_dir, exist_ok=True)
    
    return cam1_dir, cam2_dir

def extract_checkerboard_info(frame, checkerboard_size=(9, 7)):
    """
    Extract checkerboard corners and compute feature vector for frame diversity
    
    Returns:
    (success, corners, feature_vector)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Find checkerboard corners
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, 
                                          cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                          cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    if not ret:
        return False, None, None
        
    # Refine corner locations
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    
    # Compute feature vector based on the geometry of the checkerboard
    # This helps identify diverse orientations and positions
    
    # 1. Center point of checkerboard
    center = np.mean(corners2, axis=0)[0]
    
    # 2. Standard deviation of corner distances from center (shape descriptor)
    dists = np.sqrt(np.sum((corners2[:, 0, :] - center)**2, axis=1))
    dist_std = np.std(dists)
    
    # 3. Principal axis orientation (rotation descriptor)
    pts = corners2[:, 0, :]
    pts_centered = pts - center
    cov = np.cov(pts_centered.T)
    evals, evecs = np.linalg.eig(cov)
    sort_indices = np.argsort(evals)[::-1]
    principal_axis = evecs[:, sort_indices[0]]
    angle = np.arctan2(principal_axis[1], principal_axis[0])
    
    # 4. Aspect ratio of the projected checkerboard
    min_x, min_y = np.min(pts, axis=0)
    max_x, max_y = np.max(pts, axis=0)
    width = max_x - min_x
    height = max_y - min_y
    aspect_ratio = width / height if height > 0 else 1.0
    
    # Create feature vector
    feature_vector = np.array([
        center[0] / frame.shape[1],  # Normalized x position
        center[1] / frame.shape[0],  # Normalized y position
        dist_std,                    # Size/distortion measure
        np.cos(angle),               # Orientation cosine
        np.sin(angle),               # Orientation sine
        aspect_ratio                 # Perspective measure
    ])
    
    return True, corners2, feature_vector

def is_diverse_frame(feature_vector, existing_features, min_distance=0.15):
    """Check if a frame is diverse enough compared to existing frames"""
    if len(existing_features) == 0:
        return True
        
    # Calculate distances to all existing features
    distances = [distance.euclidean(feature_vector, ef) for ef in existing_features]
    min_dist = min(distances)
    
    return min_dist > min_distance

def extract_diverse_frames(video_path, output_dir, camera_id, max_frames=30, 
                         checkerboard_size=(9, 7), min_feature_distance=0.15):
    """
    Extract diverse frames from a video where checkerboard is visible
    
    Parameters:
    video_path: Path to the video file
    output_dir: Directory to save extracted frames
    camera_id: Identifier for the camera (1 or 2)
    max_frames: Maximum number of frames to extract
    checkerboard_size: Size of the checkerboard (internal corners)
    min_feature_distance: Minimum feature distance for diversity
    
    Returns:
    List of paths to extracted frames
    """
    # Get video file name without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Processing {video_name} - {total_frames} frames, {fps} FPS")
    
    # Calculate initial frame interval based on video length and desired max frames
    initial_interval = max(1, total_frames // (max_frames * 3))  # Start with 3x as many candidates
    
    # Extract candidate frames
    candidate_frames = []
    frame_count = 0
    
    while len(candidate_frames) < (max_frames * 2) and frame_count < total_frames:
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        
        if not ret:
            break
            
        # Check if checkerboard is visible and get feature vector
        success, corners, features = extract_checkerboard_info(frame, checkerboard_size)
        
        if success:
            # Store the successful candidate
            candidate_frames.append((frame_count, frame, corners, features))
            print(f"Found checkerboard in frame {frame_count}")
        
        # Move to next frame using interval
        frame_count += initial_interval
    
    cap.release()
    
    print(f"Found {len(candidate_frames)} candidate frames with visible checkerboard")
    
    # If we don't have enough candidates, try to extract more with a smaller interval
    if len(candidate_frames) < 10 and initial_interval > 1:
        print("Not enough candidates, retrying with smaller interval...")
        smaller_interval = max(1, initial_interval // 2)
        cap = cv2.VideoCapture(video_path)
        
        frame_count = smaller_interval // 2  # Offset to get different frames
        while len(candidate_frames) < (max_frames * 2) and frame_count < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()
            
            if not ret:
                break
                
            success, corners, features = extract_checkerboard_info(frame, checkerboard_size)
            
            if success:
                candidate_frames.append((frame_count, frame, corners, features))
                print(f"Found additional checkerboard in frame {frame_count}")
            
            frame_count += smaller_interval
        
        cap.release()
        print(f"Now have {len(candidate_frames)} candidate frames")
    
    # Select diverse frames from candidates
    diverse_frames = []
    existing_features = []
    
    for frame_idx, frame, corners, features in candidate_frames:
        if is_diverse_frame(features, existing_features, min_feature_distance):
            diverse_frames.append((frame_idx, frame, corners))
            existing_features.append(features)
            
            if len(diverse_frames) >= max_frames:
                break
    
    print(f"Selected {len(diverse_frames)} diverse frames for calibration")
    
    # Save the selected frames
    saved_frames = []
    for i, (frame_idx, frame, corners) in enumerate(diverse_frames):
        # Draw checkerboard corners for verification
        img_corners = frame.copy()
        cv2.drawChessboardCorners(img_corners, checkerboard_size, corners, True)
        
        # Save both original and visualized frames
        frame_path = os.path.join(output_dir, f"cam{camera_id}_frame_{i:03d}.png")
        viz_path = os.path.join(output_dir, f"cam{camera_id}_frame_{i:03d}_viz.png")
        
        cv2.imwrite(frame_path, frame)
        cv2.imwrite(viz_path, img_corners)
        
        saved_frames.append(frame_path)
        print(f"Saved frame {i+1}/{len(diverse_frames)}")
    
    return saved_frames

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract diverse frames from calibration videos')
    parser.add_argument('--cam1-video', type=str, required=True, 
                        help='Path to calibration video for camera 1')
    parser.add_argument('--cam2-video', type=str, required=True, 
                        help='Path to calibration video for camera 2')
    parser.add_argument('--output-dir', type=str, default='./calibration_data',
                        help='Base directory to save extracted frames and results')
    parser.add_argument('--max-frames', type=int, default=25, 
                        help='Maximum number of frames to extract per camera')
    parser.add_argument('--checkerboard-size', type=str, default='9,7',
                        help='Size of checkerboard as width,height of internal corners')
    args = parser.parse_args()
    
    # Parse checkerboard size
    checkerboard_size = tuple(map(int, args.checkerboard_size.split(',')))
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    cam1_dir, cam2_dir = create_output_directories(args.output_dir)
    
    # Extract frames from camera 1 video
    print("\n=== Processing Camera 1 Video ===")
    cam1_frames = extract_diverse_frames(
        args.cam1_video, cam1_dir, camera_id=1, 
        max_frames=args.max_frames, checkerboard_size=checkerboard_size
    )
    
    # Extract frames from camera 2 video
    print("\n=== Processing Camera 2 Video ===")
    cam2_frames = extract_diverse_frames(
        args.cam2_video, cam2_dir, camera_id=2, 
        max_frames=args.max_frames, checkerboard_size=checkerboard_size
    )
    
    print("\n=== Summary ===")
    print(f"Extracted {len(cam1_frames)} frames for camera 1")
    print(f"Extracted {len(cam2_frames)} frames for camera 2")
    print(f"Camera 1 frames saved to: {cam1_dir}")
    print(f"Camera 2 frames saved to: {cam2_dir}")
    print("\nNext step: Run the camera calibration script on these frames")

if __name__ == "__main__":
    main()