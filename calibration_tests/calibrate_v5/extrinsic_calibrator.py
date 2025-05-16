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

def create_directories(base_dir):
    """Create necessary directories for output."""
    os.makedirs(os.path.join(base_dir, "extrinsic_frames", "cam1"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "extrinsic_frames", "cam2"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "stereo_calibration_results"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "stereo_debug_images"), exist_ok=True)
    
    return os.path.join(base_dir, "extrinsic_frames")

def load_intrinsic_params(base_dir, camera_id):
    """
    Load intrinsic camera parameters from calibration files.
    
    Args:
        base_dir (str): Base directory
        camera_id (str): Camera identifier (cam1 or cam2)
        
    Returns:
        tuple: (camera_matrix, distortion_coefficients)
    """
    matrix_file = os.path.join(base_dir, "calibration_results", f"{camera_id}_matrix.txt")
    dist_file = os.path.join(base_dir, "calibration_results", f"{camera_id}_distortion.txt")
    
    if not os.path.exists(matrix_file) or not os.path.exists(dist_file):
        print(f"Error: Calibration files for camera {camera_id} not found at:")
        print(f"  - {matrix_file}")
        print(f"  - {dist_file}")
        return None, None
    
    camera_matrix = np.loadtxt(matrix_file)
    dist_coeffs = np.loadtxt(dist_file)
    
    print(f"Successfully loaded intrinsic calibration for camera {camera_id}")
    return camera_matrix, dist_coeffs

def extract_frames(video_path, output_dir, camera_id, frame_interval=15, max_frames=20):
    """
    Extract frames from a video at regular intervals.
    
    Args:
        video_path (str): Path to the video file
        output_dir (str): Directory to save extracted frames
        camera_id (str): Camera identifier (cam1 or cam2)
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
            # Save frame - use the explicit camera_id parameter
            frame_path = os.path.join(output_dir, camera_id, f"{camera_id}_static_{saved_count:04d}.png")
            cv2.imwrite(frame_path, frame)
            extracted_frames.append(frame_path)
            saved_count += 1
            pbar.update(1)
        
        frame_count += 1
    
    pbar.close()
    cap.release()
    
    print(f"Extracted {saved_count} frames from {camera_id}")
    return extracted_frames

def find_matching_frames(cam1_frames, cam2_frames, checkerboard_size, debug_dir):
    """
    Find matching frames from both cameras that contain the checkerboard.
    
    Args:
        cam1_frames (list): Paths to frames from camera 1
        cam2_frames (list): Paths to frames from camera 2
        checkerboard_size (tuple): Size of the checkerboard (internal corners)
        debug_dir (str): Directory to save debug images
        
    Returns:
        list: List of tuples (cam1_frame, cam2_frame) with matching frame pairs
    """
    # Sort frames to ensure they're in the same temporal order
    cam1_frames.sort()
    cam2_frames.sort()
    
    # If frame counts don't match, take the minimum number
    min_frames = min(len(cam1_frames), len(cam2_frames))
    
    # Take pairs of frames at the same temporal position
    pairs = []
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    print("\nFinding matching frame pairs with checkerboard...")
    for i in range(min_frames):
        cam1_img_path = cam1_frames[i]
        cam2_img_path = cam2_frames[i]
        
        cam1_img = cv2.imread(cam1_img_path)
        cam2_img = cv2.imread(cam2_img_path)
        
        if cam1_img is None or cam2_img is None:
            print(f"Could not read images: {cam1_img_path} or {cam2_img_path}")
            continue
        
        cam1_gray = cv2.cvtColor(cam1_img, cv2.COLOR_BGR2GRAY)
        cam2_gray = cv2.cvtColor(cam2_img, cv2.COLOR_BGR2GRAY)
        
        # Find checkerboard corners
        ret1, corners1 = cv2.findChessboardCorners(cam1_gray, checkerboard_size, 
                                                  cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                  cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        ret2, corners2 = cv2.findChessboardCorners(cam2_gray, checkerboard_size, 
                                                  cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                  cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        if ret1 and ret2:
            # Refine corner positions
            corners1 = cv2.cornerSubPix(cam1_gray, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv2.cornerSubPix(cam2_gray, corners2, (11, 11), (-1, -1), criteria)
            
            # Save debug images
            debug_img1 = cam1_img.copy()
            debug_img2 = cam2_img.copy()
            
            cv2.drawChessboardCorners(debug_img1, checkerboard_size, corners1, ret1)
            cv2.drawChessboardCorners(debug_img2, checkerboard_size, corners2, ret2)
            
            frame_num = os.path.basename(cam1_img_path).split('_')[-1]
            
            debug_path1 = os.path.join(debug_dir, f"cam1_corners_{frame_num}")
            debug_path2 = os.path.join(debug_dir, f"cam2_corners_{frame_num}")
            
            cv2.imwrite(debug_path1, debug_img1)
            cv2.imwrite(debug_path2, debug_img2)
            
            pairs.append((cam1_img_path, cam2_img_path))
    
    print(f"Found {len(pairs)} matching frame pairs with checkerboard")
    return pairs

def calibrate_stereo(matching_pairs, camera1_matrix, dist1, camera2_matrix, dist2, 
                    checkerboard_size, square_size, output_dir, debug_dir, actual_distance=None):
    """
    Perform stereo calibration to find extrinsic parameters.
    
    Args:
        matching_pairs (list): List of matching frame pairs (cam1_path, cam2_path)
        camera1_matrix, dist1: Intrinsic parameters for camera 1
        camera2_matrix, dist2: Intrinsic parameters for camera 2
        checkerboard_size (tuple): Size of the checkerboard (internal corners)
        square_size (float): Size of each checkerboard square in mm
        output_dir (str): Directory to save results
        debug_dir (str): Directory to save debug information
        actual_distance (float, optional): Actual distance between cameras in mm
        
    Returns:
        tuple: (R, T, E, F) rotation matrix, translation vector, essential matrix, fundamental matrix
    """
    # Prepare object points
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size  # Convert to actual size
    
    # Arrays to store object points and image points
    objpoints = []      # 3D points in real world space
    imgpoints1 = []     # 2D points in camera 1
    imgpoints2 = []     # 2D points in camera 2
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    img_shape = None
    
    print("\nProcessing frame pairs for stereo calibration...")
    for cam1_path, cam2_path in tqdm(matching_pairs):
        # Read images
        img1 = cv2.imread(cam1_path)
        img2 = cv2.imread(cam2_path)
        
        if img1 is None or img2 is None:
            print(f"Error reading images: {cam1_path} or {cam2_path}")
            continue
        
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        if img_shape is None:
            img_shape = gray1.shape[::-1]  # (width, height)
        
        # Find the chess board corners
        ret1, corners1 = cv2.findChessboardCorners(gray1, checkerboard_size, None)
        ret2, corners2 = cv2.findChessboardCorners(gray2, checkerboard_size, None)
        
        if ret1 and ret2:
            # Refine corner positions
            corners1_refined = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2_refined = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
            
            objpoints.append(objp)
            imgpoints1.append(corners1_refined)
            imgpoints2.append(corners2_refined)
    
    if len(objpoints) < 5:
        print(f"Error: Not enough valid frame pairs ({len(objpoints)}) for stereo calibration")
        return None, None, None, None
    
    print(f"Using {len(objpoints)} frame pairs for stereo calibration")
    
    # Stereo calibration
    flags = 0
    # You can add flags if needed, like cv2.CALIB_FIX_INTRINSIC if you trust your intrinsic calibration
    flags |= cv2.CALIB_FIX_INTRINSIC  # Use this since we trust our intrinsic calibration
    
    print("\nPerforming stereo calibration...")
    ret, camera1_matrix, dist1, camera2_matrix, dist2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints1, imgpoints2,
        camera1_matrix, dist1,
        camera2_matrix, dist2,
        img_shape, criteria=criteria, flags=flags)
    
    print(f"Stereo calibration complete with RMS error: {ret}")
    
    # Calculate camera separation distance
    camera_distance = np.linalg.norm(T)
    print(f"Camera separation distance: {camera_distance:.2f} mm")
    
    # Calculate error if actual distance is provided
    if actual_distance is not None:
        distance_error = 100 * abs(camera_distance - actual_distance) / actual_distance
        print(f"Actual distance: {actual_distance:.2f} mm")
        print(f"Distance error: {distance_error:.2f}%")
    
    # Convert rotation matrix to Euler angles (degrees)
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
        f.write(f"RMS Reprojection Error: {ret}\n\n")
        
        f.write("Rotation Matrix (Camera 2 relative to Camera 1):\n")
        for row in R:
            f.write(f"{row[0]:.8f} {row[1]:.8f} {row[2]:.8f}\n")
        
        f.write("\nTranslation Vector (mm):\n")
        f.write(f"{T[0][0]:.8f} {T[1][0]:.8f} {T[2][0]:.8f}\n")
        
        f.write(f"\nCamera Separation Distance: {camera_distance:.2f} mm\n")
        
        if actual_distance is not None:
            f.write(f"Actual camera distance: {actual_distance:.2f} mm\n")
            f.write(f"Distance error: {distance_error:.2f}%\n")
        
        f.write("\nCamera 2 orientation relative to Camera 1 (Euler angles in degrees):\n")
        f.write(f"Rotation around X: {euler_angles[0][0]:.2f}°\n")
        f.write(f"Rotation around Y: {euler_angles[1][0]:.2f}°\n")
        f.write(f"Rotation around Z: {euler_angles[2][0]:.2f}°\n")
    
    # Compute rectification parameters
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        camera1_matrix, dist1,
        camera2_matrix, dist2,
        img_shape, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.9)
    
    # Save rectification parameters
    np.savetxt(os.path.join(output_dir, 'rect_R1.txt'), R1)
    np.savetxt(os.path.join(output_dir, 'rect_R2.txt'), R2)
    np.savetxt(os.path.join(output_dir, 'rect_P1.txt'), P1)
    np.savetxt(os.path.join(output_dir, 'rect_P2.txt'), P2)
    np.savetxt(os.path.join(output_dir, 'disparity_to_depth_matrix.txt'), Q)
    
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
        ax.scatter([cam1_pos[0]], [cam1_pos[1]], [cam1_pos[2]], c='r', marker='o', s=100, label='Camera 1')
        ax.scatter([cam2_pos[0]], [cam2_pos[1]], [cam2_pos[2]], c='b', marker='o', s=100, label='Camera 2')
        
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
    parser.add_argument('--base_dir', default='.', 
                      help='Base directory for the project (default: current directory)')
    parser.add_argument('--cam1_video', default='static_video/static_cam1.mov', 
                      help='Camera 1 static checkerboard video (default: static_video/static_cam1.mov)')
    parser.add_argument('--cam2_video', default='static_video/static_cam2.mov', 
                      help='Camera 2 static checkerboard video (default: static_video/static_cam2.mov)')
    parser.add_argument('--checkerboard_size', default='9,7', 
                      help='Size of checkerboard as width,height inner corners (default: 9,7)')
    parser.add_argument('--square_size', type=float, default=25.0,
                      help='Size of checkerboard square in mm (default: 25.0)')
    parser.add_argument('--frame_interval', type=int, default=15,
                      help='Extract every Nth frame from video (default: 15)')
    parser.add_argument('--max_frames', type=int, default=20,
                      help='Maximum number of frames to extract per video (default: 20)')
    parser.add_argument('--actual_distance', type=float, default=None,
                      help='Actual distance between cameras in mm (if known)')
    
    args = parser.parse_args()
    
    # Parse checkerboard size
    cb_width, cb_height = map(int, args.checkerboard_size.split(','))
    checkerboard_size = (cb_width, cb_height)
    
    # Set up paths
    base_dir = args.base_dir
    cam1_video = os.path.join(base_dir, args.cam1_video)
    cam2_video = os.path.join(base_dir, args.cam2_video)
    
    # Create needed directories
    frames_dir = create_directories(base_dir)
    output_dir = os.path.join(base_dir, "stereo_calibration_results")
    debug_dir = os.path.join(base_dir, "stereo_debug_images")
    
    # Check if videos exist
    if not os.path.exists(cam1_video):
        print(f"Error: Camera 1 video not found at {cam1_video}")
        return
    
    if not os.path.exists(cam2_video):
        print(f"Error: Camera 2 video not found at {cam2_video}")
        return
    
    # Load intrinsic parameters
    camera1_matrix, dist1 = load_intrinsic_params(base_dir, "cam1")
    camera2_matrix, dist2 = load_intrinsic_params(base_dir, "cam2")
    
    if camera1_matrix is None or camera2_matrix is None:
        print("Error: Could not load intrinsic calibration parameters")
        return
    
    # Extract frames from videos - explicitly pass camera IDs
    cam1_frames = extract_frames(cam1_video, frames_dir, "cam1", args.frame_interval, args.max_frames)
    cam2_frames = extract_frames(cam2_video, frames_dir, "cam2", args.frame_interval, args.max_frames)
    
    if not cam1_frames or not cam2_frames:
        print("Error: Could not extract frames from videos")
        return
    
    # Find matching frames
    matching_pairs = find_matching_frames(cam1_frames, cam2_frames, checkerboard_size, debug_dir)
    
    if not matching_pairs:
        print("Error: No matching frame pairs found")
        return
    
    # Perform stereo calibration
    R, T, E, F = calibrate_stereo(
        matching_pairs, camera1_matrix, dist1, camera2_matrix, dist2, 
        checkerboard_size, args.square_size, output_dir, debug_dir, args.actual_distance
    )
    
    if R is not None:
        print("\nExtrinsic calibration complete!")
        print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()