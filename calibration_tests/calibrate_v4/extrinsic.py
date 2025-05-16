#!/usr/bin/env python3
"""
Improved Extrinsic Camera Calibration
-------------------------------------
Performs stereo calibration to determine the relative position and orientation
of two cameras using synchronized checkerboard images.
"""

import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
import json
import argparse

def load_intrinsic_params(calib_dir, camera_num):
    """Load camera matrix and distortion coefficients"""
    matrix_file = os.path.join(calib_dir, f'camera_{camera_num}_matrix.txt')
    dist_file = os.path.join(calib_dir, f'camera_{camera_num}_distortion.txt')
    
    if os.path.exists(matrix_file) and os.path.exists(dist_file):
        camera_matrix = np.loadtxt(matrix_file)
        dist_coeffs = np.loadtxt(dist_file)
        print(f"Successfully loaded calibration for camera {camera_num}")
        return camera_matrix, dist_coeffs
    else:
        print(f"Error: Calibration files for camera {camera_num} not found at:")
        print(f"  - {matrix_file}")
        print(f"  - {dist_file}")
        return None, None

def extract_stereo_frames(video1_path, video2_path, output_dir, max_frames=20):
    """
    Extract synchronized frames from stereo videos
    
    Parameters:
    video1_path, video2_path: Paths to stereo calibration videos
    output_dir: Directory to save extracted frames
    max_frames: Maximum number of frames to extract
    
    Returns:
    List of pairs of frame paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Open videos
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    
    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Could not open one or both videos")
        return []
    
    # Get video properties
    total_frames1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Determine how many frames to extract
    num_frames = min(total_frames1, total_frames2, max_frames)
    
    if num_frames < 5:
        print(f"Warning: Only {num_frames} frames available. Extrinsic calibration may not be accurate.")
    
    # Calculate intervals for frame extraction
    interval = max(1, min(total_frames1, total_frames2) // num_frames)
    
    # Extract frames
    frame_pairs = []
    for i in range(num_frames):
        frame_idx = i * interval
        
        # Set position and read frames
        cap1.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if ret1 and ret2:
            # Save frames
            frame1_path = os.path.join(output_dir, f"stereo_cam1_frame_{i:03d}.png")
            frame2_path = os.path.join(output_dir, f"stereo_cam2_frame_{i:03d}.png")
            
            cv2.imwrite(frame1_path, frame1)
            cv2.imwrite(frame2_path, frame2)
            
            frame_pairs.append((frame1_path, frame2_path))
            print(f"Saved stereo frame pair {i+1}/{num_frames}")
    
    cap1.release()
    cap2.release()
    
    return frame_pairs

def find_checkerboard_corners(image_path, checkerboard_size=(9, 7), debug_dir=None):
    """Find checkerboard corners in an image"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return False, None, None, img
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find checkerboard corners
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, 
                                           cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                           cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    if ret:
        # Refine corner locations
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # Draw corners for visualization
        img_corners = img.copy()
        cv2.drawChessboardCorners(img_corners, checkerboard_size, corners2, ret)
        
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            base_name = os.path.basename(image_path)
            debug_path = os.path.join(debug_dir, f"corners_{base_name}")
            cv2.imwrite(debug_path, img_corners)
            print(f"Saved corner visualization to {debug_path}")
        
        return ret, corners2, img.shape[:2], img_corners
    else:
        print(f"Failed to find checkerboard in {os.path.basename(image_path)}")
        
        # Save a copy of the image for manual inspection
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            base_name = os.path.basename(image_path)
            debug_path = os.path.join(debug_dir, f"failed_{base_name}")
            cv2.imwrite(debug_path, img)
            print(f"Saved failed image to {debug_path}")
        
        return False, None, None, img

def calibrate_stereo(frame_pairs, calib_dir, output_dir, 
                   checkerboard_size=(9, 7), square_size=25.0, 
                   known_distance=None):
    """
    Perform stereo calibration to find extrinsic parameters between two cameras
    
    Parameters:
    frame_pairs: List of pairs of image paths (cam1, cam2)
    calib_dir: Directory containing intrinsic calibration parameters
    output_dir: Directory to save results
    checkerboard_size: Size of the checkerboard (internal corners)
    square_size: Size of checkerboard square in mm
    known_distance: Optional known distance between cameras for validation
    
    Returns:
    R, T: Rotation matrix and translation vector
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    debug_dir = os.path.join(output_dir, "debug_images")
    os.makedirs(debug_dir, exist_ok=True)
    
    # Load intrinsic parameters
    camera1_matrix, dist1 = load_intrinsic_params(calib_dir, 1)
    camera2_matrix, dist2 = load_intrinsic_params(calib_dir, 2)
    
    if camera1_matrix is None or camera2_matrix is None:
        print("Error: Could not load intrinsic parameters")
        return None, None
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints1 = []  # 2D points in Camera 1
    imgpoints2 = []  # 2D points in Camera 2
    
    # Process each stereo pair
    successful_pairs = 0
    
    print("\nChecking checkerboard corners in stereo pairs...")
    
    # Create a PDF report
    pdf_path = os.path.join(output_dir, "stereo_calibration_report.pdf")
    with PdfPages(pdf_path) as pdf:
        for pair_idx, (cam1_img_path, cam2_img_path) in enumerate(frame_pairs):
            print(f"\nProcessing pair {pair_idx+1}/{len(frame_pairs)}:")
            print(f"  Camera 1: {os.path.basename(cam1_img_path)}")
            print(f"  Camera 2: {os.path.basename(cam2_img_path)}")
            
            # Find checkerboard corners in both images
            ret1, corners1, img1_shape, img1_corners = find_checkerboard_corners(
                cam1_img_path, checkerboard_size, debug_dir)
            ret2, corners2, img2_shape, img2_corners = find_checkerboard_corners(
                cam2_img_path, checkerboard_size, debug_dir)
            
            # Add to PDF report
            plt.figure(figsize=(12, 6))
            plt.suptitle(f"Stereo Pair {pair_idx+1}", fontsize=14)
            
            # Camera 1 image
            plt.subplot(1, 2, 1)
            if ret1:
                plt.imshow(cv2.cvtColor(img1_corners, cv2.COLOR_BGR2RGB))
                plt.title(f"Camera 1 - Corners Found")
            else:
                plt.imshow(cv2.cvtColor(img1_corners, cv2.COLOR_BGR2RGB))
                plt.title(f"Camera 1 - Corners NOT Found")
            plt.axis('off')
            
            # Camera 2 image
            plt.subplot(1, 2, 2)
            if ret2:
                plt.imshow(cv2.cvtColor(img2_corners, cv2.COLOR_BGR2RGB))
                plt.title(f"Camera 2 - Corners Found")
            else:
                plt.imshow(cv2.cvtColor(img2_corners, cv2.COLOR_BGR2RGB))
                plt.title(f"Camera 2 - Corners NOT Found")
            plt.axis('off')
            
            pdf.savefig()
            plt.close()
            
            # If both images have checkerboard corners, use them for calibration
            if ret1 and ret2:
                # Make sure we have valid image shapes
                if img1_shape is None or img2_shape is None or 0 in img1_shape or 0 in img2_shape:
                    print(f"Warning: Invalid image shape detected")
                    continue
                    
                # Set image shape for calibration
                img_shape = img1_shape
                
                # Prepare object points for this checkerboard
                objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
                objp[:,:2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
                objp *= square_size  # Convert to actual measurements
                
                objpoints.append(objp)
                imgpoints1.append(corners1)
                imgpoints2.append(corners2)
                successful_pairs += 1
                print(f"Successfully found checkerboard corners in both images")
        
        if successful_pairs < 5:
            print(f"\nWarning: Only {successful_pairs} successful stereo pairs found.")
            if successful_pairs < 3:
                print("Cannot proceed with calibration. Please provide more valid stereo pairs.")
                return None, None
        
        print(f"\nUsing {successful_pairs} stereo pairs for calibration")
        
        # Set termination criteria for stereo calibration
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
        
        # Add calibration flags
        flags = (
            cv2.CALIB_FIX_INTRINSIC +  # Use intrinsic parameters without refinement
            0  # Add other flags if needed
        )
        
        print("\nPerforming stereo calibration...")
        
        # Stereo calibration
        ret, camera1_matrix, dist1, camera2_matrix, dist2, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints1, imgpoints2,
            camera1_matrix, dist1,
            camera2_matrix, dist2,
            img_shape, criteria=criteria, flags=flags)
        
        print("\nStereo Calibration Results:")
        print(f"RMS reprojection error: {ret}")
        
        print("\nRotation Matrix (Camera 2 relative to Camera 1):")
        print(R)
        
        print("\nTranslation Vector (Camera 2 relative to Camera 1) in mm:")
        print(T)
        
        # Calculate camera separation distance
        camera_distance = np.linalg.norm(T)
        print(f"\nCamera separation distance: {camera_distance:.2f} mm")
        
        # Convert rotation matrix to Euler angles (in degrees)
        r_vec, _ = cv2.Rodrigues(R)
        euler_angles = r_vec * 180.0 / np.pi
        print("\nCamera 2 orientation relative to Camera 1 (Euler angles in degrees):")
        print(f"Rotation around X: {euler_angles[0][0]:.2f}°")
        print(f"Rotation around Y: {euler_angles[1][0]:.2f}°")
        print(f"Rotation around Z: {euler_angles[2][0]:.2f}°")
        
        # Save calibration results
        np.savetxt(os.path.join(output_dir, 'stereo_rotation_matrix.txt'), R)
        np.savetxt(os.path.join(output_dir, 'stereo_translation_vector.txt'), T)
        np.savetxt(os.path.join(output_dir, 'essential_matrix.txt'), E)
        np.savetxt(os.path.join(output_dir, 'fundamental_matrix.txt'), F)
        
        # Save data as JSON for easier reading
        stereo_data = {
            "reprojection_error": float(ret),
            "camera_distance_mm": float(camera_distance),
            "rotation_matrix": R.tolist(),
            "translation_vector_mm": T.flatten().tolist(),
            "euler_angles_degrees": euler_angles.flatten().tolist(),
            "essential_matrix": E.tolist(),
            "fundamental_matrix": F.tolist()
        }
        
        with open(os.path.join(output_dir, 'stereo_calibration_data.json'), 'w') as f:
            json.dump(stereo_data, f, indent=4)
        
        # Compute and save rectification parameters
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
        
        # Add results to PDF report
        plt.figure(figsize=(10, 8))
        plt.axis('off')
        plt.text(0.1, 0.95, "Stereo Calibration Results", fontsize=16, weight='bold')
        
        plt.text(0.1, 0.9, f"Calibration performed with {successful_pairs} stereo pairs", fontsize=12)
        plt.text(0.1, 0.85, f"RMS reprojection error: {ret:.6f} pixels", fontsize=12)
        plt.text(0.1, 0.8, f"Camera separation distance: {camera_distance:.2f} mm", fontsize=12)
        
        if known_distance is not None:
            distance_error = abs(camera_distance - known_distance)
            distance_error_pct = (distance_error / known_distance) * 100
            plt.text(0.1, 0.75, f"Known distance: {known_distance:.2f} mm", fontsize=12)
            plt.text(0.1, 0.7, f"Distance error: {distance_error:.2f} mm ({distance_error_pct:.2f}%)", fontsize=12)
        
        plt.text(0.1, 0.65, "Rotation Matrix (Camera 2 relative to Camera 1):", fontsize=12, weight='bold')
        for i in range(3):
            plt.text(0.1, 0.6 - i*0.05, 
                    f"[{R[i,0]:8.4f}, {R[i,1]:8.4f}, {R[i,2]:8.4f}]", 
                    fontsize=10, family='monospace')
        
        plt.text(0.1, 0.4, "Translation Vector (mm):", fontsize=12, weight='bold')
        t_str = ", ".join([f"{x:.2f}" for x in T.flatten()])
        plt.text(0.1, 0.35, f"[{t_str}]", fontsize=10, family='monospace')
        
        plt.text(0.1, 0.3, "Euler Angles (degrees):", fontsize=12, weight='bold')
        plt.text(0.1, 0.25, f"X: {euler_angles[0][0]:.2f}°, Y: {euler_angles[1][0]:.2f}°, Z: {euler_angles[2][0]:.2f}°", 
                fontsize=10, family='monospace')
        
        pdf.savefig()
        plt.close()
        
        # Visualize camera positions
        visualize_camera_positions(R, T, output_dir, pdf)
        
        # Visualize sample rectified images
        if successful_pairs > 0:
            # Choose a sample pair
            sample_idx = min(1, successful_pairs - 1)
            
            # Get the sample stereo pair
            img1 = cv2.imread(frame_pairs[imgpoints1.index(imgpoints1[sample_idx])][0])
            img2 = cv2.imread(frame_pairs[imgpoints2.index(imgpoints2[sample_idx])][1])
            
            # Compute rectification maps
            map1x, map1y = cv2.initUndistortRectifyMap(
                camera1_matrix, dist1, R1, P1, img_shape[::-1], cv2.CV_32FC1)
            map2x, map2y = cv2.initUndistortRectifyMap(
                camera2_matrix, dist2, R2, P2, img_shape[::-1], cv2.CV_32FC1)
            
            # Apply rectification
            rect1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
            rect2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)
            
            # Save rectified images
            cv2.imwrite(os.path.join(debug_dir, 'rectified_cam1_sample.png'), rect1)
            cv2.imwrite(os.path.join(debug_dir, 'rectified_cam2_sample.png'), rect2)
            
            # Create a side-by-side comparison with horizontal lines
            h, w = img1.shape[:2]
            rect_combined = np.zeros((h, w*2, 3), dtype=np.uint8)
            rect_combined[:, :w] = rect1
            rect_combined[:, w:] = rect2
            
            # Draw horizontal lines to check rectification (every 50 pixels)
            for y in range(0, h, 50):
                cv2.line(rect_combined, (0, y), (w*2, y), (0, 255, 0), 1)
            
            cv2.imwrite(os.path.join(debug_dir, 'rectified_side_by_side.png'), rect_combined)
            
            # Add to report
            plt.figure(figsize=(12, 8))
            plt.suptitle("Rectification Example", fontsize=14)
            
            plt.subplot(2, 2, 1)
            plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
            plt.title("Original Camera 1")
            plt.axis('off')
            
            plt.subplot(2, 2, 2)
            plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
            plt.title("Original Camera 2")
            plt.axis('off')
            
            plt.subplot(2, 2, 3)
            plt.imshow(cv2.cvtColor(rect1, cv2.COLOR_BGR2RGB))
            plt.title("Rectified Camera 1")
            plt.axis('off')
            
            plt.subplot(2, 2, 4)
            plt.imshow(cv2.cvtColor(rect2, cv2.COLOR_BGR2RGB))
            plt.title("Rectified Camera 2")
            plt.axis('off')
            
            pdf.savefig()
            plt.close()
            
            # Show side-by-side rectification
            plt.figure(figsize=(14, 8))
            plt.imshow(cv2.cvtColor(rect_combined, cv2.COLOR_BGR2RGB))
            plt.title("Rectified Images Side-by-Side with Epipolar Lines")
            plt.axis('off')
            
            pdf.savefig()
            plt.close()
        
        print(f"Stereo calibration report saved to {pdf_path}")
        print(f"Stereo calibration results saved to {output_dir}")
        
        return R, T

def visualize_camera_positions(R, T, output_dir, pdf=None):
    """Visualize the relative positions of two cameras in 3D space"""
    try:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Camera 1 is at the origin
        cam1_pos = np.array([0, 0, 0])
        
        # Camera 2 position is determined by the rotation and translation
        # We need the camera position in world coordinates
        cam2_pos = -R.T @ T.flatten()
        
        # Plot camera positions
        ax.scatter([cam1_pos[0]], [cam1_pos[1]], [cam1_pos[2]], c='r', marker='o', s=100, label='Camera 1')
        ax.scatter([cam2_pos[0]], [cam2_pos[1]], [cam2_pos[2]], c='b', marker='o', s=100, label='Camera 2')
        
        # Draw line connecting cameras
        ax.plot([cam1_pos[0], cam2_pos[0]], 
                [cam1_pos[1], cam2_pos[1]], 
                [cam1_pos[2], cam2_pos[2]], 'k-')
        
        # Plot camera orientations (principal axis)
        # For Camera 1, the principal axis is along the Z axis
        cam1_axis_length = np.linalg.norm(cam2_pos) / 2  # Scale based on camera distance
        cam1_axis = np.array([0, 0, cam1_axis_length])  # Z axis
        ax.quiver(cam1_pos[0], cam1_pos[1], cam1_pos[2], 
                  cam1_axis[0], cam1_axis[1], cam1_axis[2], 
                  color='r', arrow_length_ratio=0.1)
        
        # For Camera 2, the principal axis is rotated according to R
        cam2_axis = R.T @ np.array([0, 0, cam1_axis_length])
        ax.quiver(cam2_pos[0], cam2_pos[1], cam2_pos[2], 
                  cam2_axis[0], cam2_axis[1], cam2_axis[2], 
                  color='b', arrow_length_ratio=0.1)
        
        # Set equal aspect ratio
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title('Relative Camera Positions')
        
        # Try to set equal scaling for all axes
        max_range = np.max([
            np.max([np.abs(cam1_pos[0]), np.abs(cam2_pos[0])]),
            np.max([np.abs(cam1_pos[1]), np.abs(cam2_pos[1])]),
            np.max([np.abs(cam1_pos[2]), np.abs(cam2_pos[2])])
        ]) * 1.5
        
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)
        
        ax.legend()
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, 'camera_positions.png'))
        
        # Add to PDF if provided
        if pdf is not None:
            pdf.savefig(fig)
        
        plt.close()
        print(f"Camera positions visualization saved")
        
    except Exception as e:
        print(f"Error visualizing camera positions: {e}")

def main():
    parser = argparse.ArgumentParser(description='Improved stereo camera calibration')
    parser.add_argument('--base-dir', type=str, default='./calibration_data',
                       help='Base directory for calibration data')
    parser.add_argument('--calib-dir', type=str, 
                       help='Directory containing intrinsic calibration parameters')
    parser.add_argument('--stereo-video1', type=str, 
                       help='Path to stereo calibration video from camera 1')
    parser.add_argument('--stereo-video2', type=str, 
                       help='Path to stereo calibration video from camera 2')
    parser.add_argument('--output-dir', type=str, 
                       help='Directory to save stereo calibration results')
    parser.add_argument('--checkerboard-size', type=str, default='9,7',
                       help='Size of checkerboard as width,height of internal corners')
    parser.add_argument('--square-size', type=float, default=25.0,
                       help='Size of checkerboard square in mm')
    parser.add_argument('--known-distance', type=float, 
                       help='Known distance between cameras for validation (mm)')
    parser.add_argument('--extract-frames', action='store_true',
                       help='Extract frames from stereo videos')
    parser.add_argument('--max-frames', type=int, default=20,
                       help='Maximum number of frames to extract from stereo videos')
    
    args = parser.parse_args()
    
    # Set default directories if not specified
    base_dir = args.base_dir
    calib_dir = args.calib_dir or os.path.join(base_dir, "calibration_results")
    output_dir = args.output_dir or os.path.join(base_dir, "stereo_calibration_results")
    
    # Parse checkerboard size
    checkerboard_size = tuple(map(int, args.checkerboard_size.split(',')))
    
    # Extract frames from stereo videos if specified
    stereo_frames_dir = os.path.join(base_dir, "stereo_frames")
    if args.extract_frames and args.stereo_video1 and args.stereo_video2:
        print("Extracting frames from stereo videos...")
        frame_pairs = extract_stereo_frames(
            args.stereo_video1, args.stereo_video2, 
            stereo_frames_dir, args.max_frames
        )
    else:
        # Look for existing stereo frames
        cam1_frames = sorted(glob.glob(os.path.join(stereo_frames_dir, "stereo_cam1_*.png")))
        cam2_frames = sorted(glob.glob(os.path.join(stereo_frames_dir, "stereo_cam2_*.png")))
        
        # Match frames by index
        frame_pairs = [
            (cam1, cam2) for cam1, cam2 in zip(cam1_frames, cam2_frames)
            if os.path.basename(cam1).split('_')[-1] == os.path.basename(cam2).split('_')[-1]
        ]
        
        if not frame_pairs:
            print("Error: No stereo frame pairs found. Use --extract-frames with video paths.")
            return
    
    # Run stereo calibration
    print(f"Running stereo calibration with {len(frame_pairs)} frame pairs...")
    R, T = calibrate_stereo(
        frame_pairs, calib_dir, output_dir,
        checkerboard_size=checkerboard_size,
        square_size=args.square_size,
        known_distance=args.known_distance
    )
    
    if R is not None and T is not None:
        print("Stereo calibration completed successfully!")
    else:
        print("Stereo calibration failed.")

if __name__ == "__main__":
    main()