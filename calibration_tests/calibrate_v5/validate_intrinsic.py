#!/usr/bin/env python3
import cv2
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import glob

def load_calibration_params(base_dir, camera_id):
    """Load camera matrix and distortion coefficients from files."""
    matrix_file = os.path.join(base_dir, "calibration_results", f"{camera_id}_matrix.txt")
    dist_file = os.path.join(base_dir, "calibration_results", f"{camera_id}_distortion.txt")
    
    if not os.path.exists(matrix_file) or not os.path.exists(dist_file):
        print(f"Error: Calibration files for {camera_id} not found.")
        return None, None
    
    camera_matrix = np.loadtxt(matrix_file)
    dist_coeffs = np.loadtxt(dist_file)
    
    return camera_matrix, dist_coeffs

def undistort_image(image, camera_matrix, dist_coeffs):
    """Undistort an image using camera calibration parameters."""
    h, w = image.shape[:2]
    
    # Calculate optimal new camera matrix
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    
    # Undistort image
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, newcameramtx)
    
    # Crop the image if needed
    # x, y, w, h = roi
    # undistorted = undistorted[y:y+h, x:x+w]
    
    return undistorted

def visualize_undistortion(image_path, camera_matrix, dist_coeffs, output_dir):
    """Create a side-by-side visualization of original and undistorted images."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    # Undistort the image
    undistorted = undistort_image(image, camera_matrix, dist_coeffs)
    
    # Create side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    undistorted_rgb = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)
    
    # Draw grid lines for better distortion visualization
    h, w = image.shape[:2]
    grid_step = 50  # pixels between grid lines
    
    for x in range(0, w, grid_step):
        cv2.line(image_rgb, (x, 0), (x, h), (0, 255, 0), 1)
    for y in range(0, h, grid_step):
        cv2.line(image_rgb, (0, y), (w, y), (0, 255, 0), 1)
        
    for x in range(0, w, grid_step):
        cv2.line(undistorted_rgb, (x, 0), (x, h), (0, 255, 0), 1)
    for y in range(0, h, grid_step):
        cv2.line(undistorted_rgb, (0, y), (w, y), (0, 255, 0), 1)
    
    # Display images
    axes[0].imshow(image_rgb)
    axes[0].set_title('Original Image with Grid')
    axes[0].axis('off')
    
    axes[1].imshow(undistorted_rgb)
    axes[1].set_title('Undistorted Image with Grid')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    camera_id = os.path.basename(image_path).split('_')[0]
    frame_num = os.path.basename(image_path).split('_')[-1].split('.')[0]
    output_path = os.path.join(output_dir, f"{camera_id}_undistort_validate_{frame_num}.png")
    plt.savefig(output_path)
    
    print(f"Saved undistortion visualization to {output_path}")
    
    # Also save the undistorted image
    undistorted_path = os.path.join(output_dir, f"{camera_id}_undistorted_{frame_num}.png")
    cv2.imwrite(undistorted_path, undistorted)

def main():
    """Main function to validate camera calibration."""
    parser = argparse.ArgumentParser(description='Validate Camera Intrinsic Calibration')
    parser.add_argument('--base_dir', default='.', 
                      help='Base directory for the project (default: current directory)')
    parser.add_argument('--frames_dir', default='extracted_frames',
                      help='Directory containing extracted frames (default: extracted_frames)')
    
    args = parser.parse_args()
    
    # Set up paths
    base_dir = args.base_dir
    frames_dir = os.path.join(base_dir, args.frames_dir)
    output_dir = os.path.join(base_dir, "validation_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each camera
    for camera_id in ['cam1', 'cam2']:
        # Load calibration parameters
        camera_matrix, dist_coeffs = load_calibration_params(base_dir, camera_id)
        
        if camera_matrix is None or dist_coeffs is None:
            continue
        
        print(f"\nValidating calibration for {camera_id}...")
        
        # Get a few frames to validate
        frame_paths = glob.glob(os.path.join(frames_dir, camera_id, f"{camera_id}_frame_*.png"))
        
        if not frame_paths:
            print(f"Error: No frames found for {camera_id}")
            continue
        
        # Select frames at different positions in the video
        num_frames = len(frame_paths)
        if num_frames > 3:
            validate_indices = [0, num_frames // 2, num_frames - 1]  # Start, middle, end
            validate_frames = [frame_paths[i] for i in validate_indices]
        else:
            validate_frames = frame_paths
        
        # Visualize undistortion for each selected frame
        for frame_path in validate_frames:
            visualize_undistortion(frame_path, camera_matrix, dist_coeffs, output_dir)
    
    print("\nValidation complete! Check the validation_results directory for visualizations.")

if __name__ == "__main__":
    main()