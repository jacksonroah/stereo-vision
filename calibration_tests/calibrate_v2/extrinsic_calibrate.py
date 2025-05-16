import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_intrinsic_params(base_dir, camera_num):
    """Load camera matrix and distortion coefficients"""
    matrix_file = os.path.join(base_dir, f'camera_{camera_num}_matrix.txt')
    dist_file = os.path.join(base_dir, f'camera_{camera_num}_distortion.txt')
    
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

def match_stereo_images(cam1_folder, cam2_folder):
    """Match images from two cameras based on file modification time"""
    # Get all images from both cameras
    cam1_images = sorted(glob.glob(os.path.join(cam1_folder, '*.png')) + 
                        glob.glob(os.path.join(cam1_folder, '*.jpg')))
    cam2_images = sorted(glob.glob(os.path.join(cam2_folder, '*.png')) + 
                        glob.glob(os.path.join(cam2_folder, '*.jpg')))
    
    print(f"Found {len(cam1_images)} images in Camera 1 folder")
    print(f"Found {len(cam2_images)} images in Camera 2 folder")
    
    # If image counts don't match, we'll try to pair them by name similarity
    pairs = []
    
    # Simple pairing by index if counts match
    if len(cam1_images) == len(cam2_images):
        print("Image counts match - pairing by position in sorted list")
        pairs = list(zip(cam1_images, cam2_images))
    else:
        print("Image counts don't match - trying to pair by name similarity")
        # Extract base identifiers from filenames
        cam1_ids = [os.path.splitext(os.path.basename(img))[0] for img in cam1_images]
        cam2_ids = [os.path.splitext(os.path.basename(img))[0] for img in cam2_images]
        
        # Find matching pairs based on similarity in filenames
        for i, cam1_id in enumerate(cam1_ids):
            best_match = None
            max_similarity = 0
            
            for j, cam2_id in enumerate(cam2_ids):
                # Simple similarity measure: number of matching characters
                similarity = sum(c1 == c2 for c1, c2 in zip(cam1_id, cam2_id))
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = j
            
            if best_match is not None:
                pairs.append((cam1_images[i], cam2_images[best_match]))
    
    # Print the matched pairs for debugging
    print("\nMatched image pairs:")
    for i, (img1, img2) in enumerate(pairs[:5]):  # Print only first 5 pairs
        print(f"Pair {i+1}: {os.path.basename(img1)} - {os.path.basename(img2)}")
    
    if len(pairs) > 5:
        print(f"... and {len(pairs)-5} more pairs")
    
    return pairs

def find_checkerboard_corners(image_path, checkerboard_size=(9, 7), debug_dir=None):
    """Find checkerboard corners in an image"""

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return False, None, None, None
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Try different checkerboard sizes if specified one fails
    alternate_sizes = [(8, 6), (8, 5), (7, 6), (7, 5), (6, 5), (9, 5), (10, 7)]
    
    # First try with specified size
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, 
                                           cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                           cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    used_size = checkerboard_size
    
    # If failed, try alternate sizes
    if not ret:
        print(f"Failed to find {checkerboard_size} checkerboard in {os.path.basename(image_path)}")
      
        
        for alt_size in alternate_sizes:
            
            ret, corners = cv2.findChessboardCorners(gray, alt_size, 
                                                  cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                  cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ret:
                
                used_size = alt_size
                break
    
    if ret:
        # Refine corner locations
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # Draw and save the corners for verification
        img_corners = img.copy()
        cv2.drawChessboardCorners(img_corners, used_size, corners2, ret)
        
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            base_name = os.path.basename(image_path)
            debug_path = os.path.join(debug_dir, f"debug_{base_name}")
            cv2.imwrite(debug_path, img_corners)
            print(f"Saved corner visualization to {debug_path}")
        
        return ret, corners2, img.shape[:2], used_size
    else:
        print(f"Failed to find checkerboard in {os.path.basename(image_path)} with any size")
        
        # Save a copy of the image for manual inspection
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            base_name = os.path.basename(image_path)
            debug_path = os.path.join(debug_dir, f"failed_{base_name}")
            cv2.imwrite(debug_path, img)
            print(f"Saved failed image to {debug_path}")
        
        return False, None, None, None

def calibrate_stereo(cam1_folder, cam2_folder, output_dir, 
                    checkerboard_size=(9, 7), square_size=25.0):
    """
    Perform stereo calibration to find extrinsic parameters between two cameras
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    debug_dir = os.path.join(output_dir, "debug_images")
    os.makedirs(debug_dir, exist_ok=True)
    
    # Load intrinsic parameters
    # Updated to use the path you provided
    camera1_matrix, dist1 = load_intrinsic_params('./multi_view_calib', 1)
    camera2_matrix, dist2 = load_intrinsic_params('./multi_view_calib', 2)
    
    if camera1_matrix is None or camera2_matrix is None:
        print("Error: Could not load intrinsic parameters")
        return None, None
    
    # Find matching stereo pairs
    image_pairs = match_stereo_images(cam1_folder, cam2_folder)
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints1 = []  # 2D points in Camera 1
    imgpoints2 = []  # 2D points in Camera 2
    
    # Process each stereo pair
    successful_pairs = 0
    
    print("\nChecking checkerboard corners in image pairs...")
    for pair_idx, (cam1_img_path, cam2_img_path) in enumerate(image_pairs):
        print(f"\nProcessing pair {pair_idx+1}/{len(image_pairs)}:")
        print(f"  Camera 1: {os.path.basename(cam1_img_path)}")
        print(f"  Camera 2: {os.path.basename(cam2_img_path)}")
        
        # Find checkerboard corners in both images
        ret1, corners1, img1_shape, size1 = find_checkerboard_corners(
            cam1_img_path, checkerboard_size, debug_dir)
        ret2, corners2, img2_shape, size2 = find_checkerboard_corners(
            cam2_img_path, checkerboard_size, debug_dir)
        
        # Check if both images have the same checkerboard size
        if ret1 and ret2:
            if size1 != size2:
                print(f"Warning: Different checkerboard sizes detected ({size1} vs {size2})")
                print("Skipping this pair")
                continue

            # Make sure we have valid image shapes
            if img1_shape is None or img2_shape is None or 0 in img1_shape or 0 in img2_shape:
                print(f"Warning: Invalid image shape detected")
                continue
         
             # Use the first valid image shape for calibration
            img_shape = img1_shape
                
            # Prepare object points for this checkerboard size
            objp = np.zeros((size1[0] * size1[1], 3), np.float32)
            objp[:,:2] = np.mgrid[0:size1[0], 0:size1[1]].T.reshape(-1, 2)
            objp *= square_size  # Convert to actual measurements
            
            objpoints.append(objp)
            imgpoints1.append(corners1)
            imgpoints2.append(corners2)
            successful_pairs += 1
            print(f"Successfully found checkerboard corners in both images")
    
    if successful_pairs < 5:
        print(f"\nWarning: Only {successful_pairs} successful stereo pairs found.")
        if successful_pairs == 0:
            print("Cannot proceed with calibration. Please check your images and checkerboard.")
            return None, None
    
    print(f"\nUsing {successful_pairs} stereo pairs for calibration")
    
    # In your calibrate_stereo function, add this debugging code before the stereo calibration call:

    print(f"Debug - Using image shape for calibration: {img_shape}")
    if img_shape is None or len(img_shape) != 2 or 0 in img_shape:
        print("ERROR: Invalid image shape detected")
        # Pick a default image size from your camera
        img_shape = (1920, 1080)  # Adjust this to match your cameras
        print(f"Using fallback image shape: {img_shape}")


    # Perform stereo calibration
    if successful_pairs > 0:
        try:
            flags = 0
            # flags |= cv2.CALIB_FIX_INTRINSIC  # Use this if you trust your intrinsic calibration
            
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
            
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
            
            # Visualize camera positions
            visualize_camera_positions(R, T, output_dir)
            
            # Compute and save rectification parameters
            compute_rectification(camera1_matrix, dist1, camera2_matrix, dist2, 
                                 R, T, img1_shape, output_dir)
            
            return R, T
            
        except Exception as e:
            print(f"Error during stereo calibration: {e}")
            return None, None
    else:
        print("Error: No successful stereo pairs found")
        return None, None

def compute_rectification(camera1_matrix, dist1, camera2_matrix, dist2, 
                         R, T, image_size, output_dir):
    """Compute and save rectification parameters"""
    try:
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            camera1_matrix, dist1,
            camera2_matrix, dist2,
            image_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0.9)
        
        print("\nRectification Parameters Computed")
        
        # Save rectification parameters
        np.savetxt(os.path.join(output_dir, 'rect_R1.txt'), R1)
        np.savetxt(os.path.join(output_dir, 'rect_R2.txt'), R2)
        np.savetxt(os.path.join(output_dir, 'rect_P1.txt'), P1)
        np.savetxt(os.path.join(output_dir, 'rect_P2.txt'), P2)
        np.savetxt(os.path.join(output_dir, 'disparity_to_depth_matrix.txt'), Q)
    except Exception as e:
        print(f"Error computing rectification parameters: {e}")

def visualize_camera_positions(R, T, output_dir):
    """Visualize the relative positions of two cameras in 3D space"""
    try:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Camera 1 is at the origin
        cam1_pos = np.array([0, 0, 0])
        
        # Camera 2 position is determined by the translation vector
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
        cam1_axis = np.array([0, 0, 100])  # 100mm along Z axis
        ax.quiver(cam1_pos[0], cam1_pos[1], cam1_pos[2], 
                  cam1_axis[0], cam1_axis[1], cam1_axis[2], 
                  color='r', arrow_length_ratio=0.1)
        
        # For Camera 2, the principal axis is rotated according to R
        cam2_axis = R.T @ np.array([0, 0, 100])
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
            np.max([cam1_pos[0], cam2_pos[0]]) - np.min([cam1_pos[0], cam2_pos[0]]),
            np.max([cam1_pos[1], cam2_pos[1]]) - np.min([cam1_pos[1], cam2_pos[1]]),
            np.max([cam1_pos[2], cam2_pos[2]]) - np.min([cam1_pos[2], cam2_pos[2]])
        ])
        
        mid_x = (cam1_pos[0] + cam2_pos[0]) * 0.5
        mid_y = (cam1_pos[1] + cam2_pos[1]) * 0.5
        mid_z = (cam1_pos[2] + cam2_pos[2]) * 0.5
        
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
        
        ax.legend()
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, 'camera_positions.png'))
        plt.close()
        print(f"Camera positions visualization saved to {os.path.join(output_dir, 'camera_positions.png')}")
    except Exception as e:
        print(f"Error visualizing camera positions: {e}")

def main():
    # Directories containing calibration images
    cam1_folder = os.path.join(os.getcwd(), "camera1_calib_images")
    cam2_folder = os.path.join(os.getcwd(), "camera2_calib_images")
    
    # Output directory for stereo calibration results
    output_dir = os.path.join(os.getcwd(), "stereo_calibration_results")
    
    # Print current working directory for debugging
    print(f"Current working directory: {os.getcwd()}")
    print(f"Looking for Camera 1 images in: {cam1_folder}")
    print(f"Looking for Camera 2 images in: {cam2_folder}")
    
    # Check if the calibration image folders exist
    if not os.path.exists(cam1_folder):
        print(f"Error: Camera 1 folder not found: {cam1_folder}")
    if not os.path.exists(cam2_folder):
        print(f"Error: Camera 2 folder not found: {cam2_folder}")
    
    # Checkerboard properties
    checkerboard_size = (9, 7)  # internal corners
    square_size = 25.0  # mm (adjust this to your actual checkerboard)
    
    # Perform stereo calibration
    R, T = calibrate_stereo(cam1_folder, cam2_folder, output_dir, 
                          checkerboard_size, square_size)

if __name__ == "__main__":
    main()