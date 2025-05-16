import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt

def load_calibration_params(base_dir, camera_num):
    """Load camera matrix and distortion coefficients from files"""
    matrix_file = os.path.join(base_dir, f'camera_{camera_num}_matrix.txt')
    dist_file = os.path.join(base_dir, f'camera_{camera_num}_distortion.txt')
    
    if os.path.exists(matrix_file) and os.path.exists(dist_file):
        camera_matrix = np.loadtxt(matrix_file)
        dist_coeffs = np.loadtxt(dist_file)
        return camera_matrix, dist_coeffs
    else:
        print(f"Error: Calibration files for camera {camera_num} not found")
        return None, None

def undistort_images(camera_matrix, dist_coeffs, image_folder, output_folder, camera_num):
    """Undistort all images in a folder using calibration parameters"""
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all images in the folder
    images = glob.glob(os.path.join(image_folder, '*.png')) + \
             glob.glob(os.path.join(image_folder, '*.jpg'))
    
    if len(images) == 0:
        print(f"No images found in {image_folder}")
        return []
    
    # Process each image
    undistorted_pairs = []
    for img_path in images:
        # Read the image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read image: {img_path}")
            continue
        
        # Undistort the image
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1, (w, h))
        
        # Undistort
        dst = cv2.undistort(img, camera_matrix, dist_coeffs, None, newcameramtx)
        
        # Crop the image (optional)
        # x, y, w, h = roi
        # dst = dst[y:y+h, x:x+w]
        
        # Save the undistorted image
        base_name = os.path.basename(img_path)
        output_path = os.path.join(output_folder, f"undistorted_{base_name}")
        cv2.imwrite(output_path, dst)
        
        undistorted_pairs.append((img_path, output_path))
        print(f"Processed {base_name}")
    
    return undistorted_pairs

def compare_images(original_path, undistorted_path):
    """Display original and undistorted images side by side"""
    original = cv2.imread(original_path)
    undistorted = cv2.imread(undistorted_path)
    
    # Convert from BGR to RGB for matplotlib
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    undistorted = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)
    
    # Create a side-by-side comparison
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title('Original')
    plt.imshow(original)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title('Undistorted')
    plt.imshow(undistorted)
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save the comparison
    comparison_dir = os.path.dirname(undistorted_path)
    comparison_path = os.path.join(comparison_dir, f"comparison_{os.path.basename(original_path)}")
    plt.savefig(comparison_path)
    print(f"Saved comparison to {comparison_path}")
    
    # Also display if running in an interactive environment
    plt.show()

def main():
    base_dir = os.getcwd()  # Directory containing calibration files
    
    # Paths to calibration images
    cam1_images = os.path.join(base_dir, "camera1_calib_images")
    cam2_images = os.path.join(base_dir, "camera2_calib_images")
    
    # Output directories for undistorted images
    undistorted_dir1 = os.path.join(base_dir, "camera1_undistorted")
    undistorted_dir2 = os.path.join(base_dir, "camera2_undistorted")
    
    # Load calibration parameters
    print("Loading calibration parameters...")
    cam1_matrix, cam1_dist = load_calibration_params('./multi_view_calib', 1)
    cam2_matrix, cam2_dist = load_calibration_params('./multi_view_calib', 2)
    
    if cam1_matrix is not None:
        print("\nCalibration parameters for Camera 1:")
        print("Camera Matrix:")
        print(cam1_matrix)
        print("\nDistortion Coefficients:")
        print(cam1_dist)
        
        # Undistort Camera 1 images
        print("\nUndistorting Camera 1 images...")
        undistorted_pairs1 = undistort_images(
            cam1_matrix, cam1_dist, cam1_images, undistorted_dir1, 1)
        
        # Compare a few images (first 3)
        for i, (orig, undist) in enumerate(undistorted_pairs1[:3]):
            compare_images(orig, undist)
            if i >= 2:  # Limit to 3 comparisons
                break
    
    if cam2_matrix is not None:
        print("\nCalibration parameters for Camera 2:")
        print("Camera Matrix:")
        print(cam2_matrix)
        print("\nDistortion Coefficients:")
        print(cam2_dist)
        
        # Undistort Camera 2 images
        print("\nUndistorting Camera 2 images...")
        undistorted_pairs2 = undistort_images(
            cam2_matrix, cam2_dist, cam2_images, undistorted_dir2, 2)
        
        # Compare a few images (first 3)
        for i, (orig, undist) in enumerate(undistorted_pairs2[:3]):
            compare_images(orig, undist)
            if i >= 2:  # Limit to 3 comparisons
                break
    
    print("\nVerification complete!")
    print("Check the comparison images to see if straight lines appear straighter in the undistorted images.")
    print("This is a good visual confirmation that your calibration is working correctly.")

if __name__ == "__main__":
    main()