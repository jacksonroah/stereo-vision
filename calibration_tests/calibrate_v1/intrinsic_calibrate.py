import numpy as np
import cv2
import glob
import os
import pickle

def calibrate_single_camera(images_folder, checkerboard_size=(9,7), square_size=26.0):
    """
    Calibrate a single camera using checkerboard images
    
    Parameters:
    images_folder: folder containing calibration images
    checkerboard_size: tuple of (width, height) internal corners
    square_size: size of checkerboard square in mm
    
    Returns:
    camera_matrix, dist_coeffs: camera intrinsic parameters
    """
    print(f"Calibrating camera with images from {images_folder}")
    
    # Prepare object points
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size  # Scale to actual size
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points
    imgpoints = []  # 2D points
    
    # Get all calibration images
    images = glob.glob(os.path.join(images_folder, '*.png')) + \
             glob.glob(os.path.join(images_folder, '*.jpg'))
    
    if len(images) < 10:
        print(f"Warning: Only {len(images)} images found. Recommend at least 15-20 for good calibration.")
    
    # Process each image
    successful_images = 0
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find chess board corners
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, 
                                                cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        if ret:
            objpoints.append(objp)
            
            # Refine corner locations
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            
            # Draw and display corners (optional)
            img_corners = img.copy()
            cv2.drawChessboardCorners(img_corners, checkerboard_size, corners2, ret)
            
            # Save annotated image for verification
            output_dir = os.path.join(images_folder, 'calibration_check')
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.basename(fname)
            cv2.imwrite(os.path.join(output_dir, f"corners_{base_name}"), img_corners)
            
            successful_images += 1
            print(f"Processed {base_name} - found corners: {ret}")
        else:
            print(f"Failed to find corners in {os.path.basename(fname)}")
    
    print(f"Successfully processed {successful_images} out of {len(images)} images")
    
    if successful_images < 5:
        print("ERROR: Not enough successful calibration images!")
        return None, None
    
    # Perform calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    
    # Calculate reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    
    print(f"Calibration complete! Average reprojection error: {mean_error/len(objpoints)}")
    
    # Save the calibration results
    calibration_data = {
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coeffs,
        'rvecs': rvecs,
        'tvecs': tvecs,
        'reprojection_error': mean_error/len(objpoints)
    }
    
    with open(os.path.join(images_folder, 'calibration_results.pkl'), 'wb') as f:
        pickle.dump(calibration_data, f)
    
    return camera_matrix, dist_coeffs

def calibrate_multi_view(camera_folders, output_folder='./multi_view_calib'):
    """
    Calibrate multiple cameras and save their parameters
    
    Parameters:
    camera_folders: list of folders containing images for each camera
    output_folder: where to save the results
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Calibrate each camera individually
    camera_params = []
    for i, folder in enumerate(camera_folders):
        print(f"\n=== Calibrating Camera {i+1} ===")
        camera_matrix, dist_coeffs = calibrate_single_camera(folder)
        if camera_matrix is not None:
            camera_params.append((camera_matrix, dist_coeffs))
            
            # Save individual camera parameters
            np.savetxt(os.path.join(output_folder, f'camera_{i+1}_matrix.txt'), camera_matrix)
            np.savetxt(os.path.join(output_folder, f'camera_{i+1}_distortion.txt'), dist_coeffs)
    
    print(f"\nCalibration completed for {len(camera_params)} cameras")
    print(f"Results saved to {output_folder}")
    
    return camera_params

# Example usage:
if __name__ == "__main__":
    # Specify folders containing calibration images for each camera
    camera1_folder = "./camera1_calib_images"
    camera2_folder = "./camera2_calib_images"
    
    # Run calibration
    camera_params = calibrate_multi_view([camera1_folder, camera2_folder])
