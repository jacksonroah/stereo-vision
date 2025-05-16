import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_calibration_params(base_dir):
    """
    Load all calibration parameters (intrinsic and extrinsic)
    """
    # Intrinsic parameters
    cam1_matrix = np.loadtxt(os.path.join('./multi_view_calib/camera_1_matrix.txt'))
    cam1_dist = np.loadtxt(os.path.join('./multi_view_calib/camera_1_distortion.txt'))
    cam2_matrix = np.loadtxt(os.path.join('./multi_view_calib/camera_2_matrix.txt'))
    cam2_dist = np.loadtxt(os.path.join('./multi_view_calib/camera_2_distortion.txt'))
    

    # Extrinsic parameters
    R = np.loadtxt(os.path.join('./stereo_calibration_results/stereo_rotation_matrix.txt'))
    T = np.loadtxt(os.path.join('./stereo_calibration_results/stereo_translation_vector.txt')).reshape(3, 1)
    
    return cam1_matrix, cam1_dist, cam2_matrix, cam2_dist, R, T

def find_checkerboard_corners(image_path, checkerboard_size=(9, 7)):
    """Find checkerboard corners in an image"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return False, None, None
        
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
        
        return ret, corners2, img_corners
    
    return False, None, None

def triangulate_points(P1, P2, points1, points2):
    """
    Triangulate 3D points from 2D image points
    
    Parameters:
    P1, P2: Projection matrices for cameras 1 and 2
    points1, points2: 2D points in cameras 1 and 2 (Nx2 array)
    
    Returns:
    points3D: Triangulated 3D points in world coordinate system
    """
    points1 = points1.reshape(-1, 2).T
    points2 = points2.reshape(-1, 2).T
    
    points4D = cv2.triangulatePoints(P1, P2, points1, points2)
    points3D = points4D[:3] / points4D[3]
    
    return points3D.T

def measure_checkerboard_distance(cam1_image, cam2_image, calib_dir, checkerboard_size=(9, 7)):
    """
    Measure the distance to a checkerboard visible in both cameras
    
    Parameters:
    cam1_image, cam2_image: Paths to images from cameras 1 and 2
    calib_dir: Directory containing calibration parameters
    checkerboard_size: Size of the checkerboard (internal corners)
    
    Returns:
    distances: Distance from camera 1 to each checkerboard corner
    avg_distance: Average distance to the checkerboard
    """
    # Load calibration parameters
    cam1_matrix, cam1_dist, cam2_matrix, cam2_dist, R, T = load_calibration_params(calib_dir)
    
    # Create projection matrices
    P1 = cam1_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = cam2_matrix @ np.hstack((R, T))
    
    # Find checkerboard corners in both images
    ret1, corners1, img1_viz = find_checkerboard_corners(cam1_image, checkerboard_size)
    ret2, corners2, img2_viz = find_checkerboard_corners(cam2_image, checkerboard_size)
    
    if not ret1 or not ret2:
        print("Could not find checkerboard in one or both images")
        return None, None, None, None
    
    # Undistort the corner points
    corners1_undistorted = cv2.undistortPoints(corners1, cam1_matrix, cam1_dist, P=cam1_matrix)
    corners2_undistorted = cv2.undistortPoints(corners2, cam2_matrix, cam2_dist, P=cam2_matrix)
    
    # Triangulate 3D points
    points3D = triangulate_points(P1, P2, corners1_undistorted, corners2_undistorted)
    
    # Calculate distances from camera 1 to each point
    distances = np.linalg.norm(points3D, axis=1)
    avg_distance = np.mean(distances)
    
    print(f"Average distance to checkerboard: {avg_distance:.2f} mm")
    print(f"Min distance: {np.min(distances):.2f} mm")
    print(f"Max distance: {np.max(distances):.2f} mm")
    
    # Visualize the 3D points
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot camera positions
    ax.scatter([0], [0], [0], c='r', marker='o', s=100, label='Camera 1')
    ax.scatter([T[0][0]], [T[1][0]], [T[2][0]], c='b', marker='o', s=100, label='Camera 2')
    
    # Plot checkerboard points
    ax.scatter(points3D[:, 0], points3D[:, 1], points3D[:, 2], c='g', marker='.', s=20)
    
    # Connect checkerboard points to form a grid
    rows, cols = checkerboard_size
    for i in range(rows):
        start_idx = i * cols
        end_idx = (i + 1) * cols
        ax.plot(points3D[start_idx:end_idx, 0], 
                points3D[start_idx:end_idx, 1], 
                points3D[start_idx:end_idx, 2], 'g-')
    
    for j in range(cols):
        column_idxs = [i * cols + j for i in range(rows)]
        column_points = points3D[column_idxs]
        ax.plot(column_points[:, 0], column_points[:, 1], column_points[:, 2], 'g-')
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('3D Reconstruction of Checkerboard')
    ax.legend()
    
    plt.savefig(os.path.join(os.path.dirname(cam1_image), 'checkerboard_3d.png'))
    
    return points3D, distances, img1_viz, img2_viz

def main():
    # Directory containing calibration parameters
    calib_dir = './calibrate_v2'
    
    # Images containing checkerboard visible in both cameras
    # These should be a pair of images from cameras 1 and 2 showing the same checkerboard
    cam1_image = './camera1_calib_images/Cam1test2_0004_frame_0000.png'  # Update with your image path
    cam2_image = './camera2_calib_images/Cam2test2_0004_frame_0000.png'  # Update with your image path
    
    # Checkerboard size
    checkerboard_size = (9, 7)  # internal corners
    
    # Measure distance
    points3D, distances, img1_viz, img2_viz = measure_checkerboard_distance(
        cam1_image, cam2_image, calib_dir, checkerboard_size)
    
    if points3D is not None:
        # Save visualization images
        output_dir = './distance_measurement_results'
        os.makedirs(output_dir, exist_ok=True)
        
        if img1_viz is not None and img2_viz is not None:
            cv2.imwrite(os.path.join(output_dir, 'cam1_corners.png'), img1_viz)
            cv2.imwrite(os.path.join(output_dir, 'cam2_corners.png'), img2_viz)
        
        print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()