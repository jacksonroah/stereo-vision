import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

def validate_distance_measurement(left_frames, right_frames, left_corners, right_corners,
                              left_matrix, left_dist, right_matrix, right_dist,
                              R, T, checkerboard_size, square_size, output_dir, 
                              actual_distance=None):
    """
    Validate distance measurement by calculating the distance to a checkerboard.
    
    Args:
        left_frames, right_frames: Paths to frames from left and right cameras
        left_corners, right_corners: Detected checkerboard corners
        left_matrix, left_dist: Left camera intrinsic parameters
        right_matrix, right_dist: Right camera intrinsic parameters
        R, T: Rotation matrix and translation vector from stereo calibration
        checkerboard_size: Size of the checkerboard (width, height)
        square_size: Size of each checkerboard square in mm
        output_dir: Directory to save validation results
        actual_distance: Actual measured distance to checkerboard in mm (if known)
        
    Returns:
        Average measured distance to checkerboard in mm
    """
    
    print("\nValidating distance measurement...")
    
    if len(left_frames) == 0 or len(right_frames) == 0:
        print("Error: No frames provided for validation")
        return None
    
    # Prepare object points (3D coordinates of checkerboard corners in checkerboard coordinate system)
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size  # Scale to actual size in mm
    
    # Get image size from first frame
    img = cv2.imread(left_frames[0])
    if img is None:
        print(f"Error: Could not read frame {left_frames[0]}")
        return None
        
    img_size = img.shape[:2][::-1]  # (width, height)
    
    # Compute rectification parameters
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        left_matrix, left_dist, right_matrix, right_dist, 
        img_size, R, T, alpha=0.0)
    
    # Create undistortion maps
    map1_left, map2_left = cv2.initUndistortRectifyMap(
        left_matrix, left_dist, R1, P1, img_size, cv2.CV_32FC1)
    map1_right, map2_right = cv2.initUndistortRectifyMap(
        right_matrix, right_dist, R2, P2, img_size, cv2.CV_32FC1)
    
    # Calculate distance for each frame pair
    measured_distances = []
    reprojection_errors = []
    
    for i in tqdm(range(min(len(left_frames), len(right_frames), len(left_corners), len(right_corners)))):
        # Read and undistort frames
        left_img = cv2.imread(left_frames[i])
        right_img = cv2.imread(right_frames[i])
        
        if left_img is None or right_img is None:
            continue
            
        left_undistorted = cv2.remap(left_img, map1_left, map2_left, cv2.INTER_LINEAR)
        right_undistorted = cv2.remap(right_img, map1_right, map2_right, cv2.INTER_LINEAR)
        
        # Create a copy for visualization
        left_display = left_undistorted.copy()
        right_display = right_undistorted.copy()
        
        # Extract corners for this frame
        left_frame_corners = left_corners[i]
        right_frame_corners = right_corners[i]
        
        # Undistort corner points
        left_corners_undistorted = cv2.undistortPoints(
            left_frame_corners, left_matrix, left_dist, R=R1, P=P1)
        right_corners_undistorted = cv2.undistortPoints(
            right_frame_corners, right_matrix, right_dist, R=R2, P=P2)
        
        # Triangulate checkerboard corners to get 3D coordinates
        points_3d = []
        for j in range(left_corners_undistorted.shape[0]):
            point_left = left_corners_undistorted[j, 0]
            point_right = right_corners_undistorted[j, 0]
            
            # Triangulate using stereoRectify results
            point_4d = cv2.triangulatePoints(
                P1, P2, 
                point_left.reshape(2, 1), 
                point_right.reshape(2, 1)
            )
            
            # Convert from homogeneous coordinates
            point_3d = (point_4d[:3] / point_4d[3]).reshape(3)
            points_3d.append(point_3d)
        
        points_3d = np.array(points_3d)
        
        # Calculate distance to checkerboard center (average of all corner positions)
        checkerboard_center = np.mean(points_3d, axis=0)
        distance_to_checkerboard = np.linalg.norm(checkerboard_center)
        
        avg_error = calculate_correct_reprojection_error(
            left_corners_undistorted, right_corners_undistorted, 
            points_3d, P1, P2)
        
        # Store results
        measured_distances.append(distance_to_checkerboard)
        reprojection_errors.append(avg_error)
        
        # Visualize for the first few frames (draw the corners and distance)
        if i < 3:  # Limit to first 3 frames for visualization
            # Draw corners and distance on frame
            cv2.drawChessboardCorners(left_display, checkerboard_size, left_frame_corners, True)
            cv2.drawChessboardCorners(right_display, checkerboard_size, right_frame_corners, True)
            
            # Add distance text
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"Distance: {distance_to_checkerboard:.1f} mm"
            cv2.putText(left_display, text, (30, 60), font, 1, (0, 255, 0), 2)
            cv2.putText(right_display, text, (30, 60), font, 1, (0, 255, 0), 2)
            
            # Save visualization
            vis_filename = os.path.join(output_dir, f"distance_validation_{i+1}.png")
            combined = np.hstack((left_display, right_display))
            cv2.imwrite(vis_filename, combined)
    
    if not measured_distances:
        print("Error: Could not calculate distance for any frame")
        return None
    
    # Calculate statistics
    average_distance = np.mean(measured_distances)
    std_distance = np.std(measured_distances)
    median_distance = np.median(measured_distances)
    min_distance = np.min(measured_distances)
    max_distance = np.max(measured_distances)
    
    # Plot distance measurements
    plt.figure(figsize=(10, 6))
    plt.plot(measured_distances, 'bo-', label='Measured Distances')
    plt.axhline(y=average_distance, color='r', linestyle='-', label=f'Average: {average_distance:.1f} mm')
    
    if actual_distance is not None:
        plt.axhline(y=actual_distance, color='g', linestyle='--', label=f'Actual: {actual_distance:.1f} mm')
        
        # Calculate error percentage
        error_percent = 100 * abs(average_distance - actual_distance) / actual_distance
        plt.title(f'Distance to Checkerboard (Error: {error_percent:.2f}%)')
    else:
        plt.title('Distance to Checkerboard')
        
    plt.xlabel('Frame Number')
    plt.ylabel('Distance (mm)')
    plt.grid(True)
    plt.legend()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, 'distance_validation_plot.png'))
    plt.close()
    
    # Create a plot of distance vs. reprojection error
    plt.figure(figsize=(10, 6))
    plt.scatter(measured_distances, reprojection_errors)
    plt.xlabel('Measured Distance (mm)')
    plt.ylabel('Reprojection Error (pixels)')
    plt.title('Distance vs. Reprojection Error')
    plt.grid(True)
    
    # Save plot
    plt.savefig(os.path.join(output_dir, 'distance_vs_error_plot.png'))
    plt.close()
    
    # Save validation results
    with open(os.path.join(output_dir, 'distance_validation_results.txt'), 'w') as f:
        f.write(f"Distance Validation Results\n")
        f.write(f"==========================\n\n")
        f.write(f"Number of frames analyzed: {len(measured_distances)}\n\n")
        f.write(f"Average distance to checkerboard: {average_distance:.2f} mm\n")
        f.write(f"Standard deviation: {std_distance:.2f} mm\n")
        f.write(f"Median distance: {median_distance:.2f} mm\n")
        f.write(f"Range: {min_distance:.2f} - {max_distance:.2f} mm\n\n")
        
        if actual_distance is not None:
            f.write(f"Actual distance: {actual_distance:.2f} mm\n")
            f.write(f"Absolute error: {abs(average_distance - actual_distance):.2f} mm\n")
            f.write(f"Percentage error: {error_percent:.2f}%\n\n")
        
        f.write("Per-frame measurements:\n")
        f.write("Frame\tDistance (mm)\tReprojection Error (px)\n")
        for i, (dist, err) in enumerate(zip(measured_distances, reprojection_errors)):
            f.write(f"{i+1}\t{dist:.2f}\t{err:.4f}\n")
    
    # Print results
    print(f"\nDistance Validation Results:")
    print(f"Average distance to checkerboard: {average_distance:.2f} mm")
    print(f"Standard deviation: {std_distance:.2f} mm")
    print(f"Median distance: {median_distance:.2f} mm")
    print(f"Range: {min_distance:.2f} - {max_distance:.2f} mm")
    
    if actual_distance is not None:
        error_percent = 100 * abs(average_distance - actual_distance) / actual_distance
        print(f"Actual distance: {actual_distance:.2f} mm")
        print(f"Absolute error: {abs(average_distance - actual_distance):.2f} mm")
        print(f"Percentage error: {error_percent:.2f}%")
        
    print(f"\nResults saved to {output_dir}")
    
    return average_distance

def calculate_correct_reprojection_error(left_corners_undistorted, right_corners_undistorted, 
                                         points_3d, P1, P2):
    """
    Calculate correct reprojection error for stereo-triangulated points
    
    Args:
        left_corners_undistorted: Undistorted corner points from left camera
        right_corners_undistorted: Undistorted corner points from right camera
        points_3d: Triangulated 3D points
        P1, P2: Projection matrices from stereoRectify
        
    Returns:
        Average reprojection error in pixels
    """

    # Project 3D points to 2D using projection matrices
    reprojected_left = []
    reprojected_right = []
    
    for pt3d in points_3d:
        # Convert to homogeneous coordinates
        pt_hom = np.append(pt3d, 1.0)
        
        # Project using projection matrices
        pt_left_2d = P1 @ pt_hom
        pt_right_2d = P2 @ pt_hom
        
        # Convert from homogeneous to image coordinates
        pt_left_2d = pt_left_2d[:2] / pt_left_2d[2]
        pt_right_2d = pt_right_2d[:2] / pt_right_2d[2]
        
        reprojected_left.append(pt_left_2d)
        reprojected_right.append(pt_right_2d)
    
    reprojected_left = np.array(reprojected_left).reshape(-1, 1, 2)
    reprojected_right = np.array(reprojected_right).reshape(-1, 1, 2)
    
    # Calculate error as the average Euclidean distance between original and reprojected points
    error_left = np.mean(np.sqrt(np.sum((left_corners_undistorted - reprojected_left)**2, axis=2)))
    error_right = np.mean(np.sqrt(np.sum((right_corners_undistorted - reprojected_right)**2, axis=2)))
    avg_error = (error_left + error_right) / 2
    
    return avg_error

# Add a function to handle finding and loading videos for distance validation
def load_validation_videos(test_dir, validation_video_pattern="validation"):
    """
    Find and load validation videos from both cameras.
    
    Args:
        test_dir: Base test directory
        validation_video_pattern: Pattern to match validation video filenames
        
    Returns:
        tuple: (left_video_path, right_video_path) or (None, None) if not found
    """
    import os
    import glob
    
    print(f"Looking for validation videos with pattern '{validation_video_pattern}'...")
    
    # Define camera directories
    left_camera_dir = os.path.join(test_dir, "left")
    right_camera_dir = os.path.join(test_dir, "right")
    
    # Look for validation videos in each camera directory
    # Try various extensions and directory structures
    extensions = ['.mp4', '.mov', '.MP4', '.MOV', '.avi']
    
    left_video = None
    right_video = None
    
    # Look for videos in the main camera directories
    for ext in extensions:
        left_matches = glob.glob(os.path.join(left_camera_dir, f"*{validation_video_pattern}*{ext}"))
        right_matches = glob.glob(os.path.join(right_camera_dir, f"*{validation_video_pattern}*{ext}"))
        
        if left_matches and not left_video:
            left_video = left_matches[0]
        if right_matches and not right_video:
            right_video = right_matches[0]
    
    # If not found, look in possible subdirectories
    if not left_video or not right_video:
        subdirs = ['raw_video', 'validation', 'videos', 'calibration']
        for subdir in subdirs:
            for ext in extensions:
                if not left_video:
                    left_matches = glob.glob(os.path.join(left_camera_dir, subdir, f"*{validation_video_pattern}*{ext}"))
                    if left_matches:
                        left_video = left_matches[0]
                
                if not right_video:
                    right_matches = glob.glob(os.path.join(right_camera_dir, subdir, f"*{validation_video_pattern}*{ext}"))
                    if right_matches:
                        right_video = right_matches[0]
    
    # Report findings
    if left_video:
        print(f"Found left camera validation video: {os.path.basename(left_video)}")
    else:
        print("No validation video found for left camera")
        
    if right_video:
        print(f"Found right camera validation video: {os.path.basename(right_video)}")
    else:
        print("No validation video found for right camera")
    
    return left_video, right_video