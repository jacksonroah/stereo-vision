#!/usr/bin/env python3
import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import pickle
from scipy.spatial.distance import pdist, squareform

def print_debug(message):
    """Print debug message and flush output"""
    print(f"DEBUG: {message}")
    sys.stdout.flush()

def load_calibration(test_dir):
    """Load calibration parameters from files"""
    print_debug("Loading calibration parameters...")

    try:
        # Paths to calibration files
        intrinsic_dir = os.path.join(test_dir, "results", "intrinsic_params")
        extrinsic_dir = os.path.join(test_dir, "results", "extrinsic_params")
        
        # First try to load from pickle files (more reliable)
        left_intrinsics_path = os.path.join(intrinsic_dir, "left_intrinsics.pkl")
        right_intrinsics_path = os.path.join(intrinsic_dir, "right_intrinsics.pkl")
        extrinsic_params_path = os.path.join(extrinsic_dir, "extrinsic_params.pkl")
        
        if os.path.exists(left_intrinsics_path) and os.path.exists(right_intrinsics_path):
            # Load intrinsic parameters from pickle
            with open(left_intrinsics_path, 'rb') as f:
                left_camera_matrix, left_dist = pickle.load(f)
            
            with open(right_intrinsics_path, 'rb') as f:
                right_camera_matrix, right_dist = pickle.load(f)
        else:
            # Fall back to loading from text files
            left_camera_matrix = np.loadtxt(os.path.join(intrinsic_dir, "left_matrix.txt"))
            left_dist = np.loadtxt(os.path.join(intrinsic_dir, "left_distortion.txt"))
            right_camera_matrix = np.loadtxt(os.path.join(intrinsic_dir, "right_matrix.txt"))
            right_dist = np.loadtxt(os.path.join(intrinsic_dir, "right_distortion.txt"))
        
        if os.path.exists(extrinsic_params_path):
            # Load extrinsic parameters from pickle
            with open(extrinsic_params_path, 'rb') as f:
                extrinsic_params = pickle.load(f)
                R = extrinsic_params['R']
                T = extrinsic_params['T']
        else:
            # Fall back to loading from text files
            R = np.loadtxt(os.path.join(extrinsic_dir, "stereo_rotation_matrix.txt"))
            T = np.loadtxt(os.path.join(extrinsic_dir, "stereo_translation_vector.txt")).reshape(3, 1)
        
        print_debug("Successfully loaded all calibration parameters")
        return left_camera_matrix, left_dist, right_camera_matrix, right_dist, R, T
    except Exception as e:
        print_debug(f"Error loading calibration files: {e}")
        return None, None, None, None, None, None

def extract_frames(video_path, output_dir, camera_id):
    """Extract a frame from the video"""
    print_debug(f"Extracting frame from {video_path}")
    
    if not os.path.exists(video_path):
        print_debug(f"Video file not found: {video_path}")
        return None
    
    # Create output directory
    camera_dir = os.path.join(output_dir, camera_id)
    os.makedirs(camera_dir, exist_ok=True)
    
    # Open video and read middle frame
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print_debug(f"Could not open video: {video_path}")
        return None
    
    # Get total frames and move to middle
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
    
    # Read the frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print_debug("Failed to read frame")
        return None
    
    # Save the frame
    output_path = os.path.join(camera_dir, f"{camera_id}_frame.png")
    cv2.imwrite(output_path, frame)
    print_debug(f"Saved frame to {output_path}")
    
    return output_path

def detect_sphere(image_path, debug_dir=None):
    """
    Automatically detect a sphere in the image and extract contour points
    
    Args:
        image_path (str): Path to the image
        debug_dir (str, optional): Directory to save debug images
        
    Returns:
        tuple: (center, radius, contour_points)
    """
    print_debug(f"Detecting sphere in {image_path}")
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print_debug(f"Could not read image: {image_path}")
        return None, None, None
    
    # Create a copy for visualization
    vis_image = image.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Use multiple methods to increase detection reliability
    detected = False
    center = None
    radius = None
    contour_points = None
    
    # Method 1: Hough Circle Transform
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=100, 
        param1=50, 
        param2=30, 
        minRadius=20, 
        maxRadius=300
    )
    
    if circles is not None:
        print_debug(f"Detected {len(circles[0])} circles with Hough transform")
        # Convert coordinates to integers
        circles = np.round(circles[0, :]).astype("int")
        
        # Get the largest circle (most likely to be our sphere)
        largest_circle = sorted(circles, key=lambda x: x[2], reverse=True)[0]
        center = (largest_circle[0], largest_circle[1])
        radius = largest_circle[2]
        
        # Draw the detected circle on visualization
        cv2.circle(vis_image, center, radius, (0, 255, 0), 4)
        cv2.circle(vis_image, center, 4, (0, 0, 255), -1)
        detected = True
        
        # Generate points along the circle contour
        contour_points = []
        for angle in np.linspace(0, 2*np.pi, 16, endpoint=False):
            x = int(center[0] + radius * np.cos(angle))
            y = int(center[1] + radius * np.sin(angle))
            contour_points.append((x, y))
            # Mark these points on the visualization
            cv2.circle(vis_image, (x, y), 3, (255, 0, 0), -1)
    
    # If Hough transform fails, try contour-based detection
    if not detected:
        print_debug("Hough transform failed, trying contour detection")
        
        # Try adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 
            11, 
            2
        )
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Check if the contour is roughly circular
            perimeter = cv2.arcLength(largest_contour, True)
            area = cv2.contourArea(largest_contour)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            if circularity > 0.7:  # Threshold for circularity
                print_debug(f"Found circular contour with circularity {circularity:.2f}")
                
                # Fit a minimum enclosing circle
                (x, y), radius = cv2.minEnclosingCircle(largest_contour)
                center = (int(x), int(y))
                radius = int(radius)
                
                # Draw on visualization
                cv2.circle(vis_image, center, radius, (0, 255, 0), 4)
                cv2.circle(vis_image, center, 4, (0, 0, 255), -1)
                
                # Use the contour points directly
                # Sample 16 evenly spaced points from the contour
                step = len(largest_contour) // 16
                if step > 0:
                    contour_points = [tuple(pt[0]) for pt in largest_contour[::step][:16]]
                else:
                    contour_points = [tuple(pt[0]) for pt in largest_contour]
                
                # Mark these points on the visualization
                for pt in contour_points:
                    cv2.circle(vis_image, pt, 3, (255, 0, 0), -1)
                
                detected = True
    
    # Save debug visualization
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        debug_path = os.path.join(debug_dir, os.path.basename(image_path).replace('.png', '_detection.png'))
        cv2.imwrite(debug_path, vis_image)
        print_debug(f"Saved detection visualization to {debug_path}")
    
    if detected:
        contour_points = np.array(contour_points, dtype=np.float32)
        return center, radius, contour_points
    else:
        print_debug("Failed to detect sphere")
        return None, None, None

def triangulate_points(left_points, right_points, left_matrix, left_dist, right_matrix, right_dist, R, T, left_image, right_image):
    """Triangulate points to 3D coordinates"""
    
    # Get image dimensions
    h1, w1 = left_image.shape[:2]
    h2, w2 = right_image.shape[:2]
    
    # Calculate optimal camera matrices
    newcam1, roi1 = cv2.getOptimalNewCameraMatrix(left_matrix, left_dist, (w1, h1), 1, (w1, h1))
    newcam2, roi2 = cv2.getOptimalNewCameraMatrix(right_matrix, right_dist, (w2, h2), 1, (w2, h2))
    
    # Reshape points for undistortion
    left_points_t = left_points.reshape(-1, 1, 2)
    right_points_t = right_points.reshape(-1, 1, 2)
    
    # Undistort points
    left_undist = cv2.undistortPoints(left_points_t, left_matrix, left_dist, None, newcam1)
    right_undist = cv2.undistortPoints(right_points_t, right_matrix, right_dist, None, newcam2)
    
    # Create projection matrices
    P1 = newcam1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = newcam2 @ np.hstack((R, T))
    
    # Reshape for triangulation
    left_undist = left_undist.reshape(-1, 2).T
    right_undist = right_undist.reshape(-1, 2).T
    
    # Triangulate points
    points4D = cv2.triangulatePoints(P1, P2, left_undist, right_undist)
    
    # Convert to 3D points
    points3D = (points4D[:3] / points4D[3]).T
    
    return points3D

def validate_sphere_3d(points3D, known_circumference):
    """Validate a sphere by fitting a sphere to 3D points"""
    print_debug(f"Validating sphere with {len(points3D)} points")
    
    if len(points3D) < 8:
        print_debug("Need at least 8 points for reliable sphere validation")
        return None
    
    # Convert circumference to diameter and radius
    diameter = known_circumference / np.pi
    known_radius = diameter / 2
    
    # Simple sphere fitting: average distance from centroid
    centroid = np.mean(points3D, axis=0)
    distances = np.linalg.norm(points3D - centroid, axis=1)
    measured_radius = np.mean(distances)
    
    # Calculate measured circumference
    measured_circumference = 2 * np.pi * measured_radius
    
    # Calculate error
    error_mm = abs(measured_circumference - known_circumference)
    error_percent = 100 * error_mm / known_circumference
    
    # Calculate distance to sphere center (from origin)
    distance_to_center = np.linalg.norm(centroid)
    
    # Calculate standard deviation of distances to check sphere fit quality
    radius_std = np.std(distances)
    radius_std_percent = 100 * radius_std / measured_radius
    
    result = {
        'center': centroid,
        'center_distance': distance_to_center,
        'measured_radius': measured_radius,
        'known_radius': known_radius,
        'measured_circumference': measured_circumference,
        'known_circumference': known_circumference,
        'error_mm': error_mm,
        'error_percent': error_percent,
        'radius_std': radius_std,
        'radius_std_percent': radius_std_percent,
        'points': points3D
    }
    
    return result

def visualize_sphere_validation(results, output_dir):
    """Visualize sphere validation results in 3D"""
    try:
        center = results['center']
        radius = results['measured_radius']
        points = results['points']
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the points
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='red', marker='o', s=30, label='Detected Points')
        
        # Plot the center
        ax.scatter([center[0]], [center[1]], [center[2]], c='blue', marker='o', s=100, label='Sphere Center')
        
        # Create a wireframe sphere
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = center[0] + radius * np.cos(u) * np.sin(v)
        y = center[1] + radius * np.sin(u) * np.sin(v)
        z = center[2] + radius * np.cos(v)
        ax.plot_wireframe(x, y, z, color="green", alpha=0.2)
        
        # Plot origin (camera position)
        ax.scatter([0], [0], [0], c='black', marker='x', s=100, label='Camera Origin')
        
        # Draw a line from origin to sphere center
        ax.plot([0, center[0]], [0, center[1]], [0, center[2]], 'k--', alpha=0.5)
        
        # Add axis labels and title
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        
        title = f"Sphere Validation - {results['measured_circumference']:.1f}mm vs {results['known_circumference']:.1f}mm\n" \
                f"Error: {results['error_percent']:.2f}%, Distance: {results['center_distance']:.1f}mm"
        ax.set_title(title)
        
        # Equal aspect ratio
        max_range = max([
            np.max(points[:, 0]) - np.min(points[:, 0]),
            np.max(points[:, 1]) - np.min(points[:, 1]),
            np.max(points[:, 2]) - np.min(points[:, 2])
        ]) * 0.6
        
        mid_x = (np.max(points[:, 0]) + np.min(points[:, 0])) / 2
        mid_y = (np.max(points[:, 1]) + np.min(points[:, 1])) / 2
        mid_z = (np.max(points[:, 2]) + np.min(points[:, 2])) / 2
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        ax.legend()
        
        # Save the figure
        plot_path = os.path.join(output_dir, "sphere_validation_3d.png")
        plt.savefig(plot_path)
        print_debug(f"3D visualization saved to {plot_path}")
        
        plt.close()
        return plot_path
    
    except Exception as e:
        print_debug(f"Error creating 3D visualization: {e}")
        return None

def auto_validate_sphere(left_image_path, right_image_path, left_matrix, left_dist, right_matrix, right_dist, 
                        R, T, known_circumference, debug_dir=None, output_dir=None):
    """
    Automatically detect and validate a sphere in stereo images
    
    Args:
        left_image_path, right_image_path: Paths to stereo image pair
        left_matrix, left_dist, right_matrix, right_dist, R, T: Calibration parameters
        known_circumference: Known sphere circumference in mm
        debug_dir: Directory to save debug images
        output_dir: Directory to save results
        
    Returns:
        dict: Validation results
    """
    print("\n=== Automatic Sphere Detection and Validation ===")
    
    # Read images
    left_image = cv2.imread(left_image_path)
    right_image = cv2.imread(right_image_path)
    
    if left_image is None or right_image is None:
        print_debug("Failed to load images")
        return None
    
    # Detect sphere in left image
    print("\n1. Detecting sphere in left image...")
    left_center, left_radius, left_points = detect_sphere(left_image_path, debug_dir)
    
    if left_points is None:
        print("Failed to detect sphere in left image. Try adjusting lighting or camera position.")
        return None
    
    # Detect sphere in right image
    print("\n2. Detecting sphere in right image...")
    right_center, right_radius, right_points = detect_sphere(right_image_path, debug_dir)
    
    if right_points is None:
        print("Failed to detect sphere in right image. Try adjusting lighting or camera position.")
        return None
    
    # Ensure we have the same number of points in both images
    min_points = min(len(left_points), len(right_points))
    left_points = left_points[:min_points]
    right_points = right_points[:min_points]
    
    print(f"\n3. Triangulating {len(left_points)} points...")
    
    # Triangulate points
    points3D = triangulate_points(
        left_points, right_points, 
        left_matrix, left_dist, right_matrix, right_dist, 
        R, T, left_image, right_image
    )
    
    # Validate sphere
    print("\n4. Validating sphere measurements...")
    results = validate_sphere_3d(points3D, known_circumference)
    
    if results is None:
        print("Failed to validate sphere.")
        return None
    
    # Visualize results
    if output_dir:
        print("\n5. Creating visualizations...")
        visualize_sphere_validation(results, output_dir)
        
        # Save detailed report
        report_path = os.path.join(output_dir, "sphere_validation_report.txt")
        with open(report_path, 'w') as f:
            f.write("=== Automatic Sphere Validation Results ===\n\n")
            f.write(f"Known circumference: {results['known_circumference']:.2f} mm\n")
            f.write(f"Measured circumference: {results['measured_circumference']:.2f} mm\n")
            f.write(f"Absolute error: {results['error_mm']:.2f} mm\n")
            f.write(f"Error percentage: {results['error_percent']:.2f}%\n\n")
            
            f.write(f"Sphere center (X,Y,Z): {results['center'][0]:.2f}, {results['center'][1]:.2f}, {results['center'][2]:.2f} mm\n")
            f.write(f"Distance to sphere center: {results['center_distance']:.2f} mm\n")
            f.write(f"Measured radius: {results['measured_radius']:.2f} mm\n")
            f.write(f"Radius standard deviation: {results['radius_std']:.2f} mm ({results['radius_std_percent']:.2f}%)\n\n")
            
            f.write("3D Coordinates of Sphere Points:\n")
            for i, point in enumerate(points3D):
                f.write(f"  Point {i+1}: {point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f} mm\n")
        
        print(f"Detailed report saved to {report_path}")
    
    # Print results
    print("\n=== Sphere Validation Results ===")
    print(f"Measured circumference: {results['measured_circumference']:.2f} mm")
    print(f"Known circumference: {results['known_circumference']:.2f} mm")
    print(f"Error: {results['error_percent']:.2f}%")
    print(f"Distance to sphere center: {results['center_distance']:.2f} mm")
    print(f"Sphere quality (std dev): {results['radius_std_percent']:.2f}% (lower is better)")
    
    return results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Automatic Sphere Validator')
    parser.add_argument('--test_dir', required=True, 
                      help='Test directory name (e.g., test_001)')
    parser.add_argument('--base_dir', default='.', 
                      help='Base directory for the project (default: current directory)')
    parser.add_argument('--circumference', type=float, required=True,
                      help='Known sphere circumference in mm')
    parser.add_argument('--video_prefix', default='validation',
                      help='Prefix for validation video files (default: validation)')
    
    args = parser.parse_args()
    
    # Set up paths
    base_dir = args.base_dir
    test_path = os.path.join(base_dir, "data", args.test_dir)
    
    print("\n=== Automatic Sphere Validator ===\n")
    
    # Create output directories
    validation_dir = os.path.join(test_path, "results", "validation_results")
    os.makedirs(validation_dir, exist_ok=True)
    
    sphere_validation_dir = os.path.join(validation_dir, "sphere_auto")
    os.makedirs(sphere_validation_dir, exist_ok=True)
    
    debug_dir = os.path.join(sphere_validation_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    
    temp_dir = os.path.join(test_path, "temp", "validation")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Step 1: Load calibration parameters
    print("\nStep 1: Loading calibration parameters...")
    left_matrix, left_dist, right_matrix, right_dist, R, T = load_calibration(test_path)
    
    if left_matrix is None:
        print("Failed to load calibration parameters. Exiting.")
        return
    
    # Step 2: Extract frames from videos
    print("\nStep 2: Extracting frames from videos...")
    left_video = os.path.join(test_path, "left_camera", f"{args.video_prefix}_video.mp4")
    right_video = os.path.join(test_path, "right_camera", f"{args.video_prefix}_video.mp4")
    
    # Check for alternate video extensions if not found
    if not os.path.exists(left_video):
        for ext in ['.mov', '.avi', '.MP4', '.MOV']:
            alt_path = os.path.join(test_path, "left_camera", f"{args.video_prefix}_video{ext}")
            if os.path.exists(alt_path):
                left_video = alt_path
                break
    
    if not os.path.exists(right_video):
        for ext in ['.mov', '.avi', '.MP4', '.MOV']:
            alt_path = os.path.join(test_path, "right_camera", f"{args.video_prefix}_video{ext}")
            if os.path.exists(alt_path):
                right_video = alt_path
                break
    
    left_frame = extract_frames(left_video, temp_dir, "left")
    right_frame = extract_frames(right_video, temp_dir, "right")
    
    if left_frame is None or right_frame is None:
        print("Failed to extract frames from videos. Exiting.")
        return
    
    # Step 3: Automatic sphere detection and validation
    print("\nStep 3: Automatic sphere detection and validation...")
    auto_validate_sphere(
        left_frame, right_frame, 
        left_matrix, left_dist, right_matrix, right_dist, R, T, 
        args.circumference, debug_dir, sphere_validation_dir
    )
    
    print("\nAutomatic validation complete!")

if __name__ == "__main__":
    main()