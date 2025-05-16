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

def manual_point_entry(image_path, num_points=4, object_name="object"):
    """Display image and get manual point entry for multiple points"""
    # Make a copy of the image for drawing
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None
        
    height, width = image.shape[:2]
    
    print(f"\nImage dimensions: {width} x {height}")
    print(f"Please enter coordinates for {num_points} points on the {object_name}.")
    print(f"Format: x,y (e.g., 100,200)")
    
    points = []
    
    for i in range(num_points):
        valid_input = False
        while not valid_input:
            try:
                point_str = input(f"Point {i+1} (x,y): ")
                x, y = map(int, point_str.strip().split(','))
                
                # Validate coordinates are within image boundaries
                if 0 <= x < width and 0 <= y < height:
                    points.append((x, y))
                    valid_input = True
                else:
                    print(f"Coordinates out of bounds. Image size is {width} x {height}")
            except ValueError:
                print("Invalid format. Please use x,y (e.g., 100,200)")
    
    # Draw and save the points
    for i, pt in enumerate(points):
        cv2.circle(image, pt, 5, (0, 0, 255), -1)
        cv2.putText(image, str(i+1), (pt[0] + 10, pt[1] + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Draw lines between points to form a polygon
    for i in range(len(points)):
        cv2.line(image, points[i], points[(i+1) % len(points)], (0, 255, 0), 2)
    
    # Save the annotated image
    marked_path = image_path.replace('.png', f'_{object_name}_marked.png')
    cv2.imwrite(marked_path, image)
    print(f"Saved marked image to {marked_path}")
    
    return np.array(points, dtype=np.float32)

def triangulate_points(left_points, right_points, left_matrix, left_dist, right_matrix, right_dist, R, T, left_image, right_image):
    """Triangulate multiple points to 3D coordinates"""
    
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
    
    # Triangulate all points at once
    points4D = cv2.triangulatePoints(P1, P2, left_undist, right_undist)
    
    # Convert to 3D points
    points3D = (points4D[:3] / points4D[3]).T
    
    return points3D

def validate_sphere(points3D, known_circumference):
    """Validate a sphere by checking point distances against known circumference"""
    # Convert circumference to diameter and radius
    diameter = known_circumference / np.pi
    radius = diameter / 2
    
    # Calculate distance between all points
    if len(points3D) < 4:
        print("Need at least 4 points to validate a sphere")
        return None
    
    # Try to fit a sphere to the points
    center = np.mean(points3D, axis=0)
    
    # Calculate distances from center to each point
    distances = np.linalg.norm(points3D - center, axis=1)
    
    # Average radius from measurements
    measured_radius = np.mean(distances)
    measured_diameter = measured_radius * 2
    measured_circumference = measured_diameter * np.pi
    
    # Calculate error
    error_mm = abs(measured_circumference - known_circumference)
    error_percent = 100 * error_mm / known_circumference
    
    result = {
        'center': center,
        'measured_radius': measured_radius,
        'measured_circumference': measured_circumference,
        'known_circumference': known_circumference,
        'error_mm': error_mm,
        'error_percent': error_percent
    }
    
    return result

def validate_square(points3D, known_side_length):
    """Validate a square by checking side lengths against known value"""
    if len(points3D) != 4:
        print("Need exactly 4 points to validate a square")
        return None
    
    # Calculate side lengths (assuming points are in order)
    sides = []
    for i in range(4):
        side_length = np.linalg.norm(points3D[i] - points3D[(i+1) % 4])
        sides.append(side_length)
    
    # Calculate average side length
    measured_side_length = np.mean(sides)
    
    # Calculate error
    error_mm = abs(measured_side_length - known_side_length)
    error_percent = 100 * error_mm / known_side_length
    
    # Calculate diagonals for squareness check
    diagonal1 = np.linalg.norm(points3D[0] - points3D[2])
    diagonal2 = np.linalg.norm(points3D[1] - points3D[3])
    diagonal_ratio = diagonal1 / diagonal2
    
    result = {
        'sides': sides,
        'measured_side_length': measured_side_length,
        'known_side_length': known_side_length,
        'error_mm': error_mm,
        'error_percent': error_percent,
        'diagonal_ratio': diagonal_ratio
    }
    
    return result

def visualize_validation(points3D, results, output_dir, object_type):
    """Visualize the 3D points and validation results"""
    try:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot all points
        ax.scatter(points3D[:, 0], points3D[:, 1], points3D[:, 2], c='r', marker='o', s=100)
        
        # Label points
        for i, point in enumerate(points3D):
            ax.text(point[0], point[1], point[2], f"P{i+1}", fontsize=12)
        
        # Connect points with lines to form a polygon
        for i in range(len(points3D)):
            next_i = (i + 1) % len(points3D)
            ax.plot([points3D[i, 0], points3D[next_i, 0]], 
                    [points3D[i, 1], points3D[next_i, 1]], 
                    [points3D[i, 2], points3D[next_i, 2]], 'g-', linewidth=2)
        
        # If it's a sphere, draw the center
        if object_type == 'sphere' and 'center' in results:
            center = results['center']
            radius = results['measured_radius']
            ax.scatter([center[0]], [center[1]], [center[2]], c='b', marker='o', s=200)
            ax.text(center[0], center[1], center[2], "Center", fontsize=12)
            
            # Create a simple wireframe sphere visualization
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x = center[0] + radius * np.cos(u) * np.sin(v)
            y = center[1] + radius * np.sin(u) * np.sin(v)
            z = center[2] + radius * np.cos(v)
            ax.plot_wireframe(x, y, z, color="b", alpha=0.1)
        
        # Set labels
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        
        # Set title based on object type
        if object_type == 'sphere':
            title = f"Sphere Validation\nMeasured Circumference: {results['measured_circumference']:.2f} mm\n" \
                    f"Known Circumference: {results['known_circumference']:.2f} mm\n" \
                    f"Error: {results['error_percent']:.2f}%"
        elif object_type == 'square':
            title = f"Square Validation\nMeasured Side Length: {results['measured_side_length']:.2f} mm\n" \
                    f"Known Side Length: {results['known_side_length']:.2f} mm\n" \
                    f"Error: {results['error_percent']:.2f}%\n" \
                    f"Diagonal Ratio: {results['diagonal_ratio']:.2f} (1.0 is perfect square)"
        else:
            title = "3D Points Visualization"
            
        ax.set_title(title)
        
        # Make the 3D plot more viewable
        # Calculate the range of data
        x_range = np.ptp(points3D[:, 0])
        y_range = np.ptp(points3D[:, 1])
        z_range = np.ptp(points3D[:, 2])
        
        # Find the centroid
        centroid = np.mean(points3D, axis=0)
        
        # Set equal aspect ratio
        max_range = max(x_range, y_range, z_range) * 0.6
        ax.set_xlim(centroid[0] - max_range, centroid[0] + max_range)
        ax.set_ylim(centroid[1] - max_range, centroid[1] + max_range)
        ax.set_zlim(centroid[2] - max_range, centroid[2] + max_range)
        
        # Save the plot
        plot_path = os.path.join(output_dir, f"{object_type}_validation_3d.png")
        plt.savefig(plot_path)
        print(f"3D visualization saved to {plot_path}")
        
        plt.close()
        
        return plot_path
    except Exception as e:
        print_debug(f"Error creating 3D visualization: {e}")
        return None

def measure_object(cam1_path, cam2_path, cam1_matrix, dist1, cam2_matrix, dist2, R, T, 
                  object_type, dimension_value, num_points=4, output_dir=None):
    """Measure a 3D object using stereo triangulation"""
    print_debug(f"Starting {object_type} measurement")
    
    # Load images
    img1 = cv2.imread(cam1_path)
    img2 = cv2.imread(cam2_path)
    
    if img1 is None or img2 is None:
        print_debug("Failed to load images")
        return
    
    # Get points from user input
    print(f"\n=== Left Camera Image - {object_type.capitalize()} ===")
    print(f"Please mark {num_points} points on the {object_type}")
    points1 = manual_point_entry(cam1_path, num_points, object_type)
    
    if points1 is None:
        print(f"Error with left camera image. Exiting.")
        return
    
    print(f"\n=== Right Camera Image - {object_type.capitalize()} ===")
    print(f"Please mark the SAME {num_points} points in the SAME ORDER")
    points2 = manual_point_entry(cam2_path, num_points, object_type)
    
    if points2 is None:
        print(f"Error with right camera image. Exiting.")
        return
    
    # Triangulate the points
    points3D = triangulate_points(points1, points2, cam1_matrix, dist1, cam2_matrix, dist2, 
                                 R, T, img1, img2)
    
    # Validate based on the object type
    if object_type == 'sphere':
        results = validate_sphere(points3D, dimension_value)
        if results:
            print(f"\n=== Sphere Validation Results ===")
            print(f"Measured circumference: {results['measured_circumference']:.2f} mm")
            print(f"Known circumference: {results['known_circumference']:.2f} mm")
            print(f"Absolute error: {results['error_mm']:.2f} mm")
            print(f"Error percentage: {results['error_percent']:.2f}%")
            print(f"Distance to sphere center: {np.linalg.norm(results['center']):.2f} mm")
    
    elif object_type == 'square':
        results = validate_square(points3D, dimension_value)
        if results:
            print(f"\n=== Square Validation Results ===")
            print(f"Measured side lengths: {[f'{side:.2f}' for side in results['sides']]}")
            print(f"Measured average side length: {results['measured_side_length']:.2f} mm")
            print(f"Known side length: {results['known_side_length']:.2f} mm")
            print(f"Absolute error: {results['error_mm']:.2f} mm")
            print(f"Error percentage: {results['error_percent']:.2f}%")
            print(f"Diagonal ratio: {results['diagonal_ratio']:.2f} (1.0 is perfect square)")
    
    # Calculate distance to center of measured object
    object_center = np.mean(points3D, axis=0)
    distance_to_object = np.linalg.norm(object_center)
    print(f"Distance to object center: {distance_to_object:.2f} mm")
    
    # Visualize the results
    if output_dir:
        visualize_validation(points3D, results, output_dir, object_type)
        
        # Save detailed results to file
        report_path = os.path.join(output_dir, f"{object_type}_validation_report.txt")
        with open(report_path, 'w') as f:
            f.write(f"=== {object_type.capitalize()} Validation Results ===\n\n")
            
            # Write general information
            f.write(f"Object: {object_type.capitalize()}\n")
            f.write(f"Distance to object center: {distance_to_object:.2f} mm\n\n")
            
            # Write object-specific information
            if object_type == 'sphere':
                f.write(f"Measured circumference: {results['measured_circumference']:.2f} mm\n")
                f.write(f"Known circumference: {results['known_circumference']:.2f} mm\n")
                f.write(f"Absolute error: {results['error_mm']:.2f} mm\n")
                f.write(f"Error percentage: {results['error_percent']:.2f}%\n\n")
                f.write(f"Sphere center coordinates (X,Y,Z): {results['center'][0]:.2f}, {results['center'][1]:.2f}, {results['center'][2]:.2f} mm\n")
                f.write(f"Measured radius: {results['measured_radius']:.2f} mm\n")
            
            elif object_type == 'square':
                f.write(f"Measured side lengths:\n")
                for i, side in enumerate(results['sides']):
                    f.write(f"  Side {i+1}: {side:.2f} mm\n")
                f.write(f"Measured average side length: {results['measured_side_length']:.2f} mm\n")
                f.write(f"Known side length: {results['known_side_length']:.2f} mm\n")
                f.write(f"Absolute error: {results['error_mm']:.2f} mm\n")
                f.write(f"Error percentage: {results['error_percent']:.2f}%\n")
                f.write(f"Diagonal ratio: {results['diagonal_ratio']:.2f} (1.0 is perfect square)\n\n")
            
            # Write 3D coordinates of all points
            f.write("3D Coordinates of Marked Points:\n")
            for i, point in enumerate(points3D):
                f.write(f"  Point {i+1}: {point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f} mm\n")
        
        print(f"Detailed report saved to {report_path}")
    
    return points3D, results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Enhanced Object Validator')
    parser.add_argument('--test_dir', required=True, 
                      help='Test directory name (e.g., test_001)')
    parser.add_argument('--base_dir', default='.', 
                      help='Base directory for the project (default: current directory)')
    parser.add_argument('--object_type', choices=['sphere', 'square'], default='sphere',
                      help='Type of object to validate (default: sphere)')
    parser.add_argument('--dimension', type=float, required=True,
                      help='Known dimension in mm (circumference for sphere, side length for square)')
    parser.add_argument('--video_prefix', default='validation',
                      help='Prefix for validation video files (default: validation)')
    parser.add_argument('--num_points', type=int, default=4,
                      help='Number of points to mark on the object (default: 4)')
    
    args = parser.parse_args()
    
    # Set up paths
    base_dir = args.base_dir
    test_path = os.path.join(base_dir, "data", args.test_dir)
    
    print(f"\n=== Enhanced Object Validator - {args.object_type.capitalize()} ===\n")
    
    # Create output directory
    validation_dir = os.path.join(test_path, "results", "validation_results")
    os.makedirs(validation_dir, exist_ok=True)
    
    object_validation_dir = os.path.join(validation_dir, args.object_type)
    os.makedirs(object_validation_dir, exist_ok=True)
    
    temp_dir = os.path.join(test_path, "temp", "validation")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Step 1: Load calibration parameters
    print("\nStep 1: Loading calibration parameters...")
    cam1_matrix, dist1, cam2_matrix, dist2, R, T = load_calibration(test_path)
    
    if cam1_matrix is None:
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
    
    # Step 3: Perform object measurement and validation
    print(f"\nStep 3: Measuring and validating {args.object_type}...")
    measure_object(
        left_frame, right_frame, 
        cam1_matrix, dist1, cam2_matrix, dist2, R, T, 
        args.object_type, args.dimension, args.num_points,
        object_validation_dir
    )
    
    print("\nValidation complete!")

if __name__ == "__main__":
    main()