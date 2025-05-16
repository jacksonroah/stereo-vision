#!/usr/bin/env python3
import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def print_debug(message):
    """Print debug message and flush output"""
    print(f"DEBUG: {message}")
    sys.stdout.flush()

def load_calibration():
    """Load calibration parameters from files"""
    print_debug("Loading calibration parameters...")

    try:
        # Paths to calibration files
        base_dir = os.getcwd()
        calib_dir = os.path.join(base_dir, "calibration_results")
        stereo_dir = os.path.join(base_dir, "stereo_calibration_results")
        
        # Load intrinsic parameters
        cam1_matrix = np.loadtxt(os.path.join(calib_dir, "cam1_matrix.txt"))
        dist1 = np.loadtxt(os.path.join(calib_dir, "cam1_distortion.txt"))
        cam2_matrix = np.loadtxt(os.path.join(calib_dir, "cam2_matrix.txt"))
        dist2 = np.loadtxt(os.path.join(calib_dir, "cam2_distortion.txt"))
        
        # Load extrinsic parameters
        R = np.loadtxt(os.path.join(stereo_dir, "stereo_rotation_matrix.txt"))
        T = np.loadtxt(os.path.join(stereo_dir, "stereo_translation_vector.txt")).reshape(3, 1)
        
        print_debug("Successfully loaded all calibration parameters")
        return cam1_matrix, dist1, cam2_matrix, dist2, R, T
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

def manual_point_entry(image_path):
    """Display image and get manual point entry"""
    # Make a copy of the image for drawing
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    
    print(f"\nImage dimensions: {width} x {height}")
    print(f"Please enter coordinates for two points on the ruler.")
    print(f"Format: x,y (e.g., 100,200)")
    
    points = []
    
    for i in range(2):
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
    for pt in points:
        cv2.circle(image, pt, 5, (0, 0, 255), -1)
    
    # Draw line between points
    cv2.line(image, points[0], points[1], (0, 255, 0), 2)
    
    # Save the annotated image
    marked_path = image_path.replace('.png', '_marked.png')
    cv2.imwrite(marked_path, image)
    print(f"Saved marked image to {marked_path}")
    
    return np.array(points, dtype=np.float32)

def measure_distance(cam1_path, cam2_path, cam1_matrix, dist1, cam2_matrix, dist2, R, T, ruler_length=None):
    """Measure distance using stereo triangulation"""
    print_debug("Starting distance measurement")
    
    # Load images
    img1 = cv2.imread(cam1_path)
    img2 = cv2.imread(cam2_path)
    
    if img1 is None or img2 is None:
        print_debug("Failed to load images")
        return
    
    # Get points from user input
    print("\n=== Camera 1 Image ===")
    print(f"Please look at the image: {cam1_path}")
    points1 = manual_point_entry(cam1_path)
    
    print("\n=== Camera 2 Image ===")
    print(f"Please look at the image: {cam2_path}")
    points2 = manual_point_entry(cam2_path)
    
    # Undistort points
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Calculate optimal camera matrices
    newcam1, roi1 = cv2.getOptimalNewCameraMatrix(cam1_matrix, dist1, (w1, h1), 1, (w1, h1))
    newcam2, roi2 = cv2.getOptimalNewCameraMatrix(cam2_matrix, dist2, (w2, h2), 1, (w2, h2))
    
    # Reshape points for undistortion
    points1_t = points1.reshape(-1, 1, 2)
    points2_t = points2.reshape(-1, 1, 2)
    
    # Undistort points
    points1_undist = cv2.undistortPoints(points1_t, cam1_matrix, dist1, None, newcam1)
    points2_undist = cv2.undistortPoints(points2_t, cam2_matrix, dist2, None, newcam2)
    
    # Create projection matrices
    P1 = newcam1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = newcam2 @ np.hstack((R, T))
    
    # Reshape for triangulation
    points1_undist = points1_undist.reshape(-1, 2).T
    points2_undist = points2_undist.reshape(-1, 2).T
    
    # Triangulate
    print_debug("Triangulating points")
    points4D = cv2.triangulatePoints(P1, P2, points1_undist, points2_undist)
    
    # Convert to 3D points
    points3D = (points4D[:3] / points4D[3]).T
    
    # Calculate distance between endpoints
    measured_length = np.linalg.norm(points3D[1] - points3D[0])
    
    print(f"\n=== Measurement Results ===")
    print(f"Measured ruler length: {measured_length:.2f} mm")
    
    if ruler_length is not None:
        error_percentage = 100 * abs(measured_length - ruler_length) / ruler_length
        print(f"Known ruler length: {ruler_length:.2f} mm")
        print(f"Error percentage: {error_percentage:.2f}%")
    
    # Calculate distance to object (midpoint)
    midpoint = (points3D[0] + points3D[1]) / 2
    distance_to_object = np.linalg.norm(midpoint)
    print(f"Distance to object: {distance_to_object:.2f} mm")
    
    # Visualize in 3D
    try:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot camera positions
        ax.scatter([0], [0], [0], c='r', marker='o', s=100, label='Camera 1')
        ax.scatter([T[0][0]], [T[1][0]], [T[2][0]], c='b', marker='o', s=100, label='Camera 2')
        
        # Plot measured points
        ax.scatter(points3D[:, 0], points3D[:, 1], points3D[:, 2], c='g', marker='o', s=50)
        
        # Draw line between endpoints
        ax.plot([points3D[0, 0], points3D[1, 0]], 
                [points3D[0, 1], points3D[1, 1]], 
                [points3D[0, 2], points3D[1, 2]], 'g-', linewidth=2)
        
        # Set labels
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(f'3D Measurement - Length: {measured_length:.2f} mm')
        
        ax.legend()
        
        # Save plot
        plt.savefig("3d_measurement.png")
        print(f"3D visualization saved to 3d_measurement.png")
    except Exception as e:
        print_debug(f"Error creating 3D visualization: {e}")

def main():
    """Main function"""
    print("\n=== Simple Distance Validator ===\n")
    
    # Create output directory
    output_dir = os.path.join(os.getcwd(), "simple_validation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load calibration parameters
    print("\nStep 1: Loading calibration parameters...")
    cam1_matrix, dist1, cam2_matrix, dist2, R, T = load_calibration()
    
    if cam1_matrix is None:
        print("Failed to load calibration parameters. Exiting.")
        return
    
    # Step 2: Extract frames from videos
    print("\nStep 2: Extracting frames from videos...")
    video_dir = os.path.join(os.getcwd(), "distance_video")
    cam1_video = os.path.join(video_dir, "ruler1_cam1.mov")
    cam2_video = os.path.join(video_dir, "ruler1_cam2.mov")
    
    cam1_frame = extract_frames(cam1_video, output_dir, "cam1")
    cam2_frame = extract_frames(cam2_video, output_dir, "cam2")
    
    if cam1_frame is None or cam2_frame is None:
        print("Failed to extract frames from videos. Exiting.")
        return
    
    # Step 3: Perform distance measurement
    print("\nStep 3: Measuring distance...")
    measure_distance(cam1_frame, cam2_frame, cam1_matrix, dist1, cam2_matrix, dist2, R, T, ruler_length=310)
    
    print("\nValidation complete!")

if __name__ == "__main__":
    main()