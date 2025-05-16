import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd

def load_calibration_params():
    """
    Load all calibration parameters (intrinsic and extrinsic)
    """
    # Intrinsic parameters
    cam1_matrix = np.loadtxt('./multi_view_calib/camera_1_matrix.txt')
    cam1_dist = np.loadtxt('./multi_view_calib/camera_1_distortion.txt')
    cam2_matrix = np.loadtxt('./multi_view_calib/camera_2_matrix.txt')
    cam2_dist = np.loadtxt('./multi_view_calib/camera_2_distortion.txt')

    # Extrinsic parameters
    R = np.loadtxt('./stereo_calibration_results/stereo_rotation_matrix.txt')
    T = np.loadtxt('./stereo_calibration_results/stereo_translation_vector.txt').reshape(3, 1)
    
    print("Loaded calibration parameters:")
    print(f"Camera 1 matrix shape: {cam1_matrix.shape}")
    print(f"Camera 2 matrix shape: {cam2_matrix.shape}")
    print(f"Rotation matrix shape: {R.shape}")
    print(f"Translation vector shape: {T.shape}")
    
    return cam1_matrix, cam1_dist, cam2_matrix, cam2_dist, R, T

def extract_frames_from_video(video_path, output_dir, frame_interval=10):
    """Extract frames from a video file at regular intervals"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get video filename
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Processing {video_name} - {total_frames} frames, {fps} FPS")
    
    # Extract frames
    frame_count = 0
    saved_frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save every Nth frame
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"{video_name}_frame_{frame_count:04d}.png")
            cv2.imwrite(frame_path, frame)
            saved_frames.append(frame_path)
            print(f"Saved frame {frame_count} of {total_frames}")
        
        frame_count += 1
    
    cap.release()
    return saved_frames

def detect_ruler_endpoints(image_path):
    """
    Detect the endpoints of a ruler in an image.
    This is a simplified function - in practice, you'd likely need more
    sophisticated computer vision techniques.
    
    Returns (success, endpoints) where endpoints is [(x1,y1), (x2,y2)]
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return False, None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Blur and threshold to isolate the ruler
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area and keep the largest one (presumably the ruler)
    if not contours:
        print(f"No contours found in {image_path}")
        return False, None
    
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    ruler_contour = contours[0]
    
    # Get the bounding rectangle
    x, y, w, h = cv2.boundingRect(ruler_contour)
    
    # Define the endpoints (adjust these based on ruler orientation)
    # For a horizontal ruler:
    if w > h:  # Horizontal ruler
        endpoints = [(x, y + h//2), (x + w, y + h//2)]
    else:  # Vertical ruler
        endpoints = [(x + w//2, y), (x + w//2, y + h)]
    
    # Draw the endpoints for visualization
    img_vis = img.copy()
    cv2.circle(img_vis, endpoints[0], 5, (0, 0, 255), -1)
    cv2.circle(img_vis, endpoints[1], 5, (0, 0, 255), -1)
    cv2.line(img_vis, endpoints[0], endpoints[1], (0, 255, 0), 2)
    
    # Save visualization
    vis_dir = os.path.dirname(image_path)
    vis_path = os.path.join(vis_dir, f"ruler_detect_{os.path.basename(image_path)}")
    cv2.imwrite(vis_path, img_vis)
    
    return True, endpoints

def manual_select_ruler_endpoints(image_path):
    """
    Allow manual selection of ruler endpoints for more accurate measurements.
    """
    # Global variables for storing clicked points
    global endpoints, selecting
    endpoints = []
    selecting = True
    
    # Mouse callback function
    def mouse_callback(event, x, y, flags, param):
        global endpoints, selecting
        if event == cv2.EVENT_LBUTTONDOWN and selecting and len(endpoints) < 2:
            endpoints.append((x, y))
            # Draw point on the image
            cv2.circle(img_display, (x, y), 5, (0, 0, 255), -1)
            if len(endpoints) == 2:
                # Draw line between points
                cv2.line(img_display, endpoints[0], endpoints[1], (0, 255, 0), 2)
            cv2.imshow("Select Ruler Endpoints", img_display)
    
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return False, None
    
    # Create a window and set the callback
    img_display = img.copy()
    cv2.namedWindow("Select Ruler Endpoints")
    cv2.setMouseCallback("Select Ruler Endpoints", mouse_callback)
    
    # Display instructions
    print("\nPlease select the TWO endpoints of the ruler in the image.")
    print("Click on each endpoint. Press 'c' to confirm, 'r' to reset, or 'q' to quit.")
    
    # Show the image and wait for user input
    cv2.imshow("Select Ruler Endpoints", img_display)
    
    while selecting:
        key = cv2.waitKey(1) & 0xFF
        
        # Confirm selection
        if key == ord('c') and len(endpoints) == 2:
            selecting = False
        
        # Reset selection
        elif key == ord('r'):
            endpoints = []
            img_display = img.copy()
            cv2.imshow("Select Ruler Endpoints", img_display)
        
        # Quit without selecting
        elif key == ord('q'):
            selecting = False
            cv2.destroyAllWindows()
            return False, None
    
    cv2.destroyAllWindows()
    
    # Save visualization
    vis_dir = os.path.dirname(image_path)
    vis_path = os.path.join(vis_dir, f"ruler_manual_{os.path.basename(image_path)}")
    cv2.imwrite(vis_path, img_display)
    
    return True, endpoints

def measure_ruler_3d(cam1_image, cam2_image, cam1_matrix, cam1_dist, cam2_matrix, cam2_dist, R, T):
    """
    Measure the 3D position and length of a ruler visible in both camera images
    """
    # For a more reliable real-world application, you would want to
    # implement a robust ruler detection algorithm or use manual selection
    success1, endpoints1 = manual_select_ruler_endpoints(cam1_image)
    success2, endpoints2 = manual_select_ruler_endpoints(cam2_image)
    
    if not success1 or not success2:
        print("Failed to detect ruler in one or both images")
        return None, None, None
    
    # Convert endpoints to numpy arrays
    points1 = np.array(endpoints1, dtype=np.float32)
    points2 = np.array(endpoints2, dtype=np.float32)
    
    # Create projection matrices
    P1 = cam1_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = cam2_matrix @ np.hstack((R, T))
    
    # Undistort the points
    points1_undistorted = cv2.undistortPoints(points1.reshape(-1, 1, 2), cam1_matrix, cam1_dist, P=cam1_matrix)
    points2_undistorted = cv2.undistortPoints(points2.reshape(-1, 1, 2), cam2_matrix, cam2_dist, P=cam2_matrix)
    
    # Reshape for triangulation
    points1_undistorted = points1_undistorted.reshape(-1, 2).T
    points2_undistorted = points2_undistorted.reshape(-1, 2).T
    
    # Triangulate 3D points
    points4D = cv2.triangulatePoints(P1, P2, points1_undistorted, points2_undistorted)
    points3D = (points4D[:3] / points4D[3]).T
    
    # Calculate 3D length
    length_3d = np.linalg.norm(points3D[1] - points3D[0])
    
    # Calculate distances from camera 1 to each endpoint
    distances = np.linalg.norm(points3D, axis=1)
    avg_distance = np.mean(distances)
    
    # Calculate midpoint of ruler
    midpoint = np.mean(points3D, axis=0)
    distance_to_midpoint = np.linalg.norm(midpoint)
    
    print(f"Ruler 3D length: {length_3d:.2f} mm")
    print(f"Distance to ruler midpoint: {distance_to_midpoint:.2f} mm")
    print(f"Distance to endpoint 1: {distances[0]:.2f} mm")
    print(f"Distance to endpoint 2: {distances[1]:.2f} mm")
    
    return length_3d, distance_to_midpoint, points3D

def process_ruler_videos():
    """Process all ruler videos and measure the ruler length"""
    # Create results directory
    results_dir = os.path.join(os.getcwd(), "ruler_measurement_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Load calibration parameters
    cam1_matrix, cam1_dist, cam2_matrix, cam2_dist, R, T = load_calibration_params()
    
    # Find ruler videos
    cam1_videos = glob.glob(os.path.join("videos", "cam1", "data", "ruler", "*.mov"))
    cam2_videos = glob.glob(os.path.join("videos", "cam2", "data", "ruler", "*.mov"))
    
    print(f"Found {len(cam1_videos)} ruler videos for Camera 1")
    print(f"Found {len(cam2_videos)} ruler videos for Camera 2")
    
    # Match video pairs by name similarity
    video_pairs = []
    for cam1_video in cam1_videos:
        cam1_name = os.path.basename(cam1_video)
        best_match = None
        best_similarity = 0
        
        for cam2_video in cam2_videos:
            cam2_name = os.path.basename(cam2_video)
            # Simple similarity: count matching characters
            similarity = sum(c1 == c2 for c1, c2 in zip(cam1_name, cam2_name))
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = cam2_video
        
        if best_match:
            video_pairs.append((cam1_video, best_match))
    
    # Create dataframe to store results
    results = []
    
    # Process each video pair
    for i, (cam1_video, cam2_video) in enumerate(video_pairs):
        print(f"\nProcessing video pair {i+1}/{len(video_pairs)}:")
        print(f"Camera 1: {os.path.basename(cam1_video)}")
        print(f"Camera 2: {os.path.basename(cam2_video)}")
        
        # Create directories for frames
        cam1_frames_dir = os.path.join(results_dir, f"cam1_pair{i+1}_frames")
        cam2_frames_dir = os.path.join(results_dir, f"cam2_pair{i+1}_frames")
        
        # Extract frames
        cam1_frames = extract_frames_from_video(cam1_video, cam1_frames_dir)
        cam2_frames = extract_frames_from_video(cam2_video, cam2_frames_dir)
        
        # Use the middle frame for measurement
        if cam1_frames and cam2_frames:
            mid_idx = min(len(cam1_frames), len(cam2_frames)) // 2
            cam1_frame = cam1_frames[mid_idx]
            cam2_frame = cam2_frames[mid_idx]
            
            print(f"Measuring ruler in frame {mid_idx}...")
            
            # Measure the ruler
            ruler_length, distance, points3D = measure_ruler_3d(
                cam1_frame, cam2_frame, 
                cam1_matrix, cam1_dist, 
                cam2_matrix, cam2_dist, 
                R, T
            )
            
            if ruler_length is not None:
                # Add to results
                results.append({
                    'Pair': i+1,
                    'Camera 1 Video': os.path.basename(cam1_video),
                    'Camera 2 Video': os.path.basename(cam2_video),
                    'Ruler Length (mm)': ruler_length,
                    'Distance to Midpoint (mm)': distance
                })
                
                # Visualize the 3D points
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                # Plot camera positions
                ax.scatter([0], [0], [0], c='r', marker='o', s=100, label='Camera 1')
                ax.scatter([T[0][0]], [T[1][0]], [T[2][0]], c='b', marker='o', s=100, label='Camera 2')
                
                # Plot ruler endpoints
                ax.scatter(points3D[:, 0], points3D[:, 1], points3D[:, 2], c='g', marker='o', s=50)
                ax.plot(points3D[:, 0], points3D[:, 1], points3D[:, 2], 'g-', linewidth=2)
                
                ax.set_xlabel('X (mm)')
                ax.set_ylabel('Y (mm)')
                ax.set_zlabel('Z (mm)')
                ax.set_title(f'3D Reconstruction of Ruler - Pair {i+1}')
                ax.legend()
                
                plt.savefig(os.path.join(results_dir, f'ruler_3d_pair{i+1}.png'))
                plt.close()
    
    # Create summary table
    if results:
        df = pd.DataFrame(results)
        print("\nSummary of Ruler Measurements:")
        print(df)
        
        # Save results to CSV
        csv_path = os.path.join(results_dir, "ruler_measurements.csv")
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
        
        # Calculate real ruler length if provided
        actual_length = float(input("Enter the actual ruler length in mm for error calculation: "))
        df['Error (%)'] = ((df['Ruler Length (mm)'] - actual_length) / actual_length) * 100
        print("\nMeasurement Error Analysis:")
        print(df[['Pair', 'Ruler Length (mm)', 'Error (%)']])
        
        # Save updated results
        df.to_csv(csv_path, index=False)
    else:
        print("No successful measurements were made")

if __name__ == "__main__":
    process_ruler_videos()