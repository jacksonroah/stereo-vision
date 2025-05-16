#!/usr/bin/env python3
import cv2
import numpy as np
import os
import glob
import re
import argparse
from ultralytics import YOLO
import time
import pickle
from roboflow import Roboflow

def load_calibration(test_dir):
    """Load both intrinsic and extrinsic calibration parameters."""
    results_dir = os.path.join(test_dir, "results")
    intrinsic_dir = os.path.join(results_dir, "intrinsic_params")
    extrinsic_dir = os.path.join(results_dir, "extrinsic_params")
    
    # Check if directories exist
    if not os.path.exists(intrinsic_dir) or not os.path.exists(extrinsic_dir):
        print(f"Error: Calibration directories not found at {results_dir}")
        return None
    
    # Load intrinsic parameters
    try:
        # Try to load from pickle files first
        left_intrinsics_file = os.path.join(intrinsic_dir, "left_intrinsics.pkl")
        right_intrinsics_file = os.path.join(intrinsic_dir, "right_intrinsics.pkl")
        
        if os.path.exists(left_intrinsics_file) and os.path.exists(right_intrinsics_file):
            with open(left_intrinsics_file, 'rb') as f:
                left_matrix, left_dist = pickle.load(f)
            with open(right_intrinsics_file, 'rb') as f:
                right_matrix, right_dist = pickle.load(f)
        else:
            # Fall back to text files
            left_matrix = np.loadtxt(os.path.join(intrinsic_dir, "left_matrix.txt"))
            left_dist = np.loadtxt(os.path.join(intrinsic_dir, "left_distortion.txt"))
            right_matrix = np.loadtxt(os.path.join(intrinsic_dir, "right_matrix.txt"))
            right_dist = np.loadtxt(os.path.join(intrinsic_dir, "right_distortion.txt"))
    except Exception as e:
        print(f"Error loading intrinsic parameters: {e}")
        return None
    
    # Load extrinsic parameters
    try:
        # Try to load from pickle file first
        extrinsic_file = os.path.join(extrinsic_dir, "extrinsic_params.pkl")
        
        if os.path.exists(extrinsic_file):
            with open(extrinsic_file, 'rb') as f:
                extrinsic_data = pickle.load(f)
                R = extrinsic_data['R']
                T = extrinsic_data['T']
        else:
            # Fall back to text files
            R = np.loadtxt(os.path.join(extrinsic_dir, "stereo_rotation_matrix.txt"))
            T = np.loadtxt(os.path.join(extrinsic_dir, "stereo_translation_vector.txt"))
            
            # Reshape T if needed
            if T.shape == (3,):
                T = T.reshape(3, 1)
    except Exception as e:
        print(f"Error loading extrinsic parameters: {e}")
        return None
    
    return {
        'left_matrix': left_matrix,
        'left_dist': left_dist,
        'right_matrix': right_matrix,
        'right_dist': right_dist,
        'R': R,
        'T': T
    }

def find_validation_videos(test_dir):
    """Find validation videos in the test directory."""
    left_dir = os.path.join(test_dir, "left_camera")
    right_dir = os.path.join(test_dir, "right_camera")
    
    # Look for validation videos in each camera directory
    left_videos = []
    right_videos = []
    
    # Handle multiple possible naming conventions
    patterns = [
        "validation*.mp4",
        "validation*.mov",
        "validation*.MP4",
        "validation*.MOV",
        "val_*.mp4",
        "val_*.mov"
    ]
    
    for pattern in patterns:
        left_matches = glob.glob(os.path.join(left_dir, pattern))
        right_matches = glob.glob(os.path.join(right_dir, pattern))
        
        left_videos.extend(left_matches)
        right_videos.extend(right_matches)
    
    # Sort videos by name to ensure matching pairs
    left_videos.sort()
    right_videos.sort()
    
    # Match videos based on similar names
    video_pairs = []
    
    # If same number of videos, assume they match in order
    if len(left_videos) == len(right_videos):
        video_pairs = list(zip(left_videos, right_videos))
    else:
        # Try to match by looking for numeric identifiers
        for left_video in left_videos:
            left_basename = os.path.basename(left_video)
            left_numbers = re.findall(r'\d+', left_basename)
            
            for right_video in right_videos:
                right_basename = os.path.basename(right_video)
                right_numbers = re.findall(r'\d+', right_basename)
                
                if left_numbers and right_numbers and left_numbers[-1] == right_numbers[-1]:
                    video_pairs.append((left_video, right_video))
                    break
    
    print(f"Found {len(video_pairs)} validation video pairs")
    return video_pairs

def extract_frame_pair(left_video, right_video, frame_idx=30):
    """Extract a frame pair from videos at the specified index."""
    left_cap = cv2.VideoCapture(left_video)
    right_cap = cv2.VideoCapture(right_video)
    
    if not left_cap.isOpened() or not right_cap.isOpened():
        print(f"Error: Could not open videos")
        return None, None
    
    # Set position to desired frame
    left_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    right_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    
    # Read frames
    left_ret, left_frame = left_cap.read()
    right_ret, right_frame = right_cap.read()
    
    left_cap.release()
    right_cap.release()
    
    if not left_ret or not right_ret:
        print(f"Error: Could not read frames at position {frame_idx}")
        return None, None
    
    return left_frame, right_frame

# Cache for previously processed images
detection_cache = {}

# Cache for previously processed images
detection_cache = {}

def detect_ball_roboflow(image, model, conf_threshold=0.50):
    # If it's a numpy array (loaded image), save it temporarily
    # Create a unique key for caching (simple hash of the image)
    image_hash = hash(image.data.tobytes())
    
    # Check cache first
    if image_hash in detection_cache:
        return detection_cache[image_hash]
    
    # Need to save temporarily for Roboflow API
    temp_path = f"temp_image_{time.time()}.jpg"
    cv2.imwrite(temp_path, image)
    
    try:
        # Run prediction
        result = model.predict(temp_path, confidence=conf_threshold, overlap=30)
        predictions = result.json()["predictions"]
        
        # Process predictions
        if predictions:
            valid_predictions = []
        
            for pred in predictions:
                cx = pred["x"]
                cy = pred["y"]
                width = pred["width"]
                height = pred["height"]
                conf = pred["confidence"]
                
                # Calculate radius (average of half width and height)
                radius = (width + height) / 4
                
                # Extract region around prediction
                x1 = int(max(0, cx - width/2))
                y1 = int(max(0, cy - height/2))
                x2 = int(min(image.shape[1], cx + width/2))
                y2 = int(min(image.shape[0], cy + height/2))
                
                ball_region = image[y1:y2, x1:x2]
                
                # Check for blue/black colors in the region
                if ball_region.size > 0:
                    hsv = cv2.cvtColor(ball_region, cv2.COLOR_BGR2HSV)
                    
                    # Blue mask (adjust ranges as needed for your specific ball)
                    lower_blue = np.array([100, 50, 50])
                    upper_blue = np.array([130, 255, 255])
                    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
                    
                    # Calculate percentage of blue pixels
                    blue_percent = np.sum(blue_mask > 0) / (width * height) * 100
                    
                    # Only add prediction if it contains enough blue
                    if blue_percent > 10:  # Adjust threshold as needed
                        valid_predictions.append((cx, cy, radius, conf))
        
            # If we have valid color-filtered predictions, use the highest confidence one
            if valid_predictions:
                best_pred = max(valid_predictions, key=lambda x: x[3])  # Sort by confidence (index 3)
                cx, cy, radius, conf = best_pred
            elif predictions:
                # If no color-matching predictions, use original highest confidence
                best_pred = max(predictions, key=lambda x: x["confidence"])
                cx = best_pred["x"]
                cy = best_pred["y"]
                width = best_pred["width"]
                height = best_pred["height"]
                conf = best_pred["confidence"]
                radius = (width + height) / 4
            else:
                return None
            
            # Crop the region around the ball for contour analysis
            crop_x1 = int(max(0, cx - radius*1.5))
            crop_y1 = int(max(0, cy - radius*1.5))
            crop_x2 = int(min(image.shape[1], cx + radius*1.5))
            crop_y2 = int(min(image.shape[0], cy + radius*1.5))
            
            # Check if the crop region is valid
            if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                # Invalid crop region, use original detection
                return (cx, cy, radius, conf)
                
            ball_region = image[crop_y1:crop_y2, crop_x1:crop_x2]
            
            if ball_region.size == 0:
                # Invalid ball region, use original detection
                return (cx, cy, radius, conf)

            # Convert to grayscale and threshold
            try:
                gray_region = cv2.cvtColor(ball_region, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray_region, 100, 255, cv2.THRESH_BINARY)
                
                # Find contours
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Check circularity of contours
                for contour in contours:
                    # Only process contours with some reasonable size
                    if cv2.contourArea(contour) < 100:
                        continue
                        
                    # Calculate area and perimeter
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)
                    
                    # Calculate circularity (4π × area / perimeter²)
                    # Perfect circle has circularity of 1.0
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        
                        # If reasonably circular (adjust threshold as needed)
                        if circularity > 0.7:
                            # Get center and radius from contour
                            (x, y), refined_radius = cv2.minEnclosingCircle(contour)
                            # Convert to original image coordinates
                            refined_cx = x + crop_x1
                            refined_cy = y + crop_y1
                            
                            # Apply size check
                            if 10 <= refined_radius <= 300:  # Reasonable ball size in pixels
                                return (refined_cx, refined_cy, refined_radius, conf)
            except Exception as e:
                print(f"Error in contour analysis: {e}")
                # Fall back to original detection
            
            # If no valid contours found or contour analysis failed, use the original detection
            if 10 <= radius <= 300:  # Reasonable ball size in pixels
                # Cache the result
                result = (cx, cy, radius, conf)
                detection_cache[image_hash] = result
                return result
            
        return None
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

def calculate_3d_position(left_point, right_point, calibration_data):
    """Calculate 3D position of a point from stereo correspondence."""
    # Extract calibration parameters
    left_matrix = calibration_data['left_matrix']
    left_dist = calibration_data['left_dist']
    right_matrix = calibration_data['right_matrix']
    right_dist = calibration_data['right_dist']
    R = calibration_data['R']
    T = calibration_data['T']
    
    # Undistort points
    left_points = np.array([[left_point]], dtype=np.float32)
    right_points = np.array([[right_point]], dtype=np.float32)
    
    left_undistorted = cv2.undistortPoints(left_points, left_matrix, left_dist, P=left_matrix)
    right_undistorted = cv2.undistortPoints(right_points, right_matrix, right_dist, P=right_matrix)
    
    # Convert to format expected by triangulatePoints
    left_point_ud = left_undistorted[0, 0]
    right_point_ud = right_undistorted[0, 0]
    
    # Prepare projection matrices
    P1 = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
    P1 = left_matrix @ P1
    
    P2 = np.hstack((R, T))
    P2 = right_matrix @ P2
    
    # Triangulate point
    points_4d = cv2.triangulatePoints(P1, P2, left_point_ud, right_point_ud)
    
    # Convert from homogeneous coordinates to 3D
    point_3d = points_4d[:3] / points_4d[3]
    
    return point_3d.reshape(3)

def extract_distance_from_filename(filename):
    """Extract distance from filename like 'validation_2000.mp4'."""
    basename = os.path.basename(filename)
    matches = re.findall(r'(\d+)(?=mm|$|\.|_)', basename)
    
    if matches:
        # Get the last number in the filename
        return int(matches[-1])
    
    return None

def process_video_pair(left_video, right_video, model, calibration_data, output_dir):
    """Process a video pair to calculate ball distance."""
    print(f"\nProcessing video pair:")
    print(f"  Left: {os.path.basename(left_video)}")
    print(f"  Right: {os.path.basename(right_video)}")
    
    # Extract a frame from each video (using frame 30 to skip potential initialization frames)
    left_frame, right_frame = extract_frame_pair(left_video, right_video, frame_idx=30)
    
    if left_frame is None or right_frame is None:
        print("Failed to extract frames")
        return None
    
    # Detect ball in each frame
    left_ball = detect_ball_roboflow(left_frame, model)
    if left_ball is None:
        print("No ball detected in left frame")
        return None
    
    right_ball = detect_ball_roboflow(right_frame, model)
    if right_ball is None:
        print("No ball detected in right frame")
        return None
    
    # Extract centers
    left_center = (left_ball[0], left_ball[1])
    right_center = (right_ball[0], right_ball[1])
    
    # Calculate 3D position
    ball_3d = calculate_3d_position(left_center, right_center, calibration_data)
    
    # Calculate distance
    distance = np.linalg.norm(ball_3d)
    
    # Try to extract expected distance from filename
    expected_distance = extract_distance_from_filename(left_video)
    
    # Create visualizations
    left_vis = left_frame.copy()
    right_vis = right_frame.copy()
    
    # Draw detected ball
    cv2.circle(left_vis, (int(left_center[0]), int(left_center[1])), int(left_ball[2]), (0, 255, 0), 2)
    cv2.circle(left_vis, (int(left_center[0]), int(left_center[1])), 5, (0, 0, 255), -1)
    
    cv2.circle(right_vis, (int(right_center[0]), int(right_center[1])), int(right_ball[2]), (0, 255, 0), 2)
    cv2.circle(right_vis, (int(right_center[0]), int(right_center[1])), 5, (0, 0, 255), -1)
    
    # Add text with measurements
    cv2.putText(left_vis, f"Distance: {distance:.1f} mm", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(right_vis, f"Distance: {distance:.1f} mm", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    if expected_distance:
        error = abs(distance - expected_distance)
        error_percent = 100.0 * error / expected_distance
        
        cv2.putText(left_vis, f"Expected: {expected_distance} mm", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(left_vis, f"Error: {error_percent:.1f}%", (50, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.putText(right_vis, f"Expected: {expected_distance} mm", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(right_vis, f"Error: {error_percent:.1f}%", (50, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Save visualizations
    basename = os.path.splitext(os.path.basename(left_video))[0]
    left_output = os.path.join(output_dir, f"{basename}_left.png")
    right_output = os.path.join(output_dir, f"{basename}_right.png")
    
    cv2.imwrite(left_output, left_vis)
    cv2.imwrite(right_output, right_vis)
    
    # Create side-by-side visualization
    combined = np.hstack((left_vis, right_vis))
    combined_output = os.path.join(output_dir, f"{basename}_combined.png")
    cv2.imwrite(combined_output, combined)
    
    # Prepare result data
    result = {
        'left_video': left_video,
        'right_video': right_video,
        'ball_3d': ball_3d,
        'distance': distance,
        'left_center': left_center,
        'right_center': right_center,
        'left_output': left_output,
        'right_output': right_output,
        'combined_output': combined_output
    }
    
    if expected_distance:
        result['expected_distance'] = expected_distance
        result['error'] = error
        result['error_percent'] = error_percent
        print(f"Measured distance: {distance:.1f} mm")
        print(f"Expected distance: {expected_distance} mm")
        print(f"Error: {error:.1f} mm ({error_percent:.1f}%)")
    else:
        print(f"Measured distance: {distance:.1f} mm")
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Ball detection and distance calculation using YOLOv8")
    parser.add_argument("--test_dir", required=True, help="Test directory (e.g., test_001)")
    parser.add_argument("--base_dir", default=".", help="Base directory containing the data folder")
    parser.add_argument("--model", default="yolov8m.pt", help="Path to YOLOv8 model")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold for detection")
    parser.add_argument("--device", default="mps", help="Device to run inference on (cpu, cuda, mps)")
    
    args = parser.parse_args()
    
    # Set full test path
    test_dir = os.path.join(args.base_dir, "data", args.test_dir)
    
    # Check if test directory exists
    if not os.path.exists(test_dir):
        print(f"Error: Test directory {test_dir} not found")
        return
    
    # Create output directory
    output_dir = os.path.join(test_dir, "results", "roboflow_validation_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load calibration data
    calibration_data = load_calibration(test_dir)
    if calibration_data is None:
        print("Failed to load calibration data")
        return
    
    # Load YOLO model
    print(f"Loading YOLOv8 model {args.model}...")
    try:
        # Initialize the Roboflow API client
        rf = Roboflow(api_key="KrpgfctCqJRUl6gEiU9l")
        project = rf.workspace().project("ball-object-detection-5rz1p")
        model = project.version(1).model

        # No need to save locally - will use API but cache results
        print("Using Roboflow ball detection model")
        # Set device for inference
        if args.device == "mps" and not hasattr(model, "to"):
            print("Warning: MPS device setting not directly supported. Model will use available acceleration.")
        else:
            model.to(args.device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print("Model loaded successfully")
    
    # Find validation videos
    video_pairs = find_validation_videos(test_dir)
    
    if not video_pairs:
        print("No validation video pairs found")
        return
    
    # Process each video pair
    results = []
    for left_video, right_video in video_pairs:
        result = process_video_pair(
            left_video, right_video, model, calibration_data, output_dir
        )
        if result:
            results.append(result)
    
    # Calculate overall statistics if we have expected distances
    results_with_expected = [r for r in results if 'expected_distance' in r]
    
    if results_with_expected:
        errors = [r['error_percent'] for r in results_with_expected]
        mean_error = sum(errors) / len(errors)
        
        print("\nOverall Statistics:")
        print(f"Number of validation videos: {len(results_with_expected)}")
        print(f"Mean error: {mean_error:.2f}%")
        print(f"Min error: {min(errors):.2f}%")
        print(f"Max error: {max(errors):.2f}%")
    
    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main()