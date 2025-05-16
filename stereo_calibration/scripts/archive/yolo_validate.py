#!/usr/bin/env python3
import cv2
import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import json
import pandas as pd
from datetime import datetime
try:
    from tabulate import tabulate
except ImportError:
    # Simple fallback if tabulate is not installed
    def tabulate(data, headers, tablefmt):
        result = " | ".join(headers) + "\n"
        result += "-" * (len(result)) + "\n"
        for row in data:
            result += " | ".join(str(cell) for cell in row) + "\n"
        return result

try:
    from inference import get_model
    ROBOFLOW_AVAILABLE = True
except ImportError:
    ROBOFLOW_AVAILABLE = False

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

def extract_frames(video_path, output_dir, camera_id, session_id):
    """
    Extract a frame from the video
    
    Args:
        video_path (str): Path to the video file
        output_dir (str): Directory to save extracted frame
        camera_id (str): Left or right camera identifier
        session_id (str): Identifier for this validation session (e.g., distance)
        
    Returns:
        str: Path to the extracted frame
    """
    print_debug(f"Extracting frame from {video_path}")
    
    if not os.path.exists(video_path):
        print_debug(f"Video file not found: {video_path}")
        return None
    
    # Create output directory - create both variants for compatibility
    # First variant: /temp/left/
    camera_dir1 = os.path.join(output_dir, camera_id)
    os.makedirs(camera_dir1, exist_ok=True)
    
    # Second variant: /temp/left_camera/
    camera_dir2 = os.path.join(output_dir, f"{camera_id}_camera")
    os.makedirs(camera_dir2, exist_ok=True)
    
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
    
    # Save the frame with session ID in the filename - to both directories
    output_path1 = os.path.join(camera_dir1, f"{camera_id}_{session_id}_frame.png")
    output_path2 = os.path.join(camera_dir2, f"{camera_id}_{session_id}_frame.png")
    
    cv2.imwrite(output_path1, frame)
    cv2.imwrite(output_path2, frame)
    print_debug(f"Saved frame to {output_path1} and {output_path2}")
    
    return output_path1

def detect_ball_with_opencv(image_path, debug_dir=None, session_id=None):
    """
    Detect a ball in the image using OpenCV's Hough Circle Transform
    
    Args:
        image_path (str): Path to the image
        debug_dir (str, optional): Directory to save debug images
        session_id (str): Session identifier
        
    Returns:
        tuple: (center_x, center_y, radius, confidence)
    """
    print_debug(f"Detecting ball in {image_path} with OpenCV")
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print_debug(f"Could not read image: {image_path}")
        return None, None, None, None
    
    # Create a copy for visualization
    vis_image = image.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Try different parameters for Hough Circle Transform to find circles
    circles = None
    params = [
        # dp, minDist, param1, param2, minRadius, maxRadius
        (1, 100, 50, 30, 20, 300),  # Default
        (1, 100, 50, 20, 20, 300),  # Lower param2 (more circles)
        (1, 100, 100, 30, 20, 300), # Higher param1 (better edges)
        (1.5, 100, 50, 30, 20, 300), # Higher dp
        (1, 50, 50, 30, 20, 300),    # Lower minDist
    ]
    
    for dp, minDist, param1, param2, minRadius, maxRadius in params:
        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=dp, 
            minDist=minDist, 
            param1=param1, 
            param2=param2, 
            minRadius=minRadius, 
            maxRadius=maxRadius
        )
        
        if circles is not None and len(circles[0]) > 0:
            print_debug(f"Found {len(circles[0])} circles with parameters: dp={dp}, param1={param1}, param2={param2}")
            break
    
    if circles is not None:
        # Convert to integers
        circles = np.round(circles[0, :]).astype("int")
        
        # Find the largest circle (most likely the ball)
        largest_circle = sorted(circles, key=lambda x: x[2], reverse=True)[0]
        center_x, center_y, radius = largest_circle
        
        # Calculate a confidence value based on circle quality
        # For OpenCV detection, we don't have a real confidence score,
        # so we use a placeholder value (0.9)
        confidence = 0.9
        
        # Draw the detected circle on visualization
        cv2.circle(vis_image, (center_x, center_y), radius, (0, 255, 0), 4)
        cv2.circle(vis_image, (center_x, center_y), 4, (0, 0, 255), -1)
        
        # Add text
        cv2.putText(vis_image, f"Ball: r={radius}", (center_x - radius, center_y - radius - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Save debug visualization
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            filename = os.path.basename(image_path)
            base_name = os.path.splitext(filename)[0]
            debug_path = os.path.join(debug_dir, f"{base_name}_detection.png")
            cv2.imwrite(debug_path, vis_image)
            print_debug(f"Saved detection visualization to {debug_path}")
        
        return center_x, center_y, radius, confidence
    
    print_debug("No circles found with OpenCV")
    
    # If Hough Circle fails, try contour-based detection
    try:
        # Threshold the image
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Fit a minimum enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            center_x, center_y, radius = int(x), int(y), int(radius)
            
            # Calculate circularity (1.0 is perfect circle)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            print_debug(f"Found contour with circularity {circularity:.2f}")
            
            if circularity > 0.7:  # Reasonable threshold for a ball
                # Use circularity as confidence
                confidence = circularity
                
                # Draw results on visualization
                cv2.circle(vis_image, (center_x, center_y), radius, (0, 255, 0), 4)
                cv2.circle(vis_image, (center_x, center_y), 4, (0, 0, 255), -1)
                cv2.putText(vis_image, f"Ball: {confidence:.2f}", (center_x - radius, center_y - radius - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Save debug visualization
                if debug_dir:
                    os.makedirs(debug_dir, exist_ok=True)
                    filename = os.path.basename(image_path)
                    base_name = os.path.splitext(filename)[0]
                    debug_path = os.path.join(debug_dir, f"{base_name}_contour.png")
                    cv2.imwrite(debug_path, vis_image)
                    print_debug(f"Saved contour detection to {debug_path}")
                
                return center_x, center_y, radius, confidence
    
    except Exception as e:
        print_debug(f"Error in contour detection: {e}")
    
    return None, None, None, None

def detect_ball_with_yolo(image_path, model, debug_dir=None, session_id=None):
    """
    Detect a ball in the image using YOLO v8
    
    Args:
        image_path (str): Path to the image
        model: Loaded YOLO model
        debug_dir (str, optional): Directory to save debug images
        session_id (str): Session identifier
        
    Returns:
        tuple: (center_x, center_y, width, height, confidence)
    """
    print_debug(f"Detecting ball in {image_path} with YOLO")
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print_debug(f"Could not read image: {image_path}")
        return None, None, None, None, None
    
    # Create a copy for visualization
    vis_image = image.copy()
    
    # Run inference
    try:
        results = model.infer(image)[0]
        print_debug(f"YOLO result keys: {dir(results)}")
        
        # Handle different Roboflow API response formats
        # Try to extract detections
        predictions = None
        
        # Try different attributes based on potential response formats
        if hasattr(results, 'predictions'):
            predictions = results.predictions
        elif hasattr(results, 'boxes'):
            # Original expected format
            predictions = results.boxes
        elif hasattr(results, 'json'):
            # Try JSON method if available
            json_data = results.json()
            if 'predictions' in json_data:
                predictions = json_data['predictions']
        
        if predictions is None:
            print_debug("No predictions found in YOLO results")
            return None, None, None, None, None
        
        # Check if any balls were detected
        if len(predictions) == 0:
            print_debug("No balls detected in the image")
            return None, None, None, None, None
        
        # Try to get bbox and confidence based on potentially different formats
        best_bbox = None
        best_confidence = 0
        
        for pred in predictions:
            # Try different potential formats for bounding boxes and confidence
            confidence = None
            bbox = None
            
            # Try to get confidence
            if hasattr(pred, 'confidence'):
                confidence = pred.confidence
            elif hasattr(pred, 'conf'):
                confidence = pred.conf
            elif isinstance(pred, dict) and 'confidence' in pred:
                confidence = pred['confidence']
            
            # Try to get bounding box
            if hasattr(pred, 'bbox'):
                bbox = pred.bbox
            elif hasattr(pred, 'xyxy'):
                bbox = pred.xyxy
            elif isinstance(pred, dict) and 'bbox' in pred:
                bbox = pred['bbox']
                
                # Check if it's in x,y,w,h format or x1,y1,x2,y2
                if len(bbox) == 4 and isinstance(bbox[2], (int, float)) and bbox[2] > 0 and bbox[2] < image.shape[1]:
                    # This might be x1,y1,x2,y2 format, convert to center + width/height
                    x1, y1, x2, y2 = bbox
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    bbox = [center_x, center_y, width, height]
            
            if confidence is not None and confidence > best_confidence and bbox is not None:
                best_confidence = confidence
                best_bbox = bbox
        
        if best_bbox is None:
            print_debug("Could not extract bounding box from YOLO results")
            return None, None, None, None, None
        
        # Extract bounding box coordinates
        # Check if bbox is in x,y,w,h format or x1,y1,x2,y2 format
        if len(best_bbox) == 4 and isinstance(best_bbox[0], (int, float)):
            if isinstance(best_bbox[2], (int, float)) and best_bbox[2] > 0 and best_bbox[2] < image.shape[1]:
                # This is likely x1,y1,x2,y2 format
                x1, y1, x2, y2 = best_bbox
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
            else:
                # This is likely x,y,w,h format
                center_x, center_y, width, height = best_bbox
                x1 = center_x - width/2
                y1 = center_y - height/2
                x2 = center_x + width/2
                y2 = center_y + height/2
        else:
            print_debug(f"Unexpected bbox format: {best_bbox}")
            return None, None, None, None, None
        
        # Draw bounding box and center on visualization
        cv2.rectangle(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.circle(vis_image, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
        
        # Add confidence text
        cv2.putText(vis_image, f"Ball: {best_confidence:.2f}", (int(x1), int(y1) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Save debug visualization
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            filename = os.path.basename(image_path)
            base_name = os.path.splitext(filename)[0]
            debug_path = os.path.join(debug_dir, f"{base_name}_detection.png")
            cv2.imwrite(debug_path, vis_image)
            print_debug(f"Saved detection visualization to {debug_path}")
        
        return center_x, center_y, width, height, best_confidence
    
    except Exception as e:
        print_debug(f"Error in YOLO detection: {e}")
        return None, None, None, None, None

def detect_ball(image_path, model=None, debug_dir=None, session_id=None):
    """
    Detect a ball using available methods (YOLO or OpenCV)
    
    Args:
        image_path: Path to the image
        model: YOLO model (if available)
        debug_dir: Directory for debug images
        session_id: Session identifier
        
    Returns:
        tuple: Detection results
    """
    # Try YOLO first if available
    if model is not None:
        try:
            yolo_result = detect_ball_with_yolo(image_path, model, debug_dir, session_id)
            if yolo_result[0] is not None:
                center_x, center_y, width, height, confidence = yolo_result
                # Convert to radius for consistency
                radius = (width + height) / 4  # Average of half-width and half-height
                return center_x, center_y, radius, confidence
        except Exception as e:
            print_debug(f"YOLO detection failed: {e}")
    
    # Fall back to OpenCV if YOLO is not available or fails
    print_debug("Falling back to OpenCV detection")
    return detect_ball_with_opencv(image_path, debug_dir, session_id)

def triangulate_point(left_point, right_point, left_matrix, left_dist, right_matrix, right_dist, R, T, left_image, right_image):
    """
    Triangulate a single point to get its 3D coordinates
    
    Args:
        left_point: (x, y) coordinates in left image
        right_point: (x, y) coordinates in right image
        left_matrix, left_dist, right_matrix, right_dist, R, T: Calibration parameters
        left_image, right_image: Images for dimensions
        
    Returns:
        ndarray: 3D coordinates (X, Y, Z)
    """
    # Get image dimensions
    h1, w1 = left_image.shape[:2]
    h2, w2 = right_image.shape[:2]
    
    # Calculate optimal camera matrices
    newcam1, roi1 = cv2.getOptimalNewCameraMatrix(left_matrix, left_dist, (w1, h1), 1, (w1, h1))
    newcam2, roi2 = cv2.getOptimalNewCameraMatrix(right_matrix, right_dist, (w2, h2), 1, (w2, h2))
    
    # Reshape points for undistortion
    left_point_array = np.array([[left_point]], dtype=np.float32)
    right_point_array = np.array([[right_point]], dtype=np.float32)
    
    # Undistort points
    left_undist = cv2.undistortPoints(left_point_array, left_matrix, left_dist, None, newcam1)
    right_undist = cv2.undistortPoints(right_point_array, right_matrix, right_dist, None, newcam2)
    
    # Create projection matrices
    P1 = newcam1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = newcam2 @ np.hstack((R, T))
    
    # Reshape for triangulation
    left_undist = left_undist.reshape(-1, 2).T
    right_undist = right_undist.reshape(-1, 2).T
    
    # Triangulate point
    point4D = cv2.triangulatePoints(P1, P2, left_undist, right_undist)
    
    # Convert to 3D point
    point3D = (point4D[:3] / point4D[3]).T
    
    return point3D[0]

def validate_ball(left_detection, right_detection, left_matrix, left_dist, right_matrix, right_dist, 
                 R, T, left_image, right_image, known_diameter, session_id):
    """
    Validate a ball by triangulating the center and estimating the distance
    
    Args:
        left_detection, right_detection: Ball detection results
        left_matrix, left_dist, right_matrix, right_dist, R, T: Calibration parameters
        left_image, right_image: Images for dimensions
        known_diameter: Known ball diameter in mm
        session_id: Session identifier
        
    Returns:
        dict: Validation results
    """
    # Unpack detections
    left_center_x, left_center_y, left_radius, left_conf = left_detection
    right_center_x, right_center_y, right_radius, right_conf = right_detection
    
    # Triangulate the center point
    left_center = (left_center_x, left_center_y)
    right_center = (right_center_x, right_center_y)
    
    center3D = triangulate_point(
        left_center, right_center, 
        left_matrix, left_dist, right_matrix, right_dist, 
        R, T, left_image, right_image
    )
    
    # Calculate distance to ball center
    distance_to_center = np.linalg.norm(center3D)
    
    # Store results
    results = {
        'session_id': session_id,
        'center': center3D,
        'distance': distance_to_center,
        'known_diameter': known_diameter,
        'left_detection': {
            'center_x': left_center_x,
            'center_y': left_center_y,
            'radius': left_radius,
            'confidence': left_conf
        },
        'right_detection': {
            'center_x': right_center_x,
            'center_y': right_center_y,
            'radius': right_radius,
            'confidence': right_conf
        }
    }
    
    return results

def visualize_ball_validation(results, output_dir):
    """
    Visualize ball validation results in 3D
    
    Args:
        results: List of validation result dictionaries
        output_dir: Directory to save visualization
    """
    try:
        if not results:
            print_debug("No results to visualize")
            return False
            
        # Create 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot camera origin
        ax.scatter([0], [0], [0], c='black', marker='x', s=100, label='Camera Origin')
        
        # Different colors for different sessions
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
        
        # Plot each ball center
        for i, result in enumerate(results):
            center = result['center']
            session_id = result['session_id']
            distance = result['distance']
            color = colors[i % len(colors)]
            
            # Plot ball center
            ax.scatter([center[0]], [center[1]], [center[2]], c=color, marker='o', s=100, 
                      label=f"Session {session_id}: {distance:.0f}mm")
            
            # Draw line from origin to center
            ax.plot([0, center[0]], [0, center[1]], [0, center[2]], c=color, linestyle='--', alpha=0.7)
        
        # Set labels and title
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title('Ball Positions at Different Distances')
        
        # Add legend
        ax.legend()
        
        # Make axes equal
        max_val = max([
            max(abs(result['center'][0]) for result in results),
            max(abs(result['center'][1]) for result in results),
            max(abs(result['center'][2]) for result in results)
        ]) * 1.2
        
        ax.set_xlim(-max_val, max_val)
        ax.set_ylim(-max_val, max_val)
        ax.set_zlim(-max_val, max_val)
        
        # Save figure
        plot_path = os.path.join(output_dir, "ball_positions_3d.png")
        plt.savefig(plot_path)
        print_debug(f"Saved 3D visualization to {plot_path}")
        
        plt.close()
        
        # Create a 2D plot of distance vs session
        fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
        
        # Convert session_ids to numeric values if possible 
        distances = [result['distance'] for result in results]
        session_ids = [result['session_id'] for result in results]
        
        # Try to extract numeric values from session_ids
        numeric_session_ids = []
        for sid in session_ids:
            try:
                # Extract numbers from strings like "3000mm"
                numeric_value = ''.join(c for c in str(sid) if c.isdigit())
                if numeric_value:
                    numeric_session_ids.append(int(numeric_value))
                else:
                    # If no digits, use original
                    numeric_session_ids.append(sid)
            except:
                numeric_session_ids.append(sid)
        
        # Sort by numeric session IDs if possible
        if all(isinstance(sid, (int, float)) for sid in numeric_session_ids):
            sorted_indices = np.argsort(numeric_session_ids)
            distances = [distances[i] for i in sorted_indices]
            session_ids = [session_ids[i] for i in sorted_indices]
        
        ax.plot(session_ids, distances, 'o-', linewidth=2, markersize=10)
        
        # Set labels and title
        ax.set_xlabel('Session ID (Distance)')
        ax.set_ylabel('Measured Distance (mm)')
        ax.set_title('Measured Distance vs Session ID')
        
        # Add grid
        ax.grid(True)
        
        # Add value labels
        for i, (sid, dist) in enumerate(zip(session_ids, distances)):
            ax.annotate(f"{dist:.0f}mm", (i, dist), xytext=(0, 10), 
                       textcoords='offset points', ha='center')
        
        # Save figure
        plot_path = os.path.join(output_dir, "distance_vs_session.png")
        plt.savefig(plot_path)
        print_debug(f"Saved distance plot to {plot_path}")
        
        plt.close()
        
        return True
    
    except Exception as e:
        print_debug(f"Error creating visualization: {e}")
        return False

def validate_ball_detection(left_image_path, right_image_path, left_matrix, left_dist, right_matrix, right_dist,
                           R, T, known_diameter, model, debug_dir=None, output_dir=None, session_id=None):
    """
    Detect and validate a ball in stereo images
    
    Args:
        left_image_path, right_image_path: Paths to stereo image pair
        left_matrix, left_dist, right_matrix, right_dist, R, T: Calibration parameters
        known_diameter: Known ball diameter in mm
        model: YOLO model (can be None)
        debug_dir, output_dir: Directories for debug and output files
        session_id: Session identifier (e.g., distance)
        
    Returns:
        dict: Validation results
    """
    print(f"\n=== Ball Detection and Validation (Session {session_id}) ===")
    
    # Read images
    left_image = cv2.imread(left_image_path)
    right_image = cv2.imread(right_image_path)
    
    if left_image is None or right_image is None:
        print_debug("Failed to load images")
        # Try alternate paths with left_camera/right_camera instead of left/right
        if "left/" in left_image_path:
            alt_left_path = left_image_path.replace("left/", "left_camera/")
            print_debug(f"Trying alternate left path: {alt_left_path}")
            left_image = cv2.imread(alt_left_path)
            if left_image is not None:
                left_image_path = alt_left_path
                
        if "right/" in right_image_path:
            alt_right_path = right_image_path.replace("right/", "right_camera/")
            print_debug(f"Trying alternate right path: {alt_right_path}")
            right_image = cv2.imread(alt_right_path)
            if right_image is not None:
                right_image_path = alt_right_path
                
        if left_image is None or right_image is None:
            print_debug("Still failed to load images after trying alternate paths")
            return None
    
    # Detect ball in left image
    print(f"Detecting ball in left image...")
    left_detection = detect_ball(left_image_path, model, debug_dir, session_id)
    
    if left_detection[0] is None:
        print(f"Failed to detect ball in left image for session {session_id}")
        return None
    
    # Detect ball in right image
    print(f"Detecting ball in right image...")
    right_detection = detect_ball(right_image_path, model, debug_dir, session_id)
    
    if right_detection[0] is None:
        print(f"Failed to detect ball in right image for session {session_id}")
        return None
    
    # Validate ball
    print(f"Triangulating ball position...")
    results = validate_ball(
        left_detection, right_detection,
        left_matrix, left_dist, right_matrix, right_dist,
        R, T, left_image, right_image,
        known_diameter, session_id
    )
    
    # Print results
    print(f"\n=== Ball Validation Results (Session {session_id}) ===")
    print(f"Distance to ball center: {results['distance']:.2f} mm")
    print(f"Ball center (X,Y,Z): {results['center'][0]:.2f}, {results['center'][1]:.2f}, {results['center'][2]:.2f} mm")
    print(f"Left detection confidence: {results['left_detection']['confidence']:.2f}")
    print(f"Right detection confidence: {results['right_detection']['confidence']:.2f}")
    
    return results