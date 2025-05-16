#!/usr/bin/env python3
import cv2
import numpy as np
import os
import glob
import argparse
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from ultralytics import YOLO
import json
import pickle
from tqdm import tqdm

def load_sync_data(test_dir):
    """
    Load synchronization data from pickle file or from flash detection.
    Returns frame offset between cameras.
    """
    # Try to find sync data file
    sync_dir = os.path.join(test_dir, "results", "sync_results")
    sync_file = os.path.join(sync_dir, "sync_data.pkl")
    
    if os.path.exists(sync_file):
        try:
            with open(sync_file, 'rb') as f:
                sync_data = pickle.load(f)
            
            if 'frame_offset' in sync_data:
                print(f"Using saved synchronization data. Frame offset: {sync_data['frame_offset']}")
                return sync_data['frame_offset']
            
        except Exception as e:
            print(f"Error loading sync data: {e}")
    
    print("No synchronization data found. Running flash detection...")
    
    # Perform flash detection if sync file not found
    from flash_sync import analyze_brightness_jump
    
    # Find left and right videos
    left_video_path = find_specific_video(test_dir, "drop", "left")
    right_video_path = find_specific_video(test_dir, "drop", "right")
    
    if not left_video_path or not right_video_path:
        print("Could not find drop test videos for sync detection")
        return 0
    
    # Detect flash in both videos
    left_flash_frame = analyze_brightness_jump(left_video_path)
    right_flash_frame = analyze_brightness_jump(right_video_path)
    
    if left_flash_frame is not None and right_flash_frame is not None:
        frame_offset = right_flash_frame - left_flash_frame
        print(f"Flash detection successful. Frame offset: {frame_offset}")
        
        # Create directory if it doesn't exist
        os.makedirs(sync_dir, exist_ok=True)
        
        # Save sync data for future use
        sync_data = {
            'left_flash_frame': left_flash_frame,
            'right_flash_frame': right_flash_frame,
            'frame_offset': frame_offset,
            'sync_time': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(sync_file, 'wb') as f:
            pickle.dump(sync_data, f)
        
        return frame_offset
    
    print("Flash detection failed. Using 0 as default offset.")
    return 0

def find_specific_video(test_dir, keyword, camera):
    """Find a video with specific keyword in the filename."""
    camera_dir = os.path.join(test_dir, f"{camera}_camera")
    
    # Try different extensions
    for ext in ['.mp4', '.mov', '.MP4', '.MOV']:
        pattern = os.path.join(camera_dir, f"*{keyword}*{ext}")
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    
    return None

def detect_ball_yolo(image, model, conf_threshold=0.25):
    """
    Detect ball in image using YOLOv8 model.
    Includes improved filtering for circular objects.
    """
    results = model(image, conf=conf_threshold, verbose=False)[0]
    
    # Extract detections
    best_ball = None
    highest_conf = 0
    
    for det in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = det
        
        # We're looking for sports ball (class 32 in COCO)
        # For a custom model, adjust the class check accordingly
        if conf > conf_threshold and (cls == 32 or True):  # Accept any class for custom ball model
            # Calculate center and approximate radius
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            radius = ((x2 - x1) + (y2 - y1)) / 4
            
            # Extract region around detection for circularity check
            margin = radius * 0.5
            crop_x1 = max(0, int(x1 - margin))
            crop_y1 = max(0, int(y1 - margin))
            crop_x2 = min(image.shape[1], int(x2 + margin))
            crop_y2 = min(image.shape[0], int(y2 + margin))
            
            # Skip if crop region is invalid
            if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                continue
                
            region = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
            
            # Skip if region is empty
            if region.size == 0:
                continue
            
            # Convert to grayscale
            try:
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                
                # Apply threshold to separate foreground/background
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Find contours
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Find largest contour
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest_contour)
                    perimeter = cv2.arcLength(largest_contour, True)
                    
                    # Calculate circularity (4π × area / perimeter²)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        
                        # Adjust confidence based on circularity
                        adjusted_conf = conf * (0.5 + 0.5 * circularity)  # Scale based on circularity
                        
                        if adjusted_conf > highest_conf:
                            highest_conf = adjusted_conf
                            best_ball = (cx, cy, radius, adjusted_conf)
            except Exception as e:
                # If circularity check fails, fall back to original detection
                if conf > highest_conf:
                    highest_conf = conf
                    best_ball = (cx, cy, radius, conf)
    
    return best_ball

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

def detect_start_of_motion(positions, num_frames=5, threshold=5.0):
    """
    Detect the frame where significant vertical motion begins.
    
    Args:
        positions: List of 3D positions (x, y, z) for each frame
        num_frames: Number of consecutive frames with motion to confirm start
        threshold: Minimum change in height (mm) to consider as motion
        
    Returns:
        Index of the start frame, or 0 if no clear start detected
    """
    if len(positions) < num_frames + 1:
        return 0
    
    # Convert to numpy array for easier calculations
    if not isinstance(positions, np.ndarray):
        positions = np.array(positions)
    
    # Calculate frame-to-frame vertical displacement (y-axis in 3D space)
    y_positions = positions[:, 1]  # Y is usually up-down in 3D space
    y_diff = np.abs(np.diff(y_positions))
    
    # Look for consecutive frames with significant motion
    for i in range(len(y_diff) - num_frames + 1):
        if all(y_diff[i:i+num_frames] > threshold):
            # Found consistent motion, return the frame before motion starts
            return max(0, i - 1)
    
    # If no clear start found, look for the first significant motion
    for i, diff in enumerate(y_diff):
        if diff > threshold * 2:  # Use higher threshold for single frame detection
            return i
    
    return 0  # Default to first frame if no motion detected

def analyze_ball_drop(left_video, right_video, model, calibration_data, frame_offset, 
                     output_dir, delay, start_frame=None, max_frames=120):
    """
    Analyze a ball drop experiment from synchronized stereo videos.
    
    Args:
        left_video: Path to left camera video
        right_video: Path to right camera video
        model: YOLOv8 model for ball detection
        calibration_data: Calibration parameters
        frame_offset: Frame offset between cameras
        output_dir: Directory to save results
        start_frame: Frame to start analysis (after flash + margin)
        max_frames: Maximum number of frames to analyze
    """
    print(f"\nAnalyzing ball drop from videos:")
    print(f"  Left: {os.path.basename(left_video)}")
    print(f"  Right: {os.path.basename(right_video)}")
    
    # Open videos
    left_cap = cv2.VideoCapture(left_video)
    right_cap = cv2.VideoCapture(right_video)
    
    if not left_cap.isOpened() or not right_cap.isOpened():
        print(f"Error: Could not open videos")
        return None
    
    # Get video properties
    left_fps = left_cap.get(cv2.CAP_PROP_FPS)
    right_fps = right_cap.get(cv2.CAP_PROP_FPS)
    left_frame_count = int(left_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    right_frame_count = int(right_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info - Left: {left_fps:.2f} fps, {left_frame_count} frames")
    print(f"Video info - Right: {right_fps:.2f} fps, {right_frame_count} frames")
    
    # Determine the frame to start from (after flash + delay)
    if start_frame is None:
        # Try to load sync data to find flash frame
        sync_dir = os.path.join(os.path.dirname(os.path.dirname(left_video)), "results", "sync_results")
        sync_file = os.path.join(sync_dir, "sync_data.pkl")
        
        if os.path.exists(sync_file):
            try:
                with open(sync_file, 'rb') as f:
                    sync_data = pickle.load(f)
                
                if 'left_flash_frame' in sync_data:
                    # Start 3 seconds after the flash
                    delay_frames = int(delay * left_fps)
                    start_frame = sync_data['left_flash_frame'] + delay_frames
                    print(f"Starting analysis at frame {start_frame} ({delay} seconds after flash)")
            except Exception as e:
                print(f"Error reading sync data: {e}")
    
    if start_frame is None:
        # Default to frame 90 (about 3 seconds at 30fps) if no sync data
        start_frame = 90
        print(f"No sync data found. Starting at default frame {start_frame}")
    
    # Set video positions to start frame
    left_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    right_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + frame_offset)
    
    # For visualization
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize data storage
    frames = []
    positions_3d = []
    timestamps = []
    left_points = []
    right_points = []
    ball_detections = []
    
    frame_idx = 0
    processing_frames = min(max_frames, left_frame_count - start_frame, 
                          right_frame_count - (start_frame + frame_offset))
    
    # Process frames
    pbar = tqdm(total=processing_frames, desc="Processing frames")
    
    while frame_idx < processing_frames:
        left_ret, left_frame = left_cap.read()
        right_ret, right_frame = right_cap.read()
        
        if not left_ret or not right_ret:
            break
        
        # Detect ball in both frames
        left_ball = detect_ball_yolo(left_frame, model)
        right_ball = detect_ball_yolo(right_frame, model)
        
        if left_ball and right_ball:
            # Record frame index and timestamp
            frames.append(frame_idx)
            timestamps.append(frame_idx / left_fps)
            
            # Extract ball centers
            left_center = (left_ball[0], left_ball[1])
            right_center = (right_ball[0], right_ball[1])
            left_points.append(left_center)
            right_points.append(right_center)
            
            # Calculate 3D position
            pos_3d = calculate_3d_position(left_center, right_center, calibration_data)
            positions_3d.append(pos_3d)
            
            # Save detection data
            ball_detections.append({
                'frame': frame_idx + start_frame,
                'timestamp': frame_idx / left_fps,
                'left_center': left_center,
                'right_center': right_center,
                'left_radius': left_ball[2],
                'right_radius': right_ball[2],
                'left_conf': left_ball[3],
                'right_conf': right_ball[3],
                'position_3d': pos_3d.tolist()
            })
            
            # Save visualization of current frame
            if frame_idx % 5 == 0:  # Save every 5th frame
                vis_frame = np.hstack((left_frame, right_frame))
                cv2.circle(left_frame, (int(left_center[0]), int(left_center[1])), 
                          int(left_ball[2]), (0, 255, 0), 2)
                cv2.circle(right_frame, (int(right_center[0]), int(right_center[1])), 
                          int(right_ball[2]), (0, 255, 0), 2)
                vis_frame = np.hstack((left_frame, right_frame))
                
                # Add frame number and 3D position
                cv2.putText(vis_frame, f"Frame: {frame_idx + start_frame}", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                pos_text = f"Pos: ({pos_3d[0]:.1f}, {pos_3d[1]:.1f}, {pos_3d[2]:.1f}) mm"
                cv2.putText(vis_frame, pos_text, (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                vis_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg")
                cv2.imwrite(vis_path, vis_frame)
        
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    left_cap.release()
    right_cap.release()
    
    # Convert positions to numpy array for analysis
    if positions_3d:
        positions_3d = np.array(positions_3d)
        
        # Detect start of drop if we have enough frames
        if len(positions_3d) > 10:
            drop_start_idx = detect_start_of_motion(positions_3d)
            print(f"Detected start of drop at frame {drop_start_idx} from analysis start")
            
            # Calculate gravity from ball trajectory
            if drop_start_idx < len(positions_3d) - 10:
                # Extract drop segment
                drop_positions = positions_3d[drop_start_idx:]
                drop_frames = frames[drop_start_idx:]
                drop_times = [t - timestamps[drop_start_idx] for t in timestamps[drop_start_idx:]]
                
                # Calculate velocities
                velocities = []
                for i in range(1, len(drop_times)):
                    dt = drop_times[i] - drop_times[i-1]
                    if dt > 0:
                        velocity = (drop_positions[i] - drop_positions[i-1]) / dt
                        velocities.append(velocity)
                
                # Calculate accelerations
                accelerations = []
                for i in range(1, len(velocities)):
                    dt = drop_times[i] - drop_times[i-1]
                    if dt > 0:
                        accel = (velocities[i] - velocities[i-1]) / dt
                        accelerations.append(accel)
                
                # Calculate average gravity
                if accelerations:
                    # Assume Y-axis is vertical (may vary based on camera setup)
                    gravity_measurements = [-a[1] for a in accelerations]  # Negative because Y increases downward
                    gravity_avg = sum(gravity_measurements) / len(gravity_measurements)
                    
                    print(f"Calculated gravity: {gravity_avg:.2f} mm/s² (Expected: 9800 mm/s²)")
                    gravity_error = abs(gravity_avg - 9800) / 9800 * 100
                    print(f"Gravity measurement error: {gravity_error:.2f}%")
        
        # Create motion path visualization
        create_trajectory_visualization(positions_3d, left_points, right_points, 
                                       timestamps, output_dir)
        
        # Save all data
        save_drop_analysis_results(frames, timestamps, positions_3d, 
                                  left_points, right_points, ball_detections, 
                                  output_dir)
        
        print(f"\nBall drop analysis complete. Results saved to {output_dir}")
    else:
        print("No ball detections found in video pair")

def save_drop_analysis_results(frames, timestamps, positions_3d, 
                              left_points, right_points, ball_detections,
                              output_dir):
    """Save analysis results to files."""
    # Save raw detection data
    detections_file = os.path.join(output_dir, "ball_detections.json")
    with open(detections_file, 'w') as f:
        json.dump(ball_detections, f, indent=2)
    
    # Save trajectory data as CSV
    csv_file = os.path.join(output_dir, "ball_trajectory.csv")
    with open(csv_file, 'w') as f:
        # Write header
        f.write("frame,time,x,y,z,left_x,left_y,right_x,right_y\n")
        
        # Write data
        for i in range(len(frames)):
            f.write(f"{frames[i]},{timestamps[i]:.4f},{positions_3d[i][0]:.2f},"
                    f"{positions_3d[i][1]:.2f},{positions_3d[i][2]:.2f},"
                    f"{left_points[i][0]:.2f},{left_points[i][1]:.2f},"
                    f"{right_points[i][0]:.2f},{right_points[i][1]:.2f}\n")
    
    # Calculate and save velocities
    velocities = []
    for i in range(1, len(timestamps)):
        dt = timestamps[i] - timestamps[i-1]
        if dt > 0:
            velocity = (positions_3d[i] - positions_3d[i-1]) / dt
            velocities.append(velocity)
    
    if velocities:
        velocity_file = os.path.join(output_dir, "ball_velocity.csv")
        with open(velocity_file, 'w') as f:
            # Write header
            f.write("frame,time,vx,vy,vz,speed\n")
            
            # Write data
            for i in range(len(velocities)):
                v = velocities[i]
                speed = np.linalg.norm(v)
                f.write(f"{frames[i+1]},{timestamps[i+1]:.4f},"
                        f"{v[0]:.2f},{v[1]:.2f},{v[2]:.2f},{speed:.2f}\n")
    
    # Create summary report
    report_file = os.path.join(output_dir, "analysis_report.txt")
    with open(report_file, 'w') as f:
        f.write("Ball Drop Analysis Report\n")
        f.write("=========================\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total frames analyzed: {len(frames)}\n")
        f.write(f"Duration: {timestamps[-1]:.2f} seconds\n\n")
        
        f.write("Trajectory Information:\n")
        f.write(f"  Starting position (mm): ({positions_3d[0][0]:.2f}, {positions_3d[0][1]:.2f}, {positions_3d[0][2]:.2f})\n")
        f.write(f"  Final position (mm): ({positions_3d[-1][0]:.2f}, {positions_3d[-1][1]:.2f}, {positions_3d[-1][2]:.2f})\n")
        
        # Calculate total displacement
        total_displacement = np.linalg.norm(positions_3d[-1] - positions_3d[0])
        f.write(f"  Total displacement (mm): {total_displacement:.2f}\n")
        
        if velocities:
            # Find maximum velocity
            speeds = [np.linalg.norm(v) for v in velocities]
            max_speed = max(speeds)
            max_speed_idx = speeds.index(max_speed)
            
            f.write("\nVelocity Information:\n")
            f.write(f"  Maximum speed (mm/s): {max_speed:.2f}\n")
            f.write(f"  At time (s): {timestamps[max_speed_idx+1]:.2f}\n")
            
            # Estimate drop height if possible
            max_height_diff = max([p1[1] - p2[1] for p1, p2 in zip(positions_3d[:-1], positions_3d[1:])])
            if max_height_diff > 0:
                f.write(f"  Estimated drop height (mm): {max_height_diff:.2f}\n")
                
                # Calculate theoretical velocity for this height
                theoretical_v = np.sqrt(2 * 9800 * max_height_diff/1000) * 1000  # mm/s
                f.write(f"  Theoretical maximum speed (mm/s): {theoretical_v:.2f}\n")
                
                velocity_error = abs(max_speed - theoretical_v) / theoretical_v * 100
                f.write(f"  Velocity error: {velocity_error:.2f}%\n")

def create_trajectory_visualization(positions_3d, left_points, right_points, timestamps, output_dir):
    """Create visualizations of the ball trajectory."""
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to numpy arrays
    positions_3d = np.array(positions_3d)
    left_points = np.array(left_points)
    right_points = np.array(right_points)
    timestamps = np.array(timestamps)
    
    # 1. 3D Trajectory Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot 3D trajectory
    ax.scatter(positions_3d[:, 0], positions_3d[:, 2], -positions_3d[:, 1], c=timestamps, cmap='viridis', s=30)
    ax.plot(positions_3d[:, 0], positions_3d[:, 2], -positions_3d[:, 1], 'r-', alpha=0.7)
    
    # Mark start and end points
    ax.scatter(positions_3d[0, 0], positions_3d[0, 2], -positions_3d[0, 1], color='green', s=100, label='Start')
    ax.scatter(positions_3d[-1, 0], positions_3d[-1, 2], -positions_3d[-1, 1], color='red', s=100, label='End')
    
    # Set labels and title
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Z (mm)')
    ax.set_zlabel('Y (mm)')
    ax.set_title('3D Ball Trajectory')
    ax.legend()
    
    # Add colorbar for time
    cbar = plt.colorbar(ax.scatter(positions_3d[:, 0], positions_3d[:, 2], -positions_3d[:, 1], 
                                  c=timestamps, cmap='viridis', s=0))
    cbar.set_label('Time (s)')
    
    # Save figure
    plt.savefig(os.path.join(output_dir, "3d_trajectory.png"), dpi=300)
    plt.close()
    
    # 2. Position vs Time Plots
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Plot X position
    axes[0].plot(timestamps, positions_3d[:, 0], 'b-')
    axes[0].set_ylabel('X Position (mm)')
    axes[0].set_title('Ball Position vs Time')
    axes[0].grid(True)
    
    # Plot Y position (vertical)
    axes[1].plot(timestamps, positions_3d[:, 1], 'g-')
    axes[1].set_ylabel('Y Position (mm)')
    axes[1].grid(True)
    
    # Plot Z position (depth)
    axes[2].plot(timestamps, positions_3d[:, 2], 'r-')
    axes[2].set_ylabel('Z Position (mm)')
    axes[2].set_xlabel('Time (s)')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "position_vs_time.png"), dpi=300)
    plt.close()
    
    # 3. Calculate and plot velocity
    if len(timestamps) > 1:
        # Calculate velocities
        velocities = []
        velocity_times = []
        
        for i in range(1, len(timestamps)):
            dt = timestamps[i] - timestamps[i-1]
            if dt > 0:
                velocity = (positions_3d[i] - positions_3d[i-1]) / dt
                velocities.append(velocity)
                velocity_times.append(timestamps[i])
        
        if velocities:
            velocities = np.array(velocities)
            velocity_times = np.array(velocity_times)
            
            # Calculate speed (magnitude of velocity)
            speeds = np.linalg.norm(velocities, axis=1)
            
            # Plot speed vs time
            plt.figure(figsize=(10, 6))
            plt.plot(velocity_times, speeds, 'b-')
            plt.xlabel('Time (s)')
            plt.ylabel('Speed (mm/s)')
            plt.title('Ball Speed vs Time')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, "speed_vs_time.png"), dpi=300)
            plt.close()
            
            # Plot velocity components
            fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
            
            # Plot X velocity
            axes[0].plot(velocity_times, velocities[:, 0], 'b-')
            axes[0].set_ylabel('X Velocity (mm/s)')
            axes[0].set_title('Ball Velocity Components vs Time')
            axes[0].grid(True)
            
            # Plot Y velocity (vertical)
            axes[1].plot(velocity_times, velocities[:, 1], 'g-')
            axes[1].set_ylabel('Y Velocity (mm/s)')
            axes[1].grid(True)
            
            # Plot Z velocity (depth)
            axes[2].plot(velocity_times, velocities[:, 2], 'r-')
            axes[2].set_ylabel('Z Velocity (mm/s)')
            axes[2].set_xlabel('Time (s)')
            axes[2].grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "velocity_vs_time.png"), dpi=300)
            plt.close()
            
            # 4. Calculate and plot acceleration
            if len(velocity_times) > 1:
                # Calculate accelerations
                accelerations = []
                accel_times = []
                
                for i in range(1, len(velocity_times)):
                    dt = velocity_times[i] - velocity_times[i-1]
                    if dt > 0:
                        accel = (velocities[i] - velocities[i-1]) / dt
                        accelerations.append(accel)
                        accel_times.append(velocity_times[i])
                
                if accelerations:
                    accelerations = np.array(accelerations)
                    accel_times = np.array(accel_times)
                    
                    # Plot Y acceleration (gravity) vs time
                    plt.figure(figsize=(10, 6))
                    plt.plot(accel_times, accelerations[:, 1], 'g-')  # Y is vertical in our setup
                    plt.axhline(y=-9800, color='r', linestyle='--', label='Expected gravity (9.8 m/s²)')
                    plt.xlabel('Time (s)')
                    plt.ylabel('Y Acceleration (mm/s²)')
                    plt.title('Vertical Acceleration vs Time (Gravity Measurement)')
                    plt.grid(True)
                    plt.legend()
                    plt.savefig(os.path.join(output_dir, "gravity_measurement.png"), dpi=300)
                    plt.close()
    
    # 5. Create 2D animation of the ball drop
    create_drop_animation(positions_3d, timestamps, output_dir)

def create_drop_animation(positions_3d, timestamps, output_dir):
    """Create an animated visualization of the ball drop."""
    # Create side view (X-Y plane) animation
    fig, ax = plt.subplots(figsize=(8, 10))
    
    # Set axis limits with some padding
    x_min, x_max = np.min(positions_3d[:, 0]), np.max(positions_3d[:, 0])
    y_min, y_max = np.min(positions_3d[:, 1]), np.max(positions_3d[:, 1])
    x_pad = max(100, (x_max - x_min) * 0.1)
    y_pad = max(100, (y_max - y_min) * 0.1)
    
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_max + y_pad, y_min - y_pad)  # Invert Y-axis (top to bottom)
    
    ax.set_xlabel('X Position (mm)')
    ax.set_ylabel('Y Position (mm)')
    ax.set_title('Ball Drop Side View')
    ax.grid(True)
    
    # Initialize ball and trajectory
    ball, = ax.plot([], [], 'ro', ms=10)
    path, = ax.plot([], [], 'b-', alpha=0.7)
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes)
    
    x_data, y_data = [], []
    
    # Animation initialization function
    def init():
        ball.set_data([], [])
        path.set_data([], [])
        time_text.set_text('')
        return ball, path, time_text
    
    # Animation update function
    def update(frame):
        x_data.append(positions_3d[frame, 0])
        y_data.append(positions_3d[frame, 1])
        
        ball.set_data([positions_3d[frame, 0]], [positions_3d[frame, 1]])
        path.set_data(x_data, y_data)
        time_text.set_text(f'Time: {timestamps[frame]:.2f} s')
        return ball, path, time_text
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=len(positions_3d),
                       init_func=init, blit=True, interval=50)
    
    # Save animation
    ani.save(os.path.join(output_dir, 'ball_drop_animation.mp4'), 
             writer='ffmpeg', fps=20, dpi=100)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze ball drop motion from stereo videos")
    parser.add_argument("--test_dir", required=True, help="Test directory (e.g., motion_v1.2)")
    parser.add_argument("--base_dir", default=".", help="Base directory containing the data folder")
    parser.add_argument("--model", default="yolov8m.pt", help="Path to YOLOv8 model")
    parser.add_argument("--device", default="mps", help="Device to run inference on (cpu, cuda, mps)")
    parser.add_argument("--keyword", default="drop", help="Keyword to identify video files")
    parser.add_argument("--start_frame", type=int, default=None, help="Starting frame for analysis")
    parser.add_argument("--max_frames", type=int, default=120, help="Maximum frames to analyze")
    parser.add_argument("--delay", type=float, default=3.0, 
                  help="Delay in seconds after flash before starting analysis")
    
    args = parser.parse_args()
    
    # Set full test path
    test_dir = os.path.join(args.base_dir, "data", args.test_dir)
    
    # Check if test directory exists
    if not os.path.exists(test_dir):
        print(f"Error: Test directory {test_dir} not found")
        return
    
    # Create output directory
    output_dir = os.path.join(test_dir, "results", "ball_drop_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load calibration data
    calibration_data = load_calibration(test_dir)
    if calibration_data is None:
        print("Failed to load calibration data")
        return
    
    # Load YOLOv8 model
    print(f"Loading YOLOv8 model {args.model}...")
    try:
        model = YOLO(args.model)
        # Set device for inference
        if args.device == "mps" and not hasattr(model, "to"):
            print("Warning: MPS device setting not directly supported. Model will use available acceleration.")
        else:
            model.to(args.device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print("Model loaded successfully")
    
    # Find frame offset between cameras
    frame_offset = load_sync_data(test_dir)
    
    # Find ball drop videos
    left_video = find_specific_video(test_dir, args.keyword, "left")
    right_video = find_specific_video(test_dir, args.keyword, "right")
    
    if not left_video or not right_video:
        print(f"Error: Could not find videos with keyword '{args.keyword}'")
        return
    
    # Analyze ball drop
    analyze_ball_drop(left_video, right_video, model, calibration_data, 
                     frame_offset, output_dir, args.delay, args.start_frame, args.max_frames)

if __name__ == "__main__":
    main()