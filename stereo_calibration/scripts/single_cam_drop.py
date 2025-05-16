#!/usr/bin/env python3
import cv2
import numpy as np
import os
import glob
import argparse
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from ultralytics import YOLO
import math
import csv

def detect_ball_yolo(image, model, conf_threshold=0.25, is_orange=True):
    """
    Detect basketball in image using YOLOv8 model with color filtering.
    Optimized for orange basketball detection.
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
            
            # Extract region around detection for color & circularity check
            margin = radius * 0.2
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
            
            # Check color if we're looking for an orange ball
            if is_orange:
                try:
                    # Convert to HSV color space for better color detection
                    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
                    
                    # Orange color range in HSV
                    lower_orange = np.array([25, 50, 100])   # Lower HSV boundary for orange
                    upper_orange = np.array([38, 50, 100])  # Upper HSV boundary for orange
                    
                    # Create mask for orange pixels
                    orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
                    
                    # Calculate percentage of orange pixels in the region
                    orange_percent = np.sum(orange_mask > 0) / (region.shape[0] * region.shape[1]) * 100
                    
                    # Boost confidence if orange percentage is high
                    if orange_percent > 30:  # At least 30% orange pixels
                        color_boost = min(1.0, orange_percent / 100 + 0.3)  # Max boost factor of 1.0
                        adjusted_conf = conf * color_boost
                    else:
                        # Reduce confidence for non-orange objects
                        adjusted_conf = conf * 0.7
                except Exception as e:
                    # If color check fails, use original confidence
                    adjusted_conf = conf
            else:
                # No color filtering
                adjusted_conf = conf
            
            # Check circularity
            try:
                # Convert to grayscale
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
                        
                        # Adjust confidence based on circularity (perfect circle = 1.0)
                        circularity_factor = min(1.0, circularity + 0.2)  # Boost slightly
                        adjusted_conf = adjusted_conf * circularity_factor
            except Exception as e:
                # If circularity check fails, keep current confidence
                pass
            
            # Update best ball if this detection has higher confidence
            if adjusted_conf > highest_conf:
                highest_conf = adjusted_conf
                best_ball = (cx, cy, radius, adjusted_conf)
    
    return best_ball

def improved_ball_tracking(video_paths, model, max_frames=120):
    """
    Improved ball tracking using detection + tracking with validation.
    
    Args:
        video_paths: List of paths to videos (left, right)
        model: Detection model (YOLOv8 or similar)
        max_frames: Maximum frames to process
    """
    # Initialize OpenCV trackers
    tracker_types = ['KCF', 'CSRT']  # Try different trackers
    
    # Open videos
    caps = [cv2.VideoCapture(path) for path in video_paths]
    
    # Initialize storage
    all_detections = [[] for _ in video_paths]
    frames_processed = 0
    
    # Initial detection phase (first 5 frames)
    initial_boxes = [None for _ in video_paths]
    
    # State variables
    tracking_initialized = False
    trackers = [None for _ in video_paths]
    prev_positions = [None for _ in video_paths]
    
    # Physical constraint parameters
    max_speed_px = 50  # Maximum reasonable speed between frames
    min_conf_threshold = 0.5  # Minimum confidence for detection
    
    # Process frames
    while frames_processed < max_frames:
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                return all_detections
            frames.append(frame)
        
        # Phase 1: Initial detection to establish baseline
        if not tracking_initialized:
            all_valid = True
            for i, frame in enumerate(frames):
                # Detect with high confidence
                detection = detect_ball_yolo(frame, model, conf_threshold=0.7)
                if detection:
                    x, y, r, conf = detection
                    initial_boxes[i] = (int(x-r), int(y-r), int(2*r), int(2*r))
                else:
                    all_valid = False
            
            if all_valid:
                # Initialize trackers with high-confidence detections
                tracking_initialized = True
                for i, box in enumerate(initial_boxes):
                    if tracker_types[0] == 'KCF':
                        trackers[i] = cv2.TrackerKCF_create()
                    else:
                        trackers[i] = cv2.TrackerCSRT_create()
                    trackers[i].init(frames[i], box)
                    
                    # Store initial positions
                    x, y, w, h = box
                    center = (x + w/2, y + h/2)
                    prev_positions[i] = center
                    all_detections[i].append((center[0], center[1], w/2, 1.0))
        
        # Phase 2: Tracking with validation
        else:
            for i, frame in enumerate(frames):
                # Update tracker
                success, box = trackers[i].update(frame)
                
                if success:
                    x, y, w, h = [int(v) for v in box]
                    center = (x + w/2, y + h/2)
                    
                    # Validate tracked position (distance from previous)
                    prev_center = prev_positions[i]
                    distance = np.sqrt((center[0] - prev_center[0])**2 + 
                                      (center[1] - prev_center[1])**2)
                    
                    # If motion is reasonable, accept tracking result
                    if distance < max_speed_px:
                        all_detections[i].append((center[0], center[1], w/2, 0.9))
                        prev_positions[i] = center
                    else:
                        # Motion constraint violated, re-detect
                        detection = detect_ball_yolo(frame, model, conf_threshold=min_conf_threshold)
                        if detection:
                            x, y, r, conf = detection
                            new_box = (int(x-r), int(y-r), int(2*r), int(2*r))
                            
                            # Reinitialize tracker
                            if tracker_types[0] == 'KCF':
                                trackers[i] = cv2.TrackerKCF_create()
                            else:
                                trackers[i] = cv2.TrackerCSRT_create()
                            trackers[i].init(frame, new_box)
                            
                            # Store new position with lower confidence
                            center = (x, y)
                            all_detections[i].append((x, y, r, conf * 0.8))
                            prev_positions[i] = center
                        else:
                            # Keep previous position but mark low confidence
                            all_detections[i].append((prev_center[0], prev_center[1], 
                                                    all_detections[i][-1][2], 0.3))
                else:
                    # Tracking failed, try detection
                    detection = detect_ball_yolo(frame, model, conf_threshold=min_conf_threshold)
                    if detection:
                        x, y, r, conf = detection
                        new_box = (int(x-r), int(y-r), int(2*r), int(2*r))
                        
                        # Reinitialize tracker
                        if tracker_types[0] == 'KCF':
                            trackers[i] = cv2.TrackerKCF_create()
                        else:
                            trackers[i] = cv2.TrackerCSRT_create()
                        trackers[i].init(frame, new_box)
                        
                        # Store new position
                        all_detections[i].append((x, y, r, conf))
                        prev_positions[i] = (x, y)
                    elif prev_positions[i] is not None:
                        # Keep previous position but mark as very low confidence
                        all_detections[i].append((prev_positions[i][0], prev_positions[i][1], 
                                                all_detections[i][-1][2], 0.1))
                    else:
                        # No previous position, mark as invalid
                        all_detections[i].append(None)
        
        frames_processed += 1
    
    # Release video captures
    for cap in caps:
        cap.release()
        
    return all_detections

def analyze_ball_drop(video_path, model, output_dir, start_frame=None, max_frames=300, 
                     fps_override=None, is_orange=True, focal_length=None, ball_diameter_mm=240):
    """
    Analyze ball drop motion from a single camera.
    
    Args:
        video_path (str): Path to the video file
        model: YOLOv8 model for ball detection
        output_dir (str): Directory to save results
        start_frame (int): Frame to start analysis, if None, will auto-detect
        max_frames (int): Maximum number of frames to analyze
        fps_override (float): Override the video's reported FPS
        is_orange (bool): Whether to use orange color filtering
        focal_length (float): Camera focal length in pixels (if known)
        ball_diameter_mm (float): Actual ball diameter in mm
    """
    print(f"\nAnalyzing ball drop from video: {os.path.basename(video_path)}")
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS) if fps_override is None else fps_override
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info: {video_fps:.2f} fps, {frame_count} frames, {frame_width}x{frame_height} resolution")
    
    # Set starting frame
    if start_frame is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        print(f"Starting analysis at frame {start_frame}")
    else:
        start_frame = 0
    
    # Initialize data storage
    frames = []
    timestamps = []
    positions = []  # (x, y) in pixels
    distances = []  # Estimated distance in mm (if focal_length is provided)
    ball_sizes = []  # Radius in pixels
    
    # Process frames
    frame_idx = start_frame
    processing_frames = min(max_frames, frame_count - start_frame)
    
    pbar = tqdm(total=processing_frames, desc="Processing frames")
    
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx - start_frame >= max_frames:
            break
        
        # Detect ball in frame
        ball = detect_ball_yolo(frame, model, is_orange=is_orange)
        
        # If ball detected, record data
        if ball:
            cx, cy, radius, conf = ball
            
            # Record frame, timestamp and position
            frames.append(frame_idx)
            timestamps.append((frame_idx - start_frame) / video_fps)
            positions.append((cx, cy))
            ball_sizes.append(radius)
            
            # Estimate distance if focal length is provided
            if focal_length is not None:
                # Distance = (Actual size × Focal length) ÷ (Apparent size in pixels)
                # For a circle, apparent size is the diameter (2*radius)
                apparent_diameter = 2 * radius
                distance_mm = (ball_diameter_mm * focal_length) / apparent_diameter
                distances.append(distance_mm)
            
            # Create visualization of detected ball
            if frame_idx % 5 == 0:  # Save every 5th frame
                vis_frame = frame.copy()
                
                # Draw circle representing the ball
                cv2.circle(vis_frame, (int(cx), int(cy)), int(radius), (0, 255, 0), 2)
                cv2.circle(vis_frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                
                # Add text with frame number and confidence
                cv2.putText(vis_frame, f"Frame: {frame_idx}", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(vis_frame, f"Conf: {conf:.2f}", (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Add text with estimated distance if available
                if focal_length is not None:
                    cv2.putText(vis_frame, f"Dist: {distances[-1]:.1f} mm", (50, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Save visualization
                vis_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg")
                cv2.imwrite(vis_path, vis_frame)
        
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    print(f"Analyzed {len(frames)} frames with ball detections")
    
    # If not enough detections, return early
    if len(frames) < 10:
        print("Not enough ball detections for analysis")
        return None
    
    # Convert to numpy arrays for easier analysis
    frames = np.array(frames)
    timestamps = np.array(timestamps)
    positions = np.array(positions)
    ball_sizes = np.array(ball_sizes)
    
    if focal_length is not None:
        distances = np.array(distances)
    
    # Calculate velocities (pixels per second)
    velocities = []
    velocity_timestamps = []
    
    for i in range(1, len(timestamps)):
        dt = timestamps[i] - timestamps[i-1]
        if dt > 0:
            v = (positions[i] - positions[i-1]) / dt
            velocities.append(v)
            velocity_timestamps.append(timestamps[i])
    
    velocities = np.array(velocities)
    velocity_timestamps = np.array(velocity_timestamps)
    
    # Calculate accelerations (pixels per second^2)
    accelerations = []
    acceleration_timestamps = []
    
    for i in range(1, len(velocities)):
        dt = velocity_timestamps[i] - velocity_timestamps[i-1]
        if dt > 0:
            a = (velocities[i] - velocities[i-1]) / dt
            accelerations.append(a)
            acceleration_timestamps.append(velocity_timestamps[i])
    
    accelerations = np.array(accelerations)
    acceleration_timestamps = np.array(acceleration_timestamps)
    
    # Create visualizations
    create_trajectory_visualization(positions, timestamps, ball_sizes, 
                                  focal_length, ball_diameter_mm,
                                  velocities, velocity_timestamps,
                                  accelerations, acceleration_timestamps,
                                  output_dir)
    
    # Calculate and print key metrics
    if len(accelerations) > 0:
        # Average vertical acceleration (gravity)
        avg_y_accel = np.mean(accelerations[:, 1])
        print(f"Average vertical acceleration: {avg_y_accel:.2f} pixels/s²")
        
        # If we have focal length and ball size, convert to real units
        if focal_length is not None:
            # Convert acceleration from pixels/s² to mm/s²
            # Using the ball diameter as the scale reference
            avg_y_accel_mm = avg_y_accel * (ball_diameter_mm / (2 * np.mean(ball_sizes)))
            print(f"Average vertical acceleration: {avg_y_accel_mm:.2f} mm/s²")
            print(f"Expected value for gravity: 9800 mm/s²")
            
            gravity_error = abs(avg_y_accel_mm - 9800) / 9800 * 100
            print(f"Gravity measurement error: {gravity_error:.2f}%")
    
    # Save data to CSV for further analysis
    save_data_to_csv(frames, timestamps, positions, ball_sizes, 
                    velocities, velocity_timestamps,
                    accelerations, acceleration_timestamps,
                    focal_length, ball_diameter_mm, output_dir)
    
    # Create animation of ball drop
    create_drop_animation(positions, timestamps, ball_sizes, 
                         frame_width, frame_height, output_dir)
    
    return positions, timestamps, velocities, accelerations

def create_trajectory_visualization(positions, timestamps, ball_sizes, 
                                  focal_length, ball_diameter_mm,
                                  velocities, velocity_timestamps,
                                  accelerations, acceleration_timestamps,
                                  output_dir):
    """Create visualizations of the ball trajectory and motion data."""
    # 1. Position vs Time
    plt.figure(figsize=(12, 10))
    
    # X Position
    plt.subplot(2, 1, 1)
    plt.plot(timestamps, positions[:, 0], 'b-')
    plt.xlabel('Time (s)')
    plt.ylabel('X Position (pixels)')
    plt.title('Horizontal Position vs Time')
    plt.grid(True)
    
    # Y Position
    plt.subplot(2, 1, 2)
    plt.plot(timestamps, positions[:, 1], 'r-')
    plt.xlabel('Time (s)')
    plt.ylabel('Y Position (pixels)')
    plt.title('Vertical Position vs Time')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'position_vs_time.png'), dpi=300)
    plt.close()
    
    # 2. Ball Size vs Time
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, ball_sizes, 'g-')
    plt.xlabel('Time (s)')
    plt.ylabel('Ball Radius (pixels)')
    plt.title('Ball Size vs Time')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'ball_size_vs_time.png'), dpi=300)
    plt.close()
    
    # 3. Velocities vs Time
    plt.figure(figsize=(12, 10))
    
    # X Velocity
    plt.subplot(2, 1, 1)
    plt.plot(velocity_timestamps, velocities[:, 0], 'b-')
    plt.xlabel('Time (s)')
    plt.ylabel('X Velocity (pixels/s)')
    plt.title('Horizontal Velocity vs Time')
    plt.grid(True)
    
    # Y Velocity
    plt.subplot(2, 1, 2)
    plt.plot(velocity_timestamps, velocities[:, 1], 'r-')
    plt.xlabel('Time (s)')
    plt.ylabel('Y Velocity (pixels/s)')
    plt.title('Vertical Velocity vs Time')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'velocity_vs_time.png'), dpi=300)
    plt.close()
    
    # 4. Accelerations vs Time
    if len(accelerations) > 0:
        plt.figure(figsize=(12, 10))
        
        # X Acceleration
        plt.subplot(2, 1, 1)
        plt.plot(acceleration_timestamps, accelerations[:, 0], 'b-')
        plt.xlabel('Time (s)')
        plt.ylabel('X Acceleration (pixels/s²)')
        plt.title('Horizontal Acceleration vs Time')
        plt.grid(True)
        
        # Y Acceleration
        plt.subplot(2, 1, 2)
        plt.plot(acceleration_timestamps, accelerations[:, 1], 'r-')
        plt.xlabel('Time (s)')
        plt.ylabel('Y Acceleration (pixels/s²)')
        plt.title('Vertical Acceleration vs Time')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'acceleration_vs_time.png'), dpi=300)
        plt.close()
    
    # 5. 2D Trajectory with Time Color Mapping
    plt.figure(figsize=(10, 8))
    plt.scatter(positions[:, 0], positions[:, 1], c=timestamps, cmap='viridis', s=50)
    plt.colorbar(label='Time (s)')
    plt.plot(positions[:, 0], positions[:, 1], 'r-', alpha=0.5)
    
    # Invert y-axis to match image coordinates (origin at top-left)
    plt.gca().invert_yaxis()
    
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.title('2D Ball Trajectory')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, '2d_trajectory.png'), dpi=300)
    plt.close()
    
    # 6. Distance estimation (if focal length is provided)
    if focal_length is not None:
        # Calculate distances based on apparent size
        distances = []
        for radius in ball_sizes:
            apparent_diameter = 2 * radius
            distance_mm = (ball_diameter_mm * focal_length) / apparent_diameter
            distances.append(distance_mm)
        
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, distances, 'b-')
        plt.xlabel('Time (s)')
        plt.ylabel('Distance (mm)')
        plt.title('Estimated Distance vs Time')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'distance_vs_time.png'), dpi=300)
        plt.close()

def save_data_to_csv(frames, timestamps, positions, ball_sizes, 
                    velocities, velocity_timestamps,
                    accelerations, acceleration_timestamps,
                    focal_length, ball_diameter_mm, output_dir):
    """Save all data to CSV files for further analysis."""
    # Save position data
    position_file = os.path.join(output_dir, 'ball_positions.csv')
    with open(position_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'time', 'x', 'y', 'radius'])
        
        for i in range(len(frames)):
            writer.writerow([
                frames[i],
                f"{timestamps[i]:.4f}",
                f"{positions[i, 0]:.2f}",
                f"{positions[i, 1]:.2f}",
                f"{ball_sizes[i]:.2f}"
            ])
    
    # Save velocity data
    velocity_file = os.path.join(output_dir, 'ball_velocities.csv')
    with open(velocity_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'vx', 'vy', 'speed'])
        
        for i in range(len(velocities)):
            speed = np.linalg.norm(velocities[i])
            writer.writerow([
                f"{velocity_timestamps[i]:.4f}",
                f"{velocities[i, 0]:.2f}",
                f"{velocities[i, 1]:.2f}",
                f"{speed:.2f}"
            ])
    
    # Save acceleration data
    if len(accelerations) > 0:
        accel_file = os.path.join(output_dir, 'ball_accelerations.csv')
        with open(accel_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'ax', 'ay', 'magnitude'])
            
            for i in range(len(accelerations)):
                magnitude = np.linalg.norm(accelerations[i])
                writer.writerow([
                    f"{acceleration_timestamps[i]:.4f}",
                    f"{accelerations[i, 0]:.2f}",
                    f"{accelerations[i, 1]:.2f}",
                    f"{magnitude:.2f}"
                ])
    
    # Save distance data if focal length is provided
    if focal_length is not None:
        distance_file = os.path.join(output_dir, 'ball_distances.csv')
        with open(distance_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'distance_mm'])
            
            for i in range(len(frames)):
                apparent_diameter = 2 * ball_sizes[i]
                distance_mm = (ball_diameter_mm * focal_length) / apparent_diameter
                writer.writerow([
                    f"{timestamps[i]:.4f}",
                    f"{distance_mm:.2f}"
                ])

def create_drop_animation(positions, timestamps, ball_sizes, frame_width, frame_height, output_dir):
    """Create an animated visualization of the ball drop."""
    # Create a new figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set limits with some padding
    x_min, x_max = np.min(positions[:, 0]), np.max(positions[:, 0])
    y_min, y_max = np.min(positions[:, 1]), np.max(positions[:, 1])
    
    # Add padding as percentage of range
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1
    
    # Set axis limits (with y-axis inverted to match image coordinates)
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_max + y_padding, y_min - y_padding)
    
    ax.set_xlabel('X Position (pixels)')
    ax.set_ylabel('Y Position (pixels)')
    ax.set_title('Ball Drop Animation')
    ax.grid(True)
    
    # Initialize ball and trajectory
    trajectory, = ax.plot([], [], 'r-', alpha=0.7)
    ball = plt.Circle((0, 0), 10, color='b', alpha=0.7)
    ax.add_patch(ball)
    
    # Text annotations
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    
    # Data to update
    x_history = []
    y_history = []
    
    # Initialization function
    def init():
        trajectory.set_data([], [])
        ball.center = (0, 0)
        time_text.set_text('')
        return [trajectory, ball, time_text]
    
    # Update function
    def update(frame):
        # Update trajectory
        x_history.append(positions[frame, 0])
        y_history.append(positions[frame, 1])
        trajectory.set_data(x_history, y_history)
        
        # Update ball position and size
        ball.center = (positions[frame, 0], positions[frame, 1])
        ball.radius = ball_sizes[frame]
        
        # Update text
        time_text.set_text(f'Time: {timestamps[frame]:.2f} s')
        
        return [trajectory, ball, time_text]
    
    # Create animation
    interval = 1000 / 30  # 30 fps
    frames = len(positions)
    
    try:
        ani = FuncAnimation(fig, update, frames=frames, init_func=init, 
                          blit=True, interval=interval, repeat=False)
        
        # Save animation as mp4
        ani.save(os.path.join(output_dir, 'ball_drop_animation.mp4'), 
                writer='ffmpeg', fps=30, dpi=100, extra_args=['-vcodec', 'libx264'])
        
        print(f"Animation saved to {os.path.join(output_dir, 'ball_drop_animation.mp4')}")
    except Exception as e:
        print(f"Error creating animation: {e}")
    
    plt.close()

def estimate_focal_length(known_distance_mm, ball_diameter_mm, apparent_diameter_pixels):
    """
    Estimate the camera's focal length using a reference object of known size.
    
    Args:
        known_distance_mm: Actual distance to the object in mm
        ball_diameter_mm: Actual diameter of the ball in mm
        apparent_diameter_pixels: Apparent diameter of the ball in pixels
        
    Returns:
        Estimated focal length in pixels
    """
    # Focal length = (Apparent size × Distance) ÷ Actual size
    focal_length = (apparent_diameter_pixels * known_distance_mm) / ball_diameter_mm
    return focal_length

def main():
    parser = argparse.ArgumentParser(description="Single-camera ball drop analysis")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--output_dir", default="data/single_cam_drop/videos2/ball_drop_results", help="Output directory for results")
    parser.add_argument("--model", default="yolov8m.pt", help="Path to YOLOv8 model")
    parser.add_argument("--device", default="mps", help="Device to run inference on (cpu, cuda, mps)")
    parser.add_argument("--start_frame", type=int, default=None, help="Starting frame for analysis")
    parser.add_argument("--delay", type=float, default=None, help="Start delay in seconds after video beginning")
    parser.add_argument("--max_frames", type=int, default=300, help="Maximum frames to analyze")
    parser.add_argument("--orange", action="store_true", help="Use orange color filtering for basketball")
    parser.add_argument("--known_distance", type=float, default=None, 
                       help="Known distance to ball in mm (for focal length estimation)")
    parser.add_argument("--ball_diameter", type=float, default=240.0, 
                       help="Actual ball diameter in mm (default: 240mm)")
    parser.add_argument("--focal_length", type=float, default=None, 
                       help="Camera focal length in pixels (if known)")
    
    args = parser.parse_args()
    
    # Check if video exists
    if not os.path.exists(args.video):
        print(f"Error: Video file {args.video} not found")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load YOLOv8 model
    print(f"Loading YOLOv8 model {args.model}...")
    try:
        model = YOLO(args.model)
        # Set device for inference
        if args.device == "mps" and not hasattr(model, "to"):
            print("Warning: MPS device setting not directly supported. Model will use available acceleration.")
        else:
            model.to(args.device)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Calculate starting frame from delay if provided
    start_frame = args.start_frame
    if start_frame is None and args.delay is not None:
        # Get video FPS
        cap = cv2.VideoCapture(args.video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Convert delay to frames
        start_frame = int(args.delay * fps)
        print(f"Using delay of {args.delay} seconds ({start_frame} frames)")
    
    # Analyze ball drop
    analyze_ball_drop(
        args.video, 
        model, 
        args.output_dir,
        start_frame=start_frame,
        max_frames=args.max_frames,
        is_orange=args.orange,
        focal_length=args.focal_length,
        ball_diameter_mm=args.ball_diameter
    )

if __name__ == "__main__":
    main()