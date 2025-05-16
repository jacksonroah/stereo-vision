#!/usr/bin/env python3
import cv2
import numpy as np
import os
import glob
import argparse
import time
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import pickle
from ultralytics import YOLO

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
                return sync_data
        except Exception as e:
            print(f"Error loading sync data: {e}")
    
    return None

def analyze_brightness_jump(video_path, threshold=20, window_size=5, max_frames=900):
    """
    Detect frames with sudden brightness increases that could indicate a flash.
    Returns the frame index of detected flash or None.
    """
    print(f"Analyzing {os.path.basename(video_path)} for brightness jumps...")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    frame_count = 0
    brightness_history = []
    brightness_values = []  # Store all brightness values for visualization
    
    # Try to adapt threshold based on video characteristics
    adapt_threshold = True
    sample_brightness_values = []
    
    # First, sample some frames to determine average brightness and variance
    while cap.isOpened() and len(sample_brightness_values) < 30:  # Sample 30 frames
        ret, frame = cap.read()
        if not ret:
            break
            
        # Skip every few frames for a broader sample
        if frame_count % 10 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            sample_brightness_values.append(brightness)
        
        frame_count += 1
    
    # Reset video to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count = 0
    
    # Adapt threshold if we got enough samples
    if adapt_threshold and len(sample_brightness_values) >= 10:
        avg_brightness = np.mean(sample_brightness_values)
        std_brightness = np.std(sample_brightness_values)
        
        # Set threshold to 3 standard deviations by default, or minimum 15
        adaptive_threshold = max(15, std_brightness * 3)
        
        # For very dark or bright videos, adjust accordingly
        if avg_brightness < 50:  # Dark video
            adaptive_threshold = max(10, adaptive_threshold * 0.8)
        elif avg_brightness > 200:  # Bright video
            adaptive_threshold *= 1.5
        
        print(f"Adapting brightness threshold based on video characteristics:")
        print(f"  Average brightness: {avg_brightness:.1f}")
        print(f"  Brightness std dev: {std_brightness:.1f}")
        print(f"  Using threshold: {adaptive_threshold:.1f} (was {threshold})")
        threshold = adaptive_threshold
    
    # Main analysis loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Calculate average brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        brightness_history.append(brightness)
        brightness_values.append(brightness)
        
        # Check for jump after collecting enough frames
        if len(brightness_history) > window_size:
            # Calculate average brightness before current frame
            prev_avg = sum(brightness_history[-window_size-1:-1]) / window_size
            
            # Check if current brightness is significantly higher
            if brightness > prev_avg + threshold:
                print(f"Detected brightness jump at frame {frame_count}: {prev_avg:.1f} -> {brightness:.1f}")
                print(f"  Increase: {brightness - prev_avg:.1f} (threshold: {threshold:.1f})")
                
                # Save a visualization of brightness history
                plt.figure(figsize=(10, 6))
                plt.plot(brightness_values)
                plt.axvline(x=frame_count, color='r', linestyle='--', label=f'Flash at frame {frame_count}')
                plt.axhline(y=prev_avg + threshold, color='g', linestyle='-.', label=f'Threshold ({threshold:.1f})')
                plt.xlabel('Frame Number')
                plt.ylabel('Average Brightness')
                plt.title(f'Brightness Analysis - {os.path.basename(video_path)}')
                plt.legend()
                
                # Ensure output directory exists
                output_dir = os.path.join(os.path.dirname(os.path.dirname(video_path)), "results", "sync_results")
                os.makedirs(output_dir, exist_ok=True)
                
                plt.savefig(os.path.join(output_dir, f"brightness_analysis_{os.path.basename(video_path)}.png"))
                plt.close()
                
                cap.release()
                return frame_count
        
        frame_count += 1
        
        # Limit the number of frames to check
        if frame_count >= max_frames:
            break
    
    cap.release()
    return None

def create_synchronized_videos(left_video, right_video, sync_data, output_dir, delay):
    """Create new synchronized videos that start at the same time."""
    print("Creating synchronized videos...")
    
    left_cap = cv2.VideoCapture(left_video)
    right_cap = cv2.VideoCapture(right_video)
    
    if not left_cap.isOpened() or not right_cap.isOpened():
        print("Error: Could not open videos")
        return None, None
    
    # Get video properties
    left_fps = left_cap.get(cv2.CAP_PROP_FPS)
    right_fps = right_cap.get(cv2.CAP_PROP_FPS)
    
    # Check if frame rates match closely enough
    if abs(left_fps - right_fps) > 0.5:
        print(f"Warning: Frame rates differ significantly (left: {left_fps}, right: {right_fps})")
        print("Proceeding anyway but synchronization might not be perfect")
    
    # Calculate offsets
    frame_offset = sync_data['frame_offset']
    left_flash_frame = sync_data.get('left_flash_frame', 0)
    
    # Skip to 3 seconds after the flash frame to avoid setup frames
    frames_to_skip = int(delay * left_fps)
    left_start_frame = left_flash_frame + frames_to_skip
    right_start_frame = left_start_frame + frame_offset  # Apply frame offset
    
    left_cap.set(cv2.CAP_PROP_POS_FRAMES, left_start_frame)
    right_cap.set(cv2.CAP_PROP_POS_FRAMES, right_start_frame)
    
    # Use properties from left video for output
    width = int(left_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(left_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = left_fps
    
    # Create output video files - add timestamp to avoid overwriting
    timestamp = int(time.time())
    left_output_path = os.path.join(output_dir, f"sync_left_{timestamp}.mp4")
    right_output_path = os.path.join(output_dir, f"sync_right_{timestamp}.mp4")
    
    # Define the codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Create VideoWriter objects
    left_out = cv2.VideoWriter(left_output_path, fourcc, fps, (width, height))
    right_out = cv2.VideoWriter(right_output_path, fourcc, fps, (width, height))
    
    # Define how many frames to process after sync point
    # Default to 10 seconds of footage
    frames_to_process = 10 * int(fps)
    
    print(f"Extracting synchronized frames starting {frames_to_skip} frames after flash")
    print(f"Left starting at frame {left_start_frame}, right at frame {right_start_frame}")
    
    # Process frames
    pbar = tqdm(total=frames_to_process, desc="Saving synchronized videos")
    
    for i in range(frames_to_process):
        left_ret, left_frame = left_cap.read()
        right_ret, right_frame = right_cap.read()
        
        if not left_ret or not right_ret:
            print(f"Reached end of videos after {i} frames")
            break
        
        # Add frame number as overlay for debugging
        cv2.putText(left_frame, f"Frame: {left_start_frame + i}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(right_frame, f"Frame: {right_start_frame + i}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Write frames
        left_out.write(left_frame)
        right_out.write(right_frame)
        pbar.update(1)
    
    pbar.close()
    
    # Release resources
    left_cap.release()
    right_cap.release()
    left_out.release()
    right_out.release()
    
    print(f"Created synchronized videos:")
    print(f"Left: {left_output_path}")
    print(f"Right: {right_output_path}")
    
    return left_output_path, right_output_path

def detect_ball_yolo(image, model, conf_threshold=0.25):
    """Detect ball in image using YOLOv8 model."""
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

def process_rolling_ball(left_video, right_video, model, sync_data, output_dir, delay):
    """
    Analyze rolling ball motion from synchronized videos.
    
    Args:
        left_video: Path to left camera video
        right_video: Path to right camera video
        model: YOLOv8 model for ball detection
        sync_data: Synchronization data dictionary
        output_dir: Directory to save results
    """
    print(f"\nAnalyzing rolling ball from videos:")
    print(f"  Left: {os.path.basename(left_video)}")
    print(f"  Right: {os.path.basename(right_video)}")
    
    # Create synchronized videos
    sync_left, sync_right = create_synchronized_videos(
        left_video, right_video, sync_data, output_dir, delay
    )
    
    if not sync_left or not sync_right:
        print("Failed to create synchronized videos")
        return
    
    # Open synchronized videos
    left_cap = cv2.VideoCapture(sync_left)
    right_cap = cv2.VideoCapture(sync_right)
    
    # Get video properties
    fps = left_cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(left_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize data storage
    frames = []
    left_detections = []
    right_detections = []
    timestamps = []
    
    # Process frames
    pbar = tqdm(total=frame_count, desc="Tracking ball")
    
    while True:
        left_ret, left_frame = left_cap.read()
        right_ret, right_frame = right_cap.read()
        
        if not left_ret or not right_ret:
            break
        
        # Get current frame index
        frame_idx = len(frames)
        
        # Detect ball in both frames
        left_ball = detect_ball_yolo(left_frame, model)
        right_ball = detect_ball_yolo(right_frame, model)
        
        # Store detections
        frames.append(frame_idx)
        timestamps.append(frame_idx / fps)
        left_detections.append(left_ball)
        right_detections.append(right_ball)
        
        # Create visualization every 5 frames
        if frame_idx % 5 == 0:
            # Create copy of frames for drawing
            left_vis = left_frame.copy()
            right_vis = right_frame.copy()
            
            # Draw ball detections
            if left_ball:
                cx, cy, radius, conf = left_ball
                cv2.circle(left_vis, (int(cx), int(cy)), int(radius), (0, 255, 0), 2)
                cv2.circle(left_vis, (int(cx), int(cy)), 3, (0, 0, 255), -1)
                cv2.putText(left_vis, f"Conf: {conf:.2f}", (int(cx) + 10, int(cy) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            if right_ball:
                cx, cy, radius, conf = right_ball
                cv2.circle(right_vis, (int(cx), int(cy)), int(radius), (0, 255, 0), 2)
                cv2.circle(right_vis, (int(cx), int(cy)), 3, (0, 0, 255), -1)
                cv2.putText(right_vis, f"Conf: {conf:.2f}", (int(cx) + 10, int(cy) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Combine into single image
            vis_img = np.hstack((left_vis, right_vis))
            
            # Add frame information
            cv2.putText(vis_img, f"Frame: {frame_idx} | Time: {frame_idx/fps:.2f}s", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Save visualization
            vis_path = os.path.join(output_dir, f"tracking_{frame_idx:04d}.jpg")
            cv2.imwrite(vis_path, vis_img)
        
        pbar.update(1)
    
    pbar.close()
    left_cap.release()
    right_cap.release()
    
    # Filter detections where ball was found in both frames
    valid_frames = []
    valid_timestamps = []
    valid_left = []
    valid_right = []
    
    for i in range(len(frames)):
        if left_detections[i] and right_detections[i]:
            valid_frames.append(frames[i])
            valid_timestamps.append(timestamps[i])
            valid_left.append(left_detections[i])
            valid_right.append(right_detections[i])
    
    print(f"Total frames processed: {len(frames)}")
    print(f"Frames with ball detected in both cameras: {len(valid_frames)}")
    
    if len(valid_frames) < 5:
        print("Not enough valid detections for analysis")
        return
    
    # Save detection data
    detections_file = os.path.join(output_dir, "ball_detections.json")
    with open(detections_file, 'w') as f:
        detection_data = {
            "frames": valid_frames,
            "timestamps": valid_timestamps,
            "left_detections": [[float(x) for x in det[:3]] + [float(det[3])] for det in valid_left],
            "right_detections": [[float(x) for x in det[:3]] + [float(det[3])] for det in valid_right]
        }
        json.dump(detection_data, f, indent=2)
    
    # Create motion path visualization
    create_rolling_visualization(valid_frames, valid_timestamps, valid_left, valid_right, output_dir)

def create_rolling_visualization(frames, timestamps, left_detections, right_detections, output_dir):
    """
    Create visualizations of the rolling ball motion.
    
    Args:
        frames: List of frame indices
        timestamps: List of timestamps for each frame
        left_detections: List of ball detections in left frame (cx, cy, radius, conf)
        right_detections: List of ball detections in right frame (cx, cy, radius, conf)
        output_dir: Directory to save visualizations
    """
    # Extract ball positions
    left_x = [det[0] for det in left_detections]
    left_y = [det[1] for det in left_detections]
    right_x = [det[0] for det in right_detections]
    right_y = [det[1] for det in right_detections]
    
    # Plot 2D trajectories
    plt.figure(figsize=(12, 6))
    
    # Plot left camera trajectory
    plt.subplot(1, 2, 1)
    plt.scatter(left_x, left_y, c=timestamps, cmap='viridis', s=50)
    plt.plot(left_x, left_y, 'r-', alpha=0.7)
    plt.gca().invert_yaxis()  # Invert Y-axis to match image coordinates
    plt.title('Ball Trajectory (Left Camera)')
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.colorbar(label='Time (s)')
    plt.grid(True)
    
    # Plot right camera trajectory
    plt.subplot(1, 2, 2)
    plt.scatter(right_x, right_y, c=timestamps, cmap='viridis', s=50)
    plt.plot(right_x, right_y, 'r-', alpha=0.7)
    plt.gca().invert_yaxis()
    plt.title('Ball Trajectory (Right Camera)')
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.colorbar(label='Time (s)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trajectory_2d.png'), dpi=300)
    plt.close()
    
    # Plot positions versus time
    plt.figure(figsize=(12, 8))
    
    # X-position vs time
    plt.subplot(2, 1, 1)
    plt.plot(timestamps, left_x, 'b-', label='Left Camera')
    plt.plot(timestamps, right_x, 'r-', label='Right Camera')
    plt.title('Ball X-Position vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('X Position (pixels)')
    plt.legend()
    plt.grid(True)
    
    # Y-position vs time
    plt.subplot(2, 1, 2)
    plt.plot(timestamps, left_y, 'b-', label='Left Camera')
    plt.plot(timestamps, right_y, 'r-', label='Right Camera')
    plt.title('Ball Y-Position vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Y Position (pixels)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'position_vs_time.png'), dpi=300)
    plt.close()
    
    # Calculate velocity
    if len(timestamps) > 1:
        # Calculate velocities (pixels per second)
        left_vx = []
        left_vy = []
        right_vx = []
        right_vy = []
        velocity_timestamps = []
        
        for i in range(1, len(timestamps)):
            dt = timestamps[i] - timestamps[i-1]
            if dt > 0:
                left_vx.append((left_x[i] - left_x[i-1]) / dt)
                left_vy.append((left_y[i] - left_y[i-1]) / dt)
                right_vx.append((right_x[i] - right_x[i-1]) / dt)
                right_vy.append((right_y[i] - right_y[i-1]) / dt)
                velocity_timestamps.append(timestamps[i])
        
        # Plot velocities versus time
        plt.figure(figsize=(12, 8))
        
        # X-velocity vs time
        plt.subplot(2, 1, 1)
        plt.plot(velocity_timestamps, left_vx, 'b-', label='Left Camera')
        plt.plot(velocity_timestamps, right_vx, 'r-', label='Right Camera')
        plt.title('Ball X-Velocity vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('X Velocity (pixels/s)')
        plt.legend()
        plt.grid(True)
        
        # Y-velocity vs time
        plt.subplot(2, 1, 2)
        plt.plot(velocity_timestamps, left_vy, 'b-', label='Left Camera')
        plt.plot(velocity_timestamps, right_vy, 'r-', label='Right Camera')
        plt.title('Ball Y-Velocity vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Y Velocity (pixels/s)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'velocity_vs_time.png'), dpi=300)
        plt.close()
        
        # Calculate speed (magnitude of velocity)
        left_speed = [np.sqrt(vx**2 + vy**2) for vx, vy in zip(left_vx, left_vy)]
        right_speed = [np.sqrt(vx**2 + vy**2) for vx, vy in zip(right_vx, right_vy)]
        
        # Plot speed versus time
        plt.figure(figsize=(10, 6))
        plt.plot(velocity_timestamps, left_speed, 'b-', label='Left Camera')
        plt.plot(velocity_timestamps, right_speed, 'r-', label='Right Camera')
        plt.title('Ball Speed vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Speed (pixels/s)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'speed_vs_time.png'), dpi=300)
        plt.close()
    
    # Create 2D animation of the rolling ball
    create_rolling_animation(left_x, left_y, right_x, right_y, timestamps, output_dir)

def create_rolling_animation(left_x, left_y, right_x, right_y, timestamps, output_dir):
    """Create an animated visualization of the rolling ball."""
    # Create animation for left camera
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Set axes limits with some padding
    x_pad_left = (max(left_x) - min(left_x)) * 0.1
    y_pad_left = (max(left_y) - min(left_y)) * 0.1
    x_pad_right = (max(right_x) - min(right_x)) * 0.1
    y_pad_right = (max(right_y) - min(right_y)) * 0.1
    
    ax1.set_xlim(min(left_x) - x_pad_left, max(left_x) + x_pad_left)
    ax1.set_ylim(max(left_y) + y_pad_left, min(left_y) - y_pad_left)  # Invert Y-axis
    ax2.set_xlim(min(right_x) - x_pad_right, max(right_x) + x_pad_right)
    ax2.set_ylim(max(right_y) + y_pad_right, min(right_y) - y_pad_right)  # Invert Y-axis
    
    ax1.set_xlabel('X Position (pixels)')
    ax1.set_ylabel('Y Position (pixels)')
    ax1.set_title('Ball Motion (Left Camera)')
    ax1.grid(True)
    
    ax2.set_xlabel('X Position (pixels)')
    ax2.set_ylabel('Y Position (pixels)')
    ax2.set_title('Ball Motion (Right Camera)')
    ax2.grid(True)
    
    # Initialize balls and paths
    ball_left, = ax1.plot([], [], 'ro', ms=10)
    path_left, = ax1.plot([], [], 'b-', alpha=0.7)
    
    ball_right, = ax2.plot([], [], 'ro', ms=10)
    path_right, = ax2.plot([], [], 'b-', alpha=0.7)
    
    time_text = fig.text(0.5, 0.95, '', ha='center')
    
    left_x_data, left_y_data = [], []
    right_x_data, right_y_data = [], []
    
    # Animation initialization function
    def init():
        ball_left.set_data([], [])
        path_left.set_data([], [])
        ball_right.set_data([], [])
        path_right.set_data([], [])
        time_text.set_text('')
        return ball_left, path_left, ball_right, path_right, time_text
    
    # Animation update function
    def update(frame):
        left_x_data.append(left_x[frame])
        left_y_data.append(left_y[frame])
        right_x_data.append(right_x[frame])
        right_y_data.append(right_y[frame])
        
        ball_left.set_data([left_x[frame], left_y[frame]])
        path_left.set_data(left_x_data, left_y_data)
        
        ball_right.set_data([right_x[frame], right_y[frame]])
        path_right.set_data(right_x_data, right_y_data)
        
        time_text.set_text(f'Time: {timestamps[frame]:.2f} s')
        return ball_left, path_left, ball_right, path_right, time_text
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=len(timestamps),
                       init_func=init, blit=True, interval=50)
    
    # Save animation
    ani.save(os.path.join(output_dir, 'rolling_ball_animation.mp4'), 
             writer='ffmpeg', fps=20, dpi=100)
    plt.close()

def find_video_by_keyword(test_dir, camera_dir, keyword):
    """Find a video file containing a specific keyword in the filename."""
    camera_path = os.path.join(test_dir, camera_dir)
    
    # Check if directory exists
    if not os.path.exists(camera_path):
        print(f"Error: Camera directory {camera_path} not found")
        return None
    
    # Try different video extensions
    for ext in ['.mp4', '.mov', '.MP4', '.MOV']:
        # Search for files containing the keyword
        matches = glob.glob(os.path.join(camera_path, f"*{keyword}*{ext}"))
        if matches:
            return matches[0]
    
    return None

def synchronize_and_analyze_motion(test_dir, video_keyword, analysis_type, model, delay):
    """Main function to handle synchronization and motion analysis."""
    # Find video files
    left_video = find_video_by_keyword(test_dir, "left_camera", video_keyword)
    right_video = find_video_by_keyword(test_dir, "right_camera", video_keyword)
    
    if not left_video or not right_video:
        print(f"Error: Could not find videos with keyword '{video_keyword}'")
        return
    
    print(f"Found video pair for {analysis_type} analysis:")
    print(f"  Left: {os.path.basename(left_video)}")
    print(f"  Right: {os.path.basename(right_video)}")
    
    # Create output directory
    output_dir = os.path.join(test_dir, "results", f"{analysis_type}_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load or perform flash synchronization
    sync_data = load_sync_data(test_dir)
    
    if not sync_data:
        print("No synchronization data found. Running flash detection...")
        # Perform flash detection
        left_flash_frame = analyze_brightness_jump(left_video)
        right_flash_frame = analyze_brightness_jump(right_video)
        
        if left_flash_frame is not None and right_flash_frame is not None:
            frame_offset = right_flash_frame - left_flash_frame
            print(f"Flash detection successful. Frame offset: {frame_offset}")
            
            # Create sync data
            sync_data = {
                'left_flash_frame': left_flash_frame,
                'right_flash_frame': right_flash_frame,
                'frame_offset': frame_offset
            }
            
            # Save sync data
            sync_dir = os.path.join(test_dir, "results", "sync_results")
            os.makedirs(sync_dir, exist_ok=True)
            
            with open(os.path.join(sync_dir, "sync_data.pkl"), 'wb') as f:
                pickle.dump(sync_data, f)
        else:
            print("Flash detection failed. Cannot perform synchronized analysis.")
            return
    
    # Perform type-specific analysis
    if analysis_type == "drop":
        from ball_drop import analyze_ball_drop
        from ball_drop import load_calibration
        # Assuming ball_drop_analyzer is imported or defined
        analyze_ball_drop(left_video, right_video, model, load_calibration(test_dir), 
                         sync_data['frame_offset'], output_dir)
    
    elif analysis_type == "roll":
        process_rolling_ball(left_video, right_video, model, sync_data, output_dir, delay)
    
    elif analysis_type == "jumping":
        # Implement jumping analysis (would require human pose detection)
        print("Human jumping analysis not implemented yet")
    
    else:
        print(f"Unknown analysis type: {analysis_type}")

def main():
    parser = argparse.ArgumentParser(description="Analyze motion from stereo videos")
    parser.add_argument("--test_dir", required=True, help="Test directory (e.g., motion_v1.2)")
    parser.add_argument("--base_dir", default=".", help="Base directory containing the data folder")
    parser.add_argument("--model", default="yolov8m.pt", help="Path to YOLOv8 model")
    parser.add_argument("--device", default="mps", help="Device to run inference on (cpu, cuda, mps)")
    parser.add_argument("--type", default="roll", 
                      choices=["drop", "roll", "jumping"],
                      help="Type of motion analysis to perform")
    parser.add_argument("--keyword", default=None, 
                      help="Keyword to identify video files (default: same as type)")
    parser.add_argument("--delay", type=float, default=3.0, 
                  help="Delay in seconds after flash before starting analysis")
    
    args = parser.parse_args()
    
    # Set full test path
    test_dir = os.path.join(args.base_dir, "data", args.test_dir)
    
    # Check if test directory exists
    if not os.path.exists(test_dir):
        print(f"Error: Test directory {test_dir} not found")
        return
    
    # Use type as keyword if not specified
    if args.keyword is None:
        args.keyword = args.type
    
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
    
    # Perform synchronization and analysis
    synchronize_and_analyze_motion(test_dir, args.keyword, args.type, model, args.delay)

if __name__ == "__main__":
    main()