#!/usr/bin/env python3
import cv2
import numpy as np
import os
import argparse
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import subprocess
import json
import re
import time

def extract_timestamps_ffmpeg(video_path):
    """
    Extract timestamp metadata from a video file using ffmpeg.
    Returns a list of timestamps for each frame in seconds.
    """
    print(f"Extracting timestamps from {os.path.basename(video_path)}...")
    
    # Try to get creation time first to establish an absolute reference
    creation_time = extract_creation_time(video_path)
    if creation_time:
        base_time = creation_time.timestamp()
        print(f"Video creation time: {creation_time.isoformat()}")
    else:
        base_time = 0
        print("Could not determine video creation time, using relative timestamps only.")
    
    # Try to get more detailed metadata including timestamps
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-select_streams', 'v:0',
        '-show_entries', 'frame=pkt_pts_time,pkt_dts_time,best_effort_timestamp_time,pkt_duration_time',
        '-of', 'json',
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f"Error executing ffprobe: {result.stderr}")
            return None
        
        frame_data = json.loads(result.stdout)
        timestamps = []
        
        # Get video frame rate for estimating missing timestamps
        fps_cmd = [
            'ffprobe',
            '-v', 'quiet', 
            '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate',
            '-of', 'json',
            video_path
        ]
        fps_result = subprocess.run(fps_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        fps = 30.0  # Default fallback
        
        if fps_result.returncode == 0:
            try:
                fps_data = json.loads(fps_result.stdout)
                if 'streams' in fps_data and len(fps_data['streams']) > 0:
                    # Parse fraction like "30000/1001"
                    fraction = fps_data['streams'][0]['r_frame_rate'].split('/')
                    if len(fraction) == 2:
                        fps = float(fraction[0]) / float(fraction[1])
                    else:
                        fps = float(fraction[0])
                    print(f"Video frame rate: {fps:.3f} fps")
            except Exception as e:
                print(f"Error parsing frame rate: {e}")
        
        # Check if we have any metadata timestamps
        has_real_timestamps = False
        if 'frames' in frame_data and len(frame_data['frames']) > 0:
            sample_frame = frame_data['frames'][0]
            for key in ['pkt_pts_time', 'best_effort_timestamp_time']:
                if key in sample_frame and sample_frame[key] is not None:
                    has_real_timestamps = True
                    break
        
        if not has_real_timestamps:
            print("No embedded timestamps found in video. Using estimated timestamps based on frame rate.")
            # Generate timestamps based on frame rate
            frame_count = len(frame_data['frames'])
            frame_duration = 1.0 / fps
            for i in range(frame_count):
                timestamps.append(base_time + i * frame_duration)
            return timestamps
        
        # Extract actual timestamps where available
        for i, frame in enumerate(frame_data['frames']):
            timestamp = None
            
            # Try different timestamp fields in order of preference
            for field in ['best_effort_timestamp_time', 'pkt_pts_time', 'pkt_dts_time']:
                if field in frame and frame[field] is not None:
                    try:
                        timestamp = float(frame[field])
                        break
                    except (ValueError, TypeError):
                        pass
            
            if timestamp is not None:
                # Add base time to make it absolute if we have creation time
                if base_time > 0:
                    timestamp += base_time
                timestamps.append(timestamp)
            else:
                # If timestamp is missing, estimate based on previous timestamp
                if timestamps:
                    timestamps.append(timestamps[-1] + (1.0/fps))
                else:
                    timestamps.append(base_time)  # Start at base time if this is the first frame
        
        # Check if timestamps are meaningful (not all zeros or same value)
        if len(timestamps) > 1:
            all_same = all(t == timestamps[0] for t in timestamps)
            if all_same:
                print("Warning: All extracted timestamps are identical. Using estimated timestamps instead.")
                # Generate timestamps based on frame rate
                frame_count = len(frame_data['frames'])
                frame_duration = 1.0 / fps
                timestamps = [base_time + i * frame_duration for i in range(frame_count)]
        
        return timestamps
    
    except Exception as e:
        print(f"Error extracting timestamps: {e}")
        return None

def extract_creation_time(video_path):
    """Extract the creation time of the video file."""
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-select_streams', 'v:0',
        '-show_entries', 'format_tags=creation_time',
        '-of', 'json',
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f"Error getting creation time: {result.stderr}")
            return None
        
        data = json.loads(result.stdout)
        if 'format' in data and 'tags' in data['format'] and 'creation_time' in data['format']['tags']:
            creation_time_str = data['format']['tags']['creation_time']
            # Parse ISO format like '2023-09-15T14:30:15.000000Z'
            dt = datetime.datetime.fromisoformat(creation_time_str.replace('Z', '+00:00'))
            return dt
        
        return None
    except Exception as e:
        print(f"Error extracting creation time: {e}")
        return None

def match_frames_by_timestamp(left_timestamps, right_timestamps, max_time_diff=0.1):
    """
    Match frames from two videos based on their timestamps.
    Returns a list of (left_frame_idx, right_frame_idx) pairs.
    """
    print("Matching frames based on timestamps...")
    matched_pairs = []
    
    # Find the time offset between the two videos
    if len(left_timestamps) > 0 and len(right_timestamps) > 0:
        # Simple approach: assume linear relationship and find best offset
        # by checking the first few frames
        best_offset = 0
        min_diff = float('inf')
        
        for offset in range(-20, 21):  # Try offsets from -20 to +20 frames
            if offset < 0 and abs(offset) >= len(left_timestamps):
                continue
            if offset > 0 and offset >= len(right_timestamps):
                continue
                
            if offset < 0:
                left_idx = abs(offset)
                right_idx = 0
            else:
                left_idx = 0
                right_idx = offset
            
            # Compare 10 frames (or as many as available)
            total_diff = 0
            count = 0
            
            for i in range(10):
                if left_idx + i >= len(left_timestamps) or right_idx + i >= len(right_timestamps):
                    break
                
                time_diff = abs(left_timestamps[left_idx + i] - right_timestamps[right_idx + i])
                total_diff += time_diff
                count += 1
            
            if count > 0:
                avg_diff = total_diff / count
                if avg_diff < min_diff:
                    min_diff = avg_diff
                    best_offset = offset
        
        print(f"Best frame offset: {best_offset} (avg diff: {min_diff:.6f}s)")
        
        # Apply the best offset to match frames
        if best_offset < 0:
            left_start = abs(best_offset)
            right_start = 0
        else:
            left_start = 0
            right_start = best_offset
        
        # Match frames based on timestamp proximity
        for i in range(min(len(left_timestamps) - left_start, len(right_timestamps) - right_start)):
            left_idx = left_start + i
            right_idx = right_start + i
            
            time_diff = abs(left_timestamps[left_idx] - right_timestamps[right_idx])
            if time_diff <= max_time_diff:
                matched_pairs.append((left_idx, right_idx))
    
    if not matched_pairs:
        print("Failed to match frames using timestamps. Trying direct frame matching...")
        # Fallback: assume videos start at approximately the same time
        # and have the same frame rate
        min_frames = min(len(left_timestamps), len(right_timestamps))
        matched_pairs = [(i, i) for i in range(min_frames)]
    
    return matched_pairs

def create_synchronized_videos(left_video, right_video, left_flash_frame, right_flash_frame, output_dir):
    """Create new synchronized videos that start at the same time."""
    print("Creating synchronized videos...")
    
    left_cap = cv2.VideoCapture(left_video)
    right_cap = cv2.VideoCapture(right_video)
    
    if not left_cap.isOpened() or not right_cap.isOpened():
        print("Error: Could not open videos")
        return
    
    # Get video properties
    left_fps = left_cap.get(cv2.CAP_PROP_FPS)
    right_fps = right_cap.get(cv2.CAP_PROP_FPS)
    
    # Check if frame rates match closely enough
    if abs(left_fps - right_fps) > 0.5:
        print(f"Warning: Frame rates differ significantly (left: {left_fps}, right: {right_fps})")
        print("Proceeding anyway but synchronization might not be perfect")
    
    # Use properties from left video
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
    
    # Skip to 3 seconds after the flash frame to avoid setup frames
    frames_to_skip = 3 * int(fps)
    left_start_frame = left_flash_frame + frames_to_skip
    right_start_frame = right_flash_frame + frames_to_skip
    
    left_cap.set(cv2.CAP_PROP_POS_FRAMES, left_start_frame)
    right_cap.set(cv2.CAP_PROP_POS_FRAMES, right_start_frame)
    
    # Define how many frames to process after sync point
    # Default to 10 seconds of footage
    frames_to_process = 10 * int(fps)
    
    print(f"Extracting synchronized frames starting {frames_to_skip} frames after flash")
    print(f"Left starting at frame {left_start_frame}, right at frame {right_start_frame}")
    
    # Process frames
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
    
    # Release resources
    left_cap.release()
    right_cap.release()
    left_out.release()
    right_out.release()
    
    print(f"Created synchronized videos:")
    print(f"Left: {left_output_path}")
    print(f"Right: {right_output_path}")
    
    return left_output_path, right_output_path

def extract_frames(video_path, frame_indices):
    """Extract specific frames from a video file."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return frames
    
    max_index = max(frame_indices) if frame_indices else 0
    frame_idx = 0
    
    while cap.isOpened() and frame_idx <= max_index:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx in frame_indices:
            frames.append(frame)
        
        frame_idx += 1
    
    cap.release()
    return frames

def visualize_sync(left_frames, right_frames, matched_pairs, output_dir, sample_count=5):
    """
    Visualize the synchronized frames by showing matched pairs side by side.
    
    Args:
        left_frames: List of frames from left camera
        right_frames: List of frames from right camera
        matched_pairs: List of (left_idx, right_idx) pairs 
        output_dir: Directory to save visualizations
        sample_count: Number of visualization images to create
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure we don't try to visualize more pairs than we have frames
    max_visualizations = min(len(matched_pairs), len(left_frames), len(right_frames), sample_count)
    
    # For direct frame lists, we assume the frames and matched_pairs are already aligned
    for i in range(max_visualizations):
        left_idx, right_idx = matched_pairs[i]
        
        if i < len(left_frames) and i < len(right_frames):
            left_frame = left_frames[i]
            right_frame = right_frames[i]
            
            # Resize frames to the same height if needed
            h1, w1 = left_frame.shape[:2]
            h2, w2 = right_frame.shape[:2]
            
            if h1 != h2:
                # Resize to match height
                scale = h1 / h2
                width = int(w2 * scale)
                right_frame = cv2.resize(right_frame, (width, h1))
            
            # Create side-by-side comparison
            combined = np.hstack((left_frame, right_frame))
            
            # Add frame numbers as text
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(combined, f"Left: {left_idx}", (10, 30), font, 1, (0, 255, 0), 2)
            cv2.putText(combined, f"Right: {right_idx}", (w1 + 10, 30), font, 1, (0, 255, 0), 2)
            
            # Add timestamp if available in the filename
            if hasattr(combined, 'timestamp'):
                timestamp_str = datetime.datetime.fromtimestamp(combined.timestamp).strftime('%H:%M:%S.%f')[:-3]
                cv2.putText(combined, timestamp_str, (10, h1 - 10), font, 0.6, (255, 255, 255), 1)
            
            # Save the visualization
            output_path = os.path.join(output_dir, f"sync_visualization_{i:02d}.png")
            cv2.imwrite(output_path, combined)
            print(f"Saved visualization to {output_path}")

def analyze_brightness_jump(video_path, threshold=20, window_size=5, max_frames=900):
    """
    Detect frames with sudden brightness increases that could indicate a flash.
    Returns the frame index of detected flash or None.
    
    Args:
        video_path: Path to the video file
        threshold: Minimum brightness increase to detect (0-255 scale)
        window_size: Number of frames to average for baseline brightness
        max_frames: Maximum number of frames to check (30 seconds at 30fps)
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
    
    # If no flash found, still create visualization
    if brightness_values:
        plt.figure(figsize=(10, 6))
        plt.plot(brightness_values)
        plt.axhline(y=np.mean(brightness_values) + threshold, color='g', linestyle='-.', label=f'Threshold ({threshold:.1f})')
        plt.xlabel('Frame Number')
        plt.ylabel('Average Brightness')
        plt.title(f'Brightness Analysis - {os.path.basename(video_path)} (No Flash Detected)')
        plt.legend()
        
        # Ensure output directory exists
        output_dir = os.path.join(os.path.dirname(os.path.dirname(video_path)), "results", "sync_results")
        os.makedirs(output_dir, exist_ok=True)
        
        plt.savefig(os.path.join(output_dir, f"brightness_analysis_{os.path.basename(video_path)}.png"))
        plt.close()
    
    print("No significant brightness jump detected")
    return None

def main():
    parser = argparse.ArgumentParser(description='Test video synchronization using embedded timecodes')
    parser.add_argument('--left_video', required=True, help='Path to left camera video')
    parser.add_argument('--right_video', required=True, help='Path to right camera video')
    parser.add_argument('--output_dir', default=None, help='Directory to save visualization results (default: auto-detect)')
    parser.add_argument('--max_time_diff', type=float, default=0.1, help='Maximum time difference (in seconds) between matched frames')
    parser.add_argument('--detect_flash', action='store_true', help='Try to detect a flash/brightness jump for synchronization')
    parser.add_argument('--flash_threshold', type=float, default=20.0, help='Brightness threshold for flash detection')
    parser.add_argument('--test_dir', default=None, help='Test directory name (e.g., test_008)')
    
    args = parser.parse_args()
    
    # Ensure videos exist
    for video_path in [args.left_video, args.right_video]:
        if not os.path.exists(video_path):
            print(f"Error: Video file {video_path} does not exist")
            return
    
    # Automatically determine output directory if not specified
    if args.output_dir is None:
        # Extract test directory from video path if not specified
        if args.test_dir is None:
            # Try to extract test directory from video path
            # Assuming path like data/test_001/left_camera/video.mp4
            video_dir = os.path.dirname(args.left_video)
            parent_dir = os.path.dirname(video_dir)
            
            if os.path.basename(parent_dir).startswith('test_'):
                args.test_dir = os.path.basename(parent_dir)
                print(f"Automatically detected test directory: {args.test_dir}")
            else:
                # Try one level up
                parent_parent = os.path.dirname(parent_dir)
                if os.path.basename(parent_parent).startswith('test_'):
                    args.test_dir = os.path.basename(parent_parent)
                    print(f"Automatically detected test directory: {args.test_dir}")
                else:
                    print("Could not automatically detect test directory")
                    # Create a directory structure at the same level as the videos
                    args.output_dir = os.path.join(os.path.dirname(os.path.dirname(args.left_video)), "results", "sync_results")
        else:
            # If test_dir is specified, use it to build the output path
            args.output_dir = os.path.join(os.path.dirname(os.path.dirname(args.left_video)), "results", "sync_results")
    
    # Create output directory
    if args.output_dir is None:
        # Last resort fallback if we still don't have an output directory
        args.output_dir = os.path.join(os.getcwd(), "sync_results")
    
    print(f"Setting output directory to: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Method 1: Try flash detection first if requested
    left_flash_frame = None
    right_flash_frame = None
    
    if args.detect_flash:
        left_flash_frame = analyze_brightness_jump(args.left_video, threshold=args.flash_threshold)
        right_flash_frame = analyze_brightness_jump(args.right_video, threshold=args.flash_threshold)
        
        if left_flash_frame is not None and right_flash_frame is not None:
            print(f"Found flash frames - Left: {left_flash_frame}, Right: {right_flash_frame}")
            frame_offset = right_flash_frame - left_flash_frame
            print(f"Frame offset (right - left): {frame_offset}")
            
            # Extract frame rates to account for potential differences
            left_cap = cv2.VideoCapture(args.left_video)
            right_cap = cv2.VideoCapture(args.right_video)
            
            left_fps = left_cap.get(cv2.CAP_PROP_FPS)
            right_fps = right_cap.get(cv2.CAP_PROP_FPS)
            left_frame_count = int(left_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            right_frame_count = int(right_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            left_cap.release()
            right_cap.release()
            
            print(f"Left video: {left_fps:.2f} fps, {left_frame_count} frames")
            print(f"Right video: {right_fps:.2f} fps, {right_frame_count} frames")
            
            # Check if frame rates match
            if abs(left_fps - right_fps) > 0.1:
                print(f"WARNING: Frame rates differ by {abs(left_fps - right_fps):.2f} fps")
                print("Frame indexes may not stay perfectly synchronized over time")
                # TODO: Implement time-based matching for different frame rates
            
            # Create frame pairs based on flash detection
            max_frames = min(
                left_frame_count - left_flash_frame,
                right_frame_count - right_flash_frame
            )
            
            matched_pairs = [(left_flash_frame + i, right_flash_frame + i) for i in range(max_frames)]
            
            # Extract frames for visualization at different points in the video
            sample_positions = [0.0, 0.25, 0.5, 0.75, 0.9]  # Positions as fraction of video length
            sample_indices = [min(int(pos * len(matched_pairs)), len(matched_pairs)-1) for pos in sample_positions]
            
            left_frames = extract_frames(args.left_video, [pair[0] for pair in [matched_pairs[i] for i in sample_indices]])
            right_frames = extract_frames(args.right_video, [pair[1] for pair in [matched_pairs[i] for i in sample_indices]])
            
            # Visualize the matches
            visualize_sync(left_frames, right_frames, [matched_pairs[i] for i in sample_indices], args.output_dir)

            # Create synchronized videos
            sync_left, sync_right = create_synchronized_videos(
                args.left_video, args.right_video, 
                left_flash_frame, right_flash_frame, 
                args.output_dir
            )

            
            # Save all information needed for synchronization
            sync_info = {
                'method': 'flash_detection',
                'frame_offset': frame_offset,
                'left_flash_frame': left_flash_frame,
                'right_flash_frame': right_flash_frame,
                'left_fps': left_fps,
                'right_fps': right_fps,
                'left_frame_count': left_frame_count,
                'right_frame_count': right_frame_count,
                'left_video': os.path.basename(args.left_video),
                'right_video': os.path.basename(args.right_video),
                'sync_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'matched_pairs': matched_pairs[:100]  # Save first 100 pairs as example
            }

            sync_info['synchronized_left'] = sync_left
            sync_info['synchronized_right'] = sync_right
            
            # Save in both JSON for humans and pickle for programs
            with open(os.path.join(args.output_dir, 'sync_info.json'), 'w') as f:
                json.dump(sync_info, f, indent=2)
            
            try:
                import pickle
                with open(os.path.join(args.output_dir, 'sync_data.pkl'), 'wb') as f:
                    pickle.dump(sync_info, f)
            except Exception as e:
                print(f"Warning: Could not save pickle file: {e}")
            
            print(f"Synchronized videos based on flash detection. Frame offset: {frame_offset}")
            
            # Generate a report file with details and instructions
            with open(os.path.join(args.output_dir, 'sync_report.txt'), 'w') as f:
                f.write("Video Synchronization Report\n")
                f.write("==========================\n\n")
                f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Left video: {args.left_video}\n")
                f.write(f"Right video: {args.right_video}\n\n")
                f.write("Synchronization method: Flash detection\n")
                f.write(f"  Left flash frame: {left_flash_frame}\n")
                f.write(f"  Right flash frame: {right_flash_frame}\n")
                f.write(f"  Frame offset (right - left): {frame_offset}\n\n")
                f.write("Video properties:\n")
                f.write(f"  Left: {left_fps:.2f} fps, {left_frame_count} frames\n")
                f.write(f"  Right: {right_fps:.2f} fps, {right_frame_count} frames\n\n")
                f.write("Usage instructions:\n")
                f.write("  To get the right frame corresponding to left frame X:\n")
                f.write(f"    right_frame = X + {frame_offset}\n\n")
                f.write("  To get the left frame corresponding to right frame Y:\n")
                f.write(f"    left_frame = Y - {frame_offset}\n")
            
            return
    
if __name__ == "__main__":
    main()