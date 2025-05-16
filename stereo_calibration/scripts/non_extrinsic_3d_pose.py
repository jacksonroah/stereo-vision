#!/usr/bin/env python3
# USE THIS FOR NON CALIBRATED TWO CAMERAS
import cv2
import mediapipe as mp
import numpy as np
import math
import argparse
import os
import glob
import time
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pickle

class DualCameraPoseEstimator:
    def __init__(self, test_dir, output_dir=None):
        """Initialize the dual camera pose estimation system"""
        # Setup base directories
        self.test_dir = test_dir
        self.base_dir = os.path.join("data", test_dir)
        
        if output_dir is None:
            self.output_dir = os.path.join(self.base_dir, "results", "dual_pose")
        else:
            self.output_dir = output_dir
            
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "frames"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "data"), exist_ok=True)
        
        # Setup camera paths
        self.cam1_dir = os.path.join(self.base_dir, "cam1")
        self.cam2_dir = os.path.join(self.base_dir, "cam2")
        self.cam1_raw_dir = os.path.join(self.cam1_dir, "raw_video")
        self.cam2_raw_dir = os.path.join(self.cam2_dir, "raw_video")
        
        # Setup MediaPipe Pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
        
        # Data storage
        self.left_angle_data = []
        self.right_angle_data = []
        self.combined_angle_data = []
        self.frame_timestamps = []
        
        # Settings
        self.viz_settings = {
            'show_video': True,
            'save_frames': True,
            'plot_angles': True,
            'save_interval': 30,  # Save every 30 frames
        }
        
        # Pose processing settings
        self.smooth_window = 5  # Frames for smoothing window
        self.confidence_threshold = 0.65  # Minimum confidence for landmarks
        self.temporal_filter_weight = 0.7  # Weight for current frame (1-weight for history)
        
        # Load intrinsic calibration data (not using extrinsic calibration)
        self.intrinsic_data = self.load_intrinsic_data()
        if self.intrinsic_data is None:
            print("Warning: Intrinsic calibration data not found. Using uncalibrated camera.")
            
        # Initialize cache for temporal filtering
        self.landmark_history = {
            'left': [],
            'right': []
        }
        
        # Joint tracking statistics
        self.joint_stats = {
            'right_shoulder': {'left_visible': 0, 'right_visible': 0, 'both_visible': 0, 'neither_visible': 0},
            'left_shoulder': {'left_visible': 0, 'right_visible': 0, 'both_visible': 0, 'neither_visible': 0},
            'right_elbow': {'left_visible': 0, 'right_visible': 0, 'both_visible': 0, 'neither_visible': 0},
            'left_elbow': {'left_visible': 0, 'right_visible': 0, 'both_visible': 0, 'neither_visible': 0},
            'right_hip': {'left_visible': 0, 'right_visible': 0, 'both_visible': 0, 'neither_visible': 0},
            'left_hip': {'left_visible': 0, 'right_visible': 0, 'both_visible': 0, 'neither_visible': 0},
            'right_knee': {'left_visible': 0, 'right_visible': 0, 'both_visible': 0, 'neither_visible': 0},
            'left_knee': {'left_visible': 0, 'right_visible': 0, 'both_visible': 0, 'neither_visible': 0},
        }
        
        # Define key points for angle calculations
        self.joint_connections = {
            'right_shoulder': [
                self.mp_pose.PoseLandmark.RIGHT_HIP.value,
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                self.mp_pose.PoseLandmark.RIGHT_ELBOW.value
            ],
            'left_shoulder': [
                self.mp_pose.PoseLandmark.LEFT_HIP.value,
                self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                self.mp_pose.PoseLandmark.LEFT_ELBOW.value
            ],
            'right_elbow': [
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                self.mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                self.mp_pose.PoseLandmark.RIGHT_WRIST.value
            ],
            'left_elbow': [
                self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                self.mp_pose.PoseLandmark.LEFT_ELBOW.value,
                self.mp_pose.PoseLandmark.LEFT_WRIST.value
            ],
            'right_hip': [
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                self.mp_pose.PoseLandmark.RIGHT_HIP.value,
                self.mp_pose.PoseLandmark.RIGHT_KNEE.value
            ],
            'left_hip': [
                self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                self.mp_pose.PoseLandmark.LEFT_HIP.value,
                self.mp_pose.PoseLandmark.LEFT_KNEE.value
            ],
            'right_knee': [
                self.mp_pose.PoseLandmark.RIGHT_HIP.value,
                self.mp_pose.PoseLandmark.RIGHT_KNEE.value,
                self.mp_pose.PoseLandmark.RIGHT_ANKLE.value
            ],
            'left_knee': [
                self.mp_pose.PoseLandmark.LEFT_HIP.value,
                self.mp_pose.PoseLandmark.LEFT_KNEE.value,
                self.mp_pose.PoseLandmark.LEFT_ANKLE.value
            ]
        }
        
        print(f"Dual Camera Pose Estimator initialized for test directory: {test_dir}")
    
    def load_intrinsic_data(self):
        """Load intrinsic calibration data for both cameras"""
        intrinsic_dir = os.path.join(self.base_dir, "results", "intrinsic_params")
        
        # Check if directory exists
        if not os.path.exists(intrinsic_dir):
            print(f"Intrinsic calibration directory not found: {intrinsic_dir}")
            return None
        
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
                left_matrix_path = os.path.join(intrinsic_dir, "left_matrix.txt")
                left_dist_path = os.path.join(intrinsic_dir, "left_distortion.txt")
                right_matrix_path = os.path.join(intrinsic_dir, "right_matrix.txt")
                right_dist_path = os.path.join(intrinsic_dir, "right_distortion.txt")
                
                if not os.path.exists(left_matrix_path) or not os.path.exists(right_matrix_path):
                    print("Intrinsic calibration files not found!")
                    return None
                
                left_matrix = np.loadtxt(left_matrix_path)
                left_dist = np.loadtxt(left_dist_path)
                right_matrix = np.loadtxt(right_matrix_path)
                right_dist = np.loadtxt(right_dist_path)
            
            return {
                'left_matrix': left_matrix,
                'left_dist': left_dist,
                'right_matrix': right_matrix,
                'right_dist': right_dist
            }
            
        except Exception as e:
            print(f"Error loading intrinsic calibration data: {e}")
            return None
    
    def find_video_pairs(self, pattern):
        """Find matching video pairs from both cameras"""
        # Find videos in each camera directory matching the pattern
        cam1_videos = glob.glob(os.path.join(self.cam1_dir, "raw_video", f"*{pattern}*.MOV"))
        cam2_videos = glob.glob(os.path.join(self.cam2_dir, "raw_video", f"*{pattern}*.MOV"))
        
        # If no videos in raw_video, try the calibration directory
        if not cam1_videos:
            cam1_videos = glob.glob(os.path.join(self.cam1_dir, "calibration", f"*{pattern}*.MOV"))
        if not cam2_videos:
            cam2_videos = glob.glob(os.path.join(self.cam2_dir, "calibration", f"*{pattern}*.MOV"))
        
        # Check if we found matching videos
        if not cam1_videos or not cam2_videos:
            print(f"Could not find matching videos with pattern '{pattern}'")
            print(f"Camera 1 videos found: {len(cam1_videos)}")
            print(f"Camera 2 videos found: {len(cam2_videos)}")
            return None, None
        
        # For simplicity, use the first matching video from each camera
        return cam1_videos[0], cam2_videos[0]
    
    def analyze_brightness_jump(self, video_path, threshold=20, window_size=5, max_frames=300):
        """
        Detect frames with sudden brightness increases that could indicate a flash.
        Returns the frame index of detected flash or None.
        
        Args:
            video_path: Path to the video file
            threshold: Minimum brightness increase to detect (0-255 scale)
            window_size: Number of frames to average for baseline brightness
            max_frames: Maximum number of frames to check (10 seconds at 30fps)
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
        
        # Main analysis loop - focus on first 10 seconds (approx. 300 frames at 30fps)
        while cap.isOpened() and frame_count < max_frames:
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
                    print(f"Detected flash at frame {frame_count}: {prev_avg:.1f} -> {brightness:.1f}")
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
                    os.makedirs(os.path.join(self.output_dir, "sync"), exist_ok=True)
                    
                    plt.savefig(os.path.join(self.output_dir, "sync", f"brightness_analysis_{os.path.basename(video_path)}.png"))
                    plt.close()
                    
                    cap.release()
                    return frame_count
            
            frame_count += 1
        
        cap.release()
        print(f"No flash detected in first {max_frames} frames of {os.path.basename(video_path)}")
        return None
    
    def determine_sync_offset(self, left_video, right_video):
        """Determine frame offset between videos using flash detection"""
        print("Determining synchronization offset...")
        
        # Detect flash in first 300 frames (10 seconds at 30fps)
        left_flash_frame = self.analyze_brightness_jump(left_video, max_frames=300)
        right_flash_frame = self.analyze_brightness_jump(right_video, max_frames=300)
        
        if left_flash_frame is None or right_flash_frame is None:
            print("Could not determine sync offset - flash not detected in one or both videos")
            return 0  # Default to no offset
        
        # Calculate frame offset
        frame_offset = right_flash_frame - left_flash_frame
        print(f"Synchronization offset: {frame_offset} frames (positive means right camera starts later)")
        
        # Save synchronization information
        sync_info = {
            'method': 'flash_detection',
            'left_flash_frame': left_flash_frame,
            'right_flash_frame': right_flash_frame,
            'frame_offset': frame_offset,
            'sync_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'left_video': os.path.basename(left_video),
            'right_video': os.path.basename(right_video)
        }
        
        os.makedirs(os.path.join(self.output_dir, "sync"), exist_ok=True)
        
        # Save sync data
        with open(os.path.join(self.output_dir, "sync", "sync_data.pkl"), 'wb') as f:
            pickle.dump(sync_info, f)
        
        # Also save as JSON for human readability
        import json
        with open(os.path.join(self.output_dir, "sync", "sync_info.json"), 'w') as f:
            json.dump(sync_info, f, indent=2)
        
        return frame_offset
    
    def process_dual_videos(self, left_video, right_video, start_offset=90, max_frames=None):
        """
        Process both camera videos independently, selecting the best view for each joint.
        
        Args:
            left_video: Path to left camera video
            right_video: Path to right camera video
            start_offset: Frames to skip after flash before starting processing
            max_frames: Maximum number of frames to process (None for all)
        """
        # Determine sync offset
        frame_offset = self.determine_sync_offset(left_video, right_video)
        
        # Open video captures
        left_cap = cv2.VideoCapture(left_video)
        right_cap = cv2.VideoCapture(right_video)
        
        if not left_cap.isOpened() or not right_cap.isOpened():
            print("Error: Could not open one or both videos")
            return
        
        # Get video properties
        left_fps = left_cap.get(cv2.CAP_PROP_FPS)
        left_frame_count = int(left_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        left_width = int(left_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        left_height = int(left_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        right_fps = right_cap.get(cv2.CAP_PROP_FPS)
        right_frame_count = int(right_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Left video: {left_fps:.2f} fps, {left_frame_count} frames, {left_width}x{left_height}")
        print(f"Right video: {right_fps:.2f} fps, {right_frame_count} frames")
        
        # Adjust starting position based on sync offset and start_offset
        left_start_frame = start_offset
        right_start_frame = start_offset + frame_offset
        
        # Ensure both starting frames are valid
        if left_start_frame < 0:
            right_start_frame -= left_start_frame  # Adjust right frame if left is negative
            left_start_frame = 0
        if right_start_frame < 0:
            left_start_frame -= right_start_frame  # Adjust left frame if right is negative
            right_start_frame = 0
        
        print(f"Starting processing at frames - Left: {left_start_frame}, Right: {right_start_frame}")
        
        left_cap.set(cv2.CAP_PROP_POS_FRAMES, left_start_frame)
        right_cap.set(cv2.CAP_PROP_POS_FRAMES, right_start_frame)
        
        # Initialize MediaPipe pose detectors
        with self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,  # Use 1 for balance of accuracy and speed
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as left_pose, \
             self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as right_pose:
            
            # Initialize storage for pose data
            left_angle_sequence = []
            right_angle_sequence = []
            combined_angle_sequence = []
            timestamps = []
            
            # Frame counters
            frame_idx = 0
            left_frame_count = left_start_frame
            right_frame_count = right_start_frame
            
            # Initialize visualization window if needed
            if self.viz_settings['show_video']:
                cv2.namedWindow('Dual Pose', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Dual Pose', left_width * 2, left_height)
            
            # Main processing loop
            while left_cap.isOpened() and right_cap.isOpened():
                # Check if we've hit the max frames
                if max_frames is not None and frame_idx >= max_frames:
                    print(f"Reached maximum frame limit ({max_frames})")
                    break
                
                # Read frames
                left_ret, left_frame = left_cap.read()
                right_ret, right_frame = right_cap.read()
                
                # Check if either video ended
                if not left_ret or not right_ret:
                    print(f"End of video(s) reached after {frame_idx} frames")
                    break
                
                # Process frames with MediaPipe
                left_results = left_pose.process(cv2.cvtColor(left_frame, cv2.COLOR_BGR2RGB))
                right_results = right_pose.process(cv2.cvtColor(right_frame, cv2.COLOR_BGR2RGB))
                
                # Create annotated frames for visualization
                left_annotated = left_frame.copy()
                right_annotated = right_frame.copy()
                
                # Draw pose landmarks
                if left_results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(
                        left_annotated,
                        left_results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
                
                if right_results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(
                        right_annotated,
                        right_results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
                
                # Extract landmarks and calculate angles
                left_landmarks = None
                right_landmarks = None
                left_angles = {}
                right_angles = {}
                
                if left_results.pose_landmarks:
                    left_landmarks = self.extract_landmarks(left_results.pose_landmarks)
                    
                    # Apply temporal filtering
                    self.landmark_history['left'].append(left_landmarks)
                    if len(self.landmark_history['left']) > self.smooth_window:
                        self.landmark_history['left'].pop(0)
                    
                    if len(self.landmark_history['left']) > 1:
                        left_landmarks = self.apply_temporal_filter(self.landmark_history['left'])
                    
                    # Calculate angles
                    left_angles = self.calculate_angles(left_landmarks)
                
                if right_results.pose_landmarks:
                    right_landmarks = self.extract_landmarks(right_results.pose_landmarks)
                    
                    # Apply temporal filtering
                    self.landmark_history['right'].append(right_landmarks)
                    if len(self.landmark_history['right']) > self.smooth_window:
                        self.landmark_history['right'].pop(0)
                    
                    if len(self.landmark_history['right']) > 1:
                        right_landmarks = self.apply_temporal_filter(self.landmark_history['right'])
                    
                    # Calculate angles
                    right_angles = self.calculate_angles(right_landmarks)
                
                # Combine angles by selecting best view
                combined_angles = self.select_best_angles(left_angles, right_angles, left_landmarks, right_landmarks)
                
                # Update joint visibility statistics
                self.update_joint_stats(left_angles, right_angles)
                
                # Record data
                left_angle_sequence.append(left_angles)
                right_angle_sequence.append(right_angles)
                combined_angle_sequence.append(combined_angles)
                timestamps.append(frame_idx / left_fps)
                
                # Draw angle information on frames
                self.draw_angle_info(left_annotated, left_angles, "Left Camera")
                self.draw_angle_info(right_annotated, right_angles, "Right Camera")
                
                # Save frame at regular intervals
                if self.viz_settings['save_frames'] and frame_idx % self.viz_settings['save_interval'] == 0:
                    frame_path = os.path.join(self.output_dir, "frames", f"frame_{frame_idx:04d}.jpg")
                    combined_frame = np.hstack((left_annotated, right_annotated))
                    cv2.imwrite(frame_path, combined_frame)
                
                # Display frames
                if self.viz_settings['show_video']:
                    combined_frame = np.hstack((left_annotated, right_annotated))
                    cv2.imshow('Dual Pose', combined_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC key
                        break
                
                # Update frame counters
                frame_idx += 1
                left_frame_count += 1
                right_frame_count += 1
                
                # Print progress
                if frame_idx % 30 == 0:
                    print(f"Processed {frame_idx} frames")
            
            # Close video captures
            left_cap.release()
            right_cap.release()
            
            if self.viz_settings['show_video']:
                cv2.destroyAllWindows()
            
            # Save results
            if timestamps:
                self.save_results(left_angle_sequence, right_angle_sequence, 
                                 combined_angle_sequence, timestamps, left_fps)
                print(f"Processing complete. Processed {frame_idx} frames.")
            else:
                print("No valid pose data was detected in the videos.")
    
    def extract_landmarks(self, pose_landmarks):
        """Extract landmarks with visibility from MediaPipe results"""
        if pose_landmarks is None:
            return None
        
        landmarks = []
        for landmark in pose_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
        return np.array(landmarks)
    
    def apply_temporal_filter(self, landmark_history):
        """Apply weighted moving average filter to landmark history"""
        current = landmark_history[-1]
        previous = landmark_history[-2]
        
        # Weight current frame more heavily
        filtered = self.temporal_filter_weight * current + (1 - self.temporal_filter_weight) * previous
        return filtered
    
    def calculate_angles(self, landmarks):
        """Calculate joint angles from pose landmarks"""
        if landmarks is None:
            return {}
        
        angles = {}
        
        # For each joint defined in joint_connections
        for joint_name, points in self.joint_connections.items():
            # Check if all points have sufficient confidence
            if (self.check_point_confidence(landmarks, points[0]) and
                self.check_point_confidence(landmarks, points[1]) and
                self.check_point_confidence(landmarks, points[2])):
                
                # Calculate angle
                angle = self.calculate_angle(
                    landmarks[points[0]][:3],
                    landmarks[points[1]][:3],
                    landmarks[points[2]][:3]
                )
                
                angles[joint_name] = angle
        
        return angles
    
    def check_point_confidence(self, landmarks, point_idx, threshold=None):
        """Check if a landmark has sufficient confidence"""
        if threshold is None:
            threshold = self.confidence_threshold
            
        return landmarks[point_idx][3] >= threshold
    
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points in degrees"""
        # Create vectors
        ba = a - b
        bc = c - b
        
        # Normalize vectors
        ba_norm = np.linalg.norm(ba)
        bc_norm = np.linalg.norm(bc)
        
        if ba_norm == 0 or bc_norm == 0:
            return 0
        
        ba_normalized = ba / ba_norm
        bc_normalized = bc / bc_norm
        
        # Calculate dot product and convert to angle
        dot_product = np.dot(ba_normalized, bc_normalized)
        
        # Clamp dot product to valid range
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        angle_rad = np.arccos(dot_product)
        angle_deg = math.degrees(angle_rad)
        
        return angle_deg
    
    def select_best_angles(self, left_angles, right_angles, left_landmarks, right_landmarks):
        """Select angles from the camera with higher confidence for each joint"""
        combined_angles = {}
        
        # For each joint, select the angle from the camera with higher confidence
        for joint_name, points in self.joint_connections.items():
            left_available = joint_name in left_angles
            right_available = joint_name in right_angles
            
            if left_available and right_available:
                # Both cameras have this angle - select based on confidence
                left_conf = self.get_joint_confidence(left_landmarks, points)
                right_conf = self.get_joint_confidence(right_landmarks, points)
                
                if left_conf >= right_conf:
                    combined_angles[joint_name] = left_angles[joint_name]
                else:
                    combined_angles[joint_name] = right_angles[joint_name]
            
            elif left_available:
                # Only left camera has this angle
                combined_angles[joint_name] = left_angles[joint_name]
            
            elif right_available:
                # Only right camera has this angle
                combined_angles[joint_name] = right_angles[joint_name]
        
        return combined_angles
    
    def get_joint_confidence(self, landmarks, points):
       """Calculate average confidence for the points that define a joint angle"""
       if landmarks is None:
           return 0
           
       # Average the visibility values for the three points
       return (landmarks[points[0]][3] + landmarks[points[1]][3] + landmarks[points[2]][3]) / 3
   
    def update_joint_stats(self, left_angles, right_angles):
       """Update statistics on which camera can see each joint"""
       for joint_name in self.joint_stats.keys():
           left_visible = joint_name in left_angles
           right_visible = joint_name in right_angles
           
           if left_visible and right_visible:
               self.joint_stats[joint_name]['both_visible'] += 1
           elif left_visible:
               self.joint_stats[joint_name]['left_visible'] += 1
           elif right_visible:
               self.joint_stats[joint_name]['right_visible'] += 1
           else:
               self.joint_stats[joint_name]['neither_visible'] += 1
   
    def draw_angle_info(self, frame, angles, camera_label):
       """Draw joint angles on the frame"""
       # Add camera label
       cv2.putText(frame, camera_label, (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
       
       # Draw each angle
       for i, (joint, angle) in enumerate(angles.items()):
           cv2.putText(frame, f"{joint}: {angle:.1f}°", 
                      (10, 70 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 
                      0.8, (0, 255, 0), 2)
   
    def save_results(self, left_angles, right_angles, combined_angles, timestamps, fps):
       """Save processing results"""
       # Save raw angle data
       with open(os.path.join(self.output_dir, "data", "angle_data.pkl"), 'wb') as f:
           pickle.dump({
               'left_angles': left_angles,
               'right_angles': right_angles,
               'combined_angles': combined_angles,
               'timestamps': timestamps,
               'fps': fps,
               'joint_stats': self.joint_stats
           }, f)
       
       # Create angle statistics
       self.create_angle_statistics(left_angles, right_angles, combined_angles)
       
       # Create angle plots
       self.create_angle_plots(left_angles, right_angles, combined_angles, timestamps)
       
       # Create occlusion statistics visualization
       self.create_occlusion_visualization()
   
    def create_angle_statistics(self, left_angles, right_angles, combined_angles):
       """Calculate statistics for joint angles from each camera"""
       # Function to extract angle data for specific joints
       def extract_angle_data(angle_sequence):
           data = {}
           for joint in self.joint_connections.keys():
               data[joint] = []
           
           for frame_angles in angle_sequence:
               for joint in self.joint_connections.keys():
                   if joint in frame_angles:
                       data[joint].append(frame_angles[joint])
           
           return data
       
       # Extract angle data
       left_data = extract_angle_data(left_angles)
       right_data = extract_angle_data(right_angles)
       combined_data = extract_angle_data(combined_angles)
       
       # Calculate statistics
       def calculate_stats(data):
           stats = {}
           for joint, angles in data.items():
               if angles:  # If we have data for this joint
                   angles_array = np.array(angles)
                   stats[joint] = {
                       'mean': np.mean(angles_array),
                       'median': np.median(angles_array),
                       'std': np.std(angles_array),
                       'min': np.min(angles_array),
                       'max': np.max(angles_array)
                   }
           return stats
       
       left_stats = calculate_stats(left_data)
       right_stats = calculate_stats(right_data)
       combined_stats = calculate_stats(combined_data)
       
       # Save statistics to file
       with open(os.path.join(self.output_dir, "angle_statistics.txt"), 'w') as f:
           f.write(f"Angle Statistics\n")
           f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
           
           f.write("COMBINED VIEW (Best of Both Cameras)\n")
           f.write("=================================\n")
           for joint, joint_stats in combined_stats.items():
               f.write(f"{joint.replace('_', ' ').title()}:\n")
               f.write(f"  Mean: {joint_stats['mean']:.2f}°\n")
               f.write(f"  Median: {joint_stats['median']:.2f}°\n")
               f.write(f"  Standard Deviation: {joint_stats['std']:.2f}°\n")
               f.write(f"  Range: {joint_stats['min']:.2f}° - {joint_stats['max']:.2f}°\n")
               f.write("\n")
           
           f.write("\nLEFT CAMERA VIEW\n")
           f.write("================\n")
           for joint, joint_stats in left_stats.items():
               f.write(f"{joint.replace('_', ' ').title()}:\n")
               f.write(f"  Mean: {joint_stats['mean']:.2f}°\n")
               f.write(f"  Median: {joint_stats['median']:.2f}°\n")
               f.write(f"  Standard Deviation: {joint_stats['std']:.2f}°\n")
               f.write(f"  Range: {joint_stats['min']:.2f}° - {joint_stats['max']:.2f}°\n")
               f.write("\n")
           
           f.write("\nRIGHT CAMERA VIEW\n")
           f.write("=================\n")
           for joint, joint_stats in right_stats.items():
               f.write(f"{joint.replace('_', ' ').title()}:\n")
               f.write(f"  Mean: {joint_stats['mean']:.2f}°\n")
               f.write(f"  Median: {joint_stats['median']:.2f}°\n")
               f.write(f"  Standard Deviation: {joint_stats['std']:.2f}°\n")
               f.write(f"  Range: {joint_stats['min']:.2f}° - {joint_stats['max']:.2f}°\n")
               f.write("\n")
       
       # Also save as CSV for easier data analysis
       with open(os.path.join(self.output_dir, "angle_statistics.csv"), 'w') as f:
           f.write("joint,view,mean,median,std,min,max,count\n")
           
           # Write combined stats
           for joint, joint_stats in combined_stats.items():
               count = len([a for a in combined_angles if joint in a])
               f.write(f"{joint},combined,{joint_stats['mean']:.2f},{joint_stats['median']:.2f}," +
                       f"{joint_stats['std']:.2f},{joint_stats['min']:.2f},{joint_stats['max']:.2f},{count}\n")
           
           # Write left camera stats
           for joint, joint_stats in left_stats.items():
               count = len([a for a in left_angles if joint in a])
               f.write(f"{joint},left,{joint_stats['mean']:.2f},{joint_stats['median']:.2f}," +
                       f"{joint_stats['std']:.2f},{joint_stats['min']:.2f},{joint_stats['max']:.2f},{count}\n")
           
           # Write right camera stats
           for joint, joint_stats in right_stats.items():
               count = len([a for a in right_angles if joint in a])
               f.write(f"{joint},right,{joint_stats['mean']:.2f},{joint_stats['median']:.2f}," +
                       f"{joint_stats['std']:.2f},{joint_stats['min']:.2f},{joint_stats['max']:.2f},{count}\n")
   
    def create_angle_plots(self, left_angles, right_angles, combined_angles, timestamps):
       """Create plots comparing joint angles from left and right cameras"""
       # Extract angle data for each joint
       joint_data = {}
       for joint in self.joint_connections.keys():
           joint_data[joint] = {
               'left': {'times': [], 'angles': []},
               'right': {'times': [], 'angles': []},
               'combined': {'times': [], 'angles': []}
           }
       
       # Collect data for each joint
       for i, (left, right, combined) in enumerate(zip(left_angles, right_angles, combined_angles)):
           t = timestamps[i]
           
           for joint in self.joint_connections.keys():
               # Left camera data
               if joint in left:
                   joint_data[joint]['left']['times'].append(t)
                   joint_data[joint]['left']['angles'].append(left[joint])
               
               # Right camera data
               if joint in right:
                   joint_data[joint]['right']['times'].append(t)
                   joint_data[joint]['right']['angles'].append(right[joint])
               
               # Combined view data
               if joint in combined:
                   joint_data[joint]['combined']['times'].append(t)
                   joint_data[joint]['combined']['angles'].append(combined[joint])
       
       # Create plots for each group of joints
       self.plot_joint_group('shoulder', ['left_shoulder', 'right_shoulder'], joint_data, timestamps)
       self.plot_joint_group('elbow', ['left_elbow', 'right_elbow'], joint_data, timestamps)
       self.plot_joint_group('hip', ['left_hip', 'right_hip'], joint_data, timestamps)
       self.plot_joint_group('knee', ['left_knee', 'right_knee'], joint_data, timestamps)
       
       # Create combined plot showing all joint angles
       self.plot_all_joints(joint_data, timestamps)
       
       # Create camera comparison plot showing differences between views
       self.plot_camera_comparison(joint_data, timestamps)
   
    def plot_joint_group(self, group_name, joints, joint_data, timestamps):
       """Create plot for a group of related joints"""
       plt.figure(figsize=(12, 8))
       
       for i, joint in enumerate(joints):
           # Set up subplot
           plt.subplot(2, 1, i+1)
           
           # Plot data from each view
           if joint_data[joint]['left']['times']:
               plt.plot(joint_data[joint]['left']['times'], 
                       joint_data[joint]['left']['angles'], 
                       'r-', alpha=0.6, label='Left Camera')
           
           if joint_data[joint]['right']['times']:
               plt.plot(joint_data[joint]['right']['times'], 
                       joint_data[joint]['right']['angles'], 
                       'b-', alpha=0.6, label='Right Camera')
           
           if joint_data[joint]['combined']['times']:
               plt.plot(joint_data[joint]['combined']['times'], 
                       joint_data[joint]['combined']['angles'], 
                       'g-', linewidth=2, label='Combined View')
           
           # Add labels and grid
           plt.xlabel('Time (s)')
           plt.ylabel('Angle (degrees)')
           plt.title(f'{joint.replace("_", " ").title()} Angle')
           plt.grid(True)
           plt.legend()
       
       plt.tight_layout()
       plt.savefig(os.path.join(self.output_dir, "plots", f'{group_name}_angles.png'), dpi=300)
       plt.close()
   
    def plot_all_joints(self, joint_data, timestamps):
       """Create a plot showing all joint angles from the combined view"""
       plt.figure(figsize=(12, 8))
       
       # Plot each joint
       for joint in self.joint_connections.keys():
           if joint_data[joint]['combined']['times']:
               plt.plot(joint_data[joint]['combined']['times'], 
                       joint_data[joint]['combined']['angles'], 
                       label=joint.replace('_', ' ').title())
       
       # Add labels and grid
       plt.xlabel('Time (s)')
       plt.ylabel('Angle (degrees)')
       plt.title('All Joint Angles (Combined View)')
       plt.grid(True)
       plt.legend()
       
       plt.tight_layout()
       plt.savefig(os.path.join(self.output_dir, "plots", 'all_joints.png'), dpi=300)
       plt.close()
   
    def plot_camera_comparison(self, joint_data, timestamps):
       """Create plots comparing left and right camera measurements for the same joint"""
       plt.figure(figsize=(12, 10))
       
       # Select a few representative joints to compare
       comparison_joints = ['right_shoulder', 'left_elbow', 'right_knee', 'left_hip']
       
       for i, joint in enumerate(comparison_joints):
           # Skip if we don't have enough data
           if (not joint_data[joint]['left']['times'] or 
               not joint_data[joint]['right']['times']):
               continue
           
           # Set up subplot
           plt.subplot(2, 2, i+1)
           
           # Plot data from each camera
           plt.plot(joint_data[joint]['left']['times'], 
                   joint_data[joint]['left']['angles'], 
                   'r-', label='Left Camera')
           
           plt.plot(joint_data[joint]['right']['times'], 
                   joint_data[joint]['right']['angles'], 
                   'b-', label='Right Camera')
           
           # Add labels and grid
           plt.xlabel('Time (s)')
           plt.ylabel('Angle (degrees)')
           plt.title(f'{joint.replace("_", " ").title()} Comparison')
           plt.grid(True)
           plt.legend()
       
       plt.tight_layout()
       plt.savefig(os.path.join(self.output_dir, "plots", 'camera_comparison.png'), dpi=300)
       plt.close()
   
    def create_occlusion_visualization(self):
       """Create visualization of joint occlusion statistics"""
       # Prepare data
       joints = list(self.joint_stats.keys())
       left_only_counts = [self.joint_stats[j]['left_visible'] for j in joints]
       right_only_counts = [self.joint_stats[j]['right_visible'] for j in joints]
       both_visible_counts = [self.joint_stats[j]['both_visible'] for j in joints]
       neither_visible_counts = [self.joint_stats[j]['neither_visible'] for j in joints]
       
       # Calculate total frames for each joint
       totals = [sum([left_only_counts[i], right_only_counts[i], 
                      both_visible_counts[i], neither_visible_counts[i]]) 
                for i in range(len(joints))]
       
       # Create plot
       plt.figure(figsize=(12, 8))
       
       # Create stacked bar chart
       bar_width = 0.6
       indices = range(len(joints))
       
       # Calculate percentages
       left_only_pct = [100 * left_only_counts[i] / totals[i] if totals[i] > 0 else 0 for i in indices]
       right_only_pct = [100 * right_only_counts[i] / totals[i] if totals[i] > 0 else 0 for i in indices]
       both_pct = [100 * both_visible_counts[i] / totals[i] if totals[i] > 0 else 0 for i in indices]
       neither_pct = [100 * neither_visible_counts[i] / totals[i] if totals[i] > 0 else 0 for i in indices]
       
       # Plot stacked bars
       plt.bar(indices, both_pct, bar_width, label='Both Cameras', color='g')
       plt.bar(indices, left_only_pct, bar_width, bottom=both_pct, label='Left Camera Only', color='r')
       plt.bar(indices, right_only_pct, bar_width, 
              bottom=[both_pct[i] + left_only_pct[i] for i in indices], 
              label='Right Camera Only', color='b')
       plt.bar(indices, neither_pct, bar_width, 
              bottom=[both_pct[i] + left_only_pct[i] + right_only_pct[i] for i in indices], 
              label='Neither Camera', color='gray')
       
       # Add labels and legend
       plt.xlabel('Joint')
       plt.ylabel('Percentage of Frames')
       plt.title('Joint Visibility by Camera')
       plt.xticks(indices, [j.replace('_', ' ').title() for j in joints], rotation=45, ha='right')
       plt.legend()
       plt.grid(True, axis='y')
       
       plt.tight_layout()
       plt.savefig(os.path.join(self.output_dir, "plots", 'occlusion_statistics.png'), dpi=300)
       
       # Also save raw counts as text
       with open(os.path.join(self.output_dir, "data", 'occlusion_statistics.txt'), 'w') as f:
           f.write("Joint Visibility Statistics\n")
           f.write("==========================\n\n")
           
           for i, joint in enumerate(joints):
               f.write(f"{joint.replace('_', ' ').title()}:\n")
               f.write(f"  Visible in both cameras: {both_visible_counts[i]} frames ({both_pct[i]:.1f}%)\n")
               f.write(f"  Visible in left camera only: {left_only_counts[i]} frames ({left_only_pct[i]:.1f}%)\n")
               f.write(f"  Visible in right camera only: {right_only_counts[i]} frames ({right_only_pct[i]:.1f}%)\n")
               f.write(f"  Not visible in either camera: {neither_visible_counts[i]} frames ({neither_pct[i]:.1f}%)\n")
               f.write(f"  Total frames: {totals[i]}\n\n")
       
       # Save as CSV as well
       with open(os.path.join(self.output_dir, "data", 'occlusion_statistics.csv'), 'w') as f:
           f.write("joint,both_count,left_only_count,right_only_count,neither_count,total\n")
           for i, joint in enumerate(joints):
               f.write(f"{joint},{both_visible_counts[i]},{left_only_counts[i]}," +
                      f"{right_only_counts[i]},{neither_visible_counts[i]},{totals[i]}\n")

def main():
   parser = argparse.ArgumentParser(description='Dual Camera Pose Estimation with Independent Processing')
   parser.add_argument('--test_dir', required=True, help='Test directory name (e.g., pose_v2)')
   parser.add_argument('--video_pattern', default='pose', help='Pattern to match video filenames (default: pose)')
   parser.add_argument('--start_offset', type=int, default=90, 
                     help='Frames to skip after flash before starting processing')
   parser.add_argument('--max_frames', type=int, default=None, 
                     help='Maximum number of frames to process (default: all)')
   parser.add_argument('--no_display', action='store_true', help='Disable video display during processing')
   
   args = parser.parse_args()
   
   # Initialize pose estimator
   pose_estimator = DualCameraPoseEstimator(args.test_dir)
   
   # Set visualization options
   if args.no_display:
       pose_estimator.viz_settings['show_video'] = False
   
   # Find video pairs
   left_video, right_video = pose_estimator.find_video_pairs(args.video_pattern)
   
   if left_video is None or right_video is None:
       print(f"Could not find matching videos with pattern '{args.video_pattern}'")
       return
   
   print(f"Processing video pair:")
   print(f"  Left: {os.path.basename(left_video)}")
   print(f"  Right: {os.path.basename(right_video)}")
   
   # Process synchronized videos
   pose_estimator.process_dual_videos(
       left_video, right_video, 
       start_offset=args.start_offset,
       max_frames=args.max_frames
   )
   
   print(f"Processing complete. Results saved to {pose_estimator.output_dir}")

if __name__ == "__main__":
   main()