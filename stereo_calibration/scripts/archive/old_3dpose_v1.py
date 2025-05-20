#!/usr/bin/env python3
# USE THIS FOR 3D POSE ESTIMATION WITH TWO CALIBRATED INTR AND EXTR CAMERAS
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


class StereoPoseEstimator:
    def __init__(self, test_dir, video_pattern, output_dir=None):
        """Initialize the stereo pose estimation system"""
        # Setup base directories
        self.test_dir = test_dir
        self.base_dir = os.path.join("data", test_dir)
        
        if output_dir is None:
            self.output_dir = os.path.join(self.base_dir, "results", "stereo_pose", video_pattern)
        else:
            self.output_dir = output_dir
            
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "frames"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "3d_data"), exist_ok=True)
        
        # Setup camera paths
        self.cam1_dir = os.path.join(self.base_dir, "left")
        self.cam2_dir = os.path.join(self.base_dir, "right")
        self.cam1_raw_dir = os.path.join(self.cam1_dir, "raw_video")
        self.cam2_raw_dir = os.path.join(self.cam2_dir, "raw_video")
        
        # Setup MediaPipe Pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
        
        # Data storage
        self.pose_3d_data = {}
        self.angle_data = {}
        self.frame_timestamps = []
        
        # Settings
        self.viz_settings = {
            'show_video': True,
            'save_frames': True,
            'plot_angles': True,
            'plot_3d': True,
            'save_interval': 30,  # Save every 30 frames
        }
        
        # Pose processing settings
        self.smooth_window = 5  # Frames for smoothing window
        self.confidence_threshold = 0.65  # Minimum confidence for landmarks
        self.temporal_filter_weight = 0.7  # Weight for current frame (1-weight for history)
        
        # Load calibration data
        self.calibration_data = self.load_calibration_data()
        if self.calibration_data is None:
            print("Warning: Calibration data not found. 3D reconstruction will not be accurate.")
            
        # Initialize cache for temporal filtering
        self.landmark_history = {
            'left': [],
            'right': []
        }
        self.pose_3d_history = []
        
        print(f"Stereo Pose Estimator initialized for test directory: {test_dir}")
    
    def load_calibration_data(self):
        """Load intrinsic and extrinsic calibration data"""
        calib_dir = os.path.join(self.base_dir, "results", "calibration")
        intrinsic_dir = os.path.join(self.base_dir, "results", "intrinsic_params")
        extrinsic_dir = os.path.join(self.base_dir, "results", "extrinsic_params")
        
        # First check if calibration directory exists
        if not os.path.exists(intrinsic_dir) or not os.path.exists(extrinsic_dir):
            print(f"Calibration directories not found in {self.base_dir}/results/")
            print("Please run intrinsic_iphone.py and extrinsic.py first.")
            return None
        
        try:
            # Load intrinsic parameters
            left_matrix_path = os.path.join(intrinsic_dir, "left_matrix.txt")
            left_dist_path = os.path.join(intrinsic_dir, "left_distortion.txt")
            right_matrix_path = os.path.join(intrinsic_dir, "right_matrix.txt")
            right_dist_path = os.path.join(intrinsic_dir, "right_distortion.txt")
            
            if os.path.exists(left_matrix_path) and os.path.exists(right_matrix_path):
                left_matrix = np.loadtxt(left_matrix_path)
                left_dist = np.loadtxt(left_dist_path)
                right_matrix = np.loadtxt(right_matrix_path)
                right_dist = np.loadtxt(right_dist_path)
            else:
                print("Intrinsic calibration files not found!")
                return None
            
            # Load extrinsic parameters
            R_path = os.path.join(extrinsic_dir, "stereo_rotation_matrix.txt")
            T_path = os.path.join(extrinsic_dir, "stereo_translation_vector.txt")
            
            if os.path.exists(R_path) and os.path.exists(T_path):
                R = np.loadtxt(R_path)
                T = np.loadtxt(T_path)
                
                # Reshape T if needed
                if T.shape == (3,):
                    T = T.reshape(3, 1)
            else:
                print("Extrinsic calibration files not found!")
                return None
            
            # Create projection matrices
            P1 = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
            P1 = left_matrix @ P1
            
            P2 = np.hstack((R, T))
            P2 = right_matrix @ P2
            
            # Return all calibration parameters
            return {
                'left_matrix': left_matrix,
                'left_dist': left_dist,
                'right_matrix': right_matrix,
                'right_dist': right_dist,
                'R': R,
                'T': T,
                'P1': P1,
                'P2': P2
            }
            
        except Exception as e:
            print(f"Error loading calibration data: {e}")
            return None
    
    def find_video_pairs(self, pattern):
        """Find matching video pairs from both cameras"""
        # Find videos in each camera directory matching the pattern
        cam1_videos = glob.glob(os.path.join(self.cam1_dir, f"*{pattern}*.MOV"))
        cam2_videos = glob.glob(os.path.join(self.cam2_dir, f"*{pattern}*.MOV"))
        
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
    
    def detect_flash(self, video_path, threshold=20, window_size=5, max_frames=None):
        """
        Detect frame with sudden brightness increase indicating a flash.
        Returns the frame index of detected flash or None.
        """
        print(f"Analyzing {os.path.basename(video_path)} for flash...")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
        
        frame_count = 0
        brightness_history = []
        
        # Process frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Calculate average brightness
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            brightness_history.append(brightness)
            
            # Check for jump after collecting enough frames
            if len(brightness_history) > window_size:
                # Calculate average brightness before current frame
                prev_avg = sum(brightness_history[-window_size-1:-1]) / window_size
                
                # Check if current brightness is significantly higher
                if brightness > prev_avg + threshold:
                    print(f"Detected flash at frame {frame_count}: {prev_avg:.1f} -> {brightness:.1f}")
                    print(f"  Increase: {brightness - prev_avg:.1f} (threshold: {threshold:.1f})")
                    cap.release()
                    return frame_count
            
            frame_count += 1
            
            # Limit the number of frames to check
            if max_frames is not None and frame_count >= max_frames:
                break
            
            # Print progress
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...")
        
        cap.release()
        print(f"No flash detected in {os.path.basename(video_path)}")
        return None
    
    def determine_sync_offset(self, left_video, right_video):
        """Determine frame offset between videos using flash detection"""
        print("Determining synchronization offset...")
        
        # Detect flash in each video
        left_flash_frame = self.detect_flash(left_video)
        right_flash_frame = self.detect_flash(right_video)
        
        if left_flash_frame is None or right_flash_frame is None:
            print("Could not determine sync offset - flash not detected in one or both videos")
            return 0  # Default to no offset
        
        # Calculate frame offset
        frame_offset = right_flash_frame - left_flash_frame
        print(f"Synchronization offset: {frame_offset} frames (positive means right camera starts later)")
        
        return frame_offset
    
    def process_synchronized_videos(self, left_video, right_video, start_offset=90, max_frames=None):
        """
        Process synchronized videos from left and right cameras.
        
        Args:
            left_video: Path to left camera video
            right_video: Path to right camera video
            start_offset: Frames to skip after flash before starting processing
            max_frames: Maximum number of frames to process (None for all)
        """


        # Determine sync offset
        frame_offset = self.determine_sync_offset(left_video, right_video)

        if frame_offset == 0:
            print("Flash not detected, terminating process... Sorry!")
            return
        
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
            
            # Initialize storage for 3D pose data
            pose_3d_sequence = []
            angle_3d_sequence = []
            
            # Frame counters
            frame_idx = 0
            left_frame_count = left_start_frame
            right_frame_count = right_start_frame
            
            # Initialize visualization window if needed
            if self.viz_settings['show_video']:
                cv2.namedWindow('Stereo Pose', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Stereo Pose', left_width * 2, left_height)
            
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
                
                # Extract 3D pose if both cameras detected landmarks
                pose_3d = None
                angles_3d = None
                
                if left_results.pose_landmarks and right_results.pose_landmarks:
                    # Extract landmarks with confidence scores
                    left_landmarks = self.extract_landmarks(left_results.pose_landmarks)
                    right_landmarks = self.extract_landmarks(right_results.pose_landmarks)
                    
                    # Add to history for temporal filtering
                    self.landmark_history['left'].append(left_landmarks)
                    self.landmark_history['right'].append(right_landmarks)
                    
                    # Limit history size
                    if len(self.landmark_history['left']) > self.smooth_window:
                        self.landmark_history['left'].pop(0)
                        self.landmark_history['right'].pop(0)
                    
                    # Apply temporal filtering
                    if len(self.landmark_history['left']) > 1:
                        left_landmarks = self.apply_temporal_filter(self.landmark_history['left'])
                        right_landmarks = self.apply_temporal_filter(self.landmark_history['right'])
                    
                    # Triangulate 3D positions
                    pose_3d = self.triangulate_pose(left_landmarks, right_landmarks)
                    
                    # Add to 3D pose history
                    self.pose_3d_history.append(pose_3d)
                    if len(self.pose_3d_history) > self.smooth_window:
                        self.pose_3d_history.pop(0)
                    
                    # Apply anatomical constraints
                    pose_3d = self.apply_anatomical_constraints(pose_3d)
                    
                    # Calculate 3D angles
                    angles_3d = self.calculate_3d_angles(pose_3d)
                    
                    # Record data
                    pose_3d_sequence.append(pose_3d)
                    angle_3d_sequence.append(angles_3d)
                    
                    # Draw 3D positions and angles on frames
                    self.draw_3d_info(left_annotated, right_annotated, pose_3d, angles_3d)
                
                # Save frame at regular intervals
                if self.viz_settings['save_frames'] and frame_idx % self.viz_settings['save_interval'] == 0:
                    frame_path = os.path.join(self.output_dir, "frames", f"frame_{frame_idx:04d}.jpg")
                    combined_frame = np.hstack((left_annotated, right_annotated))
                    cv2.imwrite(frame_path, combined_frame)
                
                # Display frames
                if self.viz_settings['show_video']:
                    combined_frame = np.hstack((left_annotated, right_annotated))
                    cv2.imshow('Stereo Pose', combined_frame)
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
            if pose_3d_sequence:
                self.save_results(pose_3d_sequence, angle_3d_sequence, left_fps)
                print(f"Processing complete. Processed {frame_idx} frames with {len(pose_3d_sequence)} valid poses.")
            else:
                print("No valid pose data was detected in the synchronized videos.")
    
    def extract_landmarks(self, pose_landmarks):
        """Extract landmarks with visibility from MediaPipe results"""
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
    
    def triangulate_pose(self, left_landmarks, right_landmarks):
        """Triangulate 3D pose from left and right 2D poses with enhanced single-camera handling"""
        if self.calibration_data is None:
            print("Error: Calibration data not available for triangulation")
            return None
        
        # Get calibration matrices
        P1 = self.calibration_data['P1']
        P2 = self.calibration_data['P2']
        
        # Initialize 3D pose dictionary
        pose_3d = {}
        
        # Define key joints to triangulate
        key_joints = {
            'nose': 0,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28
        }
        
        # Triangulate each joint
        for joint_name, idx in key_joints.items():
            # Check visibility
            left_visible = left_landmarks[idx][3] > self.confidence_threshold
            right_visible = right_landmarks[idx][3] > self.confidence_threshold
            
            if left_visible and right_visible:
                # Both cameras can see the joint - use standard triangulation
                left_pt = left_landmarks[idx][:2]
                right_pt = right_landmarks[idx][:2]
                
                # Undistort points if needed
                if 'left_matrix' in self.calibration_data and 'left_dist' in self.calibration_data:
                    left_pt_array = np.array([[left_pt]], dtype=np.float32)
                    left_pt_undist = cv2.undistortPoints(left_pt_array, 
                                                    self.calibration_data['left_matrix'], 
                                                    self.calibration_data['left_dist'], 
                                                    P=self.calibration_data['left_matrix'])
                    left_pt = left_pt_undist[0, 0]
                
                if 'right_matrix' in self.calibration_data and 'right_dist' in self.calibration_data:
                    right_pt_array = np.array([[right_pt]], dtype=np.float32)
                    right_pt_undist = cv2.undistortPoints(right_pt_array, 
                                                    self.calibration_data['right_matrix'], 
                                                    self.calibration_data['right_dist'], 
                                                    P=self.calibration_data['right_matrix'])
                    right_pt = right_pt_undist[0, 0]
                
                # Triangulate
                points_4d = cv2.triangulatePoints(P1, P2, 
                                                np.array([left_pt[0], left_pt[1]]), 
                                                np.array([right_pt[0], right_pt[1]]))
                
                # Convert from homogeneous coordinates
                point_3d = points_4d[:3] / points_4d[3]
                
                # Store triangulated point
                pose_3d[joint_name] = point_3d.flatten()
                
            elif left_visible:
                # Only left camera sees the joint
                estimated_position = self.estimate_single_camera_position(
                    joint_name, 'left', left_landmarks[idx], pose_3d)
                if estimated_position is not None:
                    pose_3d[joint_name] = estimated_position
                    
            elif right_visible:
                # Only right camera sees the joint
                estimated_position = self.estimate_single_camera_position(
                    joint_name, 'right', right_landmarks[idx], pose_3d)
                if estimated_position is not None:
                    pose_3d[joint_name] = estimated_position
        
        return pose_3d

    def estimate_single_camera_position(self, joint_name, camera, landmark, current_pose_3d):
        """
        Estimate 3D position of a joint visible in only one camera.
        
        Args:
            joint_name: Name of the joint
            camera: 'left' or 'right' indicating which camera sees the joint
            landmark: 2D landmark from the camera that sees the joint
            current_pose_3d: Current 3D pose (may have some joints already triangulated)
            
        Returns:
            Estimated 3D position or None if estimation fails
        """
        # Method 1: Use historical depth if available
        if self.pose_3d_history and len(self.pose_3d_history) > 0:
            # Check if this joint exists in history
            for pose in reversed(self.pose_3d_history):  # Check most recent first
                if joint_name in pose:
                    # Use historical 3D position, but update X,Y based on current detection
                    historical_position = pose[joint_name].copy()
                    
                    # Project the current 2D point to 3D using the historical depth
                    updated_position = self.project_2d_to_3d_with_depth(
                        camera, landmark[:2], historical_position[2])
                    
                    # For stability, blend with historical position
                    alpha = 0.7  # Weight for new position (0.7 new, 0.3 historical)
                    blended_position = alpha * updated_position + (1 - alpha) * historical_position
                    
                    return blended_position
        
        # Method 2: Use anatomical constraints if we have reference joints
        estimated_position = self.estimate_from_anatomy(joint_name, camera, landmark, current_pose_3d)
        if estimated_position is not None:
            return estimated_position
        
        # Method 3: Use epipolar geometry (more complex)
        # Not implemented for simplicity, but would go here
        
        # If all else fails, return None
        return None

    def project_2d_to_3d_with_depth(self, camera, point_2d, depth):
        """
        Project a 2D point back to 3D space using a known depth.
        
        Args:
            camera: 'left' or 'right' indicating which camera
            point_2d: 2D point in image coordinates
            depth: Known depth (Z coordinate)
            
        Returns:
            3D point
        """
        # Get camera matrix for the specified camera
        if camera == 'left':
            camera_matrix = self.calibration_data['left_matrix']
        else:
            camera_matrix = self.calibration_data['right_matrix']
        
        # Extract camera parameters
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        
        # Back-project (pixel coordinates to camera coordinates)
        x = (point_2d[0] - cx) * depth / fx
        y = (point_2d[1] - cy) * depth / fy
        z = depth
        
        # If right camera, transform to left camera coordinate system
        if camera == 'right':
            # Apply inverse transformation (R^T, -R^T * T)
            R = self.calibration_data['R']
            T = self.calibration_data['T'].flatten()
            
            # R^T * [x,y,z] - R^T * T
            point = np.array([x, y, z])
            R_T = R.T
            point = R_T @ point - R_T @ T
            
            x, y, z = point
        
        return np.array([x, y, z])

    def estimate_from_anatomy(self, joint_name, camera, landmark, current_pose_3d):
        """
        Estimate 3D joint position using anatomical constraints.
        
        Args:
            joint_name: Name of the joint to estimate
            camera: Which camera sees the joint ('left' or 'right')
            landmark: 2D landmark from the camera
            current_pose_3d: Current 3D pose with some joints already triangulated
            
        Returns:
            Estimated 3D position or None
        """
        # Define anatomical constraints based on joint relationships
        joint_constraints = {
            # Shoulder-elbow-wrist relationships
            'left_elbow': [('left_shoulder', 'left_wrist', 0.5)],  # elbow is ~halfway between shoulder and wrist
            'right_elbow': [('right_shoulder', 'right_wrist', 0.5)],
            'left_wrist': [('left_elbow', 'left_shoulder', 2.0)],  # wrist is ~2x distance from shoulder as elbow
            'right_wrist': [('right_elbow', 'right_shoulder', 2.0)],
            
            # Hip-knee-ankle relationships
            'left_knee': [('left_hip', 'left_ankle', 0.5)],
            'right_knee': [('right_hip', 'right_ankle', 0.5)],
            'left_ankle': [('left_knee', 'left_hip', 2.0)],
            'right_ankle': [('right_knee', 'right_hip', 2.0)],
            
            # Shoulder-hip relationships
            'left_shoulder': [('left_hip', 'nose', 0.6)],  # shoulder is ~60% of the way from hip to nose
            'right_shoulder': [('right_hip', 'nose', 0.6)],
            'left_hip': [('left_shoulder', None, 1.7)],  # hip is ~1.7x the distance from shoulder to ground
            'right_hip': [('right_shoulder', None, 1.7)]
        }
        
        # Check if we have constraints for this joint
        if joint_name not in joint_constraints:
            return None
        
        # Try each constraint
        for ref_joint, ref_joint2, ratio in joint_constraints[joint_name]:
            # Check if reference joint is available
            if ref_joint in current_pose_3d:
                ref_pos = current_pose_3d[ref_joint]
                
                # Case 1: We have two reference joints
                if ref_joint2 and ref_joint2 in current_pose_3d:
                    ref_pos2 = current_pose_3d[ref_joint2]
                    
                    # Calculate vector between reference joints
                    vector = ref_pos2 - ref_pos
                    
                    # Estimate position based on ratio along this vector
                    estimated_pos = ref_pos + vector * ratio
                    
                    # Project the 2D detection into 3D space, constrained by this estimate
                    # This gives us the right position in the image plane but appropriate depth
                    depth = estimated_pos[2]  # Use Z from the anatomical estimate
                    
                    # Project 2D point to 3D using this depth
                    return self.project_2d_to_3d_with_depth(camera, landmark[:2], depth)
                
                # Case 2: We only have one reference joint
                # Use the historical average bone length if available
                elif self.pose_3d_history:
                    # Find the most recent frame where both joints exist
                    for pose in reversed(self.pose_3d_history):
                        if ref_joint in pose and joint_name in pose:
                            # Calculate the expected bone length
                            bone_vector = pose[joint_name] - pose[ref_joint]
                            bone_length = np.linalg.norm(bone_vector)
                            bone_direction = bone_vector / bone_length
                            
                            # Estimate new position
                            estimated_pos = ref_pos + bone_direction * bone_length
                            
                            # Project the 2D detection into 3D space, constrained by this estimate
                            depth = estimated_pos[2]
                            
                            # Project 2D point to 3D using this depth
                            return self.project_2d_to_3d_with_depth(camera, landmark[:2], depth)
        
        # If no constraints could be applied
        return None




    def apply_anatomical_constraints(self, pose_3d):
        """Apply anatomical constraints to 3D pose data"""
        if not pose_3d:
            return pose_3d
        
        # Copy the pose to avoid modifying the original
        constrained_pose = pose_3d.copy()
        
        # Define limb pairs for bone length constraints
        limb_pairs = [
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            ('left_hip', 'left_knee'),
            ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle'),
            ('left_shoulder', 'right_shoulder'),
            ('left_hip', 'right_hip')
        ]
        
        # If we have pose history, apply bone length constraints
        if self.pose_3d_history and all(joint in constrained_pose for joint in ['left_shoulder', 'left_elbow']):
            # For simplicity, just check the most recent history frame has the necessary joints
            recent_pose = self.pose_3d_history[-1]
            
            # For each limb pair, ensure consistent bone length if both joints exist
            for joint1, joint2 in limb_pairs:
                if joint1 in constrained_pose and joint2 in constrained_pose and \
                   joint1 in recent_pose and joint2 in recent_pose:
                    
                    # Calculate reference bone length from history
                    ref_vec = recent_pose[joint1] - recent_pose[joint2]
                    ref_length = np.linalg.norm(ref_vec)
                    
                    # Calculate current bone vector
                    curr_vec = constrained_pose[joint1] - constrained_pose[joint2]
                    curr_length = np.linalg.norm(curr_vec)
                    
                    # If the difference is too large, adjust the joints
                    if abs(curr_length - ref_length) / ref_length > 0.2:  # Allow 20% variation
                        # Normalize the vector
                        norm_vec = curr_vec / curr_length
                        
                        # Adjust both joints equally
                        constrained_pose[joint1] = constrained_pose[joint2] + norm_vec * ref_length
        
        return constrained_pose
    
    def calculate_3d_angles(self, pose_3d):
        """Calculate 3D joint angles from pose data"""
        if not pose_3d:
            return {}
        
        angles = {}
        
        # Calculate shoulder angles (between hip, shoulder, and elbow)
        if all(k in pose_3d for k in ['right_hip', 'right_shoulder', 'right_elbow']):
            angles['right_shoulder'] = self.calculate_angle_3d(
                pose_3d['right_hip'], 
                pose_3d['right_shoulder'], 
                pose_3d['right_elbow']
            )
        
        if all(k in pose_3d for k in ['left_hip', 'left_shoulder', 'left_elbow']):
            angles['left_shoulder'] = self.calculate_angle_3d(
                pose_3d['left_hip'], 
                pose_3d['left_shoulder'], 
                pose_3d['left_elbow']
            )
        
        # Calculate elbow angles (between shoulder, elbow, and wrist)
        if all(k in pose_3d for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
            angles['right_elbow'] = self.calculate_angle_3d(
                pose_3d['right_shoulder'], 
                pose_3d['right_elbow'], 
                pose_3d['right_wrist']
            )
        
        if all(k in pose_3d for k in ['left_shoulder', 'left_elbow', 'left_wrist']):
            angles['left_elbow'] = self.calculate_angle_3d(
                pose_3d['left_shoulder'], 
                pose_3d['left_elbow'], 
                pose_3d['left_wrist']
            )
        
        # Calculate hip angles (between shoulder, hip, and knee)
        if all(k in pose_3d for k in ['right_shoulder', 'right_hip', 'right_knee']):
            angles['right_hip'] = self.calculate_angle_3d(
                pose_3d['right_shoulder'], 
                pose_3d['right_hip'], 
                pose_3d['right_knee']
            )
        
        if all(k in pose_3d for k in ['left_shoulder', 'left_hip', 'left_knee']):
            angles['left_hip'] = self.calculate_angle_3d(
                pose_3d['left_shoulder'], 
                pose_3d['left_hip'], 
                pose_3d['left_knee']
            )
        
        # Calculate knee angles (between hip, knee, and ankle)
        if all(k in pose_3d for k in ['right_hip', 'right_knee', 'right_ankle']):
            angles['right_knee'] = self.calculate_angle_3d(
                pose_3d['right_hip'], 
                pose_3d['right_knee'], 
                pose_3d['right_ankle']
            )
        
        if all(k in pose_3d for k in ['left_hip', 'left_knee', 'left_ankle']):
            angles['left_knee'] = self.calculate_angle_3d(
                pose_3d['left_hip'], 
                pose_3d['left_knee'], 
                pose_3d['left_ankle']
            )
        
        return angles
    
    def calculate_angle_3d(self, a, b, c):
        """Calculate angle between three 3D points in degrees"""
        # Create vectors
        ba = a - b
        bc = c - b
        
        # Normalize vectors
        ba_normalized = ba / np.linalg.norm(ba)
        bc_normalized = bc / np.linalg.norm(bc)
        
        # Calculate dot product and convert to angle
        dot_product = np.dot(ba_normalized, bc_normalized)
        
        # Clamp dot product to valid range
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        angle_rad = np.arccos(dot_product)
        angle_deg = math.degrees(angle_rad)
        
        return angle_deg
    
    def draw_3d_info(self, left_frame, right_frame, pose_3d, angles_3d):
       """Draw 3D positions and angles on frames"""
       if not pose_3d or not angles_3d:
           return
       
       # Draw joint angles on left frame
       for i, (joint, angle) in enumerate(angles_3d.items()):
           cv2.putText(left_frame, f"{joint}: {angle:.1f}°", 
                      (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 
                      0.8, (0, 255, 0), 2)
       
       # Draw some selected 3D positions on right frame
       key_joints = ['nose', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
       for i, joint in enumerate([j for j in key_joints if j in pose_3d]):
           position = pose_3d[joint]
           cv2.putText(right_frame, f"{joint}: ({position[0]:.0f}, {position[1]:.0f}, {position[2]:.0f})", 
                      (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 
                      0.6, (0, 255, 0), 2)
           
       # Draw some essential information on both frames (frame count, etc.)
       cv2.putText(left_frame, "3D Angles", (10, left_frame.shape[0] - 10), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
       cv2.putText(right_frame, "3D Positions", (10, right_frame.shape[0] - 10), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
   
    def save_results(self, pose_3d_sequence, angle_3d_sequence, fps):
       """Save processing results"""
       # Create timestamps
       timestamps = [i/fps for i in range(len(pose_3d_sequence))]
       
       # Save 3D pose data
       with open(os.path.join(self.output_dir, "3d_data", "pose_3d_data.pkl"), 'wb') as f:
           pickle.dump({
               'poses': pose_3d_sequence,
               'angles': angle_3d_sequence,
               'timestamps': timestamps,
               'fps': fps
           }, f)
       
       # Create angle statistics
       self.create_angle_statistics(angle_3d_sequence)
       
       # Create angle plots
       self.create_angle_plots(angle_3d_sequence, timestamps)
       
       # Create 3D trajectory visualization
       self.create_3d_visualization(pose_3d_sequence, timestamps)
   
    def create_angle_statistics(self, angle_3d_sequence):
       """Calculate statistics for joint angles"""
       # Extract angle data for each joint
       angle_data = {}
       for joint in ['right_shoulder', 'left_shoulder', 'right_elbow', 'left_elbow',
                    'right_hip', 'left_hip', 'right_knee', 'left_knee']:
           angle_data[joint] = []
       
       # Collect angles for each joint
       for angles in angle_3d_sequence:
           for joint, angle in angles.items():
               if joint in angle_data:
                   angle_data[joint].append(angle)
       
       # Calculate statistics
       stats = {}
       for joint, angles in angle_data.items():
           if angles:  # Check if we have data
               angles_array = np.array(angles)
               stats[joint] = {
                   'mean': np.mean(angles_array),
                   'median': np.median(angles_array),
                   'std': np.std(angles_array),
                   'min': np.min(angles_array),
                   'max': np.max(angles_array)
               }
       
       # Save statistics to file
       with open(os.path.join(self.output_dir, "angle_statistics.txt"), 'w') as f:
           f.write(f"Angle Statistics for 3D Pose\n")
           f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
           
           for joint, joint_stats in stats.items():
               f.write(f"{joint.replace('_', ' ').title()}:\n")
               f.write(f"  Mean: {joint_stats['mean']:.2f}°\n")
               f.write(f"  Median: {joint_stats['median']:.2f}°\n")
               f.write(f"  Standard Deviation: {joint_stats['std']:.2f}°\n")
               f.write(f"  Range: {joint_stats['min']:.2f}° - {joint_stats['max']:.2f}°\n")
               f.write("\n")
       
       # Also save as CSV for easier data analysis
       with open(os.path.join(self.output_dir, "angle_statistics.csv"), 'w') as f:
           f.write("joint,mean,median,std,min,max\n")
           for joint, joint_stats in stats.items():
               f.write(f"{joint},{joint_stats['mean']:.2f},{joint_stats['median']:.2f}," +
                       f"{joint_stats['std']:.2f},{joint_stats['min']:.2f},{joint_stats['max']:.2f}\n")
   
    def create_angle_plots(self, angle_3d_sequence, timestamps):
       """Create plots of joint angles over time"""
       # Extract angle data for each joint
       angle_data = {}
       for joint in ['right_shoulder', 'left_shoulder', 'right_elbow', 'left_elbow',
                    'right_hip', 'left_hip', 'right_knee', 'left_knee']:
           angle_data[joint] = []
       
       # Timestamps for plotting
       plot_timestamps = []
       
       # Collect angles for each joint
       for i, angles in enumerate(angle_3d_sequence):
           # Only add timestamp if we have at least one valid angle
           if any(joint in angles for joint in angle_data.keys()):
               plot_timestamps.append(timestamps[i])
               
               # Collect data for each joint
               for joint in angle_data.keys():
                   if joint in angles:
                       angle_data[joint].append(angles[joint])
                   else:
                       # Use None for missing data
                       angle_data[joint].append(None)
       
       # Create plots only if we have data
       if not plot_timestamps:
           print("No angle data available for plotting")
           return
       
       # Create figure for angle plots
       plt.figure(figsize=(12, 10))
       
       # Upper body angles
       plt.subplot(2, 2, 1)
       self.plot_joint_angles(plot_timestamps, angle_data, 
                             ['right_shoulder', 'left_shoulder'],
                             ['r-', 'b-'],
                             ['Right Shoulder', 'Left Shoulder'])
       plt.xlabel('Time (s)')
       plt.ylabel('Angle (degrees)')
       plt.title('Shoulder Angles')
       plt.grid(True)
       plt.legend()
       
       plt.subplot(2, 2, 2)
       self.plot_joint_angles(plot_timestamps, angle_data, 
                             ['right_elbow', 'left_elbow'],
                             ['r-', 'b-'],
                             ['Right Elbow', 'Left Elbow'])
       plt.xlabel('Time (s)')
       plt.ylabel('Angle (degrees)')
       plt.title('Elbow Angles')
       plt.grid(True)
       plt.legend()
       
       # Lower body angles
       plt.subplot(2, 2, 3)
       self.plot_joint_angles(plot_timestamps, angle_data, 
                             ['right_hip', 'left_hip'],
                             ['r-', 'b-'],
                             ['Right Hip', 'Left Hip'])
       plt.xlabel('Time (s)')
       plt.ylabel('Angle (degrees)')
       plt.title('Hip Angles')
       plt.grid(True)
       plt.legend()
       
       plt.subplot(2, 2, 4)
       self.plot_joint_angles(plot_timestamps, angle_data, 
                             ['right_knee', 'left_knee'],
                             ['r-', 'b-'],
                             ['Right Knee', 'Left Knee'])
       plt.xlabel('Time (s)')
       plt.ylabel('Angle (degrees)')
       plt.title('Knee Angles')
       plt.grid(True)
       plt.legend()
       
       plt.tight_layout()
       plt.savefig(os.path.join(self.output_dir, "plots", 'angle_plots.png'), dpi=300)
       plt.close()
   
    def plot_joint_angles(self, timestamps, angle_data, joints, styles, labels):
       """Helper function to plot joint angles with proper handling of missing data"""
       for joint, style, label in zip(joints, styles, labels):
           # Extract data, handling None values
           x_vals = []
           y_vals = []
           
           for i, angle in enumerate(angle_data[joint]):
               if angle is not None:
                   x_vals.append(timestamps[i])
                   y_vals.append(angle)
           
           # Only plot if we have data
           if x_vals:
               plt.plot(x_vals, y_vals, style, label=label)
               
               # Apply smoothing if enough data points
               if len(x_vals) > 10:
                   try:
                       # Use Savitzky-Golay filter for smoothing
                       window_length = min(11, len(x_vals) - (len(x_vals) % 2 + 1))
                       if window_length > 2:
                           y_smooth = savgol_filter(y_vals, window_length, 3)
                           plt.plot(x_vals, y_smooth, style.replace('-', '--'), alpha=0.6)
                   except:
                       # Skip smoothing if it fails
                       pass
   
    def create_3d_visualization(self, pose_3d_sequence, timestamps):
       """Create 3D trajectory visualization"""
       # Check if we have enough pose data
       if len(pose_3d_sequence) < 1:
           print("Not enough pose data for 3D visualization")
           return
       
       # Extract trajectory of key joints
       trajectories = {}
       for joint in ['nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                    'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
                    'right_knee', 'left_ankle', 'right_ankle']:
           trajectories[joint] = {
               'x': [],
               'y': [],
               'z': [],
               'time': []
           }
       
       # Collect trajectory data
       for i, pose in enumerate(pose_3d_sequence):
           for joint in trajectories.keys():
               if joint in pose:
                   trajectories[joint]['x'].append(pose[joint][0])
                   trajectories[joint]['y'].append(pose[joint][1])
                   trajectories[joint]['z'].append(pose[joint][2])
                   trajectories[joint]['time'].append(timestamps[i])
       
       # Create 3D visualization for key joints
       fig = plt.figure(figsize=(15, 10))
       ax = fig.add_subplot(111, projection='3d')
       
       # Colors for different joint groups
       colors = {
           'upper_body': 'red',
           'lower_body': 'blue',
           'left_arm': 'green',
           'right_arm': 'purple',
           'left_leg': 'orange',
           'right_leg': 'cyan',
           'head': 'black'
       }
       
       # Plot trajectories
       self.plot_joint_trajectory(ax, trajectories, 'nose', colors['head'], 'Nose')
       self.plot_joint_trajectory(ax, trajectories, 'left_shoulder', colors['upper_body'], 'Left Shoulder')
       self.plot_joint_trajectory(ax, trajectories, 'right_shoulder', colors['upper_body'], 'Right Shoulder')
       self.plot_joint_trajectory(ax, trajectories, 'left_elbow', colors['left_arm'], 'Left Elbow')
       self.plot_joint_trajectory(ax, trajectories, 'right_elbow', colors['right_arm'], 'Right Elbow')
       self.plot_joint_trajectory(ax, trajectories, 'left_wrist', colors['left_arm'], 'Left Wrist')
       self.plot_joint_trajectory(ax, trajectories, 'right_wrist', colors['right_arm'], 'Right Wrist')
       self.plot_joint_trajectory(ax, trajectories, 'left_hip', colors['lower_body'], 'Left Hip')
       self.plot_joint_trajectory(ax, trajectories, 'right_hip', colors['lower_body'], 'Right Hip')
       self.plot_joint_trajectory(ax, trajectories, 'left_knee', colors['left_leg'], 'Left Knee')
       self.plot_joint_trajectory(ax, trajectories, 'right_knee', colors['right_leg'], 'Right Knee')
       self.plot_joint_trajectory(ax, trajectories, 'left_ankle', colors['left_leg'], 'Left Ankle')
       self.plot_joint_trajectory(ax, trajectories, 'right_ankle', colors['right_leg'], 'Right Ankle')
       
       # Set labels and title
       ax.set_xlabel('X (mm)')
       ax.set_ylabel('Y (mm)')
       ax.set_zlabel('Z (mm)')
       ax.set_title('3D Joint Trajectories')
       
       # Set equal aspect ratio
       max_range = self.get_max_range(trajectories)
       ax.set_xlim([-max_range, max_range])
       ax.set_ylim([-max_range, max_range])
       ax.set_zlim([-max_range, max_range])
       
       # Add legend
       ax.legend()
       
       # Add warning if calibration might be suspect
       if self.calibration_data is None:
           plt.figtext(0.5, 0.01, "WARNING: No calibration data available. 3D positions may be inaccurate.", 
                     ha='center', color='red', fontsize=12)
       
       # Save figure
       plt.savefig(os.path.join(self.output_dir, "plots", '3d_trajectories.png'), dpi=300)
       plt.close()
   
    def plot_joint_trajectory(self, ax, trajectories, joint, color, label):
       """Plot trajectory for a single joint"""
       if not trajectories[joint]['x']:
           return
       
       # Plot 3D trajectory
       ax.plot(trajectories[joint]['x'], trajectories[joint]['y'], trajectories[joint]['z'], 
              color=color, label=label)
       
       # Mark start point
       ax.scatter(trajectories[joint]['x'][0], trajectories[joint]['y'][0], trajectories[joint]['z'][0], 
                 color=color, marker='o', s=50)
   
    def get_max_range(self, trajectories):
       """Get maximum range for 3D plot scaling"""
       all_x = []
       all_y = []
       all_z = []
       
       for joint in trajectories.keys():
           all_x.extend(trajectories[joint]['x'])
           all_y.extend(trajectories[joint]['y'])
           all_z.extend(trajectories[joint]['z'])
       
       if not all_x:
           return
       
       # Calculate ranges
       x_range = max(all_x) - min(all_x)
       y_range = max(all_y) - min(all_y)
       z_range = max(all_z) - min(all_z)
       
       return max(x_range, y_range, z_range) / 2


def main():
   global quit_requested

   parser = argparse.ArgumentParser(description='Stereo Pose Estimation from Synchronized Videos')
   parser.add_argument('--test_dir', required=True, help='Test directory name (e.g., pose_v2)')
   parser.add_argument('--video_pattern', default='pose', help='Pattern to match video filenames (default: pose)')
   parser.add_argument('--start_offset', type=int, default=90, 
                     help='Frames to skip after flash before starting processing')
   parser.add_argument('--max_frames', type=int, default=None, 
                     help='Maximum number of frames to process (default: all)')
   parser.add_argument('--no_display', action='store_true', help='Disable video display during processing')
   
   args = parser.parse_args()
   
   # Initialize pose estimator
   pose_estimator = StereoPoseEstimator(args.test_dir, args.video_pattern)
   
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
   
   print("Starting calibration. Press Ctrl+C to stop gracefully.")

   # Process synchronized videos
   pose_estimator.process_synchronized_videos(
       left_video, right_video, 
       start_offset=args.start_offset,
       max_frames=args.max_frames
   )
   
   print(f"Processing complete. Results saved to {pose_estimator.output_dir}")

if __name__ == "__main__":
   main()