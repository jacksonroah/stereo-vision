#!/usr/bin/env python3
# motion_smoothing.py - Advanced motion processing for stereo vision biomechanical tracking
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import math
import os

class MotionProcessor:
    def __init__(self, config=None):
        """
        Initialize the motion processor with configuration options.
        
        Args:
            config: Dictionary with configuration parameters
        """
        # Default configuration
        self.config = {
            'filter_type': 'savgol',  # Options: 'savgol', 'kalman', 'one_euro'
            'window_size': 7,          # Window size for Savitzky-Golay filter
            'poly_order': 3,           # Polynomial order for Savitzky-Golay
            'apply_anatomical_constraints': True,
            'velocity_smoothing': True,
            'acceleration_smoothing': True,
            'min_confidence': 0.65,    # Minimum confidence for valid landmarks
        }
        
        # Update with user configuration if provided
        if config:
            self.config.update(config)
            
        # Initialize joint angle limits (anatomical constraints in degrees)
        self.joint_limits = {
            # Shoulder flexion/extension (up/down)
            'left_shoulder': {'min': 0, 'max': 180},
            'right_shoulder': {'min': 0, 'max': 180},
            
            # Elbow flexion/extension
            'left_elbow': {'min': 0, 'max': 160},  # Can't fully straighten to 180°
            'right_elbow': {'min': 0, 'max': 160},
            
            # Hip flexion/extension
            'left_hip': {'min': 0, 'max': 120},
            'right_hip': {'min': 0, 'max': 120},
            
            # Knee flexion/extension
            'left_knee': {'min': 0, 'max': 170},   # Hyperextension shouldn't exceed 10°
            'right_knee': {'min': 0, 'max': 170},
            
            # Ankle (dorsiflexion/plantarflexion)
            'left_ankle': {'min': 70, 'max': 130}, # 90° is neutral
            'right_ankle': {'min': 70, 'max': 130}
        }
        
        # Anatomical bone lengths (as ratios of height)
        self.bone_length_ratios = {
            'shoulder_width': 0.259,    # Shoulder width / height
            'upper_arm': 0.186,         # Upper arm / height
            'forearm': 0.146,           # Forearm / height
            'hip_width': 0.191,         # Hip width / height
            'thigh': 0.245,             # Thigh / height
            'shin': 0.246,              # Shin / height
        }
        
        # Kalman filter instances for each joint - will be initialized when needed
        self.kalman_filters = {}
        self.kalman_initialized = False
        
        # Storage for biomechanical data
        self.velocities = {}
        self.accelerations = {}
        self.forces = {}
        self.powers = {}
        
        # Default body mass (kg) - can be updated for specific subject
        self.body_mass = 70.0
        
        # Initialize limb mass ratios (percentage of total body mass)
        self.limb_mass_ratios = {
            'head': 0.081,
            'torso': 0.497,
            'upper_arm': 0.028,
            'forearm': 0.016,
            'hand': 0.006,
            'thigh': 0.1,
            'shin': 0.0465,
            'foot': 0.0145
        }
        
        # Previous timestamps for derivatives
        self.prev_timestamp = None
        self.prev_positions = None
        self.prev_velocities = None
    
    def initialize_kalman_filters(self, num_joints):
        """Initialize Kalman filters for 3D joint tracking"""
        for i in range(num_joints):
            # Create a filter for each joint's 3D position
            kf = KalmanFilter(dim_x=6, dim_z=3)  # State: position and velocity, Measurement: position
            
            # State transition matrix (position + velocity model)
            kf.F = np.array([
                [1, 0, 0, 1, 0, 0],  # x' = x + vx
                [0, 1, 0, 0, 1, 0],  # y' = y + vy
                [0, 0, 1, 0, 0, 1],  # z' = z + vz
                [0, 0, 0, 1, 0, 0],  # vx' = vx
                [0, 0, 0, 0, 1, 0],  # vy' = vy
                [0, 0, 0, 0, 0, 1]   # vz' = vz
            ])
            
            # Measurement matrix (we only measure position directly)
            kf.H = np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0]
            ])
            
            # Measurement noise
            kf.R = np.eye(3) * 10  # Adjusted based on expected measurement noise
            
            # Process noise
            kf.Q = np.eye(6) * 0.1  # Adjusted based on expected motion variability
            
            # Initial state covariance
            kf.P = np.eye(6) * 100   # High uncertainty initially
            
            self.kalman_filters[i] = kf
        
        self.kalman_initialized = True
    
    def get_one_euro_filter(self, fc_min=0.5, beta=0.5):
        """Create a one euro filter for real-time filtering"""
        class OneEuroFilter:
            def __init__(self, fc_min, beta):
                self.fc_min = fc_min
                self.beta = beta
                self.prev_raw_value = None
                self.prev_filtered_value = None
                self.prev_timestamp = None
                
            def smoothing_factor(self, t_e, fc):
                """Calculate smoothing factor"""
                return 1.0 / (1.0 + (2 * np.pi * fc * t_e))
            
            def exponential_smoothing(self, a, x, x_prev):
                """Apply exponential smoothing"""
                return a * x + (1 - a) * x_prev
            
            def filter(self, x, timestamp):
                """Apply one euro filter to a value"""
                if self.prev_raw_value is None:
                    self.prev_raw_value = x
                    self.prev_filtered_value = x
                    self.prev_timestamp = timestamp
                    return x
                
                # Calculate time difference
                t_e = timestamp - self.prev_timestamp
                if t_e <= 0:
                    return self.prev_filtered_value
                
                # First-order low-pass filter
                dx = (x - self.prev_raw_value) / t_e  # Derivative
                dx_hat = self.exponential_smoothing(
                    self.smoothing_factor(t_e, self.fc_min * self.beta * abs(dx)),
                    dx, 0 if self.prev_timestamp is None else 
                       (self.prev_filtered_value - self.prev_raw_value) / t_e
                )
                
                # Filter cutoff frequency
                fc = self.fc_min + self.beta * abs(dx_hat)
                
                # Filter the raw signal
                filtered_value = self.exponential_smoothing(
                    self.smoothing_factor(t_e, fc),
                    x, self.prev_filtered_value
                )
                
                # Update previous values
                self.prev_raw_value = x
                self.prev_filtered_value = filtered_value
                self.prev_timestamp = timestamp
                
                return filtered_value
        
        return OneEuroFilter(fc_min, beta)
    
    def smooth_trajectory(self, positions, timestamps=None, joint_name=None):
        """
        Apply smoothing to a trajectory of 3D positions.
        
        Args:
            positions: Array of 3D positions (N x 3)
            timestamps: Array of timestamps (optional, for velocity-based filters)
            joint_name: Name of the joint (for joint-specific filtering)
            
        Returns:
            Smoothed positions
        """
        if not isinstance(positions, np.ndarray):
            positions = np.array(positions)
        
        if len(positions) < 3:
            return positions  # Too few points to smooth
        
        if self.config['filter_type'] == 'savgol':
            # Apply Savitzky-Golay filter to each dimension separately
            window = min(self.config['window_size'], len(positions) - (1 - len(positions) % 2))
            if window > 2 and window % 2 == 1:  # Window must be odd and > 2
                smoothed = np.zeros_like(positions)
                for i in range(positions.shape[1]):
                    smoothed[:, i] = savgol_filter(
                        positions[:, i], 
                        window, 
                        self.config['poly_order']
                    )
                return smoothed
            else:
                return positions  # Can't apply filter with current settings
            
        elif self.config['filter_type'] == 'kalman':
            # Apply Kalman filter
            if not self.kalman_initialized:
                self.initialize_kalman_filters(1)  # Initialize for a single joint
            
            joint_idx = 0  # Default index
            if joint_name and joint_name in self.kalman_filters:
                joint_idx = list(self.kalman_filters.keys()).index(joint_name)
            
            kf = self.kalman_filters[joint_idx]
            smoothed = np.zeros_like(positions)
            
            # Initial state
            kf.x = np.array([positions[0, 0], positions[0, 1], positions[0, 2], 0, 0, 0])
            
            for i in range(len(positions)):
                # Predict
                kf.predict()
                
                # Update with measurement
                measurement = positions[i]
                kf.update(measurement)
                
                # Get filtered position
                smoothed[i] = kf.x[:3]
            
            return smoothed
            
        elif self.config['filter_type'] == 'one_euro':
            # Apply one euro filter (requires timestamps)
            if timestamps is None:
                # If no timestamps provided, create artificial ones
                timestamps = np.arange(len(positions)) / 30.0  # Assume 30 fps
            
            smoothed = np.zeros_like(positions)
            
            # Create filters for each dimension
            filters = [self.get_one_euro_filter() for _ in range(positions.shape[1])]
            
            for i in range(len(positions)):
                for j in range(positions.shape[1]):
                    smoothed[i, j] = filters[j].filter(positions[i, j], timestamps[i])
            
            return smoothed
        
        else:
            # No filtering
            return positions
    
    def smooth_pose_sequence(self, pose_sequence, timestamps=None):
        """
        Apply smoothing to a sequence of 3D poses.
        
        Args:
            pose_sequence: List of dictionaries, each containing 3D joint positions
            timestamps: List of timestamps (optional)
            
        Returns:
            Smoothed pose sequence
        """
        if not pose_sequence:
            return pose_sequence
        
        # Extract joint names
        joint_names = list(pose_sequence[0].keys())
        
        # Create dictionary to hold trajectories for each joint
        trajectories = {joint: [] for joint in joint_names}
        
        # Extract trajectories
        for pose in pose_sequence:
            for joint in joint_names:
                if joint in pose:
                    trajectories[joint].append(pose[joint])
        
        # Smooth each joint trajectory
        smoothed_trajectories = {}
        for joint in joint_names:
            if len(trajectories[joint]) >= 3:  # Need at least 3 points for smoothing
                # Convert to numpy array
                traj = np.array(trajectories[joint])
                
                # Apply smoothing
                smoothed_traj = self.smooth_trajectory(traj, timestamps, joint)
                
                # Store smoothed trajectory
                smoothed_trajectories[joint] = smoothed_traj
        
        # Reconstruct pose sequence
        smoothed_pose_sequence = []
        for i in range(len(pose_sequence)):
            smoothed_pose = {}
            for joint in joint_names:
                if joint in smoothed_trajectories and i < len(smoothed_trajectories[joint]):
                    smoothed_pose[joint] = smoothed_trajectories[joint][i]
            
            # Apply anatomical constraints if enabled
            if self.config['apply_anatomical_constraints']:
                smoothed_pose = self.apply_anatomical_constraints(smoothed_pose)
            
            smoothed_pose_sequence.append(smoothed_pose)
        
        return smoothed_pose_sequence

    def apply_anatomical_constraints(self, pose_3d):
        """
        Apply anatomical constraints to a 3D pose.
        
        Args:
            pose_3d: Dictionary with 3D joint positions
            
        Returns:
            Pose with constrained joint positions
        """
        if not pose_3d:
            return pose_3d
        
        # Copy the pose to avoid modifying the original
        constrained_pose = pose_3d.copy()
        
        # Apply joint angle constraints
        constrained_pose = self.apply_joint_angle_constraints(constrained_pose)
        
        # Apply bone length constraints
        constrained_pose = self.apply_bone_length_constraints(constrained_pose)
        
        # Apply symmetry constraints (optional for static poses)
        # constrained_pose = self.apply_symmetry_constraints(constrained_pose)
        
        return constrained_pose
    
    def apply_joint_angle_constraints(self, pose_3d):
        """Apply anatomical joint angle limits"""
        if not pose_3d:
            return pose_3d
        
        constrained_pose = pose_3d.copy()
        
        # Calculate and constrain joint angles
        angles = self.calculate_3d_angles(pose_3d)
        
        # Apply constraints to each joint angle
        for joint, angle in angles.items():
            if joint in self.joint_limits:
                limits = self.joint_limits[joint]
                
                # Check if angle is outside limits
                if angle < limits['min'] or angle > limits['max']:
                    # Get constrained angle
                    constrained_angle = max(limits['min'], min(angle, limits['max']))
                    
                    # Apply constrained angle by adjusting child joint
                    child_joint = self.get_child_joint(joint, pose_3d)
                    if child_joint and child_joint in constrained_pose:
                        # Get reference joints for the angle
                        parent_joint = self.get_parent_joint(joint, pose_3d)
                        if parent_joint and parent_joint in constrained_pose:
                            # Calculate vectors
                            v1 = constrained_pose[parent_joint] - constrained_pose[joint]
                            v2 = constrained_pose[child_joint] - constrained_pose[joint]
                            
                            # Get current angle
                            current_angle = self.calculate_angle_3d(
                                constrained_pose[parent_joint],
                                constrained_pose[joint],
                                constrained_pose[child_joint]
                            )
                            
                            # Calculate rotation needed
                            angle_diff = constrained_angle - current_angle
                            
                            # Create rotation axis (perpendicular to plane of the angle)
                            axis = np.cross(v1, v2)
                            axis = axis / np.linalg.norm(axis) if np.linalg.norm(axis) > 0 else np.array([0, 0, 1])
                            
                            # Create rotation matrix
                            rot = R.from_rotvec(axis * np.radians(angle_diff))
                            
                            # Rotate child joint position
                            rotated_v2 = rot.apply(v2)
                            
                            # Update child joint position
                            constrained_pose[child_joint] = constrained_pose[joint] + rotated_v2
        
        return constrained_pose
    
    def get_child_joint(self, joint, pose_3d):
        """Get child joint for a given joint"""
        joint_hierarchy = {
            'left_shoulder': 'left_elbow',
            'right_shoulder': 'right_elbow',
            'left_elbow': 'left_wrist',
            'right_elbow': 'right_wrist',
            'left_hip': 'left_knee',
            'right_hip': 'right_knee',
            'left_knee': 'left_ankle',
            'right_knee': 'right_ankle'
        }
        
        if joint in joint_hierarchy and joint_hierarchy[joint] in pose_3d:
            return joint_hierarchy[joint]
        
        return None
    
    def get_parent_joint(self, joint, pose_3d):
        """Get parent joint for a given joint"""
        joint_hierarchy = {
            'left_elbow': 'left_shoulder',
            'right_elbow': 'right_shoulder',
            'left_wrist': 'left_elbow',
            'right_wrist': 'right_elbow',
            'left_knee': 'left_hip',
            'right_knee': 'right_hip',
            'left_ankle': 'left_knee',
            'right_ankle': 'right_knee',
            'left_shoulder': 'nose',
            'right_shoulder': 'nose',
            'left_hip': 'nose',
            'right_hip': 'nose'
        }
        
        if joint in joint_hierarchy and joint_hierarchy[joint] in pose_3d:
            return joint_hierarchy[joint]
        
        return None
    
    def apply_bone_length_constraints(self, pose_3d):
        """Apply bone length constraints based on anthropometric data"""
        if not pose_3d:
            return pose_3d
        
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
        
        # Get reference height (approximate from hip to shoulder + head)
        reference_height = self.estimate_height(pose_3d)
        if reference_height is None:
            return constrained_pose  # Can't apply constraints without reference height
        
        # Define expected bone lengths based on ratios
        expected_lengths = {
            ('left_shoulder', 'left_elbow'): reference_height * self.bone_length_ratios['upper_arm'],
            ('right_shoulder', 'right_elbow'): reference_height * self.bone_length_ratios['upper_arm'],
            ('left_elbow', 'left_wrist'): reference_height * self.bone_length_ratios['forearm'],
            ('right_elbow', 'right_wrist'): reference_height * self.bone_length_ratios['forearm'],
            ('left_hip', 'left_knee'): reference_height * self.bone_length_ratios['thigh'],
            ('right_hip', 'right_knee'): reference_height * self.bone_length_ratios['thigh'],
            ('left_knee', 'left_ankle'): reference_height * self.bone_length_ratios['shin'],
            ('right_knee', 'right_ankle'): reference_height * self.bone_length_ratios['shin'],
            ('left_shoulder', 'right_shoulder'): reference_height * self.bone_length_ratios['shoulder_width'],
            ('left_hip', 'right_hip'): reference_height * self.bone_length_ratios['hip_width']
        }
        
        # For each bone, check and constrain length
        for joint1, joint2 in limb_pairs:
            if joint1 in constrained_pose and joint2 in constrained_pose:
                # Calculate current bone vector and length
                bone_vector = constrained_pose[joint2] - constrained_pose[joint1]
                current_length = np.linalg.norm(bone_vector)
                
                # Get expected length
                expected_length = expected_lengths.get((joint1, joint2))
                if expected_length is None:
                    # Try reverse order
                    expected_length = expected_lengths.get((joint2, joint1))
                
                if expected_length is not None:
                    # Allow for some variation (±20%)
                    tolerance = 0.2
                    min_length = expected_length * (1 - tolerance)
                    max_length = expected_length * (1 + tolerance)
                    
                    # Check if current length is outside valid range
                    if current_length < min_length or current_length > max_length:
                        # Normalize the bone vector
                        if current_length > 0:
                            normalized_vector = bone_vector / current_length
                        else:
                            normalized_vector = np.array([0, 0, 1])  # Default if zero length
                        
                        # Constrain to valid range
                        if current_length < min_length:
                            new_length = min_length
                        else:
                            new_length = max_length
                        
                        # Adjust the child joint position (usually distal joint)
                        child_joint = self.get_distal_joint(joint1, joint2)
                        if child_joint:
                            parent_joint = joint1 if child_joint == joint2 else joint2
                            constrained_pose[child_joint] = constrained_pose[parent_joint] + normalized_vector * new_length
        
        return constrained_pose
    
    def get_distal_joint(self, joint1, joint2):
        """Determine which joint is distal (further from center)"""
        joint_hierarchy = {
            'nose': 0,
            'left_shoulder': 1, 'right_shoulder': 1,
            'left_hip': 1, 'right_hip': 1,
            'left_elbow': 2, 'right_elbow': 2,
            'left_knee': 2, 'right_knee': 2,
            'left_wrist': 3, 'right_wrist': 3,
            'left_ankle': 3, 'right_ankle': 3
        }
        
        # Higher level is more distal
        level1 = joint_hierarchy.get(joint1, 0)
        level2 = joint_hierarchy.get(joint2, 0)
        
        if level1 >= level2:
            return joint1
        else:
            return joint2
    
    def estimate_height(self, pose_3d):
        """Estimate subject height from pose data"""
        if not pose_3d:
            return None
        
        # Check if we have key joints for height estimation
        required_joints = ['nose', 'left_hip', 'right_hip', 'left_ankle', 'right_ankle']
        if not all(joint in pose_3d for joint in required_joints):
            return None
        
        # Calculate height as distance from midpoint of ankles to nose
        left_ankle = pose_3d['left_ankle']
        right_ankle = pose_3d['right_ankle']
        ankle_midpoint = (left_ankle + right_ankle) / 2
        
        # Get nose position
        nose = pose_3d['nose']
        
        # Calculate height (vertical distance)
        height = np.linalg.norm(nose - ankle_midpoint)
        
        return height
    
    def apply_symmetry_constraints(self, pose_3d):
        """Apply left-right symmetry constraints for static poses"""
        if not pose_3d:
            return pose_3d
        
        constrained_pose = pose_3d.copy()
        
        # Define symmetric joint pairs
        symmetric_pairs = [
            ('left_shoulder', 'right_shoulder'),
            ('left_elbow', 'right_elbow'),
            ('left_wrist', 'right_wrist'),
            ('left_hip', 'right_hip'),
            ('left_knee', 'right_knee'),
            ('left_ankle', 'right_ankle')
        ]
        
        # Define plane of symmetry (sagittal plane)
        if 'nose' in pose_3d and 'left_hip' in pose_3d and 'right_hip' in pose_3d:
            # Define sagittal plane from nose and midpoint of hips
            nose = pose_3d['nose']
            hip_midpoint = (pose_3d['left_hip'] + pose_3d['right_hip']) / 2
            
            # Vector from hip midpoint to nose defines Y-axis
            y_axis = nose - hip_midpoint
            y_axis = y_axis / np.linalg.norm(y_axis)
            
            # Vector between hips defines X-axis (left to right)
            x_axis = pose_3d['right_hip'] - pose_3d['left_hip']
            x_axis = x_axis / np.linalg.norm(x_axis)
            
            # Z-axis is perpendicular to both (forward/backward)
            z_axis = np.cross(x_axis, y_axis)
            z_axis = z_axis / np.linalg.norm(z_axis)
            
            # Recompute X-axis to ensure orthogonality
            x_axis = np.cross(y_axis, z_axis)
            
            # Create rotation matrix from global to local coordinates
            rot_matrix = np.vstack((x_axis, y_axis, z_axis)).T
            
            # For each symmetric joint pair
            for left_joint, right_joint in symmetric_pairs:
                if left_joint in pose_3d and right_joint in pose_3d:
                    # Transform to local coordinates
                    left_local = rot_matrix.T @ (pose_3d[left_joint] - hip_midpoint)
                    right_local = rot_matrix.T @ (pose_3d[right_joint] - hip_midpoint)
                    
                    # Average Y and Z coordinates
                    avg_y = (left_local[1] + right_local[1]) / 2
                    avg_z = (left_local[2] + right_local[2]) / 2
                    
                    # Create symmetrical local coordinates
                    left_local_sym = np.array([-abs(left_local[0]), avg_y, avg_z])
                    right_local_sym = np.array([abs(right_local[0]), avg_y, avg_z])
                    
                    # Transform back to global coordinates
                    constrained_pose[left_joint] = rot_matrix @ left_local_sym + hip_midpoint
                    constrained_pose[right_joint] = rot_matrix @ right_local_sym + hip_midpoint
        
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
    
    def calculate_biomechanics(self, pose_sequence, timestamps, subject_mass=None):
        """
        Calculate biomechanical metrics from pose sequence.
        
        Args:
            pose_sequence: List of dictionaries with 3D joint positions
            timestamps: List of timestamps for each frame
            subject_mass: Subject mass in kg (optional)
            
        Returns:
            Dictionary with biomechanical metrics
        """
        if not pose_sequence or len(pose_sequence) < 2:
            return {}
        
        # Update body mass if provided
        if subject_mass:
            self.body_mass = subject_mass
        
        # Initialize storage
        biomechanics = {
            'velocities': {},
            'accelerations': {},
            'forces': {},
            'powers': {},
            'joint_angles': {},
            'angular_velocities': {},
            'angular_accelerations': {},
            'joint_moments': {},
            'joint_powers': {}
        }
        
        # Calculate linear velocities and accelerations
        velocities = self.calculate_velocities(pose_sequence, timestamps)
        accelerations = self.calculate_accelerations(velocities, timestamps)
        
        # Calculate forces
        forces = self.calculate_forces(accelerations)
        
        # Calculate powers
        powers = self.calculate_powers(forces, velocities)
        
        # Calculate joint angles
        angles = self.calculate_joint_angles_sequence(pose_sequence)
        
        # Calculate angular velocities and accelerations
        angular_velocities = self.calculate_angular_velocities(angles, timestamps)
        angular_accelerations = self.calculate_angular_accelerations(angular_velocities, timestamps)
        
        # Calculate joint moments and powers
        joint_moments = self.calculate_joint_moments(pose_sequence, accelerations)
        joint_powers = self.calculate_joint_powers(joint_moments, angular_velocities)
        
        # Store results
        biomechanics['velocities'] = velocities
        biomechanics['accelerations'] = accelerations
        biomechanics['forces'] = forces
        biomechanics['powers'] = powers
        biomechanics['joint_angles'] = angles
        biomechanics['angular_velocities'] = angular_velocities
        biomechanics['angular_accelerations'] = angular_accelerations
        biomechanics['joint_moments'] = joint_moments
        biomechanics['joint_powers'] = joint_powers
        
        return biomechanics
    
    def calculate_velocities(self, pose_sequence, timestamps):
        """
        Calculate linear velocities for each joint.
        
        Args:
            pose_sequence: List of dictionaries with 3D joint positions
            timestamps: List of timestamps for each frame
            
        Returns:
            Dictionary with velocities for each joint
        """
        if len(pose_sequence) < 2:
            return {}
        
        # Initialize storage
        velocities = {i: {} for i in range(len(pose_sequence) - 1)}
        
        # Get list of joints
        joints = list(pose_sequence[0].keys())
        
        # Calculate velocity for each joint
        for i in range(len(pose_sequence) - 1):
            dt = timestamps[i + 1] - timestamps[i]
            
            if dt <= 0:
                continue  # Skip invalid time differences
            
            # Calculate velocity for each joint
            for joint in joints:
                if joint in pose_sequence[i] and joint in pose_sequence[i + 1]:
                    displacement = pose_sequence[i + 1][joint] - pose_sequence[i][joint]
                    velocity = displacement / dt  # mm/s
                    velocities[i][joint] = velocity
        
        # Apply smoothing if enabled
        if self.config['velocity_smoothing'] and len(velocities) > 2:
            for joint in joints:
                joint_velocities = []
                for i in range(len(velocities)):
                    if joint in velocities[i]:
                        joint_velocities.append(velocities[i][joint])
                
                if len(joint_velocities) > 2:
                    # Convert to numpy array
                    joint_velocities_array = np.array(joint_velocities)
                    
                    # Apply smoothing
                    smoothed_velocities = self.smooth_trajectory(joint_velocities_array)
                    
                    # Update velocities
                    idx = 0
                    for i in range(len(velocities)):
                        if joint in velocities[i]:
                            velocities[i][joint] = smoothed_velocities[idx]
                            idx += 1
        
        return velocities
    
    def calculate_accelerations(self, velocities, timestamps):
        """
        Calculate linear accelerations for each joint.
        
        Args:
            velocities: Dictionary with velocities for each joint
            timestamps: List of timestamps for each frame
            
        Returns:
            Dictionary with accelerations for each joint
        """
        if len(velocities) < 2:
            return {}
        
        # Initialize storage
        accelerations = {i: {} for i in range(len(velocities) - 1)}
        
        # Get list of joints
        joints = set()
        for frame_velocities in velocities.values():
            joints.update(frame_velocities.keys())
        
        # Calculate acceleration for each joint
        for i in range(len(velocities) - 1):
            # Use center point of velocity timestamps
            t1 = timestamps[i + 1] - timestamps[i]
            t2 = timestamps[i + 2] - timestamps[i + 1]
            dt = (t1 + t2) / 2
            
            if dt <= 0:
                continue  # Skip invalid time differences
            
            # Calculate acceleration for each joint
            for joint in joints:
                if (joint in velocities[i] and 
                    joint in velocities[i + 1]):
                    
                    velocity_change = velocities[i + 1][joint] - velocities[i][joint]
                    acceleration = velocity_change / dt  # mm/s^2
                    accelerations[i][joint] = acceleration
        
        # Apply smoothing if enabled
        if self.config['acceleration_smoothing'] and len(accelerations) > 2:
            for joint in joints:
                joint_accelerations = []
                for i in range(len(accelerations)):
                    if joint in accelerations[i]:
                        joint_accelerations.append(accelerations[i][joint])
                
                if len(joint_accelerations) > 2:
                    # Convert to numpy array
                    joint_accelerations_array = np.array(joint_accelerations)
                    
                    # Apply smoothing
                    smoothed_accelerations = self.smooth_trajectory(joint_accelerations_array)
                    
                    # Update accelerations
                    idx = 0
                    for i in range(len(accelerations)):
                        if joint in accelerations[i]:
                            accelerations[i][joint] = smoothed_accelerations[idx]
                            idx += 1
        
        return accelerations
    
    def calculate_forces(self, accelerations):
        """
        Calculate forces for each joint based on accelerations.
        
        Args:
            accelerations: Dictionary with accelerations for each joint
            
        Returns:
            Dictionary with forces for each joint
        """
        if not accelerations:
            return {}
        
        # Initialize storage
        forces = {i: {} for i in range(len(accelerations))}
        
        # Get list of joints
        joints = set()
        for frame_accelerations in accelerations.values():
            joints.update(frame_accelerations.keys())
        
        # Calculate mass for each body segment
        segment_masses = {
            'head': self.body_mass * self.limb_mass_ratios['head'],
            'torso': self.body_mass * self.limb_mass_ratios['torso'],
            'left_upper_arm': self.body_mass * self.limb_mass_ratios['upper_arm'],
            'right_upper_arm': self.body_mass * self.limb_mass_ratios['upper_arm'],
            'left_forearm': self.body_mass * self.limb_mass_ratios['forearm'],
            'right_forearm': self.body_mass * self.limb_mass_ratios['forearm'],
            'left_hand': self.body_mass * self.limb_mass_ratios['hand'],
            'right_hand': self.body_mass * self.limb_mass_ratios['hand'],
            'left_thigh': self.body_mass * self.limb_mass_ratios['thigh'],
            'right_thigh': self.body_mass * self.limb_mass_ratios['thigh'],
            'left_shin': self.body_mass * self.limb_mass_ratios['shin'],
            'right_shin': self.body_mass * self.limb_mass_ratios['shin'],
            'left_foot': self.body_mass * self.limb_mass_ratios['foot'],
            'right_foot': self.body_mass * self.limb_mass_ratios['foot']
        }
        
        # Map joints to body segments
        joint_segment_map = {
            'nose': 'head',
            'left_shoulder': 'left_upper_arm',
            'right_shoulder': 'right_upper_arm',
            'left_elbow': 'left_forearm',
            'right_elbow': 'right_forearm',
            'left_wrist': 'left_hand',
            'right_wrist': 'right_hand',
            'left_hip': 'left_thigh',
            'right_hip': 'right_thigh',
            'left_knee': 'left_shin',
            'right_knee': 'right_shin',
            'left_ankle': 'left_foot',
            'right_ankle': 'right_foot'
        }
        
        # Calculate force for each joint (F = m*a)
        for i in range(len(accelerations)):
            for joint in accelerations[i]:
                if joint in joint_segment_map:
                    segment = joint_segment_map[joint]
                    mass = segment_masses.get(segment, 1.0)  # Default to 1 kg if not found
                    
                    # Convert acceleration from mm/s^2 to m/s^2
                    acceleration_ms2 = accelerations[i][joint] / 1000.0
                    
                    # Calculate force (F = m*a) in Newtons
                    force = mass * acceleration_ms2
                    
                    forces[i][joint] = force
        
        return forces
    
    def calculate_powers(self, forces, velocities):
        """
        Calculate power for each joint (P = F*v).
        
        Args:
            forces: Dictionary with forces for each joint
            velocities: Dictionary with velocities for each joint
            
        Returns:
            Dictionary with powers for each joint
        """
        if not forces or not velocities:
            return {}
        
        # Initialize storage
        powers = {i: {} for i in range(min(len(forces), len(velocities)))}
        
        # Calculate power for each joint
        for i in range(min(len(forces), len(velocities))):
            for joint in forces[i]:
                if joint in velocities[i]:
                    # Convert velocity from mm/s to m/s
                    velocity_ms = velocities[i][joint] / 1000.0
                    
                    # Calculate power (P = F*v) in Watts
                    power = np.dot(forces[i][joint], velocity_ms)
                    
                    powers[i][joint] = power
        
        return powers
    
    def calculate_joint_angles_sequence(self, pose_sequence):
        """
        Calculate joint angles for a sequence of poses.
        
        Args:
            pose_sequence: List of dictionaries with 3D joint positions
            
        Returns:
            Dictionary with joint angles for each frame
        """
        # Initialize storage
        angles = {i: {} for i in range(len(pose_sequence))}
        
        # Calculate angles for each frame
        for i, pose in enumerate(pose_sequence):
            angles[i] = self.calculate_3d_angles(pose)
        
        return angles
    
    def calculate_angular_velocities(self, angles, timestamps):
        """
        Calculate angular velocities for each joint.
        
        Args:
            angles: Dictionary with joint angles for each frame
            timestamps: List of timestamps for each frame
            
        Returns:
            Dictionary with angular velocities for each joint
        """
        if len(angles) < 2:
            return {}
        
        # Initialize storage
        angular_velocities = {i: {} for i in range(len(angles) - 1)}
        
        # Get list of joints
        joints = set()
        for frame_angles in angles.values():
            joints.update(frame_angles.keys())
        
        # Calculate angular velocity for each joint
        for i in range(len(angles) - 1):
            dt = timestamps[i + 1] - timestamps[i]
            
            if dt <= 0:
                continue  # Skip invalid time differences
            
            # Calculate angular velocity for each joint
            for joint in joints:
                if joint in angles[i] and joint in angles[i + 1]:
                    angle_change = angles[i + 1][joint] - angles[i][joint]
                    
                    # Handle angle wrapping (e.g., 350° to 10° should be -20° not +340°)
                    if angle_change > 180:
                        angle_change -= 360
                    elif angle_change < -180:
                        angle_change += 360
                    
                    angular_velocity = angle_change / dt  # deg/s
                    angular_velocities[i][joint] = angular_velocity
        
        return angular_velocities
    
    def calculate_angular_accelerations(self, angular_velocities, timestamps):
        """
        Calculate angular accelerations for each joint.
        
        Args:
            angular_velocities: Dictionary with angular velocities for each joint
            timestamps: List of timestamps for each frame
            
        Returns:
            Dictionary with angular accelerations for each joint
        """
        if len(angular_velocities) < 2:
            return {}
        
        # Initialize storage
        angular_accelerations = {i: {} for i in range(len(angular_velocities) - 1)}
        
        # Get list of joints
        joints = set()
        for frame_velocities in angular_velocities.values():
            joints.update(frame_velocities.keys())
        
        # Calculate angular acceleration for each joint
        for i in range(len(angular_velocities) - 1):
            # Use center point of velocity timestamps
            t1 = timestamps[i + 1] - timestamps[i]
            t2 = timestamps[i + 2] - timestamps[i + 1]
            dt = (t1 + t2) / 2
            
            if dt <= 0:
                continue  # Skip invalid time differences
            
            # Calculate angular acceleration for each joint
            for joint in joints:
                if (joint in angular_velocities[i] and 
                    joint in angular_velocities[i + 1]):
                    
                    velocity_change = angular_velocities[i + 1][joint] - angular_velocities[i][joint]
                    angular_acceleration = velocity_change / dt  # deg/s^2
                    angular_accelerations[i][joint] = angular_acceleration
        
        return angular_accelerations
    
    def calculate_joint_moments(self, pose_sequence, accelerations):
        """
        Calculate joint moments (torques) for each joint.
        
        Args:
            pose_sequence: List of dictionaries with 3D joint positions
            accelerations: Dictionary with accelerations for each joint
            
        Returns:
            Dictionary with joint moments for each joint
        """
        if not pose_sequence or not accelerations:
            return {}
        
        # Initialize storage
        joint_moments = {i: {} for i in range(min(len(pose_sequence), len(accelerations)))}
        
        # Define moment arms for joint torque calculation
        # This is a simplified approach - realistic biomechanical models are more complex
        moment_arms = {
            'left_shoulder': 0.05,  # 5 cm moment arm (simplified)
            'right_shoulder': 0.05,
            'left_elbow': 0.03,     # 3 cm moment arm
            'right_elbow': 0.03,
            'left_hip': 0.07,       # 7 cm moment arm
            'right_hip': 0.07,
            'left_knee': 0.04,      # 4 cm moment arm
            'right_knee': 0.04,
            'left_ankle': 0.03,     # 3 cm moment arm
            'right_ankle': 0.03
        }
        
        # Get parent-child joint relationships
        parent_child = {
            'left_shoulder': ('left_elbow', 'left_upper_arm'),
            'right_shoulder': ('right_elbow', 'right_upper_arm'),
            'left_elbow': ('left_wrist', 'left_forearm'),
            'right_elbow': ('right_wrist', 'right_forearm'),
            'left_hip': ('left_knee', 'left_thigh'),
            'right_hip': ('right_knee', 'right_thigh'),
            'left_knee': ('left_ankle', 'left_shin'),
            'right_knee': ('right_ankle', 'right_shin')
        }
        
        # Calculate segment masses
        segment_masses = {
            'left_upper_arm': self.body_mass * self.limb_mass_ratios['upper_arm'],
            'right_upper_arm': self.body_mass * self.limb_mass_ratios['upper_arm'],
            'left_forearm': self.body_mass * self.limb_mass_ratios['forearm'],
            'right_forearm': self.body_mass * self.limb_mass_ratios['forearm'],
            'left_thigh': self.body_mass * self.limb_mass_ratios['thigh'],
            'right_thigh': self.body_mass * self.limb_mass_ratios['thigh'],
            'left_shin': self.body_mass * self.limb_mass_ratios['shin'],
            'right_shin': self.body_mass * self.limb_mass_ratios['shin']
        }
        
        # Calculate joint moments for each frame
        for i in range(min(len(pose_sequence), len(accelerations))):
            pose = pose_sequence[i]
            frame_accelerations = accelerations[i]
            
            for joint in parent_child:
                child_joint, segment = parent_child[joint]
                
                if (joint in pose and child_joint in pose and
                    joint in frame_accelerations and segment in segment_masses):
                    
                    # Get segment mass
                    mass = segment_masses[segment]
                    
                    # Get joint and child joint positions
                    joint_pos = pose[joint]
                    child_pos = pose[child_joint]
                    
                    # Calculate segment center of mass (COM) - simplified as midpoint
                    com_pos = (joint_pos + child_pos) / 2
                    
                    # Get joint acceleration
                    joint_accel = frame_accelerations[joint]
                    
                    # Convert acceleration from mm/s^2 to m/s^2
                    joint_accel_ms2 = joint_accel / 1000.0
                    
                    # Calculate force at segment COM (F = m*a)
                    force = mass * joint_accel_ms2
                    
                    # Calculate moment arm vector (from joint to COM)
                    moment_arm_vec = com_pos - joint_pos
                    
                    # Convert from mm to m
                    moment_arm_vec_m = moment_arm_vec / 1000.0
                    
                    # Calculate joint moment (torque) as cross product (τ = r × F)
                    moment = np.cross(moment_arm_vec_m, force)
                    
                    # Store the result
                    joint_moments[i][joint] = moment
        
        return joint_moments
    
    def calculate_joint_powers(self, joint_moments, angular_velocities):
        """
        Calculate joint powers (P = τ * ω).
        
        Args:
            joint_moments: Dictionary with joint moments for each joint
            angular_velocities: Dictionary with angular velocities for each joint
            
        Returns:
            Dictionary with joint powers for each joint
        """
        if not joint_moments or not angular_velocities:
            return {}
        
        # Initialize storage
        joint_powers = {i: {} for i in range(min(len(joint_moments), len(angular_velocities)))}
        
        # Calculate joint power for each frame
        for i in range(min(len(joint_moments), len(angular_velocities))):
            for joint in joint_moments[i]:
                if joint in angular_velocities[i]:
                    # Convert angular velocity from degrees/s to radians/s
                    angular_velocity_rad = np.radians(angular_velocities[i][joint])
                    
                    # Angular velocity is a scalar, but we need a vector aligned with the joint axis
                    # This is a simplified approach - in a real system we'd need the true rotation axis
                    # Here we'll assume the axis is the Z-axis (vertical)
                    angular_velocity_vec = np.array([0, 0, angular_velocity_rad])
                    
                    # Calculate joint power (P = τ * ω) - dot product of moment and angular velocity
                    power = np.dot(joint_moments[i][joint], angular_velocity_vec)
                    
                    # Store the result
                    joint_powers[i][joint] = power
        
        return joint_powers
    
    def create_biomechanics_plots(self, biomechanics, timestamps, output_dir):
        """
        Create plots for biomechanical metrics.
        
        Args:
            biomechanics: Dictionary with biomechanical metrics
            timestamps: List of timestamps for each frame
            output_dir: Directory to save plots
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create plots for each metric
        self.plot_joint_angles(biomechanics['joint_angles'], timestamps, output_dir)
        self.plot_velocities(biomechanics['velocities'], timestamps, output_dir)
        self.plot_accelerations(biomechanics['accelerations'], timestamps, output_dir)
        self.plot_angular_velocities(biomechanics['angular_velocities'], timestamps, output_dir)
        self.plot_angular_accelerations(biomechanics['angular_accelerations'], timestamps, output_dir)
        self.plot_forces(biomechanics['forces'], timestamps, output_dir)
        self.plot_powers(biomechanics['powers'], timestamps, output_dir)
        self.plot_joint_moments(biomechanics['joint_moments'], timestamps, output_dir)
        self.plot_joint_powers(biomechanics['joint_powers'], timestamps, output_dir)
    
    def plot_joint_angles(self, angles, timestamps, output_dir):
        """Create plots of joint angles over time"""
        if not angles:
            return
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Split into multiple subplots
        plt.subplot(2, 2, 1)
        self.plot_metric(angles, timestamps, ['left_shoulder', 'right_shoulder'], 
                        'Shoulder Angles', 'Angle (degrees)')
        
        plt.subplot(2, 2, 2)
        self.plot_metric(angles, timestamps, ['left_elbow', 'right_elbow'], 
                        'Elbow Angles', 'Angle (degrees)')
        
        plt.subplot(2, 2, 3)
        self.plot_metric(angles, timestamps, ['left_hip', 'right_hip'], 
                        'Hip Angles', 'Angle (degrees)')
        
        plt.subplot(2, 2, 4)
        self.plot_metric(angles, timestamps, ['left_knee', 'right_knee'], 
                        'Knee Angles', 'Angle (degrees)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'joint_angles.png'), dpi=300)
        plt.close()
    
    def plot_velocities(self, velocities, timestamps, output_dir):
        """Create plots of linear velocities over time"""
        if not velocities:
            return
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Plot velocities for key joints
        key_joints = [
            ('left_wrist', 'right_wrist'),
            ('left_elbow', 'right_elbow'),
            ('left_knee', 'right_knee'),
            ('left_ankle', 'right_ankle')
        ]
        
        for i, (left_joint, right_joint) in enumerate(key_joints):
            plt.subplot(2, 2, i+1)
            
            # Calculate speed (magnitude of velocity)
            left_speeds = []
            right_speeds = []
            plot_timestamps = []
            
            for frame in range(len(velocities)):
                if left_joint in velocities[frame] and right_joint in velocities[frame]:
                    # Get frame timestamp
                    if frame < len(timestamps) - 1:
                        plot_timestamps.append(timestamps[frame])
                        
                        # Calculate speed (magnitude of velocity)
                        left_speed = np.linalg.norm(velocities[frame][left_joint])
                        right_speed = np.linalg.norm(velocities[frame][right_joint])
                        
                        left_speeds.append(left_speed)
                        right_speeds.append(right_speed)
            
            # Plot speeds
            if plot_timestamps:
                plt.plot(plot_timestamps, left_speeds, 'b-', label=f"Left {left_joint.split('_')[1]}")
                plt.plot(plot_timestamps, right_speeds, 'r-', label=f"Right {right_joint.split('_')[1]}")
                plt.title(f"{left_joint.split('_')[1].capitalize()} Speed")
                plt.xlabel('Time (s)')
                plt.ylabel('Speed (mm/s)')
                plt.grid(True)
                plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'joint_velocities.png'), dpi=300)
        plt.close()

    def plot_accelerations(self, accelerations, timestamps, output_dir):
        """Create plots of linear accelerations over time"""
        if not accelerations:
            return
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Plot accelerations for key joints
        key_joints = [
            ('left_wrist', 'right_wrist'),
            ('left_elbow', 'right_elbow'),
            ('left_knee', 'right_knee'),
            ('left_ankle', 'right_ankle')
        ]
        
        for i, (left_joint, right_joint) in enumerate(key_joints):
            plt.subplot(2, 2, i+1)
            
            # Calculate acceleration magnitude
            left_accel = []
            right_accel = []
            plot_timestamps = []
            
            for frame in range(len(accelerations)):
                if frame < len(timestamps) - 2 and left_joint in accelerations[frame] and right_joint in accelerations[frame]:
                    plot_timestamps.append(timestamps[frame+1])  # Acceleration is centered between frames
                    
                    # Calculate acceleration magnitude
                    left_accel_mag = np.linalg.norm(accelerations[frame][left_joint])
                    right_accel_mag = np.linalg.norm(accelerations[frame][right_joint])
                    
                    left_accel.append(left_accel_mag)
                    right_accel.append(right_accel_mag)
            
            # Plot acceleration magnitudes
            if plot_timestamps:
                plt.plot(plot_timestamps, left_accel, 'b-', label=f"Left {left_joint.split('_')[1]}")
                plt.plot(plot_timestamps, right_accel, 'r-', label=f"Right {right_joint.split('_')[1]}")
                plt.title(f"{left_joint.split('_')[1].capitalize()} Acceleration")
                plt.xlabel('Time (s)')
                plt.ylabel('Acceleration (mm/s²)')
                plt.grid(True)
                plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'joint_accelerations.png'), dpi=300)
        plt.close()

    def plot_angular_velocities(self, angular_velocities, timestamps, output_dir):
        """Create plots of angular velocities over time"""
        if not angular_velocities:
            return
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Define joint groups
        joint_groups = [
            (['left_shoulder', 'right_shoulder'], 'Shoulder Angular Velocity'),
            (['left_elbow', 'right_elbow'], 'Elbow Angular Velocity'),
            (['left_hip', 'right_hip'], 'Hip Angular Velocity'),
            (['left_knee', 'right_knee'], 'Knee Angular Velocity')
        ]
        
        # Plot each joint group
        for i, (joints, title) in enumerate(joint_groups):
            plt.subplot(2, 2, i+1)
            self.plot_metric(angular_velocities, timestamps, joints, title, 'Angular Velocity (deg/s)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'angular_velocities.png'), dpi=300)
        plt.close()

    def plot_angular_accelerations(self, angular_accelerations, timestamps, output_dir):
        """Create plots of angular accelerations over time"""
        if not angular_accelerations:
            return
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Define joint groups
        joint_groups = [
            (['left_shoulder', 'right_shoulder'], 'Shoulder Angular Acceleration'),
            (['left_elbow', 'right_elbow'], 'Elbow Angular Acceleration'),
            (['left_hip', 'right_hip'], 'Hip Angular Acceleration'),
            (['left_knee', 'right_knee'], 'Knee Angular Acceleration')
        ]
        
        # Plot each joint group
        for i, (joints, title) in enumerate(joint_groups):
            plt.subplot(2, 2, i+1)
            self.plot_metric(angular_accelerations, timestamps, joints, title, 'Angular Acceleration (deg/s²)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'angular_accelerations.png'), dpi=300)
        plt.close()

    def plot_forces(self, forces, timestamps, output_dir):
        """Create plots of forces over time"""
        if not forces:
            return
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Plot forces for key joints
        key_joints = [
            ('left_wrist', 'right_wrist'),
            ('left_elbow', 'right_elbow'),
            ('left_knee', 'right_knee'),
            ('left_ankle', 'right_ankle')
        ]
        
        for i, (left_joint, right_joint) in enumerate(key_joints):
            plt.subplot(2, 2, i+1)
            
            # Calculate force magnitude
            left_forces = []
            right_forces = []
            plot_timestamps = []
            
            for frame in range(len(forces)):
                if left_joint in forces[frame] and right_joint in forces[frame]:
                    if frame < len(timestamps):
                        plot_timestamps.append(timestamps[frame])
                        
                        # Calculate force magnitude (N)
                        left_force = np.linalg.norm(forces[frame][left_joint])
                        right_force = np.linalg.norm(forces[frame][right_joint])
                        
                        left_forces.append(left_force)
                        right_forces.append(right_force)
            
            # Plot forces
            if plot_timestamps:
                plt.plot(plot_timestamps, left_forces, 'b-', label=f"Left {left_joint.split('_')[1]}")
                plt.plot(plot_timestamps, right_forces, 'r-', label=f"Right {right_joint.split('_')[1]}")
                plt.title(f"{left_joint.split('_')[1].capitalize()} Force")
                plt.xlabel('Time (s)')
                plt.ylabel('Force (N)')
                plt.grid(True)
                plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'joint_forces.png'), dpi=300)
        plt.close()

    def plot_powers(self, powers, timestamps, output_dir):
        """Create plots of powers over time"""
        if not powers:
            return
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Define joint groups
        joint_groups = [
            (['left_shoulder', 'right_shoulder'], 'Shoulder Power'),
            (['left_elbow', 'right_elbow'], 'Elbow Power'),
            (['left_hip', 'right_hip'], 'Hip Power'),
            (['left_knee', 'right_knee'], 'Knee Power')
        ]
        
        # Plot each joint group
        for i, (joints, title) in enumerate(joint_groups):
            plt.subplot(2, 2, i+1)
            self.plot_metric(powers, timestamps, joints, title, 'Power (W)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'joint_powers.png'), dpi=300)
        plt.close()

    def plot_joint_moments(self, joint_moments, timestamps, output_dir):
        """Create plots of joint moments over time"""
        if not joint_moments:
            return
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Define joint groups
        joint_groups = [
            (['left_shoulder', 'right_shoulder'], 'Shoulder Moment'),
            (['left_elbow', 'right_elbow'], 'Elbow Moment'),
            (['left_hip', 'right_hip'], 'Hip Moment'),
            (['left_knee', 'right_knee'], 'Knee Moment')
        ]
        
        # Plot each joint group
        for i, (joints, title) in enumerate(joint_groups):
            plt.subplot(2, 2, i+1)
            self.plot_metric(joint_moments, timestamps, joints, title, 'Moment (Nm)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'joint_moments.png'), dpi=300)
        plt.close()

    def plot_joint_powers(self, joint_powers, timestamps, output_dir):
        """Create plots of joint powers over time"""
        if not joint_powers:
            return
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Define joint groups
        joint_groups = [
            (['left_shoulder', 'right_shoulder'], 'Shoulder Joint Power'),
            (['left_elbow', 'right_elbow'], 'Elbow Joint Power'),
            (['left_hip', 'right_hip'], 'Hip Joint Power'),
            (['left_knee', 'right_knee'], 'Knee Joint Power')
        ]
        
        # Plot each joint group
        for i, (joints, title) in enumerate(joint_groups):
            plt.subplot(2, 2, i+1)
            self.plot_metric(joint_powers, timestamps, joints, title, 'Joint Power (W)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'joint_mechanical_powers.png'), dpi=300)
        plt.close()
    
    def plot_metric(self, metric_data, timestamps, joints, title, ylabel):
        """Helper function to plot metric data for specific joints"""
        colors = {'left': 'b', 'right': 'r'}
        
        # Extract data for each joint
        for joint in joints:
            side = joint.split('_')[0]  # 'left' or 'right'
            joint_data = []
            plot_timestamps = []
            
            for frame in range(len(metric_data)):
                if joint in metric_data[frame]:
                    if isinstance(metric_data[frame][joint], (int, float)):
                        # Scalar data
                        value = metric_data[frame][joint]
                        if frame < len(timestamps):
                            plot_timestamps.append(timestamps[frame])
                            joint_data.append(value)
                    elif hasattr(metric_data[frame][joint], '__iter__'):
                        # Vector data - use magnitude
                        value = np.linalg.norm(metric_data[frame][joint])
                        if frame < len(timestamps):
                            plot_timestamps.append(timestamps[frame])
                            joint_data.append(value)
            
            # Plot data
            if plot_timestamps and joint_data:
                plt.plot(plot_timestamps, joint_data, 
                        f"{colors[side]}-", 
                        label=f"{side.capitalize()} {joint.split('_')[1]}")
        
        plt.title(title)
        plt.xlabel('Time (s)')
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.legend()
    
    def export_biomechanics_to_csv(self, biomechanics, timestamps, output_dir):
        """
        Export biomechanical metrics to CSV files.
        
        Args:
            biomechanics: Dictionary with biomechanical metrics
            timestamps: List of timestamps for each frame
            output_dir: Directory to save CSV files
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Export angles
        self.export_metric_to_csv(biomechanics['joint_angles'], timestamps, 
                                  os.path.join(output_dir, 'joint_angles.csv'),
                                  'Angle (degrees)')
        
        # Export velocities (speeds)
        self.export_vector_magnitude_to_csv(biomechanics['velocities'], timestamps,
                                          os.path.join(output_dir, 'joint_speeds.csv'),
                                          'Speed (mm/s)')
        
        # Export angular velocities
        self.export_metric_to_csv(biomechanics['angular_velocities'], timestamps,
                                 os.path.join(output_dir, 'angular_velocities.csv'),
                                 'Angular Velocity (deg/s)')
        
        # Export forces (magnitudes)
        self.export_vector_magnitude_to_csv(biomechanics['forces'], timestamps,
                                          os.path.join(output_dir, 'joint_forces.csv'),
                                          'Force (N)')
        
        # Export powers
        self.export_metric_to_csv(biomechanics['powers'], timestamps,
                                 os.path.join(output_dir, 'joint_powers.csv'),
                                 'Power (W)')
    
    def export_metric_to_csv(self, metric_data, timestamps, output_file, metric_name):
        """Export scalar metric data to CSV file"""
        if not metric_data:
            return
        
        # Get list of all joints
        joints = set()
        for frame_data in metric_data.values():
            joints.update(frame_data.keys())
        
        # Create DataFrame
        data = {'timestamp': []}
        for joint in sorted(joints):
            data[f"{joint}_{metric_name}"] = []
        
        # Fill data
        for frame in range(len(metric_data)):
            if frame < len(timestamps):
                data['timestamp'].append(timestamps[frame])
                
                for joint in sorted(joints):
                    if joint in metric_data[frame]:
                        value = metric_data[frame][joint]
                        data[f"{joint}_{metric_name}"].append(value)
                    else:
                        data[f"{joint}_{metric_name}"].append(None)
        
        # Create DataFrame and export
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
    
    def export_vector_magnitude_to_csv(self, vector_data, timestamps, output_file, metric_name):
        """Export vector magnitude data to CSV file"""
        if not vector_data:
            return
        
        # Get list of all joints
        joints = set()
        for frame_data in vector_data.values():
            joints.update(frame_data.keys())
        
        # Create DataFrame
        data = {'timestamp': []}
        for joint in sorted(joints):
            data[f"{joint}_{metric_name}"] = []
            data[f"{joint}_x"] = []
            data[f"{joint}_y"] = []
            data[f"{joint}_z"] = []
        
        # Fill data
        for frame in range(len(vector_data)):
            if frame < len(timestamps):
                data['timestamp'].append(timestamps[frame])
                
                for joint in sorted(joints):
                    if joint in vector_data[frame]:
                        # Store magnitude
                        magnitude = np.linalg.norm(vector_data[frame][joint])
                        data[f"{joint}_{metric_name}"].append(magnitude)
                        
                        # Store individual components
                        data[f"{joint}_x"].append(vector_data[frame][joint][0])
                        data[f"{joint}_y"].append(vector_data[frame][joint][1])
                        data[f"{joint}_z"].append(vector_data[frame][joint][2])
                    else:
                        data[f"{joint}_{metric_name}"].append(None)
                        data[f"{joint}_x"].append(None)
                        data[f"{joint}_y"].append(None)
                        data[f"{joint}_z"].append(None)
        
        # Create DataFrame and export
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)