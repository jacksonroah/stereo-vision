#!/usr/bin/env python3
import numpy as np
from scipy.signal import savgol_filter
import pandas as pd
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("motion_smoothing")

class MotionSmoother:
    """
    A class for applying various smoothing techniques to 3D pose data,
    with a focus on biomechanical constraints and motion filtering.
    """
    
    # Preset configurations for different camera types/frame rates
    PRESET_CONFIGS = {
        'smalliphone': {  # 30 fps iPhone
            'window_size': 9,
            'poly_order': 2,
            'limb_length_tolerance': 0.05,  # 5% tolerance
            'velocity_threshold': 50.0,     # mm per frame
            'smoothing_method': 'savgol'
        },
        'iphone': {  # 60 fps iPhone
            'window_size': 13, 
            'poly_order': 3,
            'limb_length_tolerance': 0.04,  # 4% tolerance
            'velocity_threshold': 80.0,     # mm per frame
            'smoothing_method': 'savgol'
        },
        'edger': {  # Edgertronics 480 fps
            'window_size': 31,
            'poly_order': 3,
            'limb_length_tolerance': 0.03,  # 3% tolerance
            'velocity_threshold': 100.0,    # mm per frame
            'smoothing_method': 'savgol'
        }
    }
    
    def __init__(self, preset='smalliphone', **kwargs):
        """
        Initialize the motion smoother with specified parameters.
        
        Args:
            preset (str): Preset configuration ('smalliphone', 'iphone', or 'edger')
            **kwargs: Override specific parameters from the preset
        """
        # Start with preset configuration
        if preset in self.PRESET_CONFIGS:
            self.config = self.PRESET_CONFIGS[preset].copy()
            logger.info(f"Using preset configuration: {preset}")
        else:
            logger.warning(f"Unknown preset '{preset}', defaulting to 'smalliphone'")
            self.config = self.PRESET_CONFIGS['smalliphone'].copy()
        
        # Override with any provided parameters
        self.config.update(kwargs)
        
        # Log configuration
        logger.info(f"Motion smoother configuration: {self.config}")
        
        # Initialize historical data
        self.pose_history = []
        self.joint_velocity_history = {}
        self.joint_acceleration_history = {}
        self.reference_limb_lengths = {}
        
        # Initialize stats for reporting
        self.stats = {
            'frames_processed': 0,
            'anatomical_corrections': 0,
            'velocity_corrections': 0,
            'processing_time': 0
        }
    
    def smooth_pose_sequence(self, pose_sequence):
        """
        Apply smoothing to a sequence of poses.
        
        Args:
            pose_sequence (list): List of pose dictionaries, each with joint positions
            
        Returns:
            list: Smoothed pose sequence
        """
        start_time = time.time()
        
        # Reset stats for this sequence
        self.stats = {
            'frames_processed': 0,
            'anatomical_corrections': 0,
            'velocity_corrections': 0,
            'processing_time': 0
        }
        
        # Check if we have enough frames for smoothing
        if len(pose_sequence) < self.config['window_size']:
            logger.warning(f"Sequence too short for smoothing ({len(pose_sequence)} frames, "
                         f"window size {self.config['window_size']}). Returning original sequence.")
            return pose_sequence
        
        # Convert sequence to a format suitable for filtering
        joints_data = self._extract_joint_trajectories(pose_sequence)
        
        # Apply filtering based on configuration
        smoothed_data = self._apply_smoothing(joints_data)
        
        # Calculate limb lengths before anatomical constraints
        limb_lengths_before = self._calculate_average_limb_lengths(pose_sequence)
        
        # Reconstruct poses from smoothed joint data
        smoothed_poses = self._reconstruct_poses(smoothed_data, pose_sequence)
        
        # Apply anatomical constraints
        smoothed_poses = self._apply_anatomical_constraints(smoothed_poses, limb_lengths_before)
        
        # Apply velocity constraints
        smoothed_poses = self._apply_velocity_constraints(smoothed_poses)
        
        # Calculate limb lengths after all constraints
        limb_lengths_after = self._calculate_average_limb_lengths(smoothed_poses)
        
        # Update stats
        self.stats['frames_processed'] = len(pose_sequence)
        self.stats['processing_time'] = time.time() - start_time
        
        logger.info(f"Smoothing complete: {self.stats['frames_processed']} frames processed in "
                  f"{self.stats['processing_time']:.3f}s")
        logger.info(f"Anatomical corrections: {self.stats['anatomical_corrections']}, "
                  f"Velocity corrections: {self.stats['velocity_corrections']}")
        
        return smoothed_poses
    
    def smooth_single_pose(self, pose, pose_history=None):
        """
        Apply smoothing to a single pose using historical data.
        
        Args:
            pose (dict): Current pose with joint positions
            pose_history (list, optional): Previous poses for temporal smoothing
            
        Returns:
            dict: Smoothed pose
        """
        if pose_history is not None:
            self.pose_history = pose_history
        
        # Add current pose to history
        self.pose_history.append(pose)
        
        # Apply smoothing if we have enough frames
        if len(self.pose_history) >= self.config['window_size']:
            # Extract current history window
            window = self.pose_history[-self.config['window_size']:]
            
            # Smooth the window
            smoothed_window = self.smooth_pose_sequence(window)
            
            # Return only the last (current) smoothed pose
            return smoothed_window[-1]
        else:
            # Not enough frames for smoothing yet, return original with anatomical constraints
            constrained_pose = self._apply_anatomical_constraints_to_single_pose(pose)
            return constrained_pose
    
    def get_pose_history(self):
        """
        Get the current pose history.
        
        Returns:
            list: Current pose history
        """
        return self.pose_history
    
    def get_stats(self):
        """
        Get smoothing statistics.
        
        Returns:
            dict: Statistics about the smoothing process
        """
        return self.stats
    
    def reset(self):
        """Reset the smoother's state."""
        self.pose_history = []
        self.joint_velocity_history = {}
        self.joint_acceleration_history = {}
        self.reference_limb_lengths = {}
        
        self.stats = {
            'frames_processed': 0,
            'anatomical_corrections': 0,
            'velocity_corrections': 0,
            'processing_time': 0
        }
        
        logger.info("Motion smoother reset")
    
    def _extract_joint_trajectories(self, pose_sequence):
        """
        Extract trajectories for each joint from a sequence of poses.
        
        Args:
            pose_sequence (list): List of pose dictionaries
            
        Returns:
            dict: Dictionary mapping joint names to arrays of positions
        """
        joints_data = {}
        
        # Determine all joint names present in the poses
        all_joints = set()
        for pose in pose_sequence:
            all_joints.update(pose.keys())
        
        # Extract positions for each joint
        for joint in all_joints:
            # Initialize arrays for X, Y, Z coordinates
            x_values = []
            y_values = []
            z_values = []
            
            # Extract positions for this joint from each pose
            for pose in pose_sequence:
                if joint in pose:
                    pos = pose[joint]
                    x_values.append(pos[0])
                    y_values.append(pos[1])
                    z_values.append(pos[2])
                else:
                    # If joint is missing in this pose, use NaN
                    x_values.append(np.nan)
                    y_values.append(np.nan)
                    z_values.append(np.nan)
            
            # Store trajectories
            joints_data[joint] = {
                'x': np.array(x_values),
                'y': np.array(y_values),
                'z': np.array(z_values)
            }
        
        return joints_data
    
    def _apply_smoothing(self, joints_data):
        """
        Apply chosen smoothing method to joint trajectories.
        
        Args:
            joints_data (dict): Joint trajectories
            
        Returns:
            dict: Smoothed joint trajectories
        """
        smoothed_data = {}
        method = self.config['smoothing_method']
        
        for joint, data in joints_data.items():
            smoothed_data[joint] = {}
            
            for axis in ['x', 'y', 'z']:
                values = data[axis]
                
                # Skip if not enough valid values
                if np.sum(~np.isnan(values)) < self.config['window_size']:
                    smoothed_data[joint][axis] = values
                    continue
                
                # Apply appropriate smoothing method
                if method == 'savgol':
                    smoothed_data[joint][axis] = self._apply_savgol_filter(values)
                elif method == 'moving_average':
                    smoothed_data[joint][axis] = self._apply_moving_average(values)
                elif method == 'one_euro':
                    smoothed_data[joint][axis] = self._apply_one_euro_filter(values, joint, axis)
                else:
                    # Default to Savitzky-Golay
                    smoothed_data[joint][axis] = self._apply_savgol_filter(values)
        
        return smoothed_data
    
    def _apply_savgol_filter(self, values):
        """
        Apply Savitzky-Golay filter to a series of values.
        
        Args:
            values (array): Array of values to filter
            
        Returns:
            array: Filtered values
        """
        # Handle NaN values if present
        has_nan = np.isnan(values).any()
        
        if has_nan:
            # Create a mask of valid values
            valid_mask = ~np.isnan(values)
            
            if np.sum(valid_mask) < self.config['window_size']:
                # Not enough valid points for filtering
                return values
            
            # Create a copy of the values for filtering
            filtered_values = values.copy()
            
            # Create indices for valid values
            indices = np.arange(len(values))[valid_mask]
            
            # Filter only valid values
            valid_values = values[valid_mask]
            
            # Apply Savitzky-Golay filter
            if len(valid_values) >= self.config['window_size']:
                # Ensure window size is odd
                window_size = self.config['window_size']
                if window_size % 2 == 0:
                    window_size += 1
                
                # Apply the filter
                valid_filtered = savgol_filter(
                    valid_values, 
                    window_size, 
                    self.config['poly_order']
                )
                
                # Replace filtered values
                filtered_values[indices] = valid_filtered
                
                return filtered_values
            else:
                return values
        else:
            # No NaN values, directly apply filter
            window_size = self.config['window_size']
            if window_size % 2 == 0:
                window_size += 1
                
            return savgol_filter(
                values, 
                window_size, 
                self.config['poly_order']
            )
    
    def _apply_moving_average(self, values, window_size=None):
        """
        Apply moving average filter to a series of values.
        
        Args:
            values (array): Array of values to filter
            window_size (int, optional): Window size (default: from config)
            
        Returns:
            array: Filtered values
        """
        if window_size is None:
            window_size = self.config['window_size']
        
        # Handle NaN values
        if np.isnan(values).any():
            # Use pandas for moving average with NaN handling
            series = pd.Series(values)
            return series.rolling(window=window_size, center=True, min_periods=1).mean().values
        else:
            # Simple moving average for non-NaN data
            result = np.convolve(values, np.ones(window_size)/window_size, mode='same')
            
            # Fix edge effects
            half_window = window_size // 2
            result[:half_window] = values[:half_window]
            result[-half_window:] = values[-half_window:]
            
            return result
    
    def _apply_one_euro_filter(self, values, joint, axis):
        """
        Apply One Euro Filter to a series of values.
        
        Args:
            values (array): Array of values to filter
            joint (str): Joint name (for tracking state)
            axis (str): Axis name (x, y, z)
            
        Returns:
            array: Filtered values
        """
        # One Euro Filter parameters
        min_cutoff = 1.0  # Minimum cutoff frequency
        beta = 0.1       # Speed coefficient
        
        # Initialize filtered values with original values
        filtered = np.copy(values)
        
        # Handle NaN values
        valid_mask = ~np.isnan(values)
        if np.sum(valid_mask) < 2:
            return values
        
        # Create a key for this joint+axis
        key = f"{joint}_{axis}"
        
        # Initialize state if needed
        if key not in self.joint_velocity_history:
            self.joint_velocity_history[key] = 0.0
        
        # Get valid indices
        valid_indices = np.where(valid_mask)[0]
        
        # Apply filter to valid values
        prev_value = values[valid_indices[0]]
        prev_filtered = prev_value
        prev_timestamp = 0
        
        for i in range(1, len(valid_indices)):
            idx = valid_indices[i]
            timestamp = idx  # Use frame index as timestamp
            
            # Calculate dt
            dt = timestamp - prev_timestamp
            if dt == 0:
                continue
                
            # Get current value
            value = values[idx]
            
            # Calculate cutoff frequency based on derivative
            dx = value - prev_value
            derivative = dx / dt
            
            # Update velocity
            self.joint_velocity_history[key] = derivative
            
            # Adjust cutoff frequency based on velocity
            cutoff = min_cutoff + beta * abs(derivative)
            
            # Apply low-pass filter
            alpha = 1.0 / (1.0 + (1.0 / (cutoff * dt)))
            filtered_value = alpha * value + (1 - alpha) * prev_filtered
            
            # Store result
            filtered[idx] = filtered_value
            
            # Update for next iteration
            prev_value = value
            prev_filtered = filtered_value
            prev_timestamp = timestamp
        
        return filtered
    
    def _reconstruct_poses(self, smoothed_data, original_poses):
        """
        Reconstruct poses from smoothed joint data.
        
        Args:
            smoothed_data (dict): Smoothed joint trajectories
            original_poses (list): Original poses for structure
            
        Returns:
            list: Reconstructed poses with smoothed joint positions
        """
        smoothed_poses = []
        
        for i in range(len(original_poses)):
            new_pose = {}
            
            # Copy all joints from smoothed data
            for joint in smoothed_data:
                # Check if this joint has valid data for this frame
                x = smoothed_data[joint]['x'][i]
                y = smoothed_data[joint]['y'][i]
                z = smoothed_data[joint]['z'][i]
                
                if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                    new_pose[joint] = np.array([x, y, z])
                elif joint in original_poses[i]:
                    # Use original value for missing data
                    new_pose[joint] = original_poses[i][joint]
            
            smoothed_poses.append(new_pose)
        
        return smoothed_poses
    
    def _calculate_average_limb_lengths(self, pose_sequence):
        """
        Calculate average limb lengths across a pose sequence.
        
        Args:
            pose_sequence (list): List of pose dictionaries
            
        Returns:
            dict: Average lengths for each defined limb
        """
        # Define limb pairs (joints that should maintain consistent distance)
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
            ('left_hip', 'right_hip'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip')
        ]
        
        # Initialize counters and sums
        length_sums = {pair: 0.0 for pair in limb_pairs}
        length_counts = {pair: 0 for pair in limb_pairs}
        
        # Calculate lengths for each pose
        for pose in pose_sequence:
            for joint1, joint2 in limb_pairs:
                if joint1 in pose and joint2 in pose:
                    # Calculate Euclidean distance
                    distance = np.linalg.norm(pose[joint1] - pose[joint2])
                    
                    # Only use reasonable values
                    if 10.0 < distance < 600.0:  # mm
                        length_sums[(joint1, joint2)] += distance
                        length_counts[(joint1, joint2)] += 1
        
        # Calculate averages
        avg_lengths = {}
        for pair in limb_pairs:
            if length_counts[pair] > 0:
                avg_lengths[pair] = length_sums[pair] / length_counts[pair]
        
        return avg_lengths
    
    def _apply_anatomical_constraints(self, pose_sequence, reference_lengths=None):
        """
        Apply anatomical constraints to a sequence of poses.
        
        Args:
            pose_sequence (list): List of pose dictionaries
            reference_lengths (dict, optional): Reference limb lengths
            
        Returns:
            list: Pose sequence with enforced anatomical constraints
        """
        # If no reference lengths provided, calculate from the sequence
        if reference_lengths is None:
            reference_lengths = self._calculate_average_limb_lengths(pose_sequence)
        
        # Store reference lengths for later use
        self.reference_limb_lengths = reference_lengths
        
        # Apply constraints to each pose
        constrained_poses = []
        for pose in pose_sequence:
            constrained_pose = self._apply_anatomical_constraints_to_single_pose(
                pose, reference_lengths)
            constrained_poses.append(constrained_pose)
        
        return constrained_poses
    
    def _apply_anatomical_constraints_to_single_pose(self, pose, reference_lengths=None):
        """
        Apply anatomical constraints to a single pose.
        
        Args:
            pose (dict): Pose dictionary with joint positions
            reference_lengths (dict, optional): Reference limb lengths
            
        Returns:
            dict: Pose with enforced anatomical constraints
        """
        # Make a copy to avoid modifying the original
        constrained_pose = {k: v.copy() for k, v in pose.items()}
        
        # Use provided reference lengths or stored ones
        if reference_lengths is None:
            reference_lengths = self.reference_limb_lengths
            
            # If still no reference lengths, use default reasonable values
            if not reference_lengths:
                # Default anthropometric values (in mm, approximate)
                reference_lengths = {
                    ('left_shoulder', 'left_elbow'): 300.0,
                    ('left_elbow', 'left_wrist'): 250.0,
                    ('right_shoulder', 'right_elbow'): 300.0,
                    ('right_elbow', 'right_wrist'): 250.0,
                    ('left_hip', 'left_knee'): 400.0,
                    ('left_knee', 'left_ankle'): 380.0,
                    ('right_hip', 'right_knee'): 400.0,
                    ('right_knee', 'right_ankle'): 380.0,
                    ('left_shoulder', 'right_shoulder'): 350.0,
                    ('left_hip', 'right_hip'): 250.0,
                    ('left_shoulder', 'left_hip'): 450.0,
                    ('right_shoulder', 'right_hip'): 450.0
                }
        
        # Apply limb length constraints
        corrections_made = 0
        for (joint1, joint2), ref_length in reference_lengths.items():
            if joint1 in constrained_pose and joint2 in constrained_pose:
                # Calculate current length
                current_vector = constrained_pose[joint2] - constrained_pose[joint1]
                current_length = np.linalg.norm(current_vector)
                
                # Check if adjustment needed
                tolerance = self.config['limb_length_tolerance']
                if abs(current_length - ref_length) / ref_length > tolerance:
                    # Adjust joint positions to match reference length
                    normalized_vector = current_vector / current_length
                    new_vector = normalized_vector * ref_length
                    
                    # Move both joints equally
                    midpoint = (constrained_pose[joint1] + constrained_pose[joint2]) / 2
                    constrained_pose[joint1] = midpoint - new_vector / 2
                    constrained_pose[joint2] = midpoint + new_vector / 2
                    
                    corrections_made += 1
        
        # Update stats
        self.stats['anatomical_corrections'] += corrections_made
        
        return constrained_pose
    
    def _apply_velocity_constraints(self, pose_sequence):
        """
        Apply velocity constraints to a sequence of poses.
        
        Args:
            pose_sequence (list): List of pose dictionaries
            
        Returns:
            list: Pose sequence with enforced velocity constraints
        """
        constrained_poses = pose_sequence.copy()
        
        # Cannot apply velocity constraints with fewer than 3 frames
        if len(pose_sequence) < 3:
            return constrained_poses
        
        velocity_threshold = self.config['velocity_threshold']
        
        # Process each joint for each frame
        for i in range(1, len(pose_sequence) - 1):
            prev_pose = constrained_poses[i-1]
            curr_pose = constrained_poses[i]
            next_pose = constrained_poses[i+1]
            
            for joint in curr_pose:
                if joint in prev_pose and joint in next_pose:
                    # Calculate velocities
                    velocity_prev = curr_pose[joint] - prev_pose[joint]
                    velocity_next = next_pose[joint] - curr_pose[joint]
                    
                    # Check for sudden speed changes
                    speed_prev = np.linalg.norm(velocity_prev)
                    speed_next = np.linalg.norm(velocity_next)
                    
                    if speed_prev > velocity_threshold or speed_next > velocity_threshold:
                        # Calculate average velocity
                        avg_velocity = (velocity_prev + velocity_next) / 2
                        
                        # Apply velocity constraint
                        constrained_poses[i][joint] = (prev_pose[joint] + next_pose[joint]) / 2
                        
                        self.stats['velocity_corrections'] += 1
        
        return constrained_poses
    
    def calculate_joint_velocities(self, pose_sequence, time_delta=1.0/30.0):
        """
        Calculate velocities for all joints in a pose sequence.
        
        Args:
            pose_sequence (list): List of pose dictionaries
            time_delta (float): Time between frames in seconds
            
        Returns:
            list: List of dictionaries with joint velocities
        """
        velocities = []
        
        # Need at least 2 frames to calculate velocity
        if len(pose_sequence) < 2:
            return velocities
        
        for i in range(1, len(pose_sequence)):
            prev_pose = pose_sequence[i-1]
            curr_pose = pose_sequence[i]
            
            velocity_dict = {}
            
            for joint in curr_pose:
                if joint in prev_pose:
                    # Calculate displacement
                    displacement = curr_pose[joint] - prev_pose[joint]
                    
                    # Calculate velocity (displacement / time)
                    velocity = displacement / time_delta
                    
                    # Store velocity
                    velocity_dict[joint] = velocity
            
            velocities.append(velocity_dict)
        
        return velocities
    
    def calculate_joint_accelerations(self, velocities, time_delta=1.0/30.0):
        """
        Calculate accelerations from joint velocities.
        
        Args:
            velocities (list): List of dictionaries with joint velocities
            time_delta (float): Time between frames in seconds
            
        Returns:
            list: List of dictionaries with joint accelerations
        """
        accelerations = []
        
        # Need at least 2 velocity frames to calculate acceleration
        if len(velocities) < 2:
            return accelerations
        
        for i in range(1, len(velocities)):
            prev_vel = velocities[i-1]
            curr_vel = velocities[i]
            
            accel_dict = {}
            
            for joint in curr_vel:
                if joint in prev_vel:
                    # Calculate velocity change
                    vel_change = curr_vel[joint] - prev_vel[joint]
                    
                    # Calculate acceleration (velocity change / time)
                    acceleration = vel_change / time_delta
                    
                    # Store acceleration
                    accel_dict[joint] = acceleration
            
            accelerations.append(accel_dict)
        
        return accelerations
    
    def graceful_shutdown(self):
        """Perform cleanup operations before shutdown."""
        # Log any final statistics or information
        logger.info("Motion smoother shutting down")
        logger.info(f"Total frames processed: {self.stats['frames_processed']}")
        logger.info(f"Total anatomical corrections: {self.stats['anatomical_corrections']}")
        logger.info(f"Total velocity corrections: {self.stats['velocity_corrections']}")
        
        # Reset internal state
        self.reset()
        
        return True

# Demo/test function
def test_motion_smoother():
    """Test the motion smoother with synthetic data."""
    # Create synthetic pose data with noise
    np.random.seed(42)
    frames = 100
    
    # Create a simple pendulum motion with noise
    t = np.linspace(0, 2*np.pi, frames)
    x = 300 + 100 * np.sin(t) + np.random.normal(0, 5, frames)
    y = 400 + 100 * np.cos(t) + np.random.normal(0, 5, frames)
    z = np.zeros(frames) + np.random.normal(0, 2, frames)
    
    # Create pose sequence
    poses = []
    for i in range(frames):
        pose = {
            'hand': np.array([x[i], y[i], z[i]]),
            'elbow': np.array([x[i]*0.5, y[i]*0.5, z[i]]),
            'shoulder': np.array([0, 400, 0])
        }
        poses.append(pose)
    
    # Initialize smoother with smalliphone preset
    smoother = MotionSmoother(preset='smalliphone')
    
    # Apply smoothing
    smoothed_poses = smoother.smooth_pose_sequence(poses)
    
    # Calculate RMSE for hand joint
    original_hand = np.array([pose['hand'] for pose in poses])
    smoothed_hand = np.array([pose['hand'] for pose in smoothed_poses])
    
    rmse = np.sqrt(np.mean((original_hand - smoothed_hand)**2))
    print(f"Hand joint RMSE: {rmse}")
    
    # Print stats
    print(f"Smoothing stats: {smoother.get_stats()}")
    
    return {
        'original': poses,
        'smoothed': smoothed_poses,
        'stats': smoother.get_stats()
    }

if __name__ == "__main__":
    # Run test if executed directly
    results = test_motion_smoother()
    print("Motion smoother test complete")