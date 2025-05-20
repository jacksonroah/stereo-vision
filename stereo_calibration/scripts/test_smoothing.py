#!/usr/bin/env python3
# test_smoothing.py - Test script for motion smoothing

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from motion_smoothing import MotionSmoother
import argparse

def load_pose_data(data_file):
    """Load pose data from pickle file"""
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    return data

def plot_joint_trajectory(original_data, smoothed_data, joint_name, output_dir=None):
    """Plot original vs smoothed trajectory for a specific joint"""
    # Extract positions for the specified joint
    original_positions = []
    for pose in original_data['poses']:
        if joint_name in pose:
            original_positions.append(pose[joint_name])
        else:
            original_positions.append(np.array([np.nan, np.nan, np.nan]))
    
    smoothed_positions = []
    for pose in smoothed_data['poses']:
        if joint_name in pose:
            smoothed_positions.append(pose[joint_name])
        else:
            smoothed_positions.append(np.array([np.nan, np.nan, np.nan]))
    
    # Convert to numpy arrays for easier manipulation
    original_positions = np.array(original_positions)
    smoothed_positions = np.array(smoothed_positions)
    
    # Create timestamps
    timestamps = original_data['timestamps']
    
    # Create figure with 3 subplots (x, y, z)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Plot X positions
    axes[0].plot(timestamps, original_positions[:, 0], 'r-', label='Original', alpha=0.6)
    axes[0].plot(timestamps, smoothed_positions[:, 0], 'b-', label='Smoothed', linewidth=2)
    axes[0].set_ylabel('X Position (mm)')
    axes[0].set_title(f'{joint_name} - X Trajectory')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot Y positions
    axes[1].plot(timestamps, original_positions[:, 1], 'r-', label='Original', alpha=0.6)
    axes[1].plot(timestamps, smoothed_positions[:, 1], 'b-', label='Smoothed', linewidth=2)
    axes[1].set_ylabel('Y Position (mm)')
    axes[1].set_title(f'{joint_name} - Y Trajectory')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot Z positions
    axes[2].plot(timestamps, original_positions[:, 2], 'r-', label='Original', alpha=0.6)
    axes[2].plot(timestamps, smoothed_positions[:, 2], 'b-', label='Smoothed', linewidth=2)
    axes[2].set_ylabel('Z Position (mm)')
    axes[2].set_title(f'{joint_name} - Z Trajectory')
    axes[2].set_xlabel('Time (s)')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    
    # Save figure if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{joint_name}_trajectory.png'), dpi=300)
        print(f"Saved trajectory plot to {os.path.join(output_dir, f'{joint_name}_trajectory.png')}")
    else:
        plt.show()
    
    plt.close()

def compare_limb_lengths(original_data, smoothed_data, output_dir=None):
    """Compare limb lengths between original and smoothed data"""
    # Define limb pairs
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
    
    # Calculate limb lengths for each frame
    original_lengths = {pair: [] for pair in limb_pairs}
    smoothed_lengths = {pair: [] for pair in limb_pairs}
    
    # Process each frame
    for orig_pose, smooth_pose in zip(original_data['poses'], smoothed_data['poses']):
        for pair in limb_pairs:
            joint1, joint2 = pair
            
            # Calculate length for original pose
            if joint1 in orig_pose and joint2 in orig_pose:
                dist = np.linalg.norm(orig_pose[joint1] - orig_pose[joint2])
                original_lengths[pair].append(dist)
            else:
                original_lengths[pair].append(np.nan)
            
            # Calculate length for smoothed pose
            if joint1 in smooth_pose and joint2 in smooth_pose:
                dist = np.linalg.norm(smooth_pose[joint1] - smooth_pose[joint2])
                smoothed_lengths[pair].append(dist)
            else:
                smoothed_lengths[pair].append(np.nan)
    
    # Plot limb lengths over time
    timestamps = original_data['timestamps']
    
    # Create multiple figures, 2 limbs per figure
    num_figures = (len(limb_pairs) + 1) // 2
    
    for fig_idx in range(num_figures):
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        for subplot_idx in range(2):
            pair_idx = fig_idx * 2 + subplot_idx
            
            if pair_idx < len(limb_pairs):
                pair = limb_pairs[pair_idx]
                joint1, joint2 = pair
                
                # Get limb name
                limb_name = f"{joint1.replace('_', ' ')} to {joint2.replace('_', ' ')}"
                
                # Plot original and smoothed lengths
                axes[subplot_idx].plot(timestamps, original_lengths[pair], 'r-', 
                                       label='Original', alpha=0.6)
                axes[subplot_idx].plot(timestamps, smoothed_lengths[pair], 'b-', 
                                       label='Smoothed', linewidth=2)
                
                # Calculate statistics
                orig_mean = np.nanmean(original_lengths[pair])
                orig_std = np.nanstd(original_lengths[pair])
                smooth_mean = np.nanmean(smoothed_lengths[pair])
                smooth_std = np.nanstd(smoothed_lengths[pair])
                
                # Add horizontal lines for mean values
                axes[subplot_idx].axhline(y=orig_mean, color='r', linestyle='--', alpha=0.5,
                                        label=f'Orig Mean: {orig_mean:.1f}±{orig_std:.1f}')
                axes[subplot_idx].axhline(y=smooth_mean, color='b', linestyle='--', alpha=0.5,
                                        label=f'Smooth Mean: {smooth_mean:.1f}±{smooth_std:.1f}')
                
                # Add labels
                axes[subplot_idx].set_ylabel('Length (mm)')
                axes[subplot_idx].set_title(f'{limb_name} Length')
                axes[subplot_idx].legend()
                axes[subplot_idx].grid(True)
        
        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        
        # Save figure if output directory specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f'limb_lengths_{fig_idx+1}.png'), dpi=300)
            print(f"Saved limb length plot to {os.path.join(output_dir, f'limb_lengths_{fig_idx+1}.png')}")
        else:
            plt.show()
        
        plt.close()

def analyze_jitter(original_data, smoothed_data, output_dir=None):
    """Analyze jitter reduction by calculating joint velocity"""
    # Select joints to analyze
    joints = ['left_hip', 'right_hip', 'left_knee', 'right_knee']
    
    # Calculate velocities
    timestamps = original_data['timestamps']
    fps = 1.0 / (timestamps[1] - timestamps[0]) if len(timestamps) > 1 else 30.0
    
    results = {}
    
    for joint in joints:
        # Extract positions
        orig_positions = []
        smooth_positions = []
        
        for orig_pose, smooth_pose in zip(original_data['poses'], smoothed_data['poses']):
            if joint in orig_pose:
                orig_positions.append(orig_pose[joint])
            else:
                orig_positions.append(np.array([np.nan, np.nan, np.nan]))
                
            if joint in smooth_pose:
                smooth_positions.append(smooth_pose[joint])
            else:
                smooth_positions.append(np.array([np.nan, np.nan, np.nan]))
        
        # Convert to numpy arrays
        orig_positions = np.array(orig_positions)
        smooth_positions = np.array(smooth_positions)
        
        # Calculate velocities (position differences)
        orig_velocities = np.zeros_like(orig_positions)
        smooth_velocities = np.zeros_like(smooth_positions)
        
        orig_velocities[1:] = orig_positions[1:] - orig_positions[:-1]
        smooth_velocities[1:] = smooth_positions[1:] - smooth_positions[:-1]
        
        # Calculate speed (magnitude of velocity)
        orig_speeds = np.sqrt(np.sum(orig_velocities**2, axis=1))
        smooth_speeds = np.sqrt(np.sum(smooth_velocities**2, axis=1))
        
        # Calculate jitter (sum of acceleration magnitude)
        orig_accels = np.zeros_like(orig_speeds)
        smooth_accels = np.zeros_like(smooth_speeds)
        
        orig_accels[1:] = np.abs(orig_speeds[1:] - orig_speeds[:-1])
        smooth_accels[1:] = np.abs(smooth_speeds[1:] - smooth_speeds[:-1])
        
        # Calculate jitter metrics
        orig_jitter = np.nanmean(orig_accels)
        smooth_jitter = np.nanmean(smooth_accels)
        
        # Calculate jitter reduction percentage
        if orig_jitter > 0:
            jitter_reduction = 100 * (orig_jitter - smooth_jitter) / orig_jitter
        else:
            jitter_reduction = 0
        
        # Store results
        results[joint] = {
            'original_jitter': orig_jitter,
            'smoothed_jitter': smooth_jitter,
            'reduction_percent': jitter_reduction
        }
        
        # Plot speeds
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, orig_speeds, 'r-', label='Original', alpha=0.6)
        plt.plot(timestamps, smooth_speeds, 'b-', label='Smoothed', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Speed (mm/frame)')
        plt.title(f'{joint.replace("_", " ").title()} - Speed')
        plt.legend()
        plt.grid(True)
        
        # Add jitter reduction info
        plt.figtext(0.5, 0.01, 
                   f"Jitter: Original={orig_jitter:.2f}, Smoothed={smooth_jitter:.2f}, "
                   f"Reduction: {jitter_reduction:.1f}%",
                   ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Save figure if output directory specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f'{joint}_jitter.png'), dpi=300)
            print(f"Saved jitter analysis plot to {os.path.join(output_dir, f'{joint}_jitter.png')}")
        else:
            plt.show()
        
        plt.close()
    
    # Create summary table
    print("\nJitter Reduction Summary:")
    print("=" * 70)
    print(f"{'Joint':<15} {'Original Jitter':>15} {'Smoothed Jitter':>15} {'Reduction %':>15}")
    print("-" * 70)
    
    for joint, data in results.items():
        print(f"{joint.replace('_', ' ').title():<15} "
              f"{data['original_jitter']:>15.2f} "
              f"{data['smoothed_jitter']:>15.2f} "
              f"{data['reduction_percent']:>15.1f}")
    print("=" * 70)
    
    # Save summary to file if output directory specified
    if output_dir:
        with open(os.path.join(output_dir, 'jitter_summary.txt'), 'w') as f:
            f.write("Jitter Reduction Summary:\n")
            f.write("=" * 70 + "\n")
            f.write(f"{'Joint':<15} {'Original Jitter':>15} {'Smoothed Jitter':>15} {'Reduction %':>15}\n")
            f.write("-" * 70 + "\n")
            
            for joint, data in results.items():
                f.write(f"{joint.replace('_', ' ').title():<15} "
                      f"{data['original_jitter']:>15.2f} "
                      f"{data['smoothed_jitter']:>15.2f} "
                      f"{data['reduction_percent']:>15.1f}\n")
            f.write("=" * 70 + "\n")
        
        print(f"Saved jitter summary to {os.path.join(output_dir, 'jitter_summary.txt')}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Test motion smoothing on 3D pose data')
    parser.add_argument('--data_file', required=True, help='Path to pose data pickle file')
    parser.add_argument('--output_dir', default=None, help='Directory to save output visualizations')
    parser.add_argument('--camera_type', choices=['smalliphone', 'iphone', 'edger'], 
                      default='smalliphone', help='Camera type preset')
    parser.add_argument('--window_size', type=int, default=None, help='Override window size')
    parser.add_argument('--poly_order', type=int, default=None, help='Override polynomial order')
    
    args = parser.parse_args()
    
    # Load original pose data
    print(f"Loading pose data from {args.data_file}")
    original_data = load_pose_data(args.data_file)
    
    # Create configuration
    config = {}
    if args.window_size is not None:
        config['window_size'] = args.window_size
    if args.poly_order is not None:
        config['poly_order'] = args.poly_order
    
    # Initialize motion smoother
    print(f"Initializing motion smoother with {args.camera_type} preset")
    smoother = MotionSmoother(preset=args.camera_type, **config)
    
    # Apply smoothing
    print("Applying motion smoothing...")
    smoothed_poses = smoother.smooth_pose_sequence(original_data['poses'])
    
    # Get smoothing stats
    smoothing_stats = smoother.get_stats()
    print(f"Smoothing complete: {smoothing_stats['frames_processed']} frames processed")
    print(f"  Anatomical corrections: {smoothing_stats['anatomical_corrections']}")
    print(f"  Velocity corrections: {smoothing_stats['velocity_corrections']}")
    print(f"  Processing time: {smoothing_stats['processing_time']:.2f} seconds")
    
    # Create smoothed data structure
    smoothed_data = {
        'poses': smoothed_poses,
        'timestamps': original_data['timestamps'],
        'fps': original_data.get('fps', 30.0)
    }
    
    # Create output directory if needed
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save smoothed data
        smoothed_file = os.path.join(args.output_dir, 'smoothed_pose_data.pkl')
        with open(smoothed_file, 'wb') as f:
            pickle.dump(smoothed_data, f)
        print(f"Saved smoothed pose data to {smoothed_file}")
        
        # Save smoothing stats
        stats_file = os.path.join(args.output_dir, 'smoothing_stats.txt')
        with open(stats_file, 'w') as f:
            f.write(f"Motion Smoothing Statistics\n")
            f.write(f"=========================\n\n")
            f.write(f"Preset: {args.camera_type}\n")
            if args.window_size is not None:
                f.write(f"Window Size: {args.window_size} (override)\n")
            else:
                f.write(f"Window Size: {smoother.config['window_size']}\n")
            if args.poly_order is not None:
                f.write(f"Polynomial Order: {args.poly_order} (override)\n")
            else:
                f.write(f"Polynomial Order: {smoother.config['poly_order']}\n")
            f.write(f"Frames Processed: {smoothing_stats['frames_processed']}\n")
            f.write(f"Anatomical Corrections: {smoothing_stats['anatomical_corrections']}\n")
            f.write(f"Velocity Corrections: {smoothing_stats['velocity_corrections']}\n")
            f.write(f"Processing Time: {smoothing_stats['processing_time']:.2f} seconds\n")
        print(f"Saved smoothing stats to {stats_file}")
    
    # Plot trajectory analysis for key joints
    key_joints = ['left_hip', 'right_hip', 'left_knee', 'right_knee']
    for joint in key_joints:
        plot_joint_trajectory(original_data, smoothed_data, joint, args.output_dir)
    
    # Compare limb lengths
    compare_limb_lengths(original_data, smoothed_data, args.output_dir)
    
    # Analyze jitter reduction
    analyze_jitter(original_data, smoothed_data, args.output_dir)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()