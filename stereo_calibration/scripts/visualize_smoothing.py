#!/usr/bin/env python3
# visualize_smoothing.py - Create 3D visualization of original vs smoothed poses

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from motion_smoothing import MotionSmoother
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

def load_pose_data(data_file):
    """Load pose data from pickle file"""
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    return data

def plot_3d_skeleton(ax, pose, bones=None, style='b-', alpha=0.7):
    """Plot a 3D skeleton from a pose dictionary"""
    if bones is None:
        # Define standard skeleton connections
        bones = [
            ('nose', 'left_shoulder'),
            ('nose', 'right_shoulder'),
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            ('left_hip', 'left_knee'),
            ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle')
        ]
    
    # Plot bones
    for start_joint, end_joint in bones:
        if start_joint in pose and end_joint in pose:
            start_pos = pose[start_joint]
            end_pos = pose[end_joint]
            
            # Draw line
            ax.plot([start_pos[0], end_pos[0]],
                   [start_pos[1], end_pos[1]],
                   [start_pos[2], end_pos[2]],
                   style, alpha=alpha)
    
    # Plot joints
    for joint, position in pose.items():
        ax.scatter(position[0], position[1], position[2], 
                  c=style[0], s=20, alpha=alpha)

def create_comparison_video(original_data, smoothed_data, output_dir=None, fps=10, duration=10):
    """Create a video comparing original and smoothed 3D poses"""
    # Extract poses
    original_poses = original_data['poses']
    smoothed_poses = smoothed_data['poses']
    
    # Skip if not enough frames
    if len(original_poses) < 10:
        print("Not enough frames for video animation")
        return
    
    # Calculate how many frames to include in the animation
    total_frames = min(len(original_poses), len(smoothed_poses))
    # Determine frame sampling to match desired duration
    frame_count = int(fps * duration)
    
    if frame_count >= total_frames:
        # Use all frames if we don't have enough
        frame_indices = list(range(total_frames))
    else:
        # Sample frames evenly
        frame_indices = np.linspace(0, total_frames-1, frame_count, dtype=int)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 8))
    
    # Original skeleton plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('Original Pose')
    
    # Smoothed skeleton plot
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('Smoothed Pose')
    
    # Initialize animation
    lines1 = []
    lines2 = []
    
    # Initialize animation function
    def init():
        for ax in [ax1, ax2]:
            ax.clear()
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            # Set axis limits based on data
            ax.set_xlim([-500, 500])
            ax.set_ylim([-500, 500])
            ax.set_zlim([-500, 500])
            
            # Set title
            if ax == ax1:
                ax.set_title('Original Pose')
            else:
                ax.set_title('Smoothed Pose')
        
        return []
    
    # Animation function
    def update(frame_idx):
        frame = frame_indices[frame_idx % len(frame_indices)]
        
        # Clear axes
        ax1.clear()
        ax2.clear()
        
        # Update titles with frame number
        ax1.set_title(f'Original Pose (Frame {frame})')
        ax2.set_title(f'Smoothed Pose (Frame {frame})')
        
        # Set axis properties
        for ax in [ax1, ax2]:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            # Update axis limits based on the specific frame data
            if frame < len(original_poses):
                # Get bounding box from both poses
                points = []
                for pose in [original_poses[frame], smoothed_poses[frame]]:
                    for pos in pose.values():
                        points.append(pos)
                
                if points:
                    points = np.array(points)
                    min_vals = np.min(points, axis=0)
                    max_vals = np.max(points, axis=0)
                    
                    # Add margin
                    margin = 100  # mm
                    min_vals -= margin
                    max_vals += margin
                    
                    # Set limits
                    ax.set_xlim([min_vals[0], max_vals[0]])
                    ax.set_ylim([min_vals[1], max_vals[1]])
                    ax.set_zlim([min_vals[2], max_vals[2]])
                else:
                    # Default limits if no points
                    ax.set_xlim([-500, 500])
                    ax.set_ylim([-500, 500])
                    ax.set_zlim([-500, 500])
        
        # Draw skeletons
        if frame < len(original_poses):
            plot_3d_skeleton(ax1, original_poses[frame], style='r-', alpha=0.9)
            
        if frame < len(smoothed_poses):
            plot_3d_skeleton(ax2, smoothed_poses[frame], style='b-', alpha=0.9)
        
        # Add timestamp if available
        if 'timestamps' in original_data and frame < len(original_data['timestamps']):
            timestamp = original_data['timestamps'][frame]
            time_text = f'Time: {timestamp:.2f}s'
            fig.suptitle(time_text, fontsize=14)
        
        return []
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=len(frame_indices),
                       init_func=init, blit=False, interval=1000/fps)
    
    # Save animation if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'pose_comparison.mp4')
        
        # Use ffmpeg writer
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='Motion Smoothing'), bitrate=1800)
        
        ani.save(output_file, writer=writer)
        print(f"Saved animation to {output_file}")
    else:
        plt.show()
    
    plt.close()

def create_side_by_side_comparison(original_data, smoothed_data, output_dir=None, num_frames=5):
    """Create side-by-side comparison of key frames"""
    # Extract poses
    original_poses = original_data['poses']
    smoothed_poses = smoothed_data['poses']
    
    # Calculate how many frames to include
    total_frames = min(len(original_poses), len(smoothed_poses))
    
    # Sample frames evenly
    if total_frames <= num_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    
    # Create a figure for each comparison
    for i, frame_idx in enumerate(frame_indices):
        fig = plt.figure(figsize=(16, 8))
        
        # Original skeleton plot
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.set_title(f'Original Pose (Frame {frame_idx})')
        
        # Smoothed skeleton plot
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.set_title(f'Smoothed Pose (Frame {frame_idx})')
        
        # Set axis properties
        for ax in [ax1, ax2]:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        
        # Calculate shared axis limits
        points = []
        for pose in [original_poses[frame_idx], smoothed_poses[frame_idx]]:
            for pos in pose.values():
                points.append(pos)
        
        if points:
            points = np.array(points)
            min_vals = np.min(points, axis=0)
            max_vals = np.max(points, axis=0)
            
            # Add margin
            margin = 100  # mm
            min_vals -= margin
            max_vals += margin
            
            # Set limits for both axes
            for ax in [ax1, ax2]:
                ax.set_xlim([min_vals[0], max_vals[0]])
                ax.set_ylim([min_vals[1], max_vals[1]])
                ax.set_zlim([min_vals[2], max_vals[2]])
        
        # Draw skeletons
        plot_3d_skeleton(ax1, original_poses[frame_idx], style='r-', alpha=0.9)
        plot_3d_skeleton(ax2, smoothed_poses[frame_idx], style='b-', alpha=0.9)
        
        # Add timestamp if available
        if 'timestamps' in original_data and frame_idx < len(original_data['timestamps']):
            timestamp = original_data['timestamps'][frame_idx]
            time_text = f'Time: {timestamp:.2f}s'
            fig.suptitle(time_text, fontsize=14)
        
        # Save figure if output directory specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f'pose_comparison_frame_{frame_idx:04d}.png')
            plt.savefig(output_file, dpi=300)
            print(f"Saved comparison for frame {frame_idx} to {output_file}")
        else:
            plt.show()
        
        plt.close()

def create_trajectory_visualization(original_data, smoothed_data, joint_names, output_dir=None):
    """Create 3D trajectory visualization for specific joints"""
    # Extract poses
    original_poses = original_data['poses']
    smoothed_poses = smoothed_data['poses']
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectories for each joint
    for joint in joint_names:
        # Extract positions
        orig_positions = []
        smooth_positions = []
        
        for i in range(min(len(original_poses), len(smoothed_poses))):
            if joint in original_poses[i] and joint in smoothed_poses[i]:
                orig_positions.append(original_poses[i][joint])
                smooth_positions.append(smoothed_poses[i][joint])
        
        if orig_positions and smooth_positions:
            # Convert to numpy arrays
            orig_positions = np.array(orig_positions)
            smooth_positions = np.array(smooth_positions)
            
            # Plot trajectories
            ax.plot(orig_positions[:, 0], orig_positions[:, 1], orig_positions[:, 2],
                   'r-', alpha=0.6, label=f'Original {joint}' if joint == joint_names[0] else '')
            ax.plot(smooth_positions[:, 0], smooth_positions[:, 1], smooth_positions[:, 2],
                  'b-', linewidth=2, alpha=0.9, label=f'Smoothed {joint}' if joint == joint_names[0] else '')
            
            # Plot start and end points
            ax.scatter(orig_positions[0, 0], orig_positions[0, 1], orig_positions[0, 2],
                      c='r', s=40, marker='o')
            ax.scatter(smooth_positions[0, 0], smooth_positions[0, 1], smooth_positions[0, 2],
                     c='b', s=40, marker='o')
    
    # Set labels
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('3D Joint Trajectories Comparison')
    
    # Add legend
    ax.legend()
    
    # Save figure if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        joints_str = '_'.join([j.split('_')[-1] for j in joint_names])
        output_file = os.path.join(output_dir, f'trajectory_comparison_{joints_str}.png')
        plt.savefig(output_file, dpi=300)
        print(f"Saved trajectory comparison to {output_file}")
    else:
        plt.show()
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize motion smoothing on 3D pose data')
    parser.add_argument('--data_file', required=True, help='Path to pose data pickle file')
    parser.add_argument('--output_dir', default=None, help='Directory to save output visualizations')
    parser.add_argument('--camera_type', choices=['smalliphone', 'iphone', 'edger'], 
                      default='smalliphone', help='Camera type preset')
    parser.add_argument('--window_size', type=int, default=None, help='Override window size')
    parser.add_argument('--poly_order', type=int, default=None, help='Override polynomial order')
    parser.add_argument('--comparison_frames', type=int, default=5, 
                       help='Number of frames for side-by-side comparison')
    parser.add_argument('--create_video', action='store_true', help='Create comparison video')
    parser.add_argument('--video_fps', type=int, default=10, help='FPS for video output')
    parser.add_argument('--video_duration', type=int, default=10, help='Duration for video in seconds')
    
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
    
    # Create visualization output
    print("Creating visualizations...")
    
    # Side-by-side comparison of key frames
    create_side_by_side_comparison(original_data, smoothed_data, args.output_dir, args.comparison_frames)
    
    # Create trajectory visualizations for key joints
    key_joints = [
        ['left_wrist', 'right_wrist'],
        ['left_ankle', 'right_ankle'],
        ['left_knee', 'right_knee']
    ]
    
    for joints in key_joints:
        create_trajectory_visualization(original_data, smoothed_data, joints, args.output_dir)
    
    # Create comparison video if requested
    if args.create_video:
        print(f"Creating comparison video (fps={args.video_fps}, duration={args.video_duration}s)...")
        create_comparison_video(original_data, smoothed_data, args.output_dir, 
                               args.video_fps, args.video_duration)
    
    print("Visualization complete!")

if __name__ == "__main__":
    main()