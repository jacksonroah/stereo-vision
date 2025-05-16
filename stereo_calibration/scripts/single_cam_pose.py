#!/usr/bin/env python3
import cv2
import mediapipe as mp
import numpy as np
import math
import argparse
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt

class PoseEstimator:
    def __init__(self, output_dir='pose_results'):
        """Initialize the pose estimation system"""
        # Create output directory
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup MediaPipe Pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
        
        # Dictionary to store joint angle data
        self.angle_data = {}
        
        # Initialize visualization settings
        self.viz_settings = {
            'show_video': True,
            'save_frames': True,
            'plot_angles': True,
            'save_interval': 30,  # Save every 30 frames
        }
    
    def calculate_angle(self, a, b, c):
        """
        Calculate angle between three points in degrees.
        Points should be in format [x, y] or [x, y, z].
        """
        if len(a) >= 3 and len(b) >= 3 and len(c) >= 3:
            # 3D calculation if z-coordinates are available
            a = np.array(a[:3])
            b = np.array(b[:3])
            c = np.array(c[:3])
        else:
            # 2D calculation
            a = np.array(a[:2])
            b = np.array(b[:2])
            c = np.array(c[:2])
        
        # Create vectors from points
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
    
    def process_video(self, video_path, confidence_threshold=0.7):
        """Process video for pose estimation and angle calculation"""
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {os.path.basename(video_path)}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
        
        # Create results directory
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        results_dir = os.path.join(self.output_dir, video_name)
        os.makedirs(os.path.join(results_dir, 'frames'), exist_ok=True)
        
        # Initialize pose detection
        with self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # 0, 1, or 2 (higher is more accurate but slower)
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
            
            # Initialize angle tracking dictionaries
            angles = {
                'right_shoulder': [],
                'left_shoulder': [],
                'right_elbow': [],
                'left_elbow': [],
                'right_hip': [],
                'left_hip': [],
                'right_knee': [],
                'left_knee': [],
            }
            
            frame_indices = []
            frame_count = 0
            processing_start = time.time()
            
            # Process video frames
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break
                
                # Convert to RGB for MediaPipe
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Process image with MediaPipe
                results = pose.process(image_rgb)
                
                # Skip if no pose detected
                if not results.pose_landmarks:
                    frame_count += 1
                    continue
                
                # Calculate angles if landmarks are detected with sufficient confidence
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Create visualization image
                    annotated_image = image.copy()
                    
                    # Draw the pose landmarks
                    self.mp_drawing.draw_landmarks(
                        annotated_image,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
                    
                    # Extract landmark points (convert to array for easier handling)
                    pose_landmarks = []
                    for landmark in landmarks:
                        pose_landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
                    
                    pose_landmarks = np.array(pose_landmarks)
                    
                    # Calculate and record angles if confidence is sufficient
                    if self._check_landmarks_confidence(pose_landmarks, confidence_threshold):
                        # Right shoulder angle (between right hip, shoulder, and elbow)
                        right_shoulder_angle = self.calculate_angle(
                            pose_landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value][:3],
                            pose_landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value][:3],
                            pose_landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value][:3]
                        )
                        angles['right_shoulder'].append(right_shoulder_angle)
                        
                        # Left shoulder angle
                        left_shoulder_angle = self.calculate_angle(
                            pose_landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value][:3],
                            pose_landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value][:3],
                            pose_landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value][:3]
                        )
                        angles['left_shoulder'].append(left_shoulder_angle)
                        
                        # Right elbow angle
                        right_elbow_angle = self.calculate_angle(
                            pose_landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value][:3],
                            pose_landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value][:3],
                            pose_landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value][:3]
                        )
                        angles['right_elbow'].append(right_elbow_angle)
                        
                        # Left elbow angle
                        left_elbow_angle = self.calculate_angle(
                            pose_landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value][:3],
                            pose_landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value][:3],
                            pose_landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value][:3]
                        )
                        angles['left_elbow'].append(left_elbow_angle)
                        
                        # Right hip angle
                        right_hip_angle = self.calculate_angle(
                            pose_landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value][:3],
                            pose_landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value][:3],
                            pose_landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value][:3]
                        )
                        angles['right_hip'].append(right_hip_angle)
                        
                        # Left hip angle
                        left_hip_angle = self.calculate_angle(
                            pose_landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value][:3],
                            pose_landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value][:3],
                            pose_landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value][:3]
                        )
                        angles['left_hip'].append(left_hip_angle)
                        
                        # Right knee angle
                        right_knee_angle = self.calculate_angle(
                            pose_landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value][:3],
                            pose_landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value][:3],
                            pose_landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value][:3]
                        )
                        angles['right_knee'].append(right_knee_angle)
                        
                        # Left knee angle
                        left_knee_angle = self.calculate_angle(
                            pose_landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value][:3],
                            pose_landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value][:3],
                            pose_landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value][:3]
                        )
                        angles['left_knee'].append(left_knee_angle)
                        
                        # Add frame index
                        frame_indices.append(frame_count)
                        
                        # Display angles on frame
                        for i, (joint, angle) in enumerate(zip(
                            ['R.Shoulder', 'L.Shoulder', 'R.Elbow', 'L.Elbow', 
                             'R.Hip', 'L.Hip', 'R.Knee', 'L.Knee'],
                            [right_shoulder_angle, left_shoulder_angle, 
                             right_elbow_angle, left_elbow_angle,
                             right_hip_angle, left_hip_angle,
                             right_knee_angle, left_knee_angle])):
                            cv2.putText(annotated_image, f"{joint}: {angle:.1f}°", 
                                       (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 
                                       0.8, (0, 255, 0), 2)
                    
                    # Save frame at intervals
                    if self.viz_settings['save_frames'] and frame_count % self.viz_settings['save_interval'] == 0:
                        frame_path = os.path.join(results_dir, 'frames', f"frame_{frame_count:04d}.jpg")
                        cv2.imwrite(frame_path, annotated_image)
                    
                    # Display frame
                    if self.viz_settings['show_video']:
                        cv2.imshow('Pose Estimation', annotated_image)
                        if cv2.waitKey(5) & 0xFF == 27:  # ESC key to exit
                            break
                
                frame_count += 1
                
                # Print progress
                if frame_count % 30 == 0:
                    elapsed = time.time() - processing_start
                    fps_processing = frame_count / elapsed if elapsed > 0 else 0
                    print(f"Processed {frame_count}/{total_frames} frames ({fps_processing:.1f} fps)")
            
            cap.release()
            cv2.destroyAllWindows()
            
            # Save angle data
            self.angle_data[video_name] = {
                'angles': angles,
                'frames': frame_indices
            }
            
            # Create analysis visualizations
            if self.viz_settings['plot_angles'] and frame_indices:
                self._create_angle_plots(angles, frame_indices, video_name, results_dir)
            
            # Generate statistics
            self._calculate_statistics(angles, results_dir, video_name)
            
            print(f"Completed processing {os.path.basename(video_path)}")
    
    def _check_landmarks_confidence(self, landmarks, threshold):
        """Check if key landmarks have sufficient confidence for angle calculation"""
        key_indices = [
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            self.mp_pose.PoseLandmark.RIGHT_ELBOW.value,
            self.mp_pose.PoseLandmark.LEFT_ELBOW.value,
            self.mp_pose.PoseLandmark.RIGHT_WRIST.value,
            self.mp_pose.PoseLandmark.LEFT_WRIST.value,
            self.mp_pose.PoseLandmark.RIGHT_HIP.value,
            self.mp_pose.PoseLandmark.LEFT_HIP.value,
            self.mp_pose.PoseLandmark.RIGHT_KNEE.value,
            self.mp_pose.PoseLandmark.LEFT_KNEE.value,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE.value,
            self.mp_pose.PoseLandmark.LEFT_ANKLE.value
        ]
        
        for idx in key_indices:
            if landmarks[idx][3] < threshold:  # visibility is in 4th position
                return False
        
        return True
    
    def _create_angle_plots(self, angles, frames, video_name, results_dir):
        """Create plots of joint angles over time"""
        # Convert frames to timestamps (seconds)
        timestamps = np.array(frames) / 30.0  # Assuming 30fps
        
        # Create figure for angle plots
        plt.figure(figsize=(12, 10))
        
        # Upper body angles
        plt.subplot(2, 2, 1)
        plt.plot(timestamps, angles['right_shoulder'], 'r-', label='Right Shoulder')
        plt.plot(timestamps, angles['left_shoulder'], 'b-', label='Left Shoulder')
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (degrees)')
        plt.title('Shoulder Angles')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(timestamps, angles['right_elbow'], 'r-', label='Right Elbow')
        plt.plot(timestamps, angles['left_elbow'], 'b-', label='Left Elbow')
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (degrees)')
        plt.title('Elbow Angles')
        plt.grid(True)
        plt.legend()
        
        # Lower body angles
        plt.subplot(2, 2, 3)
        plt.plot(timestamps, angles['right_hip'], 'r-', label='Right Hip')
        plt.plot(timestamps, angles['left_hip'], 'b-', label='Left Hip')
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (degrees)')
        plt.title('Hip Angles')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(2, 2, 4)
        plt.plot(timestamps, angles['right_knee'], 'r-', label='Right Knee')
        plt.plot(timestamps, angles['left_knee'], 'b-', label='Left Knee')
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (degrees)')
        plt.title('Knee Angles')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'angle_plots.png'), dpi=300)
        plt.close()
    
    def _calculate_statistics(self, angles, results_dir, video_name):
        """Calculate statistics for each joint angle"""
        stats = {}
        
        for joint, angle_list in angles.items():
            if angle_list:  # Check if we have data
                angle_array = np.array(angle_list)
                stats[joint] = {
                    'mean': np.mean(angle_array),
                    'median': np.median(angle_array),
                    'std': np.std(angle_array),
                    'min': np.min(angle_array),
                    'max': np.max(angle_array)
                }
        
        # Save statistics to file
        with open(os.path.join(results_dir, 'angle_statistics.txt'), 'w') as f:
            f.write(f"Angle Statistics for {video_name}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for joint, joint_stats in stats.items():
                f.write(f"{joint.replace('_', ' ').title()}:\n")
                f.write(f"  Mean: {joint_stats['mean']:.2f}°\n")
                f.write(f"  Median: {joint_stats['median']:.2f}°\n")
                f.write(f"  Standard Deviation: {joint_stats['std']:.2f}°\n")
                f.write(f"  Range: {joint_stats['min']:.2f}° - {joint_stats['max']:.2f}°\n")
                f.write("\n")
        
        # Also save as CSV for easier data analysis
        with open(os.path.join(results_dir, 'angle_statistics.csv'), 'w') as f:
            f.write("joint,mean,median,std,min,max\n")
            for joint, joint_stats in stats.items():
                f.write(f"{joint},{joint_stats['mean']:.2f},{joint_stats['median']:.2f}," +
                        f"{joint_stats['std']:.2f},{joint_stats['min']:.2f},{joint_stats['max']:.2f}\n")

def main():
    parser = argparse.ArgumentParser(description='Single Camera Pose Estimation for Static Poses')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, default='pose_results', help='Output directory for results')
    parser.add_argument('--confidence', type=float, default=0.7, help='Confidence threshold for landmarks')
    parser.add_argument('--no-display', action='store_true', help='Disable video display during processing')
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not os.path.exists(args.video):
        print(f"Error: Video file {args.video} not found")
        return
    
    # Initialize pose estimator
    pose_estimator = PoseEstimator(output_dir=args.output)
    
    # Set visualization options
    if args.no_display:
        pose_estimator.viz_settings['show_video'] = False
    
    # Process video
    pose_estimator.process_video(args.video, confidence_threshold=args.confidence)
    
    print(f"Pose estimation complete. Results saved to {args.output}")

if __name__ == "__main__":
    main()