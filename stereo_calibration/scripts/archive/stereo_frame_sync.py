#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np
import json
import pickle
import argparse
from pathlib import Path

class StereoFrameSync:
    """
    Utility class for working with synchronized stereo frames after running the
    sync_test.py script. This handles frame index mapping between left and right cameras.
    """
    
    def __init__(self, sync_info_path):
        """
        Initialize with a path to either the sync_info.json or sync_data.pkl file.
        
        Args:
            sync_info_path: Path to the synchronization info file
        """
        self.sync_info = self._load_sync_info(sync_info_path)
        if not self.sync_info:
            raise ValueError(f"Could not load synchronization information from {sync_info_path}")
        
        # Extract key synchronization parameters
        self.method = self.sync_info.get('method', 'unknown')
        self.left_fps = self.sync_info.get('left_fps', 30.0)
        self.right_fps = self.sync_info.get('right_fps', 30.0)
        
        # Different sync methods store the offset in different ways
        if self.method == 'flash_detection':
            self.frame_offset = self.sync_info.get('frame_offset', 0)
        elif self.method == 'timestamp_matching':
            self.frame_offset = self.sync_info.get('common_frame_offset', 0)
        elif self.method == 'creation_time':
            self.frame_offset = self.sync_info.get('frame_offset', 0)
        elif self.method == 'simple_alignment':
            self.frame_offset = 0
            self.fps_ratio = self.sync_info.get('fps_ratio', 1.0)
        else:
            self.frame_offset = self.sync_info.get('frame_offset', 0)
        
        # Get video paths if available
        self.left_video = self.sync_info.get('left_video', None)
        self.right_video = self.sync_info.get('right_video', None)
        
        # Store a small sample of matched pairs for verification
        self.sample_pairs = self.sync_info.get('matched_pairs', [])[:5]
        
        # Print synchronization info
        print(f"Loaded {self.method} synchronization")
        print(f"Frame offset (right - left): {self.frame_offset}")
        if hasattr(self, 'fps_ratio') and abs(self.fps_ratio - 1.0) > 0.01:
            print(f"Frame rate ratio (right/left): {self.fps_ratio:.4f}")
        print(f"Sample paired frames (left, right): {self.sample_pairs[:3]}")
    
    def _load_sync_info(self, sync_info_path):
        """Load synchronization information from JSON or pickle file."""
        path = Path(sync_info_path)
        if not path.exists():
            print(f"Error: Sync info file {path} does not exist")
            return None
        
        if path.suffix.lower() == '.json':
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading JSON: {e}")
                return None
        elif path.suffix.lower() == '.pkl':
            try:
                with open(path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading pickle: {e}")
                return None
        else:
            # Try to infer the correct file
            json_path = path.parent / 'sync_info.json'
            pkl_path = path.parent / 'sync_data.pkl'
            
            if json_path.exists():
                return self._load_sync_info(json_path)
            elif pkl_path.exists():
                return self._load_sync_info(pkl_path)
            else:
                print(f"Could not find synchronization files in {path.parent}")
                return None
    
    def left_to_right(self, left_frame_idx):
        """
        Convert a left camera frame index to the corresponding right camera frame index.
        
        Args:
            left_frame_idx: Frame index from the left camera
            
        Returns:
            int: Corresponding frame index for the right camera
        """
        if self.method == 'simple_alignment' and hasattr(self, 'fps_ratio') and abs(self.fps_ratio - 1.0) > 0.01:
            # Handle different frame rates
            return int(left_frame_idx * self.fps_ratio)
        else:
            # Simple offset
            return left_frame_idx + self.frame_offset
    
    def right_to_left(self, right_frame_idx):
        """
        Convert a right camera frame index to the corresponding left camera frame index.
        
        Args:
            right_frame_idx: Frame index from the right camera
            
        Returns:
            int: Corresponding frame index for the left camera
        """
        if self.method == 'simple_alignment' and hasattr(self, 'fps_ratio') and abs(self.fps_ratio - 1.0) > 0.01:
            # Handle different frame rates
            return int(right_frame_idx / self.fps_ratio)
        else:
            # Simple offset
            return right_frame_idx - self.frame_offset
    
    def get_frame_pair(self, left_frame_idx, left_video_path=None, right_video_path=None):
        """
        Extract a synchronized pair of frames from the videos.
        
        Args:
            left_frame_idx: Frame index from the left camera
            left_video_path: Path to left video (optional if set during initialization)
            right_video_path: Path to right video (optional if set during initialization)
            
        Returns:
            tuple: (left_frame, right_frame) - the synchronized frames
        """
        # Determine video paths
        left_path = left_video_path or self.left_video
        right_path = right_video_path or self.right_video
        
        if not left_path or not os.path.exists(left_path):
            raise ValueError(f"Left video path not provided or does not exist: {left_path}")
        if not right_path or not os.path.exists(right_path):
            raise ValueError(f"Right video path not provided or does not exist: {right_path}")
        
        # Get corresponding right frame index
        right_frame_idx = self.left_to_right(left_frame_idx)
        
        # Check if indices are valid
        right_cap = cv2.VideoCapture(right_path)
        right_frame_count = int(right_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        right_cap.release()
        
        if right_frame_idx < 0 or right_frame_idx >= right_frame_count:
            raise ValueError(f"Invalid right frame index: {right_frame_idx} (valid range: 0-{right_frame_count-1})")
        
        # Extract frames
        left_frame = self._extract_frame(left_path, left_frame_idx)
        right_frame = self._extract_frame(right_path, right_frame_idx)
        
        return left_frame, right_frame
    
    def _extract_frame(self, video_path, frame_idx):
        """Extract a specific frame from a video file."""
        if frame_idx < 0:
            raise ValueError(f"Invalid frame index: {frame_idx}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")
        
        # Check if frame index is valid
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_idx >= total_frames:
            cap.release()
            raise ValueError(f"Frame index {frame_idx} out of range (max: {total_frames-1})")
        
        # Set position to the requested frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise IOError(f"Could not read frame {frame_idx} from {video_path}")
        
        return frame
    
    def extract_frame_sequence(self, start_left_idx, num_frames, step=1, 
                              left_video_path=None, right_video_path=None):
        """
        Extract a sequence of synchronized frame pairs.
        
        Args:
            start_left_idx: Starting frame index from left camera
            num_frames: Number of frame pairs to extract
            step: Frame step size (default: 1)
            left_video_path: Path to left video (optional if set during initialization)
            right_video_path: Path to right video (optional if set during initialization)
            
        Returns:
            tuple: (left_frames, right_frames) - lists of synchronized frames
        """
        # Determine video paths
        left_path = left_video_path or self.left_video
        right_path = right_video_path or self.right_video
        
        if not left_path or not os.path.exists(left_path):
            raise ValueError(f"Left video path not provided or does not exist: {left_path}")
        if not right_path or not os.path.exists(right_path):
            raise ValueError(f"Right video path not provided or does not exist: {right_path}")
        
        left_frames = []
        right_frames = []
        
        # Open video captures
        left_cap = cv2.VideoCapture(left_path)
        right_cap = cv2.VideoCapture(right_path)
        
        if not left_cap.isOpened() or not right_cap.isOpened():
            raise IOError(f"Could not open one or both videos")
        
        try:
            # Extract each frame pair
            for i in range(num_frames):
                left_idx = start_left_idx + i * step
                right_idx = self.left_to_right(left_idx)
                
                # Extract left frame
                left_cap.set(cv2.CAP_PROP_POS_FRAMES, left_idx)
                ret_left, left_frame = left_cap.read()
                if not ret_left:
                    print(f"Warning: Could not read left frame {left_idx}")
                    break
                
                # Extract right frame
                right_cap.set(cv2.CAP_PROP_POS_FRAMES, right_idx)
                ret_right, right_frame = right_cap.read()
                if not ret_right:
                    print(f"Warning: Could not read right frame {right_idx}")
                    break
                
                left_frames.append(left_frame)
                right_frames.append(right_frame)
        finally:
            # Ensure captures are released
            left_cap.release()
            right_cap.release()
        
        return left_frames, right_frames
    
    def verify_sync(self, output_dir=None, num_samples=5, left_video_path=None, right_video_path=None):
        """
        Verify synchronization by visualizing matched frame pairs across the video.
        
        Args:
            output_dir: Directory to save visualization images (default: current directory)
            num_samples: Number of sample points to check
            left_video_path: Path to left video (optional if set during initialization)
            right_video_path: Path to right video (optional if set during initialization)
            
        Returns:
            bool: True if verification was successful
        """
        # Determine video paths
        left_path = left_video_path or self.left_video
        right_path = right_video_path or self.right_video
        
        if not left_path or not os.path.exists(left_path):
            raise ValueError(f"Left video path not provided or does not exist: {left_path}")
        if not right_path or not os.path.exists(right_path):
            raise ValueError(f"Right video path not provided or does not exist: {right_path}")
        
        # Set output directory
        if output_dir is None:
            output_dir = os.getcwd()
        os.makedirs(output_dir, exist_ok=True)
        
        # Get video information
        left_cap = cv2.VideoCapture(left_path)
        right_cap = cv2.VideoCapture(right_path)
        left_frame_count = int(left_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        right_frame_count = int(right_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        left_cap.release()
        right_cap.release()
        
        print(f"Left video: {left_frame_count} frames")
        print(f"Right video: {right_frame_count} frames")
        
        # Determine the valid frame range based on the offset
        if self.frame_offset >= 0:
            # Right frames start after left frames
            # Valid left range: 0 to (left_max - 1)
            # Valid right range: offset to (right_max - 1)
            min_left_idx = 0
            max_left_idx = min(left_frame_count - 1, right_frame_count - 1 - self.frame_offset)
        else:
            # Left frames start after right frames
            # Valid left range: |offset| to (left_max - 1)
            # Valid right range: 0 to (right_max - 1)
            min_left_idx = abs(self.frame_offset)
            max_left_idx = left_frame_count - 1
        
        print(f"Valid left frame range: {min_left_idx} to {max_left_idx}")
        
        # Choose sample points spread throughout the valid range
        left_indices = []
        valid_range = max_left_idx - min_left_idx
        if valid_range <= 0:
            print("Error: No valid frame range for verification")
            return False
            
        for i in range(num_samples):
            if num_samples > 1:
                # Spread evenly through the valid range
                frame_position = min_left_idx + int((i / (num_samples - 1)) * valid_range)
            else:
                # Just use the middle point if only one sample
                frame_position = min_left_idx + valid_range // 2
            left_indices.append(frame_position)
        
        print(f"Sampling left frames at indices: {left_indices}")
        
        # Extract and visualize frame pairs
        for i, left_idx in enumerate(left_indices):
            try:
                right_idx = self.left_to_right(left_idx)
                
                # Double-check indices are valid
                if right_idx < 0 or right_idx >= right_frame_count:
                    print(f"Warning: Right frame index {right_idx} out of range (0-{right_frame_count-1})")
                    continue
                    
                # Extract frames
                left_frame = self._extract_frame(left_path, left_idx)
                right_frame = self._extract_frame(right_path, right_idx)
                
                # Create side-by-side visualization
                h1, w1 = left_frame.shape[:2]
                h2, w2 = right_frame.shape[:2]
                
                # Resize to match height if needed
                if h1 != h2:
                    scale = h1 / h2
                    width = int(w2 * scale)
                    right_frame = cv2.resize(right_frame, (width, h1))
                    w2 = width
                
                combined = np.zeros((h1, w1 + w2, 3), dtype=np.uint8)
                combined[:, :w1] = left_frame
                combined[:, w1:] = right_frame
                
                # Add frame numbers as text
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(combined, f"Left: {left_idx}", (10, 30), font, 1, (0, 255, 0), 2)
                cv2.putText(combined, f"Right: {right_idx}", (w1 + 10, 30), font, 1, (0, 255, 0), 2)
                
                # Add position indicator
                pos_percent = ((left_idx - min_left_idx) / valid_range * 100) if valid_range > 0 else 0
                cv2.putText(combined, f"Position: {pos_percent:.1f}%", (10, h1 - 10), font, 0.7, (255, 255, 255), 2)
                
                # Save the visualization
                output_path = os.path.join(output_dir, f"sync_verification_{i:02d}.png")
                cv2.imwrite(output_path, combined)
                print(f"Saved verification image to {output_path}")
                
            except Exception as e:
                print(f"Error verifying frame pair at left index {left_idx}: {e}")
                continue
        
        print(f"Successfully verified synchronized frame pairs")
        return True

def main():
    parser = argparse.ArgumentParser(description='Stereo Frame Synchronization Utility')
    parser.add_argument('--sync_info', required=True, 
                        help='Path to sync_info.json or sync_data.pkl file')
    parser.add_argument('--left_video', default=None,
                        help='Path to left camera video (optional if specified in sync info)')
    parser.add_argument('--right_video', default=None,
                        help='Path to right camera video (optional if specified in sync info)')
    parser.add_argument('--output_dir', default='sync_verification',
                        help='Directory to save verification images')
    parser.add_argument('--verify', action='store_true',
                        help='Verify synchronization by visualizing frame pairs')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of sample points to check for verification')
    
    args = parser.parse_args()
    
    try:
        # Initialize synchronization
        sync = StereoFrameSync(args.sync_info)
        
        # Verify if requested
        if args.verify:
            sync.verify_sync(
                output_dir=args.output_dir,
                num_samples=args.num_samples,
                left_video_path=args.left_video,
                right_video_path=args.right_video
            )
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())