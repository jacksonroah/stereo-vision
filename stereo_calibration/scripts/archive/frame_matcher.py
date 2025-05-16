#!/usr/bin/env python3
import cv2
import os
import glob
import argparse
import numpy as np
from tqdm import tqdm
import time

def find_videos(camera_dir, pattern="*.mp4"):
    """
    Find all videos in a camera directory matching a pattern.
    
    Args:
        camera_dir (str): Path to camera directory
        pattern (str): Glob pattern to match video files
        
    Returns:
        list: List of video file paths
    """
    video_paths = []
    
    # Try common video extensions if specific pattern not provided
    if pattern == "*.mp4":
        extensions = ['.mp4', '.mov', '.avi', '.MP4', '.MOV', '.AVI']
        for ext in extensions:
            video_paths.extend(glob.glob(os.path.join(camera_dir, f"*{ext}")))
    else:
        video_paths = glob.glob(os.path.join(camera_dir, pattern))
    
    # Sort by filename for consistent ordering
    video_paths.sort()
    
    return video_paths

def find_matching_videos(left_dir, right_dir, pattern=None):
    """
    Find matching video pairs between left and right camera directories.
    Uses filename similarity to match videos.
    
    Args:
        left_dir (str): Path to left camera directory
        right_dir (str): Path to right camera directory
        pattern (str, optional): Glob pattern to filter videos
        
    Returns:
        list: List of tuples (left_video_path, right_video_path)
    """
    # Find all videos in both directories
    left_videos = find_videos(left_dir, pattern if pattern else "*.mp4")
    right_videos = find_videos(right_dir, pattern if pattern else "*.mp4")
    
    if not left_videos or not right_videos:
        print(f"Warning: No videos found in one or both directories")
        print(f"Left directory ({left_dir}): {len(left_videos)} videos")
        print(f"Right directory ({right_dir}): {len(right_videos)} videos")
        return []
    
    print(f"Found {len(left_videos)} videos in left camera directory")
    print(f"Found {len(right_videos)} videos in right camera directory")
    
    # Match videos based on filename similarity
    matched_pairs = []
    
    # If both directories have the same number of videos, assume they match in order
    if len(left_videos) == len(right_videos):
        print("Same number of videos in both directories, matching by position")
        return list(zip(left_videos, right_videos))
    
    # Try to match by filename similarity
    print("Different number of videos, matching by filename similarity")
    
    # Extract basenames without extensions
    left_names = [os.path.splitext(os.path.basename(v))[0] for v in left_videos]
    right_names = [os.path.splitext(os.path.basename(v))[0] for v in right_videos]
    
    # Try different matching strategies
    for strategy in ['exact', 'common_prefix', 'numbers', 'longest_common']:
        print(f"Trying matching strategy: {strategy}")
        pairs = []
        
        if strategy == 'exact':
            # Match by exact filename (without extension)
            for i, left_name in enumerate(left_names):
                if left_name in right_names:
                    j = right_names.index(left_name)
                    pairs.append((left_videos[i], right_videos[j]))
        
        elif strategy == 'common_prefix':
            # Match by common prefix (e.g., "validation_001")
            for i, left_name in enumerate(left_names):
                for j, right_name in enumerate(right_names):
                    # Try to find the longest common prefix
                    common_length = 0
                    for k in range(min(len(left_name), len(right_name))):
                        if left_name[k] == right_name[k]:
                            common_length += 1
                        else:
                            break
                    
                    # If common prefix is at least 5 characters
                    if common_length >= 5:
                        pairs.append((left_videos[i], right_videos[j]))
                        break
        
        elif strategy == 'numbers':
            # Match by numbers in filenames
            import re
            
            left_numbers = {}
            right_numbers = {}
            
            # Extract numbers from filenames
            for i, name in enumerate(left_names):
                nums = re.findall(r'\d+', name)
                if nums:
                    # Use the last number in the filename
                    left_numbers[nums[-1]] = i
            
            for i, name in enumerate(right_names):
                nums = re.findall(r'\d+', name)
                if nums:
                    # Use the last number in the filename
                    right_numbers[nums[-1]] = i
            
            # Match by shared numbers
            for num in left_numbers:
                if num in right_numbers:
                    pairs.append((left_videos[left_numbers[num]], right_videos[right_numbers[num]]))
        
        elif strategy == 'longest_common':
            # Match by longest common substring
            from difflib import SequenceMatcher
            
            for i, left_name in enumerate(left_names):
                best_match = None
                best_score = 0
                
                for j, right_name in enumerate(right_names):
                    # Calculate similarity score
                    matcher = SequenceMatcher(None, left_name, right_name)
                    score = matcher.ratio()
                    
                    if score > best_score and score > 0.6:  # Minimum similarity threshold
                        best_score = score
                        best_match = j
                
                if best_match is not None:
                    pairs.append((left_videos[i], right_videos[best_match]))
        
        # If we found matching pairs, return them
        if pairs:
            print(f"Found {len(pairs)} matching pairs using {strategy} strategy")
            return pairs
    
    # If no matches found, return empty list
    print("No matching pairs found through any strategy")
    return []

def compute_frame_similarity(left_frame, right_frame):
    """
    Compute similarity between two frames for synchronization.
    
    Args:
        left_frame (ndarray): Frame from left camera
        right_frame (ndarray): Frame from right camera
        
    Returns:
        float: Similarity score (higher means more similar)
    """
    # Convert to grayscale
    left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
    
    # Resize to common size for comparison if needed
    if left_gray.shape != right_gray.shape:
        height = min(left_gray.shape[0], right_gray.shape[0])
        width = min(left_gray.shape[1], right_gray.shape[1])
        left_gray = cv2.resize(left_gray, (width, height))
        right_gray = cv2.resize(right_gray, (width, height))
    
    # Calculate similarity using structural similarity index (SSIM)
    try:
        # OpenCV 4.x has SSIM built-in, if available use it
        if hasattr(cv2, 'PSNR'):
            ssim = cv2.PSNR(left_gray, right_gray)
            return ssim
        else:
            # Otherwise use Mean Squared Error (MSE)
            err = np.sum((left_gray.astype("float") - right_gray.astype("float")) ** 2)
            err /= float(left_gray.shape[0] * left_gray.shape[1])
            # Convert to a similarity score (higher is better)
            return 100.0 / (1.0 + err)
    except Exception as e:
        print(f"Error computing frame similarity: {e}")
        # Fallback to a simple absolute difference
        diff = cv2.absdiff(left_gray, right_gray)
        score = 100.0 - (np.mean(diff) * 100.0 / 255.0)
        return score

def find_matching_frames(left_video, right_video, output_dir, max_frames=10, search_window=30):
    """
    Extract synchronous frame pairs from left and right videos.
    Uses a sliding window approach to find the most similar frames.
    
    Args:
        left_video (str): Path to left camera video
        right_video (str): Path to right camera video
        output_dir (str): Directory to save extracted frames
        max_frames (int): Maximum number of frame pairs to extract
        search_window (int): Search window size for synchronization
        
    Returns:
        list: List of tuples (left_frame_path, right_frame_path)
    """
    print(f"\nMatching frames between:")
    print(f"  Left:  {os.path.basename(left_video)}")
    print(f"  Right: {os.path.basename(right_video)}")
    
    # Create output directories
    left_dir = os.path.join(output_dir, "left")
    right_dir = os.path.join(output_dir, "right")
    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)
    
    # Open video captures
    left_cap = cv2.VideoCapture(left_video)
    right_cap = cv2.VideoCapture(right_video)
    
    if not left_cap.isOpened() or not right_cap.isOpened():
        print("Error: Could not open one or both videos")
        return []
    
    # Get video info
    left_frames = int(left_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    right_frames = int(right_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    left_fps = left_cap.get(cv2.CAP_PROP_FPS)
    right_fps = right_cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Left video:  {left_frames} frames @ {left_fps:.2f} FPS")
    print(f"Right video: {right_frames} frames @ {right_fps:.2f} FPS")
    
    # Calculate frame intervals to target the desired number of frame pairs
    left_interval = max(1, left_frames // (max_frames * 3))
    right_interval = max(1, right_frames // (max_frames * 3))
    
    # Calculate FPS ratio to adjust synchronization
    fps_ratio = left_fps / right_fps
    
    # Skip first second (30 frames) to avoid initialization frames
    skip_frames = min(30, min(left_frames, right_frames) // 4)
    for _ in range(skip_frames):
        left_cap.read()
        right_cap.read()
    
    # Collect frame pairs
    frame_pairs = []
    
    # First pass: Sample frames at regular intervals
    left_sample_frames = []
    right_sample_frames = []
    
    # Sample frames from left video
    left_cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)
    for i in range(skip_frames, left_frames, left_interval):
        left_cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = left_cap.read()
        if not ret:
            break
        
        left_sample_frames.append((i, frame))
    
    # Sample frames from right video
    right_cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)
    for i in range(skip_frames, right_frames, right_interval):
        right_cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = right_cap.read()
        if not ret:
            break
        
        right_sample_frames.append((i, frame))
    
    print(f"Collected {len(left_sample_frames)} left frames and {len(right_sample_frames)} right frames")
    
    # Match frames by similarity
    print("Matching frames by visual similarity...")
    
    # For each left frame, find the most similar right frame
    matches = []
    
    for left_idx, (left_frame_num, left_frame) in enumerate(tqdm(left_sample_frames)):
        best_similarity = -1
        best_match = None
        
        # Estimate corresponding right frame using FPS ratio
        estimated_right_idx = int(left_idx * fps_ratio)
        
        # Define search window
        start_idx = max(0, estimated_right_idx - search_window)
        end_idx = min(len(right_sample_frames), estimated_right_idx + search_window + 1)
        
        for j in range(start_idx, end_idx):
            right_frame_num, right_frame = right_sample_frames[j]
            
            # Compute similarity between frames
            similarity = compute_frame_similarity(left_frame, right_frame)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = (right_frame_num, right_frame, j)
        
        if best_match:
            matches.append((left_frame_num, left_frame, best_match[0], best_match[1], best_similarity))
    
    # Sort matches by similarity and take the top max_frames
    matches.sort(key=lambda x: x[4], reverse=True)
    top_matches = matches[:max_frames]
    
    # Save matched frame pairs
    frame_pairs = []
    
    for i, (left_frame_num, left_frame, right_frame_num, right_frame, similarity) in enumerate(top_matches):
        # Save frames
        left_frame_path = os.path.join(left_dir, f"left_{i:04d}.png")
        right_frame_path = os.path.join(right_dir, f"right_{i:04d}.png")
        
        cv2.imwrite(left_frame_path, left_frame)
        cv2.imwrite(right_frame_path, right_frame)
        
        # Create a side-by-side visualization
        vis_img = np.hstack((left_frame, right_frame))
        vis_path = os.path.join(output_dir, f"stereo_pair_{i:04d}.png")
        cv2.imwrite(vis_path, vis_img)
        
        frame_pairs.append((left_frame_path, right_frame_path))
        
        print(f"Pair {i+1}: Left frame {left_frame_num}, Right frame {right_frame_num}, Similarity: {similarity:.2f}")
    
    left_cap.release()
    right_cap.release()
    
    print(f"Extracted {len(frame_pairs)} matched frame pairs")
    return frame_pairs

def main():
    """Main function to find and extract matched frame pairs."""
    parser = argparse.ArgumentParser(description='Extract Matched Frame Pairs from Stereo Videos')
    parser.add_argument('--test_dir', required=True, 
                      help='Test directory name (e.g., test_001)')
    parser.add_argument('--base_dir', default='.', 
                      help='Base directory for the project (default: current directory)')
    parser.add_argument('--video_pattern', default=None,
                      help='Pattern to match video files (e.g., "validation*.mp4")')
    parser.add_argument('--max_frames', type=int, default=10,
                      help='Maximum number of frame pairs to extract per video (default: 10)')
    parser.add_argument('--search_window', type=int, default=30,
                      help='Search window size for synchronization (default: 30 frames)')
    
    args = parser.parse_args()
    
    # Set up paths
    base_dir = args.base_dir
    test_dir = os.path.join(base_dir, "data", args.test_dir)
    left_dir = os.path.join(test_dir, "left_camera")
    right_dir = os.path.join(test_dir, "right_camera")
    
    # Create output directory
    output_dir = os.path.join(test_dir, "temp", "matched_frames")
    os.makedirs(output_dir, exist_ok=True)
    
    # Find matching video pairs
    video_pairs = find_matching_videos(left_dir, right_dir, args.video_pattern)
    
    if not video_pairs:
        print("No matching video pairs found")
        return
    
    print(f"Found {len(video_pairs)} matching video pairs")
    
    # Process each video pair
    all_frame_pairs = []
    
    for i, (left_video, right_video) in enumerate(video_pairs):
        # Create a subdirectory for each video pair
        pair_dir = os.path.join(output_dir, f"pair_{i+1:02d}")
        os.makedirs(pair_dir, exist_ok=True)
        
        # Extract matched frame pairs
        frame_pairs = find_matching_frames(
            left_video, right_video, pair_dir, 
            max_frames=args.max_frames,
            search_window=args.search_window
        )
        
        all_frame_pairs.extend(frame_pairs)
    
    print(f"\nExtracted a total of {len(all_frame_pairs)} matched frame pairs")
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()