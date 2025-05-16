import cv2
import os
import glob
import argparse

def create_output_directories(base_dir):
    """Create output directories for calibration images"""
    cam1_dir = os.path.join(base_dir, "camera1_calib_images")
    cam2_dir = os.path.join(base_dir, "camera2_calib_images")
    
    os.makedirs(cam1_dir, exist_ok=True)
    os.makedirs(cam2_dir, exist_ok=True)
    
    return cam1_dir, cam2_dir

def extract_frames(video_path, output_dir, frame_interval=30):
    """
    Extract frames from a video file at regular intervals
    
    Parameters:
    video_path: path to the video file
    output_dir: directory to save extracted frames
    frame_interval: extract every Nth frame
    
    Returns:
    list of paths to extracted frames
    """
    # Get video file name without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Processing {video_name} - {total_frames} frames, {fps} FPS")
    
    # Extract frames
    frame_count = 0
    saved_frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Save every Nth frame
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"{video_name}_frame_{frame_count:04d}.png")
            cv2.imwrite(frame_path, frame)
            saved_frames.append(frame_path)
            print(f"Saved {frame_path}")
            
        frame_count += 1
    
    cap.release()
    return saved_frames

def process_videos_in_folder(folder, output_dir):
    """Process all MOV files in a folder"""
    video_files = glob.glob(os.path.join(folder, "*.mov"))
    all_frames = []
    
    for video in video_files:
        # Skip files in 'data' subdirectories
        if 'data' in video.lower():
            continue
            
        frames = extract_frames(video, output_dir)
        all_frames.extend(frames)
        
    return all_frames

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract frames from Edgertronic videos for calibration')
    parser.add_argument('--interval', type=int, default=30, 
                        help='Extract every Nth frame (default: 30)')
    args = parser.parse_args()
    
    # Base directory (current directory where script is run)
    base_dir = os.getcwd()
    
    # Create output directories
    cam1_dir, cam2_dir = create_output_directories(base_dir)
    
    # Process cam1 videos
    print("\nProcessing Camera 1 videos...")
    cam1_folder = os.path.join(base_dir, "videos", "cam1")
    if os.path.exists(cam1_folder):
        cam1_frames = process_videos_in_folder(cam1_folder, cam1_dir)
        print(f"Extracted {len(cam1_frames)} frames from Camera 1 videos")
    else:
        print(f"Warning: Camera 1 folder not found at {cam1_folder}")
    
    # Process cam2 videos
    print("\nProcessing Camera 2 videos...")
    cam2_folder = os.path.join(base_dir, "videos", "cam2")
    if os.path.exists(cam2_folder):
        cam2_frames = process_videos_in_folder(cam2_folder, cam2_dir)
        print(f"Extracted {len(cam2_frames)} frames from Camera 2 videos")
    else:
        print(f"Warning: Camera 2 folder not found at {cam2_folder}")
    
    print("\nFrame extraction complete!")
    print(f"Camera 1 frames saved to: {cam1_dir}")
    print(f"Camera 2 frames saved to: {cam2_dir}")
    print("\nNext step: Run the camera calibration script on these images")

if __name__ == "__main__":
    main()