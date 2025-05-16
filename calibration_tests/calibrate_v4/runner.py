
#!/usr/bin/env python3
"""
Complete Stereo Calibration Workflow
-----------------------------------
This script orchestrates the entire stereo camera calibration process:
1. Extract optimal frames from calibration videos
2. Perform intrinsic calibration for both cameras
3. Perform extrinsic stereo calibration
4. Validate the calibration with distance measurements

It provides a streamlined workflow to achieve accurate calibration with minimal effort.
"""

import os
import argparse
import subprocess
import time
import glob
import json
import numpy as np

def run_command(command, description):
    """Run a shell command and print output"""
    print(f"\n{'='*80}")
    print(f"STEP: {description}")
    print(f"{'='*80}")
    print(f"Running command: {' '.join(command)}")
    print()
    
    start_time = time.time()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                              universal_newlines=True)
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    elapsed_time = time.time() - start_time
    
    print(f"\nCommand completed in {elapsed_time:.2f} seconds with return code: {process.returncode}")
    return process.returncode == 0

def check_calibration_quality(calib_dir):
    """Check the quality of intrinsic calibration results"""
    # Load summary file if it exists
    summary_file = os.path.join(calib_dir, "calibration_summary.json")
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            summary = json.load(f)
            
            # Check reprojection errors
            cam1_error = summary["camera1"]["avg_reprojection_error"]
            cam2_error = summary["camera2"]["avg_reprojection_error"]
            
            print(f"\nCamera 1 reprojection error: {cam1_error:.6f} pixels")
            print(f"Camera 2 reprojection error: {cam2_error:.6f} pixels")
            
            # Evaluate quality
            if cam1_error > 1.0 or cam2_error > 1.0:
                print("\nWARNING: Reprojection errors are high (>1.0 pixel).")
                print("This may indicate poor calibration quality. Consider re-calibrating with better images.")
                return False
            elif cam1_error > 0.5 or cam2_error > 0.5:
                print("\nNote: Reprojection errors are moderate (>0.5 pixel).")
                print("Calibration is acceptable but could be improved with better images.")
                return True
            else:
                print("\nCalibration quality is good (errors <0.5 pixel).")
                return True
    else:
        print("\nWARNING: Calibration summary file not found!")
        return False

def check_stereo_calibration_quality(stereo_dir):
    """Check the quality of stereo calibration results"""
    # Load stereo calibration data if it exists
    stereo_file = os.path.join(stereo_dir, "stereo_calibration_data.json")
    if os.path.exists(stereo_file):
        with open(stereo_file, 'r') as f:
            stereo_data = json.load(f)
            
            # Check reprojection error
            reprojection_error = stereo_data["reprojection_error"]
            print(f"\nStereo calibration reprojection error: {reprojection_error:.6f} pixels")
            
            # Camera distance
            camera_distance = stereo_data["camera_distance_mm"]
            print(f"Camera separation distance: {camera_distance:.2f} mm")
            
            # Evaluate quality
            if reprojection_error > 1.0:
                print("\nWARNING: Stereo reprojection error is high (>1.0 pixel).")
                print("This may indicate poor stereo calibration quality. Consider re-calibrating.")
                return False
            elif reprojection_error > 0.5:
                print("\nNote: Stereo reprojection error is moderate (>0.5 pixel).")
                print("Stereo calibration is acceptable but could be improved.")
                return True
            else:
                print("\nStereo calibration quality is good (error <0.5 pixel).")
                return True
    else:
        print("\nWARNING: Stereo calibration data file not found!")
        return False

def main():
    parser = argparse.ArgumentParser(description='Complete stereo camera calibration workflow')
    
    # General options
    parser.add_argument('--base-dir', type=str, default='./calibration_data',
                       help='Base directory for all calibration data and outputs')
    parser.add_argument('--checkerboard-size', type=str, default='9,7',
                       help='Size of checkerboard as width,height of internal corners')
    parser.add_argument('--square-size', type=float, default=25.0,
                       help='Size of checkerboard square in mm')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip steps if output files already exist')
    
    # Video files for calibration
    parser.add_argument('--cam1-intrinsic-video', type=str, 
                       help='Path to video for camera 1 intrinsic calibration')
    parser.add_argument('--cam2-intrinsic-video', type=str, 
                       help='Path to video for camera 2 intrinsic calibration')
    parser.add_argument('--stereo-video1', type=str, 
                       help='Path to stereo calibration video from camera 1')
    parser.add_argument('--stereo-video2', type=str, 
                       help='Path to stereo calibration video from camera 2')
    
    # Validation options
    parser.add_argument('--validation-img1', type=str, 
                       help='Path to validation image from camera 1')
    parser.add_argument('--validation-img2', type=str, 
                       help='Path to validation image from camera 2')
    parser.add_argument('--known-distance', type=float, 
                       help='Known distance for validation (mm)')
    parser.add_argument('--known-distances-file', type=str, 
                       help='File with known distances for validation')
    
    # Step selection
    parser.add_argument('--extract-only', action='store_true',
                       help='Only extract frames from videos')
    parser.add_argument('--intrinsic-only', action='store_true',
                       help='Only perform intrinsic calibration')
    parser.add_argument('--extrinsic-only', action='store_true',
                       help='Only perform extrinsic calibration')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only perform validation')
    
    args = parser.parse_args()
    
    # Setup directories
    base_dir = args.base_dir
    os.makedirs(base_dir, exist_ok=True)
    
    cam1_frames_dir = os.path.join(base_dir, "camera1_calib_images")
    cam2_frames_dir = os.path.join(base_dir, "camera2_calib_images")
    stereo_frames_dir = os.path.join(base_dir, "stereo_frames")
    calib_results_dir = os.path.join(base_dir, "calibration_results")
    stereo_results_dir = os.path.join(base_dir, "stereo_calibration_results")
    validation_dir = os.path.join(base_dir, "validation_results")
    
    # Create directories
    for d in [cam1_frames_dir, cam2_frames_dir, stereo_frames_dir, 
              calib_results_dir, stereo_results_dir, validation_dir]:
        os.makedirs(d, exist_ok=True)
    
    # Parse checkerboard size
    checkerboard_size = args.checkerboard_size
    
    # Determine which steps to run
    extract_frames = not (args.intrinsic_only or args.extrinsic_only or args.validate_only)
    run_intrinsic = not (args.extract_only or args.extrinsic_only or args.validate_only)
    run_extrinsic = not (args.extract_only or args.intrinsic_only or args.validate_only)
    run_validation = not (args.extract_only or args.intrinsic_only or args.extrinsic_only)
    
    # If no specific steps are selected, run all steps
    if not any([extract_frames, run_intrinsic, run_extrinsic, run_validation]):
        extract_frames = run_intrinsic = run_extrinsic = run_validation = True
    
    # Check for existing files if skip_existing is enabled
    if args.skip_existing:
        # Check intrinsic calibration files
        cam1_matrix_file = os.path.join(calib_results_dir, "camera_1_matrix.txt")
        cam2_matrix_file = os.path.join(calib_results_dir, "camera_2_matrix.txt")
        if os.path.exists(cam1_matrix_file) and os.path.exists(cam2_matrix_file):
            print("Intrinsic calibration files already exist.")
            run_intrinsic = False
        
        # Check stereo calibration files
        stereo_R_file = os.path.join(stereo_results_dir, "stereo_rotation_matrix.txt")
        stereo_T_file = os.path.join(stereo_results_dir, "stereo_translation_vector.txt")
        if os.path.exists(stereo_R_file) and os.path.exists(stereo_T_file):
            print("Stereo calibration files already exist.")
            run_extrinsic = False
    
    success = True
    
    # STEP 1: Extract frames from videos
    if extract_frames:
        # Extract frames for intrinsic calibration
        if args.cam1_intrinsic_video and args.cam2_intrinsic_video:
            extract_cmd = [
                "python", "optimal_frame_extractor.py",
                "--cam1-video", args.cam1_intrinsic_video,
                "--cam2-video", args.cam2_intrinsic_video,
                "--output-dir", base_dir,
                "--checkerboard-size", checkerboard_size
            ]
            success = run_command(extract_cmd, "Extracting frames for intrinsic calibration")
            if not success:
                print("Error: Frame extraction for intrinsic calibration failed!")
        else:
            print("Skipping intrinsic frame extraction (videos not provided)")
        
        # Extract frames for stereo calibration
        if args.stereo_video1 and args.stereo_video2:
            extract_stereo_cmd = [
                "python", "improved_extrinsic_calibration.py",
                "--base-dir", base_dir,
                "--stereo-video1", args.stereo_video1,
                "--stereo-video2", args.stereo_video2,
                "--extract-frames",
                "--checkerboard-size", checkerboard_size
            ]
            success = run_command(extract_stereo_cmd, "Extracting frames for stereo calibration")
            if not success:
                print("Error: Frame extraction for stereo calibration failed!")
    
    # STEP 2: Run intrinsic calibration
    if run_intrinsic and success:
        intrinsic_cmd = [
            "python", "improved_intrinsic_calibration.py",
            "--base-dir", base_dir,
            "--cam1-images", cam1_frames_dir,
            "--cam2-images", cam2_frames_dir,
            "--output-dir", calib_results_dir,
            "--checkerboard-size", checkerboard_size,
            "--square-size", str(args.square_size)
        ]
        success = run_command(intrinsic_cmd, "Running intrinsic calibration")
        if not success:
            print("Error: Intrinsic calibration failed!")
        else:
            # Check calibration quality
            check_calibration_quality(calib_results_dir)
    
    # STEP 3: Run stereo calibration
    if run_extrinsic and success:
        stereo_cmd = [
            "python", "improved_extrinsic_calibration.py",
            "--base-dir", base_dir,
            "--calib-dir", calib_results_dir,
            "--output-dir", stereo_results_dir,
            "--checkerboard-size", checkerboard_size,
            "--square-size", str(args.square_size)
        ]
        
        if args.stereo_video1 and args.stereo_video2:
            stereo_cmd.extend([
                "--stereo-video1", args.stereo_video1,
                "--stereo-video2", args.stereo_video2,
                "--extract-frames"
            ])
        
        success = run_command(stereo_cmd, "Running stereo calibration")
        if not success:
            print("Error: Stereo calibration failed!")
        else:
            # Check stereo calibration quality
            check_stereo_calibration_quality(stereo_results_dir)
    
    # STEP 4: Run validation
    if run_validation and success and args.validation_img1 and args.validation_img2:
        validation_cmd = [
            "python", "improved_distance_measurement.py",
            "--calib-dir", calib_results_dir,
            "--stereo-dir", stereo_results_dir,
            "--cam1-image", args.validation_img1,
            "--cam2-image", args.validation_img2,
            "--output-dir", validation_dir
        ]
        
        if args.known_distance:
            validation_cmd.extend(["--known-distance", str(args.known_distance)])
        
        if args.known_distances_file:
            validation_cmd.extend([
                "--validation",
                "--known-distances", args.known_distances_file
            ])
        
        success = run_command(validation_cmd, "Running validation measurements")
        if not success:
            print("Error: Validation failed!")
    
    # Print final summary
    if success:
        print("\n" + "="*80)
        print("CALIBRATION WORKFLOW COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nCalibration files are located in:")
        print(f"- Intrinsic calibration: {calib_results_dir}")
        print(f"- Stereo calibration: {stereo_results_dir}")
        
        if run_validation:
            print(f"- Validation results: {validation_dir}")
        
        print("\nYou can now use these for measuring distances with:")
        print("python improved_distance_measurement.py --calib-dir", calib_results_dir, 
              "--stereo-dir", stereo_results_dir, 
              "--cam1-image /path/to/cam1.png --cam2-image /path/to/cam2.png")
    else:
        print("\n" + "="*80)
        print("CALIBRATION WORKFLOW FAILED")
        print("="*80)
        print("\nReview the error messages above and try again.")

if __name__ == "__main__":
    main()