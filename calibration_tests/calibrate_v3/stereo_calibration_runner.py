#!/usr/bin/env python3
"""
Stereo Calibration Runner Script

This script executes the improved stereo calibration workflow using the
videos available in the specified directories.
"""

import os
import sys
import traceback
import argparse
import numpy as np
from stereo_calibration_workflow import StereoCalibrationWorkflow

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run stereo calibration workflow')
    
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'intrinsic', 'extrinsic', 'validate'],
                        help='Which mode to run (full, intrinsic, extrinsic, or validate)')
    
    parser.add_argument('--use-existing', action='store_true',
                        help='Use existing intrinsic calibration files')
    
    parser.add_argument('--output-dir', type=str, default='calibration_results',
                        help='Directory to save calibration results')
    
    parser.add_argument('--checkerboard-size', type=str, default='9,7',
                        help='Checkerboard size as width,height')
    
    parser.add_argument('--square-size', type=float, default=25.0,
                        help='Size of checkerboard squares in mm')
    
    parser.add_argument('--max-frames', type=int, default=30,
                        help='Maximum number of frames to use for calibration')
    
    parser.add_argument('--cam1-calib-video', type=str,
                        default='./videos/cam1/calib1.mov',
                        help='Camera 1 video for intrinsic calibration')
    
    parser.add_argument('--cam2-calib-video', type=str, 
                        default='./videos/cam2/calib2.mov',
                        help='Camera 2 video for intrinsic calibration')
    
    parser.add_argument('--cam1-stereo-video', type=str,
                        default='./videos/cam1/static1.mov',
                        help='Camera 1 video for stereo calibration')
    
    parser.add_argument('--cam2-stereo-video', type=str,
                        default='./videos/cam2/static1.mov',
                        help='Camera 2 video for stereo calibration')
    
    parser.add_argument('--cam1-validation-video', type=str,
                        default='./videos/cam1/data/ruler/ruler1.mov',
                        help='Camera 1 video for validation')
    
    parser.add_argument('--cam2-validation-video', type=str,
                        default='./videos/cam2/data/ruler/ruler1.mov',
                        help='Camera 2 video for validation')
    
    parser.add_argument('--cam1-params-dir', type=str,
                        default='./parameters/cam1',
                        help='Directory with Camera 1 existing parameters')
    
    parser.add_argument('--cam2-params-dir', type=str,
                        default='./parameters/cam2',
                        help='Directory with Camera 2 existing parameters')
    
    args = parser.parse_args()
    
    # Parse checkerboard size
    width, height = map(int, args.checkerboard_size.split(','))
    checkerboard_size = (width, height)
    
    try:
        # Initialize the workflow
        workflow = StereoCalibrationWorkflow(
            base_dir=".",
            output_dir=args.output_dir,
            checkerboard_size=checkerboard_size,
            square_size=args.square_size,
            max_frames=args.max_frames
        )
        
        # Run the selected mode
        if args.mode == 'full':
            # Run full workflow
            workflow.run_complete_workflow(
                args.cam1_calib_video, args.cam2_calib_video,
                args.cam1_stereo_video, args.cam2_stereo_video,
                args.cam1_validation_video, args.cam2_validation_video,
                args.use_existing, args.cam1_params_dir, args.cam2_params_dir
            )
        
        elif args.mode == 'intrinsic':
            # Run only intrinsic calibration
            if args.use_existing:
                workflow.load_existing_calibration(args.cam1_params_dir, args.cam2_params_dir)
            else:
                workflow.run_intrinsic_calibration(args.cam1_calib_video, args.cam2_calib_video)
        
        elif args.mode == 'extrinsic':
            # Run only extrinsic calibration (requires intrinsic parameters)
            if not args.use_existing:
                print("Error: Extrinsic-only mode requires --use-existing flag")
                return
            
            success = workflow.load_existing_calibration(args.cam1_params_dir, args.cam2_params_dir)
            if success:
                workflow.run_extrinsic_calibration(args.cam1_stereo_video, args.cam2_stereo_video)
        
        elif args.mode == 'validate':
            # Run only validation (requires full calibration)
            if not args.use_existing:
                print("Error: Validate-only mode requires --use-existing flag")
                return
            
            success = workflow.load_existing_calibration(args.cam1_params_dir, args.cam2_params_dir)
            if success:
                # Also need to load extrinsic parameters
                R_path = os.path.join(args.output_dir, 'stereo_rotation_matrix.txt')
                T_path = os.path.join(args.output_dir, 'stereo_translation_vector.txt')
                
                if os.path.exists(R_path) and os.path.exists(T_path):
                    workflow.R = np.loadtxt(R_path)
                    workflow.T = np.loadtxt(T_path).reshape(3, 1)
                    workflow.validate_with_ruler(args.cam1_validation_video, args.cam2_validation_video)
                else:
                    print("Error: Extrinsic parameters not found. Run extrinsic calibration first.")

    except Exception as e:
        print("\n*** ERROR OCCURRED ***")
        print(f"Error message: {str(e)}")
        print("\nDetailed traceback:")
        traceback.print_exc()
        print("\nCheck the error message above and make corrections.")
        print("If it's a format or configuration error, you may need to adjust the script.")
        sys.exit(1)

if __name__ == "__main__":
    main()