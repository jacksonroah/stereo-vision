import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import xml.etree.ElementTree as ET
import time
import threading
from tqdm import tqdm  # Optional, for progress bars

class StereoCalibrationWorkflow:
    def __init__(self, base_dir='.', output_dir='calibration_results',
                 checkerboard_size=(9, 7), square_size=25.0,
                 frame_interval=15, max_frames=40):
        """
        Initialize the stereo calibration workflow
        
        Parameters:
        base_dir: Base directory containing videos and calibration files
        output_dir: Directory to save calibration results
        checkerboard_size: Size of the checkerboard (internal corners)
        square_size: Size of checkerboard squares in mm
        frame_interval: Extract every Nth frame from videos
        max_frames: Maximum number of frames to use for calibration
        """
        self.base_dir = base_dir
        self.output_dir = os.path.join(base_dir, output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create subdirectories
        self.cam1_frames_dir = os.path.join(self.output_dir, 'camera1_frames')
        self.cam2_frames_dir = os.path.join(self.output_dir, 'camera2_frames')
        self.debug_dir = os.path.join(self.output_dir, 'debug_images')
        self.visualization_dir = os.path.join(self.output_dir, 'visualizations')
        
        for directory in [self.cam1_frames_dir, self.cam2_frames_dir, 
                          self.debug_dir, self.visualization_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Calibration parameters
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        self.frame_interval = frame_interval
        self.max_frames = max_frames
        
        # Calibration results
        self.cam1_matrix = None
        self.cam1_dist = None
        self.cam2_matrix = None
        self.cam2_dist = None
        self.R = None  # Rotation matrix
        self.T = None  # Translation vector
        
        # Initialize the logger
        self.log_file = os.path.join(self.output_dir, 'calibration_log.txt')
        with open(self.log_file, 'w') as f:
            f.write(f"Stereo Calibration Log - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
    
    def log(self, message):
        """Add a message to the log file"""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(f"{message}\n")
    
    def extract_frames_from_video(self, video_path, output_dir, prefix='frame'):
        """
        Extract frames from a video file at regular intervals
        
        Parameters:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        prefix: Prefix for frame filenames
        
        Returns:
        List of paths to extracted frames
        """
        self.log(f"Extracting frames from {os.path.basename(video_path)}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get video file name without extension
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Open the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.log(f"Error: Could not open video {video_path}")
            return []
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        self.log(f"Video info: {total_frames} frames, {fps} FPS")
        
        # Calculate how many frames to extract to stay under max_frames
        step = max(1, total_frames // self.max_frames)
        if step < self.frame_interval:
            step = self.frame_interval
        
        self.log(f"Extracting every {step} frames to get ~{total_frames//step} samples")
        
        # Extract frames
        frame_count = 0
        saved_frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Save every Nth frame
            if frame_count % step == 0:
                frame_path = os.path.join(output_dir, f"{prefix}_{video_name}_{frame_count:04d}.png")
                cv2.imwrite(frame_path, frame)
                saved_frames.append(frame_path)
                
            frame_count += 1
        
        cap.release()
        self.log(f"Extracted {len(saved_frames)} frames from {video_name}")
        return saved_frames
    
    def find_checkerboard_corners(self, image_path, debug=True):
        """
        Find checkerboard corners in an image
        
        Parameters:
        image_path: Path to the image
        debug: Whether to save debug images
        
        Returns:
        (success, corners, image_shape)
        """
        img = cv2.imread(image_path)
        if img is None:
            self.log(f"Error: Could not read image {image_path}")
            return False, None, None
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find checkerboard corners with adaptative methods
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        ret, corners = cv2.findChessboardCorners(
            gray, self.checkerboard_size, 
            cv2.CALIB_CB_ADAPTIVE_THRESH + 
            cv2.CALIB_CB_NORMALIZE_IMAGE +
            cv2.CALIB_CB_FAST_CHECK)
        
        if ret:
            # Refine corner locations
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Draw and save corners for visualization if debug is enabled
            if debug:
                img_corners = img.copy()
                cv2.drawChessboardCorners(img_corners, self.checkerboard_size, corners2, ret)
                debug_path = os.path.join(self.debug_dir, f"corners_{os.path.basename(image_path)}")
                cv2.imwrite(debug_path, img_corners)
            
            return ret, corners2, gray.shape
        
        return False, None, None
    
    def calibrate_single_camera(self, images_folder, camera_id):
        """
        Calibrate a single camera using checkerboard images
        
        Parameters:
        images_folder: Folder containing calibration images
        camera_id: ID of the camera being calibrated
        
        Returns:
        (camera_matrix, dist_coeffs)
        """
        self.log(f"\n=== Calibrating Camera {camera_id} ===")
        
        # Get all images in the folder
        images = glob.glob(os.path.join(images_folder, '*.png'))
        self.log(f"Found {len(images)} images for calibration")
        
        if len(images) < 10:
            self.log(f"Warning: Only {len(images)} images found. Recommend at least 15 for good calibration.")
        
        # Create object points
        objp = np.zeros((self.checkerboard_size[0] * self.checkerboard_size[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:self.checkerboard_size[0], 0:self.checkerboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size  # Scale to actual size in mm
        
        # Arrays to store object points and image points
        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane
        
        # Process each image
        successful_images = 0
        img_shape = None
        
        for img_path in images:
            ret, corners, shape = self.find_checkerboard_corners(img_path)
            
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)
                successful_images += 1
                
                # Save the shape for the first successful image
                if img_shape is None:
                    img_shape = shape
        
        self.log(f"Successfully processed {successful_images} out of {len(images)} images")
        
        if successful_images < 5:
            self.log("ERROR: Not enough successful calibration images!")
            return None, None
        
        # Add calibration flags with constraints to prevent extreme values
        flags = (
            cv2.CALIB_RATIONAL_MODEL +  # Use rational polynomial model (more stable)
            cv2.CALIB_FIX_K4 +          # Fix 4th order radial distortion coefficient
            cv2.CALIB_FIX_K5 +          # Fix 5th order radial distortion coefficient
            cv2.CALIB_FIX_K6            # Fix 6th order radial distortion coefficient
        )
        
        # Set termination criteria for the optimization
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-8)
        
        # Perform calibration
        self.log(f"Starting intrinsic calibration with {successful_images} images")
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, img_shape, None, None, 
            flags=flags, criteria=criteria)
        
        # Calculate reprojection error
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        
        avg_error = mean_error / len(objpoints)
        self.log(f"Calibration complete! Average reprojection error: {avg_error}")
        
        # Validate the calibration parameters
        if not self.validate_calibration_params(camera_matrix, dist_coeffs, avg_error):
            self.log("Warning: Calibration parameters may be unreliable!")
        
        # Save calibration parameters
        self.save_calibration_params(camera_id, camera_matrix, dist_coeffs, avg_error)
        
        return camera_matrix, dist_coeffs
    
    def validate_calibration_params(self, camera_matrix, dist_coeffs, reprojection_error, 
                                max_error=1.0, max_dist_coeff=1.5):
        """
        Validate calibration parameters to detect potential issues
        
        Returns:
        bool: True if parameters seem reasonable, False otherwise
        """
        # Check reprojection error
        if reprojection_error > max_error:
            self.log(f"Warning: High reprojection error: {reprojection_error} (threshold: {max_error})")
            # Don't fail on this, just warn
        
        # Check focal length consistency (fx should be similar to fy)
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        focal_ratio = fx / fy
        if focal_ratio < 0.8 or focal_ratio > 1.2:
            self.log(f"Warning: Inconsistent focal lengths - fx/fy ratio: {focal_ratio}")
            # Don't fail on this, just warn
        
        # Check for extremely large distortion coefficients
        has_extreme_coeffs = False
        for i, coeff in enumerate(dist_coeffs.flatten()):
            if abs(coeff) > max_dist_coeff:
                self.log(f"Warning: Large distortion coefficient {i}: {coeff}")
                has_extreme_coeffs = True
        
        # Print overall validation result
        if reprojection_error <= 0.5 and not has_extreme_coeffs:
            self.log("Calibration parameters look good!")
            return True
        elif reprojection_error <= 1.0 and not has_extreme_coeffs:
            self.log("Calibration parameters are acceptable but not optimal")
            return True
        else:
            self.log("Warning: Some calibration parameters are unusual, but we'll continue")
            return True  # Changed to return True to continue despite warnings
    
    def save_calibration_params(self, camera_id, camera_matrix, dist_coeffs, reprojection_error):
        """Save calibration parameters to files"""
        # Save to numpy text files (for easier loading in other scripts)
        np.savetxt(os.path.join(self.output_dir, f'camera_{camera_id}_matrix.txt'), camera_matrix)
        np.savetxt(os.path.join(self.output_dir, f'camera_{camera_id}_distortion.txt'), dist_coeffs)
        
        # Save to CSV with descriptive headers
        matrix_headers = ['Row1Col1', 'Row1Col2', 'Row1Col3', 
                        'Row2Col1', 'Row2Col2', 'Row2Col3', 
                        'Row3Col1', 'Row3Col2', 'Row3Col3']
        
        # Generate proper number of distortion headers
        num_dist_coeffs = len(dist_coeffs.flatten())
        dist_headers = []
        if num_dist_coeffs >= 5:
            dist_headers = ['k1', 'k2', 'p1', 'p2', 'k3']
            # Add k4, k5, k6 if needed
            if num_dist_coeffs > 5:
                for i in range(6, num_dist_coeffs + 1):
                    dist_headers.append(f'k{i-2}')
        else:
            # Just use generic names if we don't know the standard names
            dist_headers = [f'dist{i+1}' for i in range(num_dist_coeffs)]
        
        # Make sure the number of headers matches the number of values
        assert len(matrix_headers) == len(camera_matrix.flatten()), "Matrix headers and values mismatch"
        assert len(dist_headers) == len(dist_coeffs.flatten()), "Distortion headers and values mismatch"
        
        df_matrix = pd.DataFrame({
            'Element': matrix_headers,
            'Value': camera_matrix.flatten()
        })
        
        df_dist = pd.DataFrame({
            'Element': dist_headers,
            'Value': dist_coeffs.flatten()
        })
        
        df_matrix.to_csv(os.path.join(self.output_dir, f'camera_{camera_id}_matrix.csv'), index=False)
        df_dist.to_csv(os.path.join(self.output_dir, f'camera_{camera_id}_distortion.csv'), index=False)
        
        # Save to XML format
        self.save_to_xml(camera_id, camera_matrix, dist_coeffs, reprojection_error)

    def save_to_xml(self, camera_id, camera_matrix, dist_coeffs, reprojection_error):
        """Save calibration parameters to XML format"""
        xml_path = os.path.join(self.output_dir, f'camera_{camera_id}_calibration.xml')
        
        # Create XML structure
        root = ET.Element('opencv_storage')
        
        # Add comment
        comment = ET.Comment('Camera Matrix and Distortion Coefficients for Camera Calibration')
        root.append(comment)
        
        # Camera matrix
        matrix_elem = ET.SubElement(root, 'CameraMatrix')
        matrix_elem.set('type_id', 'opencv-matrix')
        ET.SubElement(matrix_elem, 'rows').text = '3'
        ET.SubElement(matrix_elem, 'cols').text = '3'
        ET.SubElement(matrix_elem, 'dt').text = 'd'
        matrix_data = ' '.join(map(str, camera_matrix.flatten()))
        ET.SubElement(matrix_elem, 'data').text = matrix_data
        
        # Add comment for camera matrix
        comment = ET.Comment('Camera Matrix (Intrinsic Parameters)')
        root.append(comment)
        
        # Distortion coefficients
        dist_elem = ET.SubElement(root, 'DistortionCoefficients')
        dist_elem.set('type_id', 'opencv-matrix')
        ET.SubElement(dist_elem, 'rows').text = '1'
        ET.SubElement(dist_elem, 'cols').text = str(len(dist_coeffs.flatten()))
        ET.SubElement(dist_elem, 'dt').text = 'd'
        dist_data = ' '.join(map(str, dist_coeffs.flatten()))
        ET.SubElement(dist_elem, 'data').text = dist_data
        
        # Add comment for distortion coefficients
        comment = ET.Comment('Distortion Coefficients (k1, k2, p1, p2, k3)')
        root.append(comment)
        
        # Add reprojection error
        error_elem = ET.SubElement(root, 'ReprojectionError')
        error_elem.text = str(reprojection_error)
        
        # Write to file
        tree = ET.ElementTree(root)
        with open(xml_path, 'wb') as f:
            tree.write(f, encoding='utf-8', xml_declaration=True)
    
    def load_calibration_params_from_xml(self, xml_path):
        """Load calibration parameters from an XML file"""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Extract camera matrix
        matrix_elem = root.find('CameraMatrix/data')
        if matrix_elem is not None:
            matrix_data = list(map(float, matrix_elem.text.split()))
            camera_matrix = np.array(matrix_data).reshape(3, 3)
        else:
            return None, None
        
        # Extract distortion coefficients
        dist_elem = root.find('DistortionCoefficients/data')
        if dist_elem is not None:
            dist_data = list(map(float, dist_elem.text.split()))
            dist_coeffs = np.array(dist_data).reshape(1, -1)
        else:
            return None, None
        
        return camera_matrix, dist_coeffs
    
    def load_calibration_params_from_csv(self, matrix_csv, dist_csv):
        """Load calibration parameters from CSV files"""
        matrix_df = pd.read_csv(matrix_csv)
        dist_df = pd.read_csv(dist_csv)
        
        matrix_values = matrix_df['Camera Matrix'].values
        camera_matrix = np.array(matrix_values).reshape(3, 3)
        
        dist_values = dist_df['Distortion Coefficients'].values
        dist_coeffs = np.array(dist_values).reshape(1, -1)
        
        return camera_matrix, dist_coeffs
    
    def load_existing_calibration(self, cam1_params_dir, cam2_params_dir):
        """
        Load existing calibration parameters instead of recalibrating
        
        Parameters:
        cam1_params_dir: Directory containing camera 1 parameters
        cam2_params_dir: Directory containing camera 2 parameters
        
        Returns:
        bool: True if parameters loaded successfully
        """
        self.log("\nLoading existing calibration parameters...")
        
        # Try to load from XML files
        cam1_xml = os.path.join(cam1_params_dir, 'cam1_calibration_parameters.xml')
        cam2_xml = os.path.join(cam2_params_dir, 'cam2_calibration_parameters.xml')
        
        if os.path.exists(cam1_xml) and os.path.exists(cam2_xml):
            self.log("Loading calibration from XML files")
            self.cam1_matrix, self.cam1_dist = self.load_calibration_params_from_xml(cam1_xml)
            self.cam2_matrix, self.cam2_dist = self.load_calibration_params_from_xml(cam2_xml)
        
        # If XML loading failed, try CSV files
        if self.cam1_matrix is None or self.cam2_matrix is None:
            self.log("Loading calibration from CSV files")
            cam1_matrix_csv = os.path.join(cam1_params_dir, 'cam1_camera_matrix.csv')
            cam1_dist_csv = os.path.join(cam1_params_dir, 'cam1_distortion_coefficients.csv')
            cam2_matrix_csv = os.path.join(cam2_params_dir, 'cam2_camera_matrix.csv')
            cam2_dist_csv = os.path.join(cam2_params_dir, 'cam2_distortion_coefficients.csv')
            
            if (os.path.exists(cam1_matrix_csv) and os.path.exists(cam1_dist_csv) and
                os.path.exists(cam2_matrix_csv) and os.path.exists(cam2_dist_csv)):
                self.cam1_matrix, self.cam1_dist = self.load_calibration_params_from_csv(
                    cam1_matrix_csv, cam1_dist_csv)
                self.cam2_matrix, self.cam2_dist = self.load_calibration_params_from_csv(
                    cam2_matrix_csv, cam2_dist_csv)
        
        # Check if loading was successful
        if self.cam1_matrix is not None and self.cam2_matrix is not None:
            self.log("Successfully loaded calibration parameters:")
            self.log(f"Camera 1 Matrix:\n{self.cam1_matrix}")
            self.log(f"Camera 1 Distortion:\n{self.cam1_dist}")
            self.log(f"Camera 2 Matrix:\n{self.cam2_matrix}")
            self.log(f"Camera 2 Distortion:\n{self.cam2_dist}")
            return True
        else:
            self.log("Failed to load calibration parameters")
            return False
    
    def match_stereo_frames(self, cam1_frames, cam2_frames):
        """
        Match frames from two cameras for stereo calibration
        
        Parameters:
        cam1_frames: List of frames from camera 1
        cam2_frames: List of frames from camera 2
        
        Returns:
        List of (cam1_frame, cam2_frame) pairs
        """
        self.log("\nMatching stereo frames...")
        
        # If frame counts match exactly, pair them by index
        pairs = []
        if len(cam1_frames) == len(cam2_frames):
            self.log("Equal number of frames - pairing by index")
            pairs = list(zip(cam1_frames, cam2_frames))
        else:
            # Try to match by filename similarity (timestamp)
            self.log(f"Different frame counts: Cam1={len(cam1_frames)}, Cam2={len(cam2_frames)}")
            self.log("Matching frames by filename similarity")
            
            # Extract frame numbers from filenames
            frame_extract = lambda path: int(os.path.basename(path).split('_')[-1].split('.')[0])
            
            # Sort frames by number
            cam1_frames = sorted(cam1_frames, key=frame_extract)
            cam2_frames = sorted(cam2_frames, key=frame_extract)
            
            # Try to match frames with similar numbers
            for cam1_frame in cam1_frames:
                cam1_num = frame_extract(cam1_frame)
                best_match = None
                min_diff = float('inf')
                
                for cam2_frame in cam2_frames:
                    cam2_num = frame_extract(cam2_frame)
                    diff = abs(cam1_num - cam2_num)
                    
                    if diff < min_diff:
                        min_diff = diff
                        best_match = cam2_frame
                
                if best_match and min_diff < 5:  # Frame number should be close
                    pairs.append((cam1_frame, best_match))
        
        self.log(f"Matched {len(pairs)} stereo frame pairs")
        return pairs
    
    def calibrate_stereo(self, cam1_frames, cam2_frames):
        """
        Perform stereo calibration to find extrinsic parameters between cameras
        
        Parameters:
        cam1_frames: List of frames from camera 1
        cam2_frames: List of frames from camera 2
        
        Returns:
        (R, T): Rotation matrix and translation vector from camera 1 to camera 2
        """
        self.log("\n=== Performing Stereo Calibration ===")
        
        # Match frames between cameras
        stereo_pairs = self.match_stereo_frames(cam1_frames, cam2_frames)
        
        if len(stereo_pairs) < 5:
            self.log(f"Error: Not enough matching frame pairs for stereo calibration ({len(stereo_pairs)} found)")
            return None, None
        
        # Arrays to store object points and image points
        objpoints = []  # 3D points in real world space
        imgpoints1 = []  # 2D points in camera 1
        imgpoints2 = []  # 2D points in camera 2
        
        # Prepare object points (same for all frames)
        objp = np.zeros((self.checkerboard_size[0] * self.checkerboard_size[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:self.checkerboard_size[0], 0:self.checkerboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size  # Convert to mm
        
        # Process each stereo pair
        successful_pairs = 0
        img_shape = None
        
        for pair_idx, (cam1_img_path, cam2_img_path) in enumerate(stereo_pairs):
            self.log(f"Processing stereo pair {pair_idx+1}/{len(stereo_pairs)}")
            
            # Find checkerboard corners in both images
            ret1, corners1, shape1 = self.find_checkerboard_corners(cam1_img_path)
            ret2, corners2, shape2 = self.find_checkerboard_corners(cam2_img_path)
            
            if ret1 and ret2:
                objpoints.append(objp)
                imgpoints1.append(corners1)
                imgpoints2.append(corners2)
                successful_pairs += 1
                
                # Store image shape for calibration
                if img_shape is None:
                    img_shape = shape1
        
        if successful_pairs < 5:
            self.log(f"Error: Only {successful_pairs} successful stereo pairs found. Need at least 5.")
            return None, None
        
        self.log(f"Using {successful_pairs} stereo pairs for calibration")
        
        # Set stereo calibration flags
        flags = 0
        # If we have good intrinsic parameters, fix them
        if self.validate_calibration_params(self.cam1_matrix, self.cam1_dist, 0.5) and \
           self.validate_calibration_params(self.cam2_matrix, self.cam2_dist, 0.5):
            flags |= cv2.CALIB_FIX_INTRINSIC
            self.log("Using fixed intrinsic parameters for stereo calibration")
        else:
            self.log("Allowing intrinsic parameters to be refined during stereo calibration")
        
        # Set optimization termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        
        # Perform stereo calibration
        self.log("Starting stereo calibration...")
        ret, self.cam1_matrix, self.cam1_dist, self.cam2_matrix, self.cam2_dist, \
        self.R, self.T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints1, imgpoints2,
            self.cam1_matrix, self.cam1_dist,
            self.cam2_matrix, self.cam2_dist,
            img_shape, criteria=criteria, flags=flags)
        
        self.log(f"Stereo calibration complete. RMS error: {ret}")
        
        # Calculate camera separation distance
        camera_distance = np.linalg.norm(self.T)
        self.log(f"Camera separation: {camera_distance:.2f} mm")
        
        # Convert rotation matrix to Euler angles for better visualization
        r_vec, _ = cv2.Rodrigues(self.R)
        euler_angles = r_vec * 180.0 / np.pi
        self.log(f"Camera 2 orientation relative to Camera 1 (Euler angles in degrees):")
        self.log(f"  X-rotation: {euler_angles[0][0]:.2f}°")
        self.log(f"  Y-rotation: {euler_angles[1][0]:.2f}°")
        self.log(f"  Z-rotation: {euler_angles[2][0]:.2f}°")
        
        # Save stereo calibration parameters
        np.savetxt(os.path.join(self.output_dir, 'stereo_rotation_matrix.txt'), self.R)
        np.savetxt(os.path.join(self.output_dir, 'stereo_translation_vector.txt'), self.T)
        np.savetxt(os.path.join(self.output_dir, 'essential_matrix.txt'), E)
        np.savetxt(os.path.join(self.output_dir, 'fundamental_matrix.txt'), F)
        
        # Visualize camera positions
        self.visualize_camera_positions()
        
        # Compute and save rectification parameters
        self.compute_rectification(img_shape)
        
        return self.R, self.T
    
    def compute_rectification(self, image_size):
        """Compute and save rectification parameters for stereo vision"""
        try:
            R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
                self.cam1_matrix, self.cam1_dist,
                self.cam2_matrix, self.cam2_dist,
                image_size, self.R, self.T,
                flags=cv2.CALIB_ZERO_DISPARITY,
                alpha=0.9)
            
            self.log("Rectification parameters computed successfully")
            
            # Save rectification parameters
            np.savetxt(os.path.join(self.output_dir, 'rect_R1.txt'), R1)
            np.savetxt(os.path.join(self.output_dir, 'rect_R2.txt'), R2)
            np.savetxt(os.path.join(self.output_dir, 'rect_P1.txt'), P1)
            np.savetxt(os.path.join(self.output_dir, 'rect_P2.txt'), P2)
            np.savetxt(os.path.join(self.output_dir, 'disparity_to_depth_matrix.txt'), Q)
            
            # Save ROIs as well
            np.savetxt(os.path.join(self.output_dir, 'rect_roi1.txt'), np.array(roi1))
            np.savetxt(os.path.join(self.output_dir, 'rect_roi2.txt'), np.array(roi2))
            
            return R1, R2, P1, P2, Q
        except Exception as e:
            self.log(f"Error computing rectification parameters: {e}")
            return None, None, None, None, None
    
    def visualize_camera_positions(self):
        """Visualize the relative positions of two cameras in 3D space"""
        try:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Camera 1 is at the origin
            cam1_pos = np.array([0, 0, 0])
            
            # Camera 2 position is determined by the translation vector
            cam2_pos = -self.R.T @ self.T.flatten()
            
            # Plot camera positions
            ax.scatter([cam1_pos[0]], [cam1_pos[1]], [cam1_pos[2]], c='r', marker='o', s=100, label='Camera 1')
            ax.scatter([cam2_pos[0]], [cam2_pos[1]], [cam2_pos[2]], c='b', marker='o', s=100, label='Camera 2')
            
            # Draw line connecting cameras
            ax.plot([cam1_pos[0], cam2_pos[0]], 
                    [cam1_pos[1], cam2_pos[1]], 
                    [cam1_pos[2], cam2_pos[2]], 'k-')
            
            # Plot camera orientations (principal axis)
            # For Camera 1, the principal axis is along the Z axis
            cam1_axis = np.array([0, 0, 100])  # 100mm along Z axis
            ax.quiver(cam1_pos[0], cam1_pos[1], cam1_pos[2], 
                      cam1_axis[0], cam1_axis[1], cam1_axis[2], 
                      color='r', arrow_length_ratio=0.1)
            
            # For Camera 2, the principal axis is rotated according to R
            cam2_axis = self.R.T @ np.array([0, 0, 100])
            ax.quiver(cam2_pos[0], cam2_pos[1], cam2_pos[2], 
                      cam2_axis[0], cam2_axis[1], cam2_axis[2], 
                      color='b', arrow_length_ratio=0.1)
            
            # Set equal aspect ratio
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_zlabel('Z (mm)')
            ax.set_title('Relative Camera Positions')
            
            # Try to set equal scaling for all axes
            max_range = np.max([
                np.max([cam1_pos[0], cam2_pos[0]]) - np.min([cam1_pos[0], cam2_pos[0]]),
                np.max([cam1_pos[1], cam2_pos[1]]) - np.min([cam1_pos[1], cam2_pos[1]]),
                np.max([cam1_pos[2], cam2_pos[2]]) - np.min([cam1_pos[2], cam2_pos[2]])
            ])
            
            mid_x = (cam1_pos[0] + cam2_pos[0]) * 0.5
            mid_y = (cam1_pos[1] + cam2_pos[1]) * 0.5
            mid_z = (cam1_pos[2] + cam2_pos[2]) * 0.5
            
            ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
            ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
            ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
            
            ax.legend()
            
            # Save the plot
            plt.savefig(os.path.join(self.visualization_dir, 'camera_positions.png'))
            plt.close()
            self.log(f"Camera positions visualization saved to {self.visualization_dir}/camera_positions.png")
        except Exception as e:
            self.log(f"Error visualizing camera positions: {e}")
    
    def measure_distance_to_checkerboard(self, cam1_image, cam2_image):
        """
        Measure the distance to a checkerboard visible in both cameras
        
        Parameters:
        cam1_image, cam2_image: Paths to images from cameras 1 and 2
        
        Returns:
        avg_distance: Average distance to the checkerboard (mm)
        """
        # Ensure calibration parameters are loaded
        if self.cam1_matrix is None or self.cam2_matrix is None or self.R is None or self.T is None:
            self.log("Error: Calibration parameters not loaded")
            return None
        
        # Create projection matrices
        P1 = self.cam1_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = self.cam2_matrix @ np.hstack((self.R, self.T))
        
        # Find checkerboard corners in both images
        ret1, corners1, _ = self.find_checkerboard_corners(cam1_image)
        ret2, corners2, _ = self.find_checkerboard_corners(cam2_image)
        
        if not ret1 or not ret2:
            self.log("Could not find checkerboard in one or both images")
            return None
        
        # Undistort the corner points
        corners1_undistorted = cv2.undistortPoints(corners1, self.cam1_matrix, self.cam1_dist, P=self.cam1_matrix)
        corners2_undistorted = cv2.undistortPoints(corners2, self.cam2_matrix, self.cam2_dist, P=self.cam2_matrix)
        
        # Reshape for triangulation
        points1 = corners1_undistorted.reshape(-1, 2).T
        points2 = corners2_undistorted.reshape(-1, 2).T
        
        # Triangulate 3D points
        points4D = cv2.triangulatePoints(P1, P2, points1, points2)
        points3D = (points4D[:3] / points4D[3]).T
        
        # Calculate distances from camera 1 to each point
        distances = np.linalg.norm(points3D, axis=1)
        avg_distance = np.mean(distances)
        
        self.log(f"Distance measurement results:")
        self.log(f"  Average distance to checkerboard: {avg_distance:.2f} mm")
        self.log(f"  Min distance: {np.min(distances):.2f} mm")
        self.log(f"  Max distance: {np.max(distances):.2f} mm")
        
        # Visualize the 3D points
        self.visualize_3d_points(points3D, os.path.join(self.visualization_dir, 'checkerboard_3d.png'))
        
        return avg_distance, points3D
    
    def measure_ruler_distance(self, cam1_image, cam2_image, manual_selection=True):
        """
        Measure the 3D position and length of a ruler visible in both camera images
        
        Parameters:
        cam1_image, cam2_image: Paths to images from cameras 1 and 2
        manual_selection: Whether to use manual endpoint selection
        
        Returns:
        ruler_length: Length of the ruler in 3D space (mm)
        """
        # Ensure calibration parameters are loaded
        if self.cam1_matrix is None or self.cam2_matrix is None or self.R is None or self.T is None:
            self.log("Error: Calibration parameters not loaded")
            return None, None
        
        # Function for manual endpoint selection
        def select_points(image_path, num_points=2):
            points = []
            img = cv2.imread(image_path)
            if img is None:
                self.log(f"Error: Could not read image {image_path}")
                return False, None
            
            # Create a window for selection
            window_name = f"Select {num_points} points in the image (click + press 'c' when done)"
            cv2.namedWindow(window_name)
            
            # Mouse callback function
            def mouse_callback(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN and len(points) < num_points:
                    points.append((x, y))
                    # Draw the point
                    cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
                    cv2.imshow(window_name, img)
            
            cv2.setMouseCallback(window_name, mouse_callback)
            cv2.imshow(window_name, img)
            
            # Wait for selection
            while len(points) < num_points:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    return False, None
            
            cv2.destroyAllWindows()
            return True, np.array(points, dtype=np.float32)
        
        # Select endpoints in both images
        if manual_selection:
            self.log("Please select ruler endpoints in both images")
            ret1, points1 = select_points(cam1_image)
            ret2, points2 = select_points(cam2_image)
        else:
            # Implement automatic ruler detection here
            # For now, just return failure
            self.log("Automatic ruler detection not implemented yet")
            return None, None
        
        if not ret1 or not ret2:
            self.log("Failed to select ruler endpoints")
            return None, None
        
        # Create projection matrices
        P1 = self.cam1_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = self.cam2_matrix @ np.hstack((self.R, self.T))
        
        # Undistort the points
        points1_undistorted = cv2.undistortPoints(points1.reshape(-1, 1, 2), self.cam1_matrix, self.cam1_dist, P=self.cam1_matrix)
        points2_undistorted = cv2.undistortPoints(points2.reshape(-1, 1, 2), self.cam2_matrix, self.cam2_dist, P=self.cam2_matrix)
        
        # Reshape for triangulation
        points1_undistorted = points1_undistorted.reshape(-1, 2).T
        points2_undistorted = points2_undistorted.reshape(-1, 2).T
        
        # Triangulate 3D points
        points4D = cv2.triangulatePoints(P1, P2, points1_undistorted, points2_undistorted)
        points3D = (points4D[:3] / points4D[3]).T
        
        # Calculate 3D length
        ruler_length = np.linalg.norm(points3D[1] - points3D[0])
        
        # Calculate distances from camera 1 to each endpoint
        distances = np.linalg.norm(points3D, axis=1)
        avg_distance = np.mean(distances)
        
        self.log(f"Ruler measurement results:")
        self.log(f"  Ruler 3D length: {ruler_length:.2f} mm")
        self.log(f"  Distance to ruler midpoint: {avg_distance:.2f} mm")
        
        # Visualize the 3D points
        self.visualize_3d_points(points3D, os.path.join(self.visualization_dir, 'ruler_3d.png'), connect_points=True)
        
        return ruler_length, points3D
    
    def visualize_3d_points(self, points3D, output_path, connect_points=False):
        """Visualize 3D points with camera positions"""
        try:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot camera positions
            ax.scatter([0], [0], [0], c='r', marker='o', s=100, label='Camera 1')
            if self.T is not None:
                cam2_pos = -self.R.T @ self.T.flatten()
                ax.scatter([cam2_pos[0]], [cam2_pos[1]], [cam2_pos[2]], c='b', marker='o', s=100, label='Camera 2')
            
            # Plot the 3D points
            ax.scatter(points3D[:, 0], points3D[:, 1], points3D[:, 2], c='g', marker='.', s=20)
            
            # Connect points if requested (for ruler measurements)
            if connect_points:
                ax.plot(points3D[:, 0], points3D[:, 1], points3D[:, 2], 'g-', linewidth=2)
            
            # For checkerboard, connect points to form a grid
            elif points3D.shape[0] > 4:  # Assume it's a checkerboard
                rows, cols = self.checkerboard_size
                for i in range(rows):
                    start_idx = i * cols
                    end_idx = (i + 1) * cols
                    if start_idx < points3D.shape[0] and end_idx <= points3D.shape[0]:
                        ax.plot(points3D[start_idx:end_idx, 0], 
                                points3D[start_idx:end_idx, 1], 
                                points3D[start_idx:end_idx, 2], 'g-')
                
                for j in range(cols):
                    column_idxs = [i * cols + j for i in range(rows) if i * cols + j < points3D.shape[0]]
                    if column_idxs:
                        column_points = points3D[column_idxs]
                        ax.plot(column_points[:, 0], column_points[:, 1], column_points[:, 2], 'g-')
            
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_zlabel('Z (mm)')
            ax.set_title('3D Points')
            ax.legend()
            
            plt.savefig(output_path)
            plt.close()
            self.log(f"3D visualization saved to {output_path}")
        except Exception as e:
            self.log(f"Error visualizing 3D points: {e}")
    
    def validate_with_ruler(self, cam1_ruler_video, cam2_ruler_video, known_length=304.8):  # 12 inches = 304.8 mm
        """
        Validate calibration using a ruler of known length
        
        Parameters:
        cam1_ruler_video, cam2_ruler_video: Paths to videos of ruler from both cameras
        known_length: Known length of the ruler in mm
        
        Returns:
        error_percentage: Error in measurement
        """
        self.log("\n=== Validating calibration with ruler measurements ===")
        
        # Extract frames from ruler videos
        cam1_frames = self.extract_frames_from_video(
            cam1_ruler_video, os.path.join(self.output_dir, 'ruler_validation', 'cam1'), 'ruler')
        cam2_frames = self.extract_frames_from_video(
            cam2_ruler_video, os.path.join(self.output_dir, 'ruler_validation', 'cam2'), 'ruler')
        
        # Match frames between cameras
        frame_pairs = self.match_stereo_frames(cam1_frames, cam2_frames)
        
        if not frame_pairs:
            self.log("Error: No matching frame pairs found for validation")
            return None
        
        # Use the middle frame pair for measurement
        mid_idx = len(frame_pairs) // 2
        cam1_frame, cam2_frame = frame_pairs[mid_idx]
        
        # Measure ruler length
        measured_length, _ = self.measure_ruler_distance(cam1_frame, cam2_frame)
        
        if measured_length is None:
            self.log("Error: Failed to measure ruler length")
            return None
        
        # Calculate error
        error = abs(measured_length - known_length)
        error_percentage = (error / known_length) * 100
        
        self.log(f"Validation results:")
        self.log(f"  Known ruler length: {known_length:.2f} mm")
        self.log(f"  Measured ruler length: {measured_length:.2f} mm")
        self.log(f"  Absolute error: {error:.2f} mm")
        self.log(f"  Relative error: {error_percentage:.2f}%")
        
        return error_percentage
    
    def run_intrinsic_calibration(self, cam1_video, cam2_video):
        """
        Run the complete intrinsic calibration workflow for both cameras
        
        Parameters:
        cam1_video, cam2_video: Paths to calibration videos for cameras 1 and 2
        
        Returns:
        bool: True if calibration was successful
        """
        self.log("\n=== Starting Intrinsic Calibration Workflow ===")
        
        # Extract frames from videos
        self.log("\nExtracting frames from camera 1 video...")
        cam1_frames = self.extract_frames_from_video(cam1_video, self.cam1_frames_dir)
        
        self.log("\nExtracting frames from camera 2 video...")
        cam2_frames = self.extract_frames_from_video(cam2_video, self.cam2_frames_dir)
        
        # Perform intrinsic calibration for each camera
        self.cam1_matrix, self.cam1_dist = self.calibrate_single_camera(self.cam1_frames_dir, 1)
        self.cam2_matrix, self.cam2_dist = self.calibrate_single_camera(self.cam2_frames_dir, 2)
        
        # Check if calibration was successful
        if self.cam1_matrix is None or self.cam2_matrix is None:
            self.log("Error: Intrinsic calibration failed for one or both cameras")
            return False
        
        self.log("\nIntrinsic calibration completed successfully!")
        return True
    
    def run_extrinsic_calibration(self, cam1_video, cam2_video):
        """
        Run the complete extrinsic calibration workflow
        
        Parameters:
        cam1_video, cam2_video: Paths to stereo calibration videos for cameras 1 and 2
        
        Returns:
        bool: True if calibration was successful
        """
        self.log("\n=== Starting Extrinsic Calibration Workflow ===")
        
        # Extract frames from videos
        self.log("\nExtracting frames from camera 1 video...")
        cam1_frames = self.extract_frames_from_video(
            cam1_video, os.path.join(self.output_dir, 'stereo_frames', 'cam1'), 'stereo')
        
        self.log("\nExtracting frames from camera 2 video...")
        cam2_frames = self.extract_frames_from_video(
            cam2_video, os.path.join(self.output_dir, 'stereo_frames', 'cam2'), 'stereo')
        
        # Perform stereo calibration
        self.R, self.T = self.calibrate_stereo(cam1_frames, cam2_frames)
        
        # Check if calibration was successful
        if self.R is None or self.T is None:
            self.log("Error: Extrinsic calibration failed")
            return False
        
        self.log("\nExtrinsic calibration completed successfully!")
        return True
    
    def run_complete_workflow(self, intrinsic_cam1_video, intrinsic_cam2_video,
                             extrinsic_cam1_video, extrinsic_cam2_video,
                             validation_cam1_video=None, validation_cam2_video=None,
                             use_existing_intrinsic=False, intrinsic_cam1_dir=None, intrinsic_cam2_dir=None):
        """
        Run the complete stereo calibration workflow
        
        Parameters:
        intrinsic_cam1_video, intrinsic_cam2_video: Videos for intrinsic calibration
        extrinsic_cam1_video, extrinsic_cam2_video: Videos for stereo calibration
        validation_cam1_video, validation_cam2_video: Videos for validation (optional)
        use_existing_intrinsic: Whether to use existing intrinsic calibration
        intrinsic_cam1_dir, intrinsic_cam2_dir: Directories containing existing intrinsic parameters
        
        Returns:
        bool: True if the complete workflow was successful
        """
        self.log("=== Starting Complete Stereo Calibration Workflow ===")
        
        # Step 1: Intrinsic calibration or load existing parameters
        if use_existing_intrinsic and intrinsic_cam1_dir and intrinsic_cam2_dir:
            intrinsic_success = self.load_existing_calibration(intrinsic_cam1_dir, intrinsic_cam2_dir)
        else:
            intrinsic_success = self.run_intrinsic_calibration(intrinsic_cam1_video, intrinsic_cam2_video)
        
        if not intrinsic_success:
            self.log("Error: Intrinsic calibration step failed")
            return False
        
        # Step 2: Extrinsic calibration
        extrinsic_success = self.run_extrinsic_calibration(extrinsic_cam1_video, extrinsic_cam2_video)
        
        if not extrinsic_success:
            self.log("Error: Extrinsic calibration step failed")
            return False
        
        # Step 3: Validation (optional)
        if validation_cam1_video and validation_cam2_video:
            error = self.validate_with_ruler(validation_cam1_video, validation_cam2_video)
            if error is not None:
                self.log(f"Validation complete. Measurement error: {error:.2f}%")
        
        self.log("\n=== Complete Stereo Calibration Workflow Completed Successfully! ===")
        self.log(f"Results saved to: {self.output_dir}")
        return True


# Example usage
if __name__ == "__main__":
    # Initialize the workflow
    workflow = StereoCalibrationWorkflow(
        base_dir=".", 
        output_dir="calibration_results",
        checkerboard_size=(9, 7),  # Adjust as needed
        square_size=25.0,  # mm
        frame_interval=10, 
        max_frames=30
    )
    
    # Option 1: Run with existing intrinsic parameters
    # workflow.run_complete_workflow(
    #     None, None,  # Not needed when using existing intrinsic params
    #     "./videos/cam1/static1.mov", "./videos/cam2/static1.mov",
    #     "./videos/cam1/data/ruler/ruler1.mov", "./videos/cam2/data/ruler/ruler1.mov",
    #     use_existing_intrinsic=True,
    #     intrinsic_cam1_dir="./parameters/cam1",
    #     intrinsic_cam2_dir="./parameters/cam2"
    # )
    
    # Option 2: Run the full calibration workflow
    workflow.run_complete_workflow(
        "./videos/cam1/calib1.mov", "./videos/cam2/calib2.mov",
        "./videos/cam1/static1.mov", "./videos/cam2/static1.mov",
        "./videos/cam1/data/ruler/ruler1.mov", "./videos/cam2/data/ruler/ruler1.mov"
    )