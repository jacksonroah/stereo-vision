#!/usr/bin/env python3
"""
Improved Distance Measurement Tool
---------------------------------
Uses calibrated stereo cameras to measure distances between points
and provides visualization and validation utilities.
"""

import numpy as np
import cv2
import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
import json

class StereoCameraSystem:
    """Class to manage a calibrated stereo camera system"""
    
    def __init__(self, calib_dir, stereo_dir):
        """
        Initialize the stereo camera system with calibration parameters
        
        Parameters:
        calib_dir: Directory containing intrinsic calibration files
        stereo_dir: Directory containing stereo calibration files
        """
        self.calib_dir = calib_dir
        self.stereo_dir = stereo_dir
        
        # Load intrinsic parameters
        self.cam1_matrix, self.cam1_dist = self._load_intrinsic_params(calib_dir, 1)
        self.cam2_matrix, self.cam2_dist = self._load_intrinsic_params(calib_dir, 2)
        
        # Load extrinsic parameters
        success = self._load_extrinsic_params(stereo_dir)
        
        if success and self.cam1_matrix is not None and self.cam2_matrix is not None:
            # Create projection matrices
            self.P1 = self.cam1_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
            self.P2 = self.cam2_matrix @ np.hstack((self.R, self.T))
            
            # Compute rectification parameters
            if hasattr(self, 'R') and hasattr(self, 'T'):
                self._compute_rectification()
            
            print("Stereo camera system initialized successfully")
        else:
            print("Error: Could not initialize stereo camera system")
            
    def _load_intrinsic_params(self, calib_dir, camera_num):
        """Load camera matrix and distortion coefficients"""
        matrix_file = os.path.join(calib_dir, f'camera_{camera_num}_matrix.txt')
        dist_file = os.path.join(calib_dir, f'camera_{camera_num}_distortion.txt')
        
        if os.path.exists(matrix_file) and os.path.exists(dist_file):
            camera_matrix = np.loadtxt(matrix_file)
            dist_coeffs = np.loadtxt(dist_file)
            print(f"Loaded calibration for camera {camera_num}")
            return camera_matrix, dist_coeffs
        else:
            print(f"Error: Calibration files for camera {camera_num} not found")
            return None, None
    
    def _load_extrinsic_params(self, stereo_dir):
        """Load stereo camera extrinsic parameters"""
        R_path = os.path.join(stereo_dir, 'stereo_rotation_matrix.txt')
        T_path = os.path.join(stereo_dir, 'stereo_translation_vector.txt')
        
        if os.path.exists(R_path) and os.path.exists(T_path):
            self.R = np.loadtxt(R_path)
            self.T = np.loadtxt(T_path).reshape(3, 1)
            print("Loaded stereo extrinsic parameters")
            return True
        else:
            print("Error: Stereo extrinsic parameters not found")
            return False
    
    def _compute_rectification(self):
        """Compute stereo rectification parameters"""
        # We'll need an image size for this - use a default value if not available
        dummy_size = (1920, 1080)  # Default size, adjust based on your camera
        
        self.R1, self.R2, self.P1_rect, self.P2_rect, self.Q, self.roi1, self.roi2 = cv2.stereoRectify(
            self.cam1_matrix, self.cam1_dist,
            self.cam2_matrix, self.cam2_dist,
            dummy_size, self.R, self.T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0.9)
    
    def is_calibrated(self):
        """Check if the system is properly calibrated"""
        return (hasattr(self, 'cam1_matrix') and 
                hasattr(self, 'cam2_matrix') and 
                hasattr(self, 'R') and 
                hasattr(self, 'T') and 
                hasattr(self, 'P1') and 
                hasattr(self, 'P2'))
    
    def undistort_images(self, img1, img2):
        """
        Undistort a pair of stereo images
        
        Parameters:
        img1, img2: Images from cameras 1 and 2
        
        Returns:
        (undistorted1, undistorted2)
        """
        if not self.is_calibrated():
            print("Error: System not calibrated")
            return None, None
        
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Get optimal new camera matrices
        newcameramtx1, roi1 = cv2.getOptimalNewCameraMatrix(
            self.cam1_matrix, self.cam1_dist, (w1, h1), 1, (w1, h1))
        newcameramtx2, roi2 = cv2.getOptimalNewCameraMatrix(
            self.cam2_matrix, self.cam2_dist, (w2, h2), 1, (w2, h2))
        
        # Undistort
        undist1 = cv2.undistort(img1, self.cam1_matrix, self.cam1_dist, None, newcameramtx1)
        undist2 = cv2.undistort(img2, self.cam2_matrix, self.cam2_dist, None, newcameramtx2)
        
        return undist1, undist2
    
    def rectify_images(self, img1, img2):
        """
        Rectify a pair of stereo images
        
        Parameters:
        img1, img2: Images from cameras 1 and 2
        
        Returns:
        (rectified1, rectified2)
        """
        if not self.is_calibrated() or not hasattr(self, 'R1'):
            print("Error: System not calibrated or rectification not computed")
            return None, None
        
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Compute rectification maps
        map1x, map1y = cv2.initUndistortRectifyMap(
            self.cam1_matrix, self.cam1_dist, self.R1, self.P1_rect, (w1, h1), cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(
            self.cam2_matrix, self.cam2_dist, self.R2, self.P2_rect, (w2, h2), cv2.CV_32FC1)
        
        # Apply rectification
        rect1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
        rect2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)
        
        return rect1, rect2
    
    def triangulate_points(self, points1, points2, undistort=True):
        """
        Triangulate 3D points from 2D point correspondences
        
        Parameters:
        points1, points2: 2D points in cameras 1 and 2 (N x 2 arrays)
        undistort: Whether to undistort the points before triangulation
        
        Returns:
        points3D: Triangulated 3D points
        """
        if not self.is_calibrated():
            print("Error: System not calibrated")
            return None
        
        # Convert points to numpy arrays if needed
        points1 = np.array(points1, dtype=np.float32)
        points2 = np.array(points2, dtype=np.float32)
        
        # Reshape if needed
        if points1.ndim == 1:
            points1 = points1.reshape(1, -1)
        if points2.ndim == 1:
            points2 = points2.reshape(1, -1)
        
        # Ensure points are the right shape
        if points1.shape[1] != 2 or points2.shape[1] != 2:
            print("Error: Points should be Nx2 arrays")
            return None
        
        # Undistort the points if requested
        if undistort:
            points1_undist = cv2.undistortPoints(
                points1.reshape(-1, 1, 2), self.cam1_matrix, self.cam1_dist, P=self.cam1_matrix)
            points2_undist = cv2.undistortPoints(
                points2.reshape(-1, 1, 2), self.cam2_matrix, self.cam2_dist, P=self.cam2_matrix)
            
            points1 = points1_undist.reshape(-1, 2)
            points2 = points2_undist.reshape(-1, 2)
        
        # Reshape for triangulation
        points1_t = points1.T
        points2_t = points2.T
        
        # Triangulate 3D points
        points4D = cv2.triangulatePoints(self.P1, self.P2, points1_t, points2_t)
        points3D = (points4D[:3] / points4D[3]).T
        
        return points3D
    
    def measure_distance(self, points3D_1, points3D_2=None):
        """
        Measure distance between 3D points or from camera to a point
        
        Parameters:
        points3D_1: First 3D point or array of points
        points3D_2: Second 3D point (optional)
        
        Returns:
        distance: Measured distance
        """
        if points3D_2 is None:
            # Measure distance from origin (camera 1) to point
            return np.linalg.norm(points3D_1, axis=1 if points3D_1.ndim > 1 else 0)
        else:
            # Measure distance between two points
            return np.linalg.norm(points3D_2 - points3D_1, axis=1 if points3D_1.ndim > 1 else 0)

class DistanceMeasurementTool:
    """Tool for measuring distances in stereo camera setups"""
    
    def __init__(self, stereo_system):
        """
        Initialize the distance measurement tool
        
        Parameters:
        stereo_system: StereoCameraSystem object
        """
        self.stereo_system = stereo_system
    
    def select_points(self, image, window_name="Select Points", num_points=1, 
                     point_labels=None, instructions=None, point_color=(0, 0, 255)):
        """
        Select points in an image using mouse clicks
        
        Parameters:
        image: Image to select points in
        window_name: Name for the display window
        num_points: Number of points to select
        point_labels: Optional labels for each point
        instructions: Optional instructions to display
        point_color: Color for the point markers
        
        Returns:
        (success, points): Boolean success flag and selected points
        """
        # Set default point labels if not provided
        if point_labels is None:
            point_labels = [f"Point {i+1}" for i in range(num_points)]
        
        # Create a copy of the image for drawing
        img_display = image.copy()
        
        # Variables to store selected points
        points = []
        current_point = 0
        
        # Mouse callback function
        def mouse_callback(event, x, y, flags, param):
            nonlocal current_point, img_display, points
            
            # Draw instructions if provided
            if instructions is not None:
                cv2.putText(img_display, instructions, (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Left button click to select a point
            if event == cv2.EVENT_LBUTTONDOWN and current_point < num_points:
                # Add the point
                points.append((x, y))
                
                # Draw the point and label
                cv2.circle(img_display, (x, y), 5, point_color, -1)
                cv2.putText(img_display, point_labels[current_point], (x + 10, y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, point_color, 2)
                
                current_point += 1
                cv2.imshow(window_name, img_display)
                
                # If all points are selected, wait for key press
                if current_point >= num_points:
                    print("All points selected. Press any key to continue.")
        
        # Create a window and set mouse callback
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(image.shape[1], 1280), min(image.shape[0], 720))
        cv2.setMouseCallback(window_name, mouse_callback)
        
        # Initial instructions
        if instructions is None:
            instructions = "Click to select points. Press 'r' to reset, 'q' to quit."
        
        img_with_text = img_display.copy()
        cv2.putText(img_with_text, instructions, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(window_name, img_with_text)
        
        # Main loop
        while current_point < num_points:
            key = cv2.waitKey(1) & 0xFF
            
            # Quit
            if key == ord('q'):
                cv2.destroyWindow(window_name)
                return False, None
            
            # Reset
            if key == ord('r'):
                points = []
                current_point = 0
                img_display = image.copy()
                cv2.putText(img_display, instructions, (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow(window_name, img_display)
        
        # Wait for a key press and then close window
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)
        
        return True, np.array(points, dtype=np.float32)
    
    def measure_point_to_point_distance(self, img1, img2, output_dir=None, 
                                       known_distance=None, point_labels=None,
                                       create_report=True):
        """
        Measure the distance between two manually selected points in stereo images
        
        Parameters:
        img1, img2: Images from cameras 1 and 2
        output_dir: Directory to save output files
        known_distance: Optional known distance for validation
        point_labels: Optional labels for the points
        create_report: Whether to create a PDF report
        
        Returns:
        results: Dictionary with measurement results
        """
        if output_dir is None:
            output_dir = os.getcwd()
        os.makedirs(output_dir, exist_ok=True)
        
        # Default point labels
        if point_labels is None:
            point_labels = ["Point A", "Point B"]
        
        # Undistort the images
        undist1, undist2 = self.stereo_system.undistort_images(img1, img2)
        
        if undist1 is None or undist2 is None:
            print("Error: Failed to undistort images")
            return None
        
        # Save undistorted images
        cv2.imwrite(os.path.join(output_dir, "undistorted_cam1.png"), undist1)
        cv2.imwrite(os.path.join(output_dir, "undistorted_cam2.png"), undist2)
        
        # Select first point in both images
        print("\nSelect the first point in Camera 1 image:")
        ret1_1, point1_1 = self.select_points(
            undist1, "Camera 1 - Select Point A", 1, [point_labels[0]],
            "Click to select Point A. Press 'r' to reset, 'q' to quit.")
        
        if not ret1_1:
            print("Point selection canceled")
            return None
        
        print("\nSelect the same point in Camera 2 image:")
        ret1_2, point1_2 = self.select_points(
            undist2, "Camera 2 - Select Point A", 1, [point_labels[0]],
            "Click to select the SAME Point A. Press 'r' to reset, 'q' to quit.")
        
        if not ret1_2:
            print("Point selection canceled")
            return None
        
        # Select second point in both images
        print("\nSelect the second point in Camera 1 image:")
        ret2_1, point2_1 = self.select_points(
            undist1, "Camera 1 - Select Point B", 1, [point_labels[1]],
            "Click to select Point B. Press 'r' to reset, 'q' to quit.")
        
        if not ret2_1:
            print("Point selection canceled")
            return None
        
        print("\nSelect the same point in Camera 2 image:")
        ret2_2, point2_2 = self.select_points(
            undist2, "Camera 2 - Select Point B", 1, [point_labels[1]],
            "Click to select the SAME Point B. Press 'r' to reset, 'q' to quit.")
        
        if not ret2_2:
            print("Point selection canceled")
            return None
        
        # Prepare points for triangulation
        points1 = np.vstack([point1_1, point2_1])
        points2 = np.vstack([point1_2, point2_2])
        
        # Triangulate 3D points
        points3D = self.stereo_system.triangulate_points(points1, points2)
        
        if points3D is None:
            print("Error: Failed to triangulate points")
            return None
        
        # Calculate 3D distance
        distance = self.stereo_system.measure_distance(points3D[0], points3D[1])
        
        # Calculate positions relative to camera 1
        distances_from_camera = self.stereo_system.measure_distance(points3D)
        
        # Create results dictionary
        results = {
            "point_names": point_labels,
            "point3D_A": points3D[0].tolist(),
            "point3D_B": points3D[1].tolist(),
            "distance_mm": float(distance),
            "distance_from_camera_A_mm": float(distances_from_camera[0]),
            "distance_from_camera_B_mm": float(distances_from_camera[1])
        }
        
        # If known distance is provided, calculate error
        if known_distance is not None:
            error = abs(distance - known_distance)
            error_percent = (error / known_distance) * 100
            results["known_distance_mm"] = float(known_distance)
            results["absolute_error_mm"] = float(error)
            results["relative_error_percent"] = float(error_percent)
        
        # Print results
        print("\nMeasurement Results:")
        print(f"3D coordinates of {point_labels[0]}: {points3D[0]}")
        print(f"3D coordinates of {point_labels[1]}: {points3D[1]}")
        print(f"Distance between points: {distance:.2f} mm")
        print(f"Distance from camera to {point_labels[0]}: {distances_from_camera[0]:.2f} mm")
        print(f"Distance from camera to {point_labels[1]}: {distances_from_camera[1]:.2f} mm")
        
        if known_distance is not None:
            print(f"Known distance: {known_distance:.2f} mm")
            print(f"Absolute error: {error:.2f} mm")
            print(f"Relative error: {error_percent:.2f}%")
        
        # Save results to JSON
        results_file = os.path.join(output_dir, "distance_measurement_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        # Draw points on images for visualization
        img1_points = undist1.copy()
        img2_points = undist2.copy()
        
        for i, (pt1, pt2) in enumerate(zip(points1, points2)):
            cv2.circle(img1_points, (int(pt1[0]), int(pt1[1])), 5, (0, 0, 255), -1)
            cv2.putText(img1_points, point_labels[i], (int(pt1[0]) + 10, int(pt1[1]) + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.circle(img2_points, (int(pt2[0]), int(pt2[1])), 5, (0, 0, 255), -1)
            cv2.putText(img2_points, point_labels[i], (int(pt2[0]) + 10, int(pt2[1]) + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.imwrite(os.path.join(output_dir, "points_cam1.png"), img1_points)
        cv2.imwrite(os.path.join(output_dir, "points_cam2.png"), img2_points)
        
        # Create visualization
        self._visualize_distance_measurement(
            points3D, distance, distances_from_camera, point_labels,
            os.path.join(output_dir, "distance_measurement_3d.png")
        )
        
        # Create PDF report if requested
        if create_report:
            self._create_measurement_report(
                img1, img2, undist1, undist2, img1_points, img2_points,
                points3D, distance, distances_from_camera, results,
                os.path.join(output_dir, "distance_measurement_report.pdf")
            )
        
        return results
    
    def validate_with_known_distances(self, img1, img2, known_distances, point_labels=None, output_dir=None):
        """
        Validate the stereo system with multiple known distances
        
        Parameters:
        img1, img2: Images from cameras 1 and 2
        known_distances: List of (point_name, distance_mm) tuples
        point_labels: Optional list of point names
        output_dir: Directory to save output files
        
        Returns:
        validation_results: Dictionary with validation results
        """
        if output_dir is None:
            output_dir = os.getcwd()
        os.makedirs(output_dir, exist_ok=True)
        
        # Use provided point labels or generate default ones
        if point_labels is None:
            point_labels = [f"Point {i+1}" for i in range(len(known_distances) + 1)]
        
        # Undistort the images
        undist1, undist2 = self.stereo_system.undistort_images(img1, img2)
        
        if undist1 is None or undist2 is None:
            print("Error: Failed to undistort images")
            return None
        
        # Select reference point (origin) in both images
        print("\nSelect the reference point (origin) in Camera 1 image:")
        ret0_1, point0_1 = self.select_points(
            undist1, "Camera 1 - Select Reference Point", 1, [point_labels[0]],
            "Click to select the reference point. Press 'r' to reset, 'q' to quit.")
        
        if not ret0_1:
            print("Point selection canceled")
            return None
        
        print("\nSelect the same reference point in Camera 2 image:")
        ret0_2, point0_2 = self.select_points(
            undist2, "Camera 2 - Select Reference Point", 1, [point_labels[0]],
            "Click to select the SAME reference point. Press 'r' to reset, 'q' to quit.")
        
        if not ret0_2:
            print("Point selection canceled")
            return None
        
        # Triangulate reference point
        ref_point3D = self.stereo_system.triangulate_points(point0_1, point0_2)[0]
        
        # Initialize arrays for all points
        points1 = [point0_1[0]]
        points2 = [point0_2[0]]
        point3D_list = [ref_point3D]
        
        # Measure each known distance
        results = []
        
        for i, (point_name, known_dist) in enumerate(known_distances):
            point_idx = i + 1  # Index in point_labels
            
            print(f"\nSelect point '{point_name}' in Camera 1 image:")
            ret_1, pt_1 = self.select_points(
                undist1, f"Camera 1 - Select {point_name}", 1, [point_name],
                f"Click to select {point_name}. Press 'r' to reset, 'q' to quit.")
            
            if not ret_1:
                print("Point selection canceled")
                continue
            
            print(f"\nSelect the same point '{point_name}' in Camera 2 image:")
            ret_2, pt_2 = self.select_points(
                undist2, f"Camera 2 - Select {point_name}", 1, [point_name],
                f"Click to select the SAME {point_name}. Press 'r' to reset, 'q' to quit.")
            
            if not ret_2:
                print("Point selection canceled")
                continue
            
            # Add points to arrays
            points1.append(pt_1[0])
            points2.append(pt_2[0])
            
            # Triangulate 3D point
            curr_point3D = self.stereo_system.triangulate_points(pt_1, pt_2)[0]
            point3D_list.append(curr_point3D)
            
            # Calculate measured distance
            measured_dist = self.stereo_system.measure_distance(ref_point3D, curr_point3D)
            
            # Calculate error
            abs_error = abs(measured_dist - known_dist)
            rel_error = (abs_error / known_dist) * 100
            
            result = {
                "point_name": point_name,
                "point3D": curr_point3D.tolist(),
                "known_distance_mm": float(known_dist),
                "measured_distance_mm": float(measured_dist),
                "absolute_error_mm": float(abs_error),
                "relative_error_percent": float(rel_error)
            }
            
            results.append(result)
            
            print(f"\nValidation for {point_name}:")
            print(f"Known distance: {known_dist:.2f} mm")
            print(f"Measured distance: {measured_dist:.2f} mm")
            print(f"Absolute error: {abs_error:.2f} mm")
            print(f"Relative error: {rel_error:.2f}%")
        
        # Convert to numpy arrays for easier handling
        points1 = np.array(points1)
        points2 = np.array(points2)
        point3D_list = np.array(point3D_list)
        
        # Calculate average error
        if results:
            avg_abs_error = sum(r["absolute_error_mm"] for r in results) / len(results)
            avg_rel_error = sum(r["relative_error_percent"] for r in results) / len(results)
            
            print("\nValidation Summary:")
            print(f"Average absolute error: {avg_abs_error:.2f} mm")
            print(f"Average relative error: {avg_rel_error:.2f}%")
        else:
            avg_abs_error = None
            avg_rel_error = None
            print("\nNo valid measurements were completed.")
        
        # Create validation results dictionary
        validation_results = {
            "reference_point": point_labels[0],
            "reference_point3D": ref_point3D.tolist(),
            "measurements": results,
            "average_absolute_error_mm": avg_abs_error,
            "average_relative_error_percent": avg_rel_error
        }
        
        # Save results to JSON
        results_file = os.path.join(output_dir, "validation_results.json")
        with open(results_file, 'w') as f:
            json.dump(validation_results, f, indent=4)
        
        # Draw points on images for visualization
        img1_points = undist1.copy()
        img2_points = undist2.copy()
        
        # Draw reference point
        cv2.circle(img1_points, (int(points1[0][0]), int(points1[0][1])), 5, (0, 255, 0), -1)
        cv2.putText(img1_points, point_labels[0], (int(points1[0][0]) + 10, int(points1[0][1]) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.circle(img2_points, (int(points2[0][0]), int(points2[0][1])), 5, (0, 255, 0), -1)
        cv2.putText(img2_points, point_labels[0], (int(points2[0][0]) + 10, int(points2[0][1]) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Draw the other points
        for i in range(1, len(points1)):
            cv2.circle(img1_points, (int(points1[i][0]), int(points1[i][1])), 5, (0, 0, 255), -1)
            cv2.putText(img1_points, known_distances[i-1][0], (int(points1[i][0]) + 10, int(points1[i][1]) + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.circle(img2_points, (int(points2[i][0]), int(points2[i][1])), 5, (0, 0, 255), -1)
            cv2.putText(img2_points, known_distances[i-1][0], (int(points2[i][0]) + 10, int(points2[i][1]) + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.imwrite(os.path.join(output_dir, "validation_points_cam1.png"), img1_points)
        cv2.imwrite(os.path.join(output_dir, "validation_points_cam2.png"), img2_points)
        
        # Create 3D visualization
        self._visualize_validation_measurements(
            point3D_list, validation_results,
            os.path.join(output_dir, "validation_measurement_3d.png")
        )
        
        # Create PDF report
        self._create_validation_report(
            img1, img2, undist1, undist2, img1_points, img2_points,
            point3D_list, validation_results,
            os.path.join(output_dir, "validation_report.pdf")
        )
        
        return validation_results
    
    def _visualize_distance_measurement(self, points3D, distance, distances_from_camera, 
                                      point_labels, output_path):
        """
        Visualize 3D points and the distance between them
        
        Parameters:
        points3D: 3D points to visualize
        distance: Distance between points
        distances_from_camera: Distances from camera to each point
        point_labels: Labels for the points
        output_path: Path to save the visualization
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot camera positions
        ax.scatter([0], [0], [0], c='r', marker='o', s=100, label='Camera 1')
        
        if hasattr(self.stereo_system, 'T'):
            # Calculate Camera 2 position from rotation and translation
            cam2_pos = -self.stereo_system.R.T @ self.stereo_system.T.flatten()
            ax.scatter([cam2_pos[0]], [cam2_pos[1]], [cam2_pos[2]], c='b', marker='o', s=100, label='Camera 2')
        
        # Plot the 3D points
        ax.scatter(points3D[:, 0], points3D[:, 1], points3D[:, 2], c='g', marker='o', s=50)
        
        # Connect the points with a line
        ax.plot(points3D[:, 0], points3D[:, 1], points3D[:, 2], 'g-', linewidth=2)
        
        # Add text labels for the points
        for i, point in enumerate(points3D):
            ax.text(point[0], point[1], point[2], point_labels[i], color='black', fontsize=10)
        
        # Add distance label
        if len(points3D) >= 2:
            # Calculate midpoint for label placement
            midpoint = (points3D[0] + points3D[1]) / 2
            ax.text(midpoint[0], midpoint[1], midpoint[2], 
                   f"{distance:.2f} mm", color='blue', fontsize=12, 
                   horizontalalignment='center', verticalalignment='center')
        
        # Set axis labels and title
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title('3D Distance Measurement')
        
        # Add legend
        ax.legend()
        
        # Auto-scale axes to make visualization better
        max_range = np.max([
            np.max(points3D[:, 0]) - np.min(points3D[:, 0]),
            np.max(points3D[:, 1]) - np.min(points3D[:, 1]),
            np.max(points3D[:, 2]) - np.min(points3D[:, 2])
        ])
        
        mid_x = (np.max(points3D[:, 0]) + np.min(points3D[:, 0])) * 0.5
        mid_y = (np.max(points3D[:, 1]) + np.min(points3D[:, 1])) * 0.5
        mid_z = (np.max(points3D[:, 2]) + np.min(points3D[:, 2])) * 0.5
        
        ax.set_xlim(mid_x - max_range*0.6, mid_x + max_range*0.6)
        ax.set_ylim(mid_y - max_range*0.6, mid_y + max_range*0.6)
        ax.set_zlim(mid_z - max_range*0.6, mid_z + max_range*0.6)
        
        # Save the visualization
        plt.savefig(output_path)
        plt.close(fig)
        print(f"3D visualization saved to {output_path}")
    
    def _visualize_validation_measurements(self, points3D, validation_results, output_path):
        """
        Visualize 3D points for validation measurements
        
        Parameters:
        points3D: 3D points including reference point
        validation_results: Validation results dictionary
        output_path: Path to save the visualization
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot camera positions
        ax.scatter([0], [0], [0], c='r', marker='o', s=100, label='Camera 1')
        
        if hasattr(self.stereo_system, 'T'):
            cam2_pos = -self.stereo_system.R.T @ self.stereo_system.T.flatten()
            ax.scatter([cam2_pos[0]], [cam2_pos[1]], [cam2_pos[2]], c='b', marker='o', s=100, label='Camera 2')
        
        # Plot reference point
        ref_point = points3D[0]
        ax.scatter([ref_point[0]], [ref_point[1]], [ref_point[2]], c='g', marker='o', s=80, label='Reference Point')
        ax.text(ref_point[0], ref_point[1], ref_point[2], validation_results["reference_point"], 
               color='black', fontsize=10)
        
        # Plot measurement points
        for i, measurement in enumerate(validation_results["measurements"]):
            point = np.array(measurement["point3D"])
            point_name = measurement["point_name"]
            
            # Plot the point
            ax.scatter([point[0]], [point[1]], [point[2]], c='m', marker='o', s=50)
            ax.text(point[0], point[1], point[2], point_name, color='black', fontsize=10)
            
            # Connect to reference point with a line
            ax.plot([ref_point[0], point[0]], [ref_point[1], point[1]], [ref_point[2], point[2]], 'g--', linewidth=1)
            
            # Add distance label
            midpoint = (ref_point + point) / 2
            measured = measurement["measured_distance_mm"]
            known = measurement["known_distance_mm"]
            error_pct = measurement["relative_error_percent"]
            
            label = f"{point_name}: {measured:.1f}mm\n(Known: {known:.1f}mm, Err: {error_pct:.1f}%)"
            ax.text(midpoint[0], midpoint[1], midpoint[2], label, color='blue', fontsize=9)
        
        # Set axis labels and title
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title('Validation Measurements')
        
        # Add legend
        ax.legend()
        
        # Auto-scale axes
        points = np.vstack([points3D] + [np.array(m["point3D"]) for m in validation_results["measurements"]])
        max_range = np.max([
            np.max(points[:, 0]) - np.min(points[:, 0]),
            np.max(points[:, 1]) - np.min(points[:, 1]),
            np.max(points[:, 2]) - np.min(points[:, 2])
        ])
        
        mid_x = (np.max(points[:, 0]) + np.min(points[:, 0])) * 0.5
        mid_y = (np.max(points[:, 1]) + np.min(points[:, 1])) * 0.5
        mid_z = (np.max(points[:, 2]) + np.min(points[:, 2])) * 0.5
        
        ax.set_xlim(mid_x - max_range*0.6, mid_x + max_range*0.6)
        ax.set_ylim(mid_y - max_range*0.6, mid_y + max_range*0.6)
        ax.set_zlim(mid_z - max_range*0.6, mid_z + max_range*0.6)
        
        # Save the visualization
        plt.savefig(output_path)
        plt.close(fig)
        print(f"Validation visualization saved to {output_path}")
    
    def _create_measurement_report(self, img1, img2, undist1, undist2, img1_points, img2_points,
                                  points3D, distance, distances_from_camera, results, output_path):
        """Create a PDF report for distance measurement"""
        with PdfPages(output_path) as pdf:
            # Title page
            plt.figure(figsize=(10, 8))
            plt.axis('off')
            plt.text(0.5, 0.5, "Distance Measurement Report", fontsize=20, 
                    horizontalalignment='center', verticalalignment='center')
            plt.text(0.5, 0.4, f"Date: {plt.datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    fontsize=12, horizontalalignment='center')
            pdf.savefig()
            plt.close()
            
            # Original and undistorted images
            plt.figure(figsize=(12, 8))
            plt.suptitle("Original and Undistorted Images", fontsize=14)
            
            plt.subplot(2, 2, 1)
            plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
            plt.title("Original Camera 1")
            plt.axis('off')
            
            plt.subplot(2, 2, 2)
            plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
            plt.title("Original Camera 2")
            plt.axis('off')
            
            plt.subplot(2, 2, 3)
            plt.imshow(cv2.cvtColor(undist1, cv2.COLOR_BGR2RGB))
            plt.title("Undistorted Camera 1")
            plt.axis('off')
            
            plt.subplot(2, 2, 4)
            plt.imshow(cv2.cvtColor(undist2, cv2.COLOR_BGR2RGB))
            plt.title("Undistorted Camera 2")
            plt.axis('off')
            
            pdf.savefig()
            plt.close()
            
            # Points selection
            plt.figure(figsize=(12, 6))
            plt.suptitle("Selected Points", fontsize=14)
            
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(img1_points, cv2.COLOR_BGR2RGB))
            plt.title("Points in Camera 1")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(img2_points, cv2.COLOR_BGR2RGB))
            plt.title("Points in Camera 2")
            plt.axis('off')
            
            pdf.savefig()
            plt.close()
            
            # Create 3D visualization
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot camera positions
            ax.scatter([0], [0], [0], c='r', marker='o', s=100, label='Camera 1')
            
            if hasattr(self.stereo_system, 'T'):
                cam2_pos = -self.stereo_system.R.T @ self.stereo_system.T.flatten()
                ax.scatter([cam2_pos[0]], [cam2_pos[1]], [cam2_pos[2]], c='b', marker='o', s=100, label='Camera 2')
            
            # Plot the 3D points
            ax.scatter(points3D[:, 0], points3D[:, 1], points3D[:, 2], c='g', marker='o', s=50)
            
            # Connect the points with a line
            ax.plot(points3D[:, 0], points3D[:, 1], points3D[:, 2], 'g-', linewidth=2)
            
            # Add text labels for the points
            for i, point in enumerate(points3D):
                ax.text(point[0], point[1], point[2], results["point_names"][i], color='black', fontsize=10)
            
            # Add distance label
            if len(points3D) >= 2:
                midpoint = (points3D[0] + points3D[1]) / 2
                ax.text(midpoint[0], midpoint[1], midpoint[2], 
                       f"{distance:.2f} mm", color='blue', fontsize=12, 
                       horizontalalignment='center', verticalalignment='center')
            
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_zlabel('Z (mm)')
            ax.set_title('3D Distance Measurement')
            ax.legend()
            
            # Auto-scale axes
            max_range = np.max([
                np.max(points3D[:, 0]) - np.min(points3D[:, 0]),
                np.max(points3D[:, 1]) - np.min(points3D[:, 1]),
                np.max(points3D[:, 2]) - np.min(points3D[:, 2])
            ])
            
            mid_x = (np.max(points3D[:, 0]) + np.min(points3D[:, 0])) * 0.5
            mid_y = (np.max(points3D[:, 1]) + np.min(points3D[:, 1])) * 0.5
            mid_z = (np.max(points3D[:, 2]) + np.min(points3D[:, 2])) * 0.5
            
            ax.set_xlim(mid_x - max_range*0.6, mid_x + max_range*0.6)
            ax.set_ylim(mid_y - max_range*0.6, mid_y + max_range*0.6)
            ax.set_zlim(mid_z - max_range*0.6, mid_z + max_range*0.6)
            
            pdf.savefig()
            plt.close()
            
            # Results page
            plt.figure(figsize=(10, 8))
            plt.axis('off')
            plt.text(0.1, 0.95, "Measurement Results", fontsize=16, weight='bold')
            
            # Point coordinates
            plt.text(0.1, 0.85, f"3D Coordinates:", fontsize=12, weight='bold')
            for i, name in enumerate(results["point_names"]):
                point = points3D[i]
                plt.text(0.1, 0.80 - i*0.05, 
                        f"{name}: ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}) mm", 
                        fontsize=10)
            
            # Distance results
            plt.text(0.1, 0.65, "Distance Measurements:", fontsize=12, weight='bold')
            plt.text(0.1, 0.60, f"Distance between points: {distance:.2f} mm", fontsize=10)
            plt.text(0.1, 0.55, f"Distance from camera to {results['point_names'][0]}: {distances_from_camera[0]:.2f} mm", fontsize=10)
            plt.text(0.1, 0.50, f"Distance from camera to {results['point_names'][1]}: {distances_from_camera[1]:.2f} mm", fontsize=10)
            
            # Error analysis if known distance was provided
            if "known_distance_mm" in results:
                plt.text(0.1, 0.40, "Error Analysis:", fontsize=12, weight='bold')
                plt.text(0.1, 0.35, f"Known distance: {results['known_distance_mm']:.2f} mm", fontsize=10)
                plt.text(0.1, 0.30, f"Absolute error: {results['absolute_error_mm']:.2f} mm", fontsize=10)
                plt.text(0.1, 0.25, f"Relative error: {results['relative_error_percent']:.2f}%", fontsize=10)
            
            pdf.savefig()
            plt.close()
        
        print(f"Measurement report saved to {output_path}")
    
    def _create_validation_report(self, img1, img2, undist1, undist2, img1_points, img2_points,
                                 points3D, validation_results, output_path):
        """Create a PDF report for validation measurements"""
        with PdfPages(output_path) as pdf:
            # Title page
            plt.figure(figsize=(10, 8))
            plt.axis('off')
            plt.text(0.5, 0.5, "Validation Measurement Report", fontsize=20, 
                    horizontalalignment='center', verticalalignment='center')
            plt.text(0.5, 0.4, f"Date: {plt.datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    fontsize=12, horizontalalignment='center')
            pdf.savefig()
            plt.close()
            
            # Original and undistorted images
            plt.figure(figsize=(12, 8))
            plt.suptitle("Original and Undistorted Images", fontsize=14)
            
            plt.subplot(2, 2, 1)
            plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
            plt.title("Original Camera 1")
            plt.axis('off')
            
            plt.subplot(2, 2, 2)
            plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
            plt.title("Original Camera 2")
            plt.axis('off')
            
            plt.subplot(2, 2, 3)
            plt.imshow(cv2.cvtColor(undist1, cv2.COLOR_BGR2RGB))
            plt.title("Undistorted Camera 1")
            plt.axis('off')
            
            plt.subplot(2, 2, 4)
            plt.imshow(cv2.cvtColor(undist2, cv2.COLOR_BGR2RGB))
            plt.title("Undistorted Camera 2")
            plt.axis('off')
            
            pdf.savefig()
            plt.close()
            
            # Points selection
            plt.figure(figsize=(12, 6))
            plt.suptitle("Selected Points", fontsize=14)
            
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(img1_points, cv2.COLOR_BGR2RGB))
            plt.title("Points in Camera 1")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(img2_points, cv2.COLOR_BGR2RGB))
            plt.title("Points in Camera 2")
            plt.axis('off')
            
            pdf.savefig()
            plt.close()
            
            # Create 3D visualization for validation
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot camera positions
            ax.scatter([0], [0], [0], c='r', marker='o', s=100, label='Camera 1')
            
            if hasattr(self.stereo_system, 'T'):
                cam2_pos = -self.stereo_system.R.T @ self.stereo_system.T.flatten()
                ax.scatter([cam2_pos[0]], [cam2_pos[1]], [cam2_pos[2]], c='b', marker='o', s=100, label='Camera 2')
            
            # Plot reference point
            ref_point = points3D[0]
            ax.scatter([ref_point[0]], [ref_point[1]], [ref_point[2]], c='g', marker='o', s=80, label='Reference Point')
            ax.text(ref_point[0], ref_point[1], ref_point[2], validation_results["reference_point"], 
                   color='black', fontsize=10)
            
            # Plot measurement points
            for i, measurement in enumerate(validation_results["measurements"]):
                point = np.array(measurement["point3D"])
                point_name = measurement["point_name"]
                
                # Plot the point
                ax.scatter([point[0]], [point[1]], [point[2]], c='m', marker='o', s=50)
                ax.text(point[0], point[1], point[2], point_name, color='black', fontsize=10)
                
                # Connect to reference point with a line
                ax.plot([ref_point[0], point[0]], [ref_point[1], point[1]], [ref_point[2], point[2]], 'g--', linewidth=1)
                
                # Add distance label
                midpoint = (ref_point + point) / 2
                measured = measurement["measured_distance_mm"]
                known = measurement["known_distance_mm"]
                error_pct = measurement["relative_error_percent"]
                
                label = f"{point_name}: {measured:.1f}mm\n(Known: {known:.1f}mm, Err: {error_pct:.1f}%)"
                ax.text(midpoint[0], midpoint[1], midpoint[2], label, color='blue', fontsize=9)
            
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_zlabel('Z (mm)')
            ax.set_title('Validation Measurements')
            ax.legend()
            
            # Auto-scale axes
            all_points = [points3D[0]] + [np.array(m["point3D"]) for m in validation_results["measurements"]]
            points = np.vstack(all_points)
            
            max_range = np.max([
                np.max(points[:, 0]) - np.min(points[:, 0]),
                np.max(points[:, 1]) - np.min(points[:, 1]),
                np.max(points[:, 2]) - np.min(points[:, 2])
            ])
            
            mid_x = (np.max(points[:, 0]) + np.min(points[:, 0])) * 0.5
            mid_y = (np.max(points[:, 1]) + np.min(points[:, 1])) * 0.5
            mid_z = (np.max(points[:, 2]) + np.min(points[:, 2])) * 0.5
            
            ax.set_xlim(mid_x - max_range*0.6, mid_x + max_range*0.6)
            ax.set_ylim(mid_y - max_range*0.6, mid_y + max_range*0.6)
            ax.set_zlim(mid_z - max_range*0.6, mid_z + max_range*0.6)
            
            pdf.savefig()
            plt.close()
            
            # Results table
            plt.figure(figsize=(10, 8))
            plt.axis('off')
            plt.text(0.1, 0.95, "Validation Results", fontsize=16, weight='bold')
            
            # Reference point
            plt.text(0.1, 0.9, f"Reference point: {validation_results['reference_point']}", fontsize=12)
            ref_point3D = validation_results["reference_point3D"]
            plt.text(0.1, 0.85, f"Reference point coordinates: ({ref_point3D[0]:.2f}, {ref_point3D[1]:.2f}, {ref_point3D[2]:.2f}) mm", 
                    fontsize=10)
            
            # Measurement table headers
            y_pos = 0.75
            plt.text(0.1, y_pos, "Point", fontsize=11, weight='bold')
            plt.text(0.25, y_pos, "Known (mm)", fontsize=11, weight='bold')
            plt.text(0.45, y_pos, "Measured (mm)", fontsize=11, weight='bold')
            plt.text(0.65, y_pos, "Abs. Error (mm)", fontsize=11, weight='bold')
            plt.text(0.85, y_pos, "Rel. Error (%)", fontsize=11, weight='bold')
            
            # Measurement results
            y_pos -= 0.05
            for measurement in validation_results["measurements"]:
                y_pos -= 0.05
                plt.text(0.1, y_pos, measurement["point_name"], fontsize=10)
                plt.text(0.25, y_pos, f"{measurement['known_distance_mm']:.2f}", fontsize=10)
                plt.text(0.45, y_pos, f"{measurement['measured_distance_mm']:.2f}", fontsize=10)
                plt.text(0.65, y_pos, f"{measurement['absolute_error_mm']:.2f}", fontsize=10)
                plt.text(0.85, y_pos, f"{measurement['relative_error_percent']:.2f}", fontsize=10)
            
            # Average error
            if "average_absolute_error_mm" in validation_results and validation_results["average_absolute_error_mm"] is not None:
                y_pos -= 0.1
                plt.text(0.1, y_pos, "Average", fontsize=10, weight='bold')
                plt.text(0.65, y_pos, f"{validation_results['average_absolute_error_mm']:.2f}", fontsize=10, weight='bold')
                plt.text(0.85, y_pos, f"{validation_results['average_relative_error_percent']:.2f}", fontsize=10, weight='bold')
            
            pdf.savefig()
            plt.close()
            
            # Error bar chart
            if validation_results["measurements"]:
                plt.figure(figsize=(10, 6))
                plt.suptitle("Measurement Errors", fontsize=14)
                
                # Extract data for plotting
                points = [m["point_name"] for m in validation_results["measurements"]]
                abs_errors = [m["absolute_error_mm"] for m in validation_results["measurements"]]
                rel_errors = [m["relative_error_percent"] for m in validation_results["measurements"]]
                
                # Plot absolute errors
                plt.subplot(1, 2, 1)
                plt.bar(points, abs_errors)
                plt.axhline(y=validation_results["average_absolute_error_mm"], color='r', linestyle='--',
                           label=f'Avg: {validation_results["average_absolute_error_mm"]:.2f} mm')
                plt.ylabel('Absolute Error (mm)')
                plt.title('Absolute Errors')
                plt.legend()
                
                # Plot relative errors
                plt.subplot(1, 2, 2)
                plt.bar(points, rel_errors)
                plt.axhline(y=validation_results["average_relative_error_percent"], color='r', linestyle='--',
                           label=f'Avg: {validation_results["average_relative_error_percent"]:.2f}%')
                plt.ylabel('Relative Error (%)')
                plt.title('Relative Errors')
                plt.legend()
                
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                pdf.savefig()
                plt.close()
        
        print(f"Validation report saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Improved distance measurement tool')
    parser.add_argument('--calib-dir', type=str, required=True,
                       help='Directory containing intrinsic calibration parameters')
    parser.add_argument('--stereo-dir', type=str, required=True,
                       help='Directory containing stereo calibration parameters')
    parser.add_argument('--cam1-image', type=str, required=True,
                       help='Path to image from camera 1 for measurement')
    parser.add_argument('--cam2-image', type=str, required=True,
                       help='Path to image from camera 2 for measurement')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save output files')
    parser.add_argument('--known-distance', type=float, default=None,
                       help='Known distance between points for validation (mm)')
    parser.add_argument('--validation', action='store_true',
                       help='Run validation with multiple known distances')
    parser.add_argument('--known-distances', type=str, default=None,
                       help='File with known distances in format: "label,distance_mm" per line')
    
    args = parser.parse_args()
    
    # Initialize stereo camera system
    stereo_system = StereoCameraSystem(args.calib_dir, args.stereo_dir)
    
    if not stereo_system.is_calibrated():
        print("Error: Stereo camera system not properly calibrated")
        return
    
    # Create measurement tool
    tool = DistanceMeasurementTool(stereo_system)
    
    # Load images
    img1 = cv2.imread(args.cam1_image)
    img2 = cv2.imread(args.cam2_image)
    
    if img1 is None or img2 is None:
        print("Error: Could not read one or both images")
        return
    
    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(os.path.dirname(args.cam1_image), "measurement_results")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Perform measurement or validation
    if args.validation:
        # Parse known distances file
        known_distances = []
        if args.known_distances:
            with open(args.known_distances, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split(',')
                        if len(parts) == 2:
                            label = parts[0].strip()
                            try:
                                distance = float(parts[1].strip())
                                known_distances.append((label, distance))
                            except ValueError:
                                print(f"Warning: Could not parse distance value in line: {line}")
        
        # If no file provided or parsing failed, use command line value
        if not known_distances and args.known_distance is not None:
            known_distances = [("Point B", args.known_distance)]
        
        if not known_distances:
            print("Error: No known distances provided for validation")
            return
        
        # Run validation
        results = tool.validate_with_known_distances(
            img1, img2, known_distances, output_dir=output_dir)
        
        if results:
            print("\nValidation completed successfully!")
            print(f"Average relative error: {results['average_relative_error_percent']:.2f}%")
            print(f"Results saved to {output_dir}")
    else:
        # Single distance measurement
        results = tool.measure_point_to_point_distance(
            img1, img2, output_dir=output_dir, known_distance=args.known_distance)
        
        if results:
            print("\nMeasurement completed successfully!")
            print(f"Measured distance: {results['distance_mm']:.2f} mm")
            if args.known_distance is not None:
                print(f"Relative error: {results['relative_error_percent']:.2f}%")
            print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()