#!/usr/bin/env python3
"""
Distance Measurement Tool

This script provides utilities for measuring distances between points in stereo images,
using a calibrated stereo camera setup.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from mpl_toolkits.mplot3d import Axes3D
from calibration_utils import load_calibration

class DistanceMeasurementTool:
    def __init__(self, calib_dir):
        """
        Initialize the distance measurement tool
        
        Parameters:
        calib_dir: Directory containing calibration parameters
        """
        self.calib_dir = calib_dir
        
        # Load calibration parameters
        self.cam1_matrix, self.cam1_dist = load_calibration(calib_dir, 1)
        self.cam2_matrix, self.cam2_dist = load_calibration(calib_dir, 2)
        
        # Load extrinsic parameters
        R_path = os.path.join(calib_dir, 'stereo_rotation_matrix.txt')
        T_path = os.path.join(calib_dir, 'stereo_translation_vector.txt')
        
        if os.path.exists(R_path) and os.path.exists(T_path):
            self.R = np.loadtxt(R_path)
            self.T = np.loadtxt(T_path).reshape(3, 1)
        else:
            print("Error: Extrinsic parameters not found")
            self.R = None
            self.T = None
            
        # Create projection matrices
        if self.cam1_matrix is not None and self.cam2_matrix is not None and self.R is not None and self.T is not None:
            self.P1 = self.cam1_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
            self.P2 = self.cam2_matrix @ np.hstack((self.R, self.T))
            print("Projection matrices created successfully")
        else:
            self.P1 = None
            self.P2 = None
            print("Error: Could not create projection matrices")
    
    def is_calibrated(self):
        """Check if the system is properly calibrated"""
        return (self.cam1_matrix is not None and self.cam2_matrix is not None and 
                self.R is not None and self.T is not None and 
                self.P1 is not None and self.P2 is not None)
    
    def select_points(self, image_path, num_points=2, point_labels=None):
        """
        Select points in an image using mouse clicks
        
        Parameters:
        image_path: Path to the image
        num_points: Number of points to select
        point_labels: Optional labels for each point
        
        Returns:
        (success, points)
        """
        # Set default point labels if not provided
        if point_labels is None:
            point_labels = [f"Point {i+1}" for i in range(num_points)]
        
        points = []
        current_point = 0
        
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image {image_path}")
            return False, None
        
        # Create a copy for drawing
        img_display = img.copy()
        
        # Create a window
        window_name = "Select Points"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(img.shape[1], 1280), min(img.shape[0], 720))
        
        # Mouse callback function
        def mouse_callback(event, x, y, flags, param):
            nonlocal current_point, img_display
            
            if event == cv2.EVENT_LBUTTONDOWN and current_point < num_points:
                # Add the point
                points.append((x, y))
                
                # Draw the point and label
                cv2.circle(img_display, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(img_display, point_labels[current_point], (x + 10, y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                current_point += 1
                cv2.imshow(window_name, img_display)
                
                # If all points are selected, wait for key press
                if current_point >= num_points:
                    print("All points selected. Press any key to continue.")
        
        # Set mouse callback
        cv2.setMouseCallback(window_name, mouse_callback)
        
        # Display instructions
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
                cv2.destroyAllWindows()
                return False, None
            
            # Reset
            if key == ord('r'):
                points = []
                current_point = 0
                img_display = img.copy()
                cv2.putText(img_display, instructions, (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow(window_name, img_display)
        
        # Wait for a key press and then close windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save the image with points
        img_name = os.path.basename(image_path)
        output_path = os.path.join(os.path.dirname(image_path), f"points_{img_name}")
        cv2.imwrite(output_path, img_display)
        print(f"Image with points saved to {output_path}")
        
        return True, np.array(points, dtype=np.float32)
    
    def triangulate_points(self, points1, points2):
        """
        Triangulate 3D points from 2D point correspondences
        
        Parameters:
        points1, points2: 2D points in cameras 1 and 2
        
        Returns:
        points3D: Triangulated 3D points
        """
        if not self.is_calibrated():
            print("Error: System not calibrated properly")
            return None
        
        # Undistort the points
        points1_undistorted = cv2.undistortPoints(
            points1.reshape(-1, 1, 2), self.cam1_matrix, self.cam1_dist, P=self.cam1_matrix)
        points2_undistorted = cv2.undistortPoints(
            points2.reshape(-1, 1, 2), self.cam2_matrix, self.cam2_dist, P=self.cam2_matrix)
        
        # Reshape for triangulation
        points1_undistorted = points1_undistorted.reshape(-1, 2).T
        points2_undistorted = points2_undistorted.reshape(-1, 2).T
        
        # Triangulate 3D points
        points4D = cv2.triangulatePoints(self.P1, self.P2, points1_undistorted, points2_undistorted)
        points3D = (points4D[:3] / points4D[3]).T
        
        return points3D
    
    def measure_distance(self, cam1_image, cam2_image, output_dir=None, known_distance=None):
        """
        Measure the distance between two manually selected points in a pair of stereo images
        
        Parameters:
        cam1_image, cam2_image: Paths to images from cameras 1 and 2
        output_dir: Directory to save output files (default: same as cam1_image)
        known_distance: Optional known distance between points for error calculation
        
        Returns:
        distance: Measured distance between the points
        """
        if not self.is_calibrated():
            print("Error: System not calibrated properly")
            return None
        
        # Set output directory
        if output_dir is None:
            output_dir = os.path.dirname(cam1_image)
        os.makedirs(output_dir, exist_ok=True)
        
        # Select points in camera 1 image
        print("\nSelect two points in the Camera 1 image:")
        ret1, points1 = self.select_points(cam1_image, 2, ["Point A", "Point B"])
        
        if not ret1:
            print("Point selection canceled for Camera 1")
            return None
        
        # Select points in camera 2 image
        print("\nSelect the same two points in the Camera 2 image:")
        ret2, points2 = self.select_points(cam2_image, 2, ["Point A", "Point B"])
        
        if not ret2:
            print("Point selection canceled for Camera 2")
            return None
        
        # Triangulate 3D points
        points3D = self.triangulate_points(points1, points2)
        
        if points3D is None:
            return None
        
        # Calculate 3D distance
        distance = np.linalg.norm(points3D[1] - points3D[0])
        
        # Calculate positions relative to camera 1
        distances_from_camera = np.linalg.norm(points3D, axis=1)
        
        print("\nMeasurement results:")
        print(f"3D coordinates of Point A: {points3D[0]}")
        print(f"3D coordinates of Point B: {points3D[1]}")
        print(f"Distance between points: {distance:.2f} mm")
        print(f"Distance from camera to Point A: {distances_from_camera[0]:.2f} mm")
        print(f"Distance from camera to Point B: {distances_from_camera[1]:.2f} mm")
        
        # If known distance is provided, calculate error
        if known_distance is not None:
            error = abs(distance - known_distance)
            error_percent = (error / known_distance) * 100
            print(f"Known distance: {known_distance:.2f} mm")
            print(f"Absolute error: {error:.2f} mm")
            print(f"Relative error: {error_percent:.2f}%")
        
        # Visualize the 3D points
        self.visualize_distance_measurement(points3D, os.path.join(output_dir, "distance_measurement_3d.png"))
        
        return distance
    
    def measure_object_dimensions(self, cam1_image, cam2_image, num_points=4, output_dir=None):
        """
        Measure the dimensions of an object by selecting multiple points
        
        Parameters:
        cam1_image, cam2_image: Paths to images from cameras 1 and 2
        num_points: Number of points to select (default: 4 for rectangular object)
        output_dir: Directory to save output files
        
        Returns:
        dimensions: List of measured dimensions
        """
        if not self.is_calibrated():
            print("Error: System not calibrated properly")
            return None
        
        # Set output directory
        if output_dir is None:
            output_dir = os.path.dirname(cam1_image)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create point labels
        point_labels = [f"Point {i+1}" for i in range(num_points)]
        
        # Select points in camera 1 image
        print(f"\nSelect {num_points} points in the Camera 1 image:")
        ret1, points1 = self.select_points(cam1_image, num_points, point_labels)
        
        if not ret1:
            print("Point selection canceled for Camera 1")
            return None
        
        # Select points in camera 2 image
        print(f"\nSelect the same {num_points} points in the Camera 2 image:")
        ret2, points2 = self.select_points(cam2_image, num_points, point_labels)
        
        if not ret2:
            print("Point selection canceled for Camera 2")
            return None
        
        # Triangulate 3D points
        points3D = self.triangulate_points(points1, points2)
        
        if points3D is None:
            return None
        
        # Calculate all pairwise distances
        dimensions = []
        for i in range(num_points):
            for j in range(i+1, num_points):
                dist = np.linalg.norm(points3D[i] - points3D[j])
                dimensions.append((i, j, dist))
                print(f"Distance between Point {i+1} and Point {j+1}: {dist:.2f} mm")
        
        # Visualize the 3D points
        self.visualize_3d_object(points3D, dimensions, os.path.join(output_dir, "object_dimensions_3d.png"))
        
        return dimensions
    
    def visualize_distance_measurement(self, points3D, output_path):
        """
        Visualize 3D points and the distance between them
        
        Parameters:
        points3D: 3D points to visualize
        output_path: Path to save the visualization
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot camera positions
        ax.scatter([0], [0], [0], c='r', marker='o', s=100, label='Camera 1')
        if self.T is not None:
            cam2_pos = -self.R.T @ self.T.flatten()
            ax.scatter([cam2_pos[0]], [cam2_pos[1]], [cam2_pos[2]], c='b', marker='o', s=100, label='Camera 2')
        
        # Plot the 3D points
        ax.scatter(points3D[:, 0], points3D[:, 1], points3D[:, 2], c='g', marker='o', s=50)
        
        # Connect the points with a line
        ax.plot(points3D[:, 0], points3D[:, 1], points3D[:, 2], 'g-', linewidth=2)
        
        # Add text labels for the points
        for i, point in enumerate(points3D):
            ax.text(point[0], point[1], point[2], f"Point {chr(65+i)}", color='black')
        
        # Add distance label
        if len(points3D) >= 2:
            # Calculate midpoint for label placement
            midpoint = (points3D[0] + points3D[1]) / 2
            distance = np.linalg.norm(points3D[1] - points3D[0])
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
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        # Save the visualization
        plt.savefig(output_path)
        plt.close(fig)
        print(f"3D visualization saved to {output_path}")
    
    def visualize_3d_object(self, points3D, dimensions, output_path):
        """
        Visualize 3D points representing an object and its dimensions
        
        Parameters:
        points3D: 3D points to visualize
        dimensions: List of (i, j, distance) tuples
        output_path: Path to save the visualization
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot camera positions
        ax.scatter([0], [0], [0], c='r', marker='o', s=100, label='Camera 1')
        if self.T is not None:
            cam2_pos = -self.R.T @ self.T.flatten()
            ax.scatter([cam2_pos[0]], [cam2_pos[1]], [cam2_pos[2]], c='b', marker='o', s=100, label='Camera 2')
        
        # Plot the 3D points
        ax.scatter(points3D[:, 0], points3D[:, 1], points3D[:, 2], c='g', marker='o', s=50)
        
        # Connect the points to form an object (if 4 or more points, assume rectangular)
        if len(points3D) >= 4:
            # Connect points to form a shape
            # For a rectangular object, connect in sequence plus the diagonal
            for i in range(len(points3D)):
                next_idx = (i + 1) % len(points3D)
                ax.plot([points3D[i, 0], points3D[next_idx, 0]],
                        [points3D[i, 1], points3D[next_idx, 1]],
                        [points3D[i, 2], points3D[next_idx, 2]], 'g-', linewidth=2)
        
        # Add text labels for the points
        for i, point in enumerate(points3D):
            ax.text(point[0], point[1], point[2], f"P{i+1}", color='black', fontsize=10)
        
        # Add distance labels for each dimension
        for i, j, dist in dimensions:
            # Calculate midpoint for label placement
            midpoint = (points3D[i] + points3D[j]) / 2
            ax.text(midpoint[0], midpoint[1], midpoint[2], 
                   f"{dist:.2f} mm", color='blue', fontsize=9, 
                   horizontalalignment='center', verticalalignment='center')
            
            # Draw a dashed line between these points
            ax.plot([points3D[i, 0], points3D[j, 0]],
                    [points3D[i, 1], points3D[j, 1]],
                    [points3D[i, 2], points3D[j, 2]], 'r--', linewidth=1)
        
        # Set axis labels and title
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title('3D Object Dimensions')
        
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
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        # Save the visualization
        plt.savefig(output_path)
        plt.close(fig)
        print(f"3D object visualization saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Distance Measurement Tool')
    
    parser.add_argument('--calib-dir', type=str, required=True,
                        help='Directory containing calibration parameters')
    
    parser.add_argument('--mode', type=str, choices=['distance', 'object'], 
                        default='distance',
                        help='Measurement mode: "distance" for point-to-point, "object" for object dimensions')
    
    parser.add_argument('--cam1-image', type=str, required=True,
                        help='Path to image from camera 1')
    
    parser.add_argument('--cam2-image', type=str, required=True,
                        help='Path to image from camera 2')
    
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save output files')
    
    parser.add_argument('--known-distance', type=float, default=None,
                        help='Known distance between points for validation (mm)')
    
    parser.add_argument('--num-points', type=int, default=4,
                        help='Number of points to select for object mode')
    
    args = parser.parse_args()
    
    # Initialize the measurement tool
    tool = DistanceMeasurementTool(args.calib_dir)
    
    if not tool.is_calibrated():
        print("Error: Calibration parameters not loaded correctly")
        return
    
    # Run the requested measurement mode
    if args.mode == 'distance':
        tool.measure_distance(args.cam1_image, args.cam2_image, 
                             args.output_dir, args.known_distance)
    else:  # object mode
        tool.measure_object_dimensions(args.cam1_image, args.cam2_image, 
                                      args.num_points, args.output_dir)
    
if __name__ == "__main__":
    main()