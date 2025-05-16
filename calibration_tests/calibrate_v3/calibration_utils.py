#!/usr/bin/env python3
"""
Calibration Parameter Utilities

This module provides utilities for working with camera calibration parameters
in various formats, including XML, CSV, and NumPy files.
"""

import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import cv2

def load_calibration_from_xml(xml_path):
    """
    Load camera calibration parameters from an XML file
    
    Parameters:
    xml_path: Path to the XML file
    
    Returns:
    camera_matrix, dist_coeffs
    """
    print(f"Loading calibration from XML: {xml_path}")
    
    if not os.path.exists(xml_path):
        print(f"Error: XML file not found: {xml_path}")
        return None, None
    
    try:
        # Try parsing with ElementTree
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Extract camera matrix
        matrix_elem = root.find('.//CameraMatrix/data')
        if matrix_elem is not None:
            matrix_data = list(map(float, matrix_elem.text.split()))
            camera_matrix = np.array(matrix_data).reshape(3, 3)
        else:
            print("Error: Camera matrix not found in XML")
            return None, None
        
        # Extract distortion coefficients
        dist_elem = root.find('.//DistortionCoefficients/data')
        if dist_elem is not None:
            dist_data = list(map(float, dist_elem.text.split()))
            dist_coeffs = np.array(dist_data).reshape(1, -1)
        else:
            print("Error: Distortion coefficients not found in XML")
            return None, None
        
        print(f"Successfully loaded camera matrix ({camera_matrix.shape}) and distortion coefficients ({dist_coeffs.shape})")
        return camera_matrix, dist_coeffs
        
    except ET.ParseError:
        print(f"Error parsing XML file: {xml_path}")
        
        # Fallback to OpenCV's FileStorage
        try:
            fs = cv2.FileStorage(xml_path, cv2.FILE_STORAGE_READ)
            camera_matrix = fs.getNode("CameraMatrix").mat()
            dist_coeffs = fs.getNode("DistortionCoefficients").mat()
            fs.release()
            
            print(f"Successfully loaded with OpenCV: camera matrix ({camera_matrix.shape}) and distortion coefficients ({dist_coeffs.shape})")
            return camera_matrix, dist_coeffs
            
        except Exception as e:
            print(f"OpenCV FileStorage also failed: {e}")
            return None, None

def load_calibration_from_csv(matrix_csv, dist_csv):
    """
    Load camera calibration parameters from CSV files
    
    Parameters:
    matrix_csv: Path to the camera matrix CSV
    dist_csv: Path to the distortion coefficients CSV
    
    Returns:
    camera_matrix, dist_coeffs
    """
    print(f"Loading calibration from CSV: {matrix_csv} and {dist_csv}")
    
    if not os.path.exists(matrix_csv) or not os.path.exists(dist_csv):
        print(f"Error: CSV files not found")
        return None, None
    
    try:
        # Load camera matrix
        matrix_df = pd.read_csv(matrix_csv)
        
        # Handle different column names
        value_col = None
        for possible_col in ['Value', 'Camera Matrix']:
            if possible_col in matrix_df.columns:
                value_col = possible_col
                break
        
        if value_col is None:
            print(f"Error: Could not find value column in matrix CSV")
            return None, None
            
        matrix_values = matrix_df[value_col].values
        
        if matrix_values.shape[0] != 9:
            print(f"Error: Camera matrix should have 9 values, found {matrix_values.shape[0]}")
            return None, None
            
        camera_matrix = matrix_values.reshape(3, 3)
        
        # Load distortion coefficients
        dist_df = pd.read_csv(dist_csv)
        
        # Handle different column names
        value_col = None
        for possible_col in ['Value', 'Distortion Coefficients']:
            if possible_col in dist_df.columns:
                value_col = possible_col
                break
        
        if value_col is None:
            print(f"Error: Could not find value column in distortion CSV")
            return None, None
            
        dist_values = dist_df[value_col].values
        dist_coeffs = dist_values.reshape(1, -1)
        
        print(f"Successfully loaded camera matrix ({camera_matrix.shape}) and distortion coefficients ({dist_coeffs.shape})")
        return camera_matrix, dist_coeffs
        
    except Exception as e:
        print(f"Error loading from CSV: {e}")
        return None, None

def load_calibration_from_txt(matrix_txt, dist_txt):
    """
    Load camera calibration parameters from NumPy text files
    
    Parameters:
    matrix_txt: Path to the camera matrix text file
    dist_txt: Path to the distortion coefficients text file
    
    Returns:
    camera_matrix, dist_coeffs
    """
    print(f"Loading calibration from TXT: {matrix_txt} and {dist_txt}")
    
    if not os.path.exists(matrix_txt) or not os.path.exists(dist_txt):
        print(f"Error: TXT files not found")
        return None, None
    
    try:
        camera_matrix = np.loadtxt(matrix_txt)
        dist_coeffs = np.loadtxt(dist_txt)
        
        # Ensure proper shapes
        if camera_matrix.shape != (3, 3):
            print(f"Warning: Camera matrix has shape {camera_matrix.shape}, reshaping to (3, 3)")
            camera_matrix = camera_matrix.reshape(3, 3)
        
        if len(dist_coeffs.shape) == 1:
            dist_coeffs = dist_coeffs.reshape(1, -1)
        
        print(f"Successfully loaded camera matrix ({camera_matrix.shape}) and distortion coefficients ({dist_coeffs.shape})")
        return camera_matrix, dist_coeffs
        
    except Exception as e:
        print(f"Error loading from TXT: {e}")
        return None, None

def load_calibration(params_dir, camera_id):
    """
    Load camera calibration parameters from a directory, trying different formats
    
    Parameters:
    params_dir: Directory containing calibration files
    camera_id: Camera identifier (usually 1 or 2)
    
    Returns:
    camera_matrix, dist_coeffs
    """
    print(f"Attempting to load calibration for camera {camera_id} from {params_dir}")
    
    # Define possible file paths
    xml_path = os.path.join(params_dir, f"cam{camera_id}_calibration_parameters.xml")
    matrix_csv = os.path.join(params_dir, f"cam{camera_id}_camera_matrix.csv")
    dist_csv = os.path.join(params_dir, f"cam{camera_id}_distortion_coefficients.csv")
    matrix_txt = os.path.join(params_dir, f"camera_{camera_id}_matrix.txt")
    dist_txt = os.path.join(params_dir, f"camera_{camera_id}_distortion.txt")
    
    # Try loading from XML first
    if os.path.exists(xml_path):
        camera_matrix, dist_coeffs = load_calibration_from_xml(xml_path)
        if camera_matrix is not None:
            return camera_matrix, dist_coeffs
    
    # Try loading from CSV next
    if os.path.exists(matrix_csv) and os.path.exists(dist_csv):
        camera_matrix, dist_coeffs = load_calibration_from_csv(matrix_csv, dist_csv)
        if camera_matrix is not None:
            return camera_matrix, dist_coeffs
    
    # Finally, try loading from TXT
    if os.path.exists(matrix_txt) and os.path.exists(dist_txt):
        camera_matrix, dist_coeffs = load_calibration_from_txt(matrix_txt, dist_txt)
        if camera_matrix is not None:
            return camera_matrix, dist_coeffs
    
    print(f"Failed to load calibration for camera {camera_id} from any format")
    return None, None

def save_calibration(output_dir, camera_id, camera_matrix, dist_coeffs, reprojection_error=None):
    """
    Save camera calibration parameters in multiple formats
    
    Parameters:
    output_dir: Directory to save the files
    camera_id: Camera identifier (usually 1 or 2)
    camera_matrix: 3x3 camera matrix
    dist_coeffs: Distortion coefficients
    reprojection_error: Optional reprojection error value
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to NumPy text files
    np.savetxt(os.path.join(output_dir, f'camera_{camera_id}_matrix.txt'), camera_matrix)
    np.savetxt(os.path.join(output_dir, f'camera_{camera_id}_distortion.txt'), dist_coeffs)
    
    # Save to CSV with descriptive headers
    matrix_headers = ['Row1Col1', 'Row1Col2', 'Row1Col3', 
                      'Row2Col1', 'Row2Col2', 'Row2Col3', 
                      'Row3Col1', 'Row3Col2', 'Row3Col3']
    dist_headers = ['k1', 'k2', 'p1', 'p2', 'k3']
    
    df_matrix = pd.DataFrame({
        'Element': matrix_headers,
        'Value': camera_matrix.flatten()
    })
    df_dist = pd.DataFrame({
        'Element': dist_headers[:len(dist_coeffs.flatten())],
        'Value': dist_coeffs.flatten()
    })
    
    df_matrix.to_csv(os.path.join(output_dir, f'cam{camera_id}_camera_matrix.csv'), index=False)
    df_dist.to_csv(os.path.join(output_dir, f'cam{camera_id}_distortion_coefficients.csv'), index=False)
    
    # Save to XML format
    save_to_xml(output_dir, camera_id, camera_matrix, dist_coeffs, reprojection_error)
    
    print(f"Calibration parameters for camera {camera_id} saved to {output_dir}")

def save_to_xml(output_dir, camera_id, camera_matrix, dist_coeffs, reprojection_error=None):
    """
    Save calibration parameters to XML format
    
    Parameters:
    output_dir: Directory to save the file
    camera_id: Camera identifier
    camera_matrix: 3x3 camera matrix
    dist_coeffs: Distortion coefficients
    reprojection_error: Optional reprojection error value
    """
    xml_path = os.path.join(output_dir, f'cam{camera_id}_calibration_parameters.xml')
    
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
    
    # Add reprojection error if provided
    if reprojection_error is not None:
        error_elem = ET.SubElement(root, 'ReprojectionError')
        error_elem.text = str(reprojection_error)
    
    # Write to file
    tree = ET.ElementTree(root)
    with open(xml_path, 'wb') as f:
        tree.write(f, encoding='utf-8', xml_declaration=True)
    
    print(f"Calibration parameters saved to XML: {xml_path}")

if __name__ == "__main__":
    # Simple test/example
    import argparse
    
    parser = argparse.ArgumentParser(description='Test calibration parameter loading/saving')
    parser.add_argument('--params-dir', type=str, required=True,
                        help='Directory containing camera parameters')
    parser.add_argument('--camera-id', type=int, required=True,
                        help='Camera ID (1 or 2)')
    
    args = parser.parse_args()
    
    # Load and display calibration parameters
    camera_matrix, dist_coeffs = load_calibration(args.params_dir, args.camera_id)
    
    if camera_matrix is not None:
        print("\nLoaded parameters:")
        print("Camera Matrix:")
        print(camera_matrix)
        print("\nDistortion Coefficients:")
        print(dist_coeffs)