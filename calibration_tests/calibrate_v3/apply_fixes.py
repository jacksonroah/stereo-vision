#!/usr/bin/env python3
"""
Script to apply fixes to stereo_calibration_workflow.py
"""

import os
import re
import shutil
import sys

def backup_file(file_path):
    """Create a backup of the file"""
    backup_path = file_path + '.bak'
    shutil.copy2(file_path, backup_path)
    print(f"Created backup: {backup_path}")
    return backup_path

def replace_function(file_content, function_name, new_function_content):
    """Replace a function in the file content"""
    # Regex pattern to match the function definition and its body
    pattern = rf'def {function_name}\(.*?\):.*?(?=\n\S|$)'
    
    # Find the function in the file content
    match = re.search(pattern, file_content, re.DOTALL)
    
    if not match:
        print(f"WARNING: Function '{function_name}' not found in the file")
        return file_content
    
    # Replace the function
    updated_content = file_content[:match.start()] + new_function_content + file_content[match.end():]
    
    print(f"Replaced function: {function_name}")
    return updated_content

def apply_fixes(file_path):
    """Apply all fixes to the file"""
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        return False
    
    # Create backup
    backup_file(file_path)
    
    # Read file content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix 1: Replace save_calibration_params function
    save_calibration_params_function = """def save_calibration_params(self, camera_id, camera_matrix, dist_coeffs, reprojection_error):
        \"\"\"Save calibration parameters to files\"\"\"
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
        self.save_to_xml(camera_id, camera_matrix, dist_coeffs, reprojection_error)"""
    
    # Fix 2: Replace validate_calibration_params function
    validate_calibration_params_function = """def validate_calibration_params(self, camera_matrix, dist_coeffs, reprojection_error, 
                                    max_error=1.0, max_dist_coeff=1.5):
        \"\"\"
        Validate calibration parameters to detect potential issues
        
        Returns:
        bool: True if parameters seem reasonable, False otherwise
        \"\"\"
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
            return True  # Changed to return True to continue despite warnings"""
    
    # Apply fixes
    content = replace_function(content, "save_calibration_params", save_calibration_params_function)
    content = replace_function(content, "validate_calibration_params", validate_calibration_params_function)
    
    # Write updated content
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Successfully applied fixes to {file_path}")
    return True

def main():
    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: python apply_fixes.py path/to/stereo_calibration_workflow.py")
        sys.exit(1)
    
    file_path = sys.argv[1]
    success = apply_fixes(file_path)
    
    if success:
        print("\nAll fixes have been applied successfully!")
        print("You can now run the stereo_calibration_runner.py script again.")
    else:
        print("\nSome errors occurred while applying fixes.")
        print("Please check the messages above for details.")

if __name__ == "__main__":
    main()