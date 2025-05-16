#!/usr/bin/env python3
import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from tabulate import tabulate
import re

def find_validation_results(test_dir):
    """Find all validation result files in the results directory."""
    results_dir = os.path.join(test_dir, "results", "validation_results")
    
    if not os.path.exists(results_dir):
        print(f"Error: Validation results directory not found: {results_dir}")
        return []
    
    # Find all pickle files with validation results
    result_files = glob.glob(os.path.join(results_dir, "*_validation_results.pkl"))
    
    # Sort files by validation number
    result_files.sort()
    
    return result_files

def extract_distance_from_filename(filename):
    """Try to extract distance from the filename if it contains a distance indicator."""
    basename = os.path.basename(filename)
    
    # Try to find a pattern like "dist_1000mm" or "1000mm" or "dist_1000" in the filename
    patterns = [
        r'dist[_-]?(\d+)mm',     # dist_1000mm or dist1000mm
        r'dist[_-]?(\d+)',       # dist_1000 or dist1000
        r'(\d+)mm',              # 1000mm
        r'(\d+)cm',              # 100cm (will be converted to mm)
        r'm[_-]?(\d+)'           # m_100 or m100 (assuming meters)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, basename, re.IGNORECASE)
        if match:
            distance = int(match.group(1))
            
            # Convert to mm if needed
            if pattern.endswith('cm'):
                distance *= 10
            elif pattern.startswith('m'):
                distance *= 1000
                
            return distance
    
    return None

def load_results(result_files):
    """Load validation results from pickle files."""
    all_results = []
    
    for file_path in result_files:
        try:
            with open(file_path, 'rb') as f:
                result = pickle.load(f)
                
                # Add file path for reference
                result['file_path'] = file_path
                
                # Add filename
                result['filename'] = os.path.basename(file_path)
                
                # Try to extract distance from filename if not already in results
                if 'actual_distance_mm' not in result:
                    distance = extract_distance_from_filename(file_path)
                    if distance is not None:
                        result['actual_distance_mm'] = distance
                        
                        # Calculate error metrics
                        error_mm = abs(result['distance_mm'] - distance)
                        error_percent = 100.0 * error_mm / distance
                        
                        result['distance_error_mm'] = error_mm
                        result['distance_error_percent'] = error_percent
                
                all_results.append(result)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return all_results

def create_summary_table(results):
    """Create a summary table of validation results."""
    # Extract key information for the table
    table_data = []
    
    for result in results:
        row = {
            'Video': result.get('video_name', 'Unknown'),
            'Measured Distance (mm)': f"{result['distance_mm']:.1f}",
            'Ball Diameter (mm)': f"{result.get('ball_diameter_mm', 'Unknown')}"
        }
        
        # Add actual distance and error if available
        if 'actual_distance_mm' in result:
            row['Actual Distance (mm)'] = f"{result['actual_distance_mm']:.1f}"
            row['Error (mm)'] = f"{result.get('distance_error_mm', 0):.1f}"
            row['Error (%)'] = f"{result.get('distance_error_percent', 0):.2f}%"
        
        table_data.append(row)
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(table_data)
    
    # Return both the DataFrame and a formatted table
    return df, tabulate(df, headers='keys', tablefmt='pipe', showindex=False)

def plot_distance_error(results, output_dir):
    """Create plots of distance error vs actual distance."""
    # Extract results that have actual distance information
    valid_results = [r for r in results if 'actual_distance_mm' in r]
    
    if not valid_results:
        print("No results with actual distance information found")
        return
    
    # Sort by actual distance
    valid_results.sort(key=lambda x: x['actual_distance_mm'])
    
    # Extract data for plotting
    distances = [r['actual_distance_mm'] for r in valid_results]
    measured = [r['distance_mm'] for r in valid_results]
    errors_mm = [r.get('distance_error_mm', 0) for r in valid_results]
    errors_percent = [r.get('distance_error_percent', 0) for r in valid_results]
    
    # Convert distances to meters for better readability
    distances_m = [d/1000.0 for d in distances]
    measured_m = [m/1000.0 for m in measured]
    
    # Plot 1: Actual vs. Measured Distance
    plt.figure(figsize=(10, 6))
    plt.plot(distances_m, distances_m, 'k--', label='Perfect Measurement')
    plt.plot(distances_m, measured_m, 'bo-', label='Measured Distance')
    
    plt.xlabel('Actual Distance (m)')
    plt.ylabel('Distance (m)')
    plt.title('Actual vs. Measured Distance')
    plt.grid(True)
    plt.legend()
    
    # Add error values as annotations
    for i, (x, y, err) in enumerate(zip(distances_m, measured_m, errors_percent)):
        plt.annotate(f"{err:.1f}%", (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distance_comparison.png'))
    
    # Plot 2: Error vs. Distance
    plt.figure(figsize=(10, 6))
    
    # Plot both absolute and percentage error
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color1 = 'tab:blue'
    ax1.set_xlabel('Actual Distance (m)')
    ax1.set_ylabel('Absolute Error (mm)', color=color1)
    ax1.plot(distances_m, errors_mm, 'o-', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Percentage Error (%)', color=color2)
    ax2.plot(distances_m, errors_percent, 's-', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.title('Distance Measurement Error')
    plt.grid(True)
    fig.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'distance_error.png'))
    
    # Plot 3: Error histogram
    plt.figure(figsize=(10, 6))
    plt.hist(errors_percent, bins=min(10, len(errors_percent)), alpha=0.7, color='skyblue')
    plt.xlabel('Percentage Error (%)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Measurement Errors')
    plt.grid(True)
    
    plt.axvline(np.mean(errors_percent), color='r', linestyle='dashed', linewidth=2, 
               label=f'Mean: {np.mean(errors_percent):.2f}%')
    plt.axvline(np.median(errors_percent), color='g', linestyle='dashed', linewidth=2,
               label=f'Median: {np.median(errors_percent):.2f}%')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_distribution.png'))
    
    # Close all plots
    plt.close('all')
    
    print(f"Distance error plots saved to {output_dir}")

def calculate_overall_metrics(results):
    """Calculate overall accuracy metrics."""
    # Extract results that have actual distance information
    valid_results = [r for r in results if 'actual_distance_mm' in r]
    
    if not valid_results:
        return None
    
    errors_percent = [r.get('distance_error_percent', 0) for r in valid_results]
    errors_mm = [r.get('distance_error_mm', 0) for r in valid_results]
    
    metrics = {
        'count': len(valid_results),
        'mean_error_percent': np.mean(errors_percent),
        'median_error_percent': np.median(errors_percent),
        'min_error_percent': np.min(errors_percent),
        'max_error_percent': np.max(errors_percent),
        'std_error_percent': np.std(errors_percent),
        
        'mean_error_mm': np.mean(errors_mm),
        'median_error_mm': np.median(errors_mm),
        'min_error_mm': np.min(errors_mm),
        'max_error_mm': np.max(errors_mm),
        'std_error_mm': np.std(errors_mm),
    }
    
    return metrics

def generate_report(results, metrics, summary_table, output_dir):
    """Generate a comprehensive validation report."""
    report_path = os.path.join(output_dir, 'validation_report.md')
    
    with open(report_path, 'w') as f:
        f.write('# Stereo Vision Validation Report\n\n')
        
        f.write('## Summary\n\n')
        
        if metrics:
            f.write(f"* **Number of validation tests:** {metrics['count']}\n")
            f.write(f"* **Mean percentage error:** {metrics['mean_error_percent']:.2f}%\n")
            f.write(f"* **Median percentage error:** {metrics['median_error_percent']:.2f}%\n")
            f.write(f"* **Error range:** {metrics['min_error_percent']:.2f}% to {metrics['max_error_percent']:.2f}%\n")
            f.write(f"* **Standard deviation of error:** {metrics['std_error_percent']:.2f}%\n\n")
            
            f.write(f"* **Mean absolute error:** {metrics['mean_error_mm']:.2f}mm\n")
            f.write(f"* **Median absolute error:** {metrics['median_error_mm']:.2f}mm\n")
            f.write(f"* **Absolute error range:** {metrics['min_error_mm']:.2f}mm to {metrics['max_error_mm']:.2f}mm\n\n")
        else:
            f.write("*No validation tests with known distances found*\n\n")
        
        f.write('## Detailed Results\n\n')
        
        # Add the summary table
        f.write(summary_table)
        f.write('\n\n')
        
        # Add graphics if available
        if os.path.exists(os.path.join(output_dir, 'distance_comparison.png')):
            f.write('## Visualization\n\n')
            f.write('### Actual vs. Measured Distance\n\n')
            f.write('![Distance Comparison](distance_comparison.png)\n\n')
            
            f.write('### Distance Measurement Error\n\n')
            f.write('![Distance Error](distance_error.png)\n\n')
            
            f.write('### Error Distribution\n\n')
            f.write('![Error Distribution](error_distribution.png)\n\n')
        
        f.write('## Recommendations\n\n')
        
        # Add recommendations based on results
        if metrics:
            if metrics['mean_error_percent'] < 5.0:
                f.write("* The calibration is excellent with an average error below 5%.\n")
                f.write("* The system is ready for precise measurements.\n")
            elif metrics['mean_error_percent'] < 10.0:
                f.write("* The calibration is good with an average error below 10%.\n")
                f.write("* Consider additional calibration with more checkerboard positions for higher precision.\n")
            else:
                f.write("* The calibration shows significant error (>10%).\n")
                f.write("* Recommendations:\n")
                f.write("  - Redo the intrinsic calibration with more varied checkerboard positions\n")
                f.write("  - Ensure consistent checkerboard corner detection\n")
                f.write("  - Record more extrinsic calibration positions\n")
                f.write("  - Verify the actual distance measurements\n")
    
    print(f"Validation report saved to {report_path}")

def main():
    """Main function to analyze validation results."""
    parser = argparse.ArgumentParser(description='Analyze Stereo Vision Validation Results')
    parser.add_argument('--test_dir', required=True, 
                      help='Test directory name (e.g., test_001)')
    parser.add_argument('--base_dir', default='.', 
                      help='Base directory for the project (default: current directory)')
    
    args = parser.parse_args()
    
    # Set up paths
    base_dir = args.base_dir
    test_dir = os.path.join(base_dir, "data", args.test_dir)
    results_dir = os.path.join(test_dir, "results", "roboflow_validation_results")
    
    # Find all validation result files
    result_files = find_validation_results(test_dir)
    
    if not result_files:
        print("No validation result files found")
        return
    
    print(f"Found {len(result_files)} validation result files")
    
    # Load all results
    results = load_results(result_files)
    
    # Create analysis output directory
    analysis_dir = os.path.join(results_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Create summary table
    df, summary_table = create_summary_table(results)
    print("\nValidation Results Summary:")
    print(summary_table)
    
    # Save summary table as CSV
    df.to_csv(os.path.join(analysis_dir, 'validation_summary.csv'), index=False)
    
    # Calculate overall metrics
    metrics = calculate_overall_metrics(results)
    
    if metrics:
        print("\nOverall Metrics:")
        print(f"Mean error: {metrics['mean_error_percent']:.2f}%")
        print(f"Median error: {metrics['median_error_percent']:.2f}%")
        print(f"Error range: {metrics['min_error_percent']:.2f}% to {metrics['max_error_percent']:.2f}%")
    
    # Create visualizations
    plot_distance_error(results, analysis_dir)
    
    # Generate comprehensive report
    generate_report(results, metrics, summary_table, analysis_dir)
    
    print(f"\nAnalysis complete! Results saved to {analysis_dir}")

if __name__ == "__main__":
    main()