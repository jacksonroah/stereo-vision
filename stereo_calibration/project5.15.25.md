Stereo Vision Biomechanical Tracking System
Project Status Update - May 15, 2025
Current Capabilities

Robust Calibration Pipeline: Successfully calibrated stereo cameras with <1% distance error (0.22% measured in recent tests)
3D Pose Estimation: Implemented complete pipeline from 2D detection to 3D reconstruction
Enhanced Triangulation: Improved handling of joints visible in only one camera
Flash Synchronization: Working method for temporal alignment between cameras
Basic Anatomical Constraints: Initial implementation of limb length consistency
Basic Temporal Filtering: Simple weighted averaging filter for landmark tracking
Joint Angle Measurement: Successful calculation and visualization of 3D joint angles
Data Export: Generating statistics and CSV files for further analysis

Recent Progress

Jump Test Success: Completed successful validation using a box jump test
Angle Statistics: Generated comprehensive angle measurements with expected ranges
3D Reconstruction: Achieved consistent 3D tracking throughout the movement sequence
Camera Setup: Optimized camera positioning for maximum shared field of view
Pose Recognition: MediaPipe successfully tracked key joints through the entire sequence
Single Camera Recovery: Implemented successful estimation of joints visible in only one camera
Trajectory Visualization: Created 3D trajectory plots showing the full motion path

Hardware Configuration

Camera Setup: Two iPhones in stereo configuration
Camera Separation: ~3.5 meters (measured accurately during calibration)
Calibration Tools: Large checkerboard with good corner detection
Lighting: Controlled indoor environment with good illumination
Flash Sync: Using phone flash for precise frame synchronization

Key Measurements

Calibration Accuracy: 0.22% error in baseline measurement
Joint Angle Ranges:

Shoulder: 9.8° - 115.5° (right), 10.6° - 108.4° (left)
Elbow: 0.1° - 178.6° (right), 0.1° - 171.9° (left)
Hip: 53.7° - 176.9° (right), 57.1° - 178.6° (left)
Knee: 0.1° - 179.7° (right), 0.0° - 179.7° (left)


Frame Processing: Successfully processed full test video sequence

Current Challenges

Joint Tracking Jitter: Noticeable jitter in joint angles and positions
Anatomical Inconsistencies: Occasional unrealistic joint positions and angles
Temporal Consistency: Current simple filtering insufficient for smooth motion
Background Object Detection: Occasional confusion with background objects
Flash Detection Threshold: Current fixed threshold not optimal for all lighting conditions

Current Focus: Motion Smoothing Integration
We're now working on integrating a dedicated motion smoothing module to address the jitter issues and impose stronger anatomical constraints:

Created comprehensive motion_smoothing.py module with:

Multiple filtering options (Savitzky-Golay, One-Euro, Moving Average)
Anatomical joint constraints based on biomechanical limits
Limb length consistency enforcement
Velocity-based constraints to prevent unrealistic movements


Planning integration steps:

Add the module to the existing codebase
Modify key processing points to apply enhanced filtering
Implement graceful shutdown function
Maintain backward compatibility with existing pipeline
Add new command-line parameters for smoothing options



Next Steps

Complete integration of motion_smoothing.py with 3dpose.py
Conduct comparative testing with and without smoothing
Fine-tune filtering parameters for optimal balance between smoothness and responsiveness
Implement biomechanical calculations in a separate module
Develop visualization improvements to show raw vs. filtered data

Long-Term Development Path
The project continues to follow the structured progression from fundamental tracking to advanced biomechanical analysis:

Core Tracking Refinement (Current phase)

Improve joint tracking stability and accuracy
Add robust filtering and constraints


Biomechanical Analysis (Next phase)

Calculate velocities, accelerations, and forces
Implement sport-specific metrics


Performance Metrics (Future phase)

Develop scoring systems for movement quality
Create real-time feedback mechanisms