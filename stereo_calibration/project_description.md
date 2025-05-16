Project Summary: Stereo Vision Calibration and Tracking System
Project Overview
We've developed a comprehensive stereo vision system for accurate 3D position, distance, and eventually motion tracking using two cameras. The system is designed to be flexible, working with both specialized high-speed Edgertronic cameras and more accessible iPhones.
Where We Started
The project began with a need to measure human movement metrics (position, velocity, acceleration) using computer vision rather than wearable sensors. Initial challenges included:

Establishing accurate camera calibration
Developing reliable object detection
Calculating 3D positions and distances
Measuring error rates and validating the system

Key Components Developed
1. Calibration Pipeline

Intrinsic Calibration: Individual camera parameters calibration using checkerboard patterns
Extrinsic Calibration: Relative camera position calibration with static checkerboard positions
Corner Order Correction: Fixed critical issue with inconsistent checkerboard corner detection

2. Object Detection

Started with simple Hough circle detection for balls
Evolved to using YOLOv8 general object detection
Finally implemented a specialized ball detection model using Roboflow
Added caching mechanism for more efficient API usage

3. Distance Calculation

Implemented accurate triangulation for 3D position calculation
Created validation using objects at known distances
Developed error metrics and visualization tools

Current Status

Calibration system works reliably with 4% error in extrinsic calibration
Ball detection works well with the Roboflow specialized model
Distance validation shows 12.73% average error across different distances (2m-4m)
The system can process video frames and calculate 3D positions

Today's Achievements

Implemented a specialized ball detection model using Roboflow's API
Created an efficient caching mechanism for API calls
Improved accuracy from ~18% to ~12.7% error
Successfully validated the system across multiple distances
Visualized detection results and error metrics

Next Steps
1. Motion Tracking Implementation

Extract sequential frames from videos
Track ball position across frames
Calculate velocity and acceleration
Implement smoothing and filtering

2. System Improvements

Add center refinement for more precise ball center detection
Implement multi-frame averaging for noise reduction
Create a more robust frame synchronization mechanism

3. Validation Extensions

Test with objects moving at known speeds
Compare with ground truth measurements
Create comprehensive error profiles at different distances and speeds

4. Human Motion Applications

Extend detection to human keypoints (using pose estimation)
Create biomechanical models for performance metrics
Implement user-friendly interface for sports/medical applications

Technical Improvements Needed

Ball Detection: Refine center detection for sub-pixel accuracy
Frame Extraction: Implement synchronized frame extraction from stereo videos
Error Handling: Better handling of detection failures and outliers
Performance: Optimize for real-time processing where possible
Documentation: Comprehensive documentation of calibration best practices

Conclusion
The stereo vision system has proven its capability for accurate distance measurement with an acceptable error rate of 12.73%. This provides a solid foundation for implementing motion tracking and expanding to human movement analysis. The combination of proper calibration and specialized object detection has demonstrated that consumer-grade equipment can achieve research-quality measurements for biomechanical applications.