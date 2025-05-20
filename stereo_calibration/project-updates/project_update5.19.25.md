Stereo Vision Biomechanical Tracking System - Project Journal
INSTRUCTIONS
This document serves as a continuously updated project journal for the Stereo Vision Biomechanical Tracking System. At the end of each conversation, paste this document with the command "Update Project Journal" to maintain an accurate record of the project's status, progress, and plans.
The assistant will update all relevant sections based on the latest discussion, removing outdated information and adding new developments, ensuring this journal always reflects the current state of the project.
Current Status (Last Updated: May 19, 2025)
System Capabilities

Functional intrinsic and extrinsic camera calibration pipeline with ~6% error rates
Implemented distance validation for stereo calibration
3D pose reconstruction from stereo cameras with MediaPipe-based landmark detection
Joint angle calculations for major body segments (shoulders, elbows, hips, knees)
Basic motion smoothing implemented with Savitzky-Golay filters
Rudimentary occlusion handling when joints are visible in only one camera
Git repository established for source code version control (excluding large data files)

Available Resources

iPhone cameras (30fps) currently used for development and testing
Access to Edgertronic high-speed cameras with genlock capability
OpenBiomechanics database from Driveline Research for reference data
Force plates available for validation of biomechanical measurements
Upgraded 100mm 7x4 internal corners checkerboard for improved calibration

Current Focus

Optimizing Savitzky-Golay filter parameters for different capture frame rates
Transitioning to Edgertronic cameras with genlock for improved synchronization
Establishing improved validation methods for biomechanical measurements
Implementing anatomical constraints and joint velocity restrictions

Recent Progress
Calibration Improvements

Upgraded to larger checkerboard: Implemented a 100mm 7x4 internal corners checkerboard for more accurate calibration
Implemented distance validation: Added functionality to measure and validate distance to checkerboard in extrinsic.py
Reduced reprojection error: Achieved < 1 pixel reprojection error (average 0.73 pixels) in stereo calibration
Distance measurement validation: Achieved 6% error in distance measurement (2157mm measured vs 2295mm actual)
Fixed coordinate system mismatch: Corrected reprojection error calculation to properly handle rectified coordinate system
Created visualization tools: Enhanced debug outputs for distance validation with plots showing measured vs. actual distances

Motion Smoothing Implementation

Implemented Savitzky-Golay filtering with parameter variations:

iPhone (30fps): Window size 7-9, polynomial order 2
Future iPhone (60fps): Window size 11-15, polynomial order 3
Edgertronic (480fps): Window size 21-31, polynomial order 3-4


Initial results show mixed performance - effective at removing jitter but possible over-smoothing of rapid movements
Created comparison of different filtering approaches (SG, One-Euro, Moving Average)
Statistics: 3,966 anatomical corrections across 798 frames in recent test

Edgertronic Integration Planning

Designed camera configuration for genlock setup:

90° angle between cameras for optimal triangulation
Calibration videos: 1080p at 120fps
Motion capture: 1080p at 480fps, exposure 1/1000s - 1/3000s


Developed simplified workflow leveraging hardware synchronization
Created optimal setup guides for different movements (pitching, batting, jumping)

Source Code Management

Established Git repository structure with data files excluded
Implemented modular code organization for maintainability
Created documentation structure for tracking development decisions
Planned feature branching workflow for experimental development

Testing Results

Jump test showed challenges when subject moved out of frame
Lateral step test showed promising results for hip tracking
Current smoothing parameters may be too aggressive, altering biomechanical signals
Substantial differences in angle statistics before and after smoothing

Technical Exploration

Investigated combining Extended Kalman Filter with SG filtering for better predictions
Explored physical motion models to constrain movement during occlusion
Researched subject-specific skeletal models via T-pose calibration
Compared filtering approaches from recent research literature
Evaluated joint velocity constraints as additional smoothing method

Ongoing Challenges
Technical Hurdles

Occlusion handling: Current methods struggle when joints become partially or fully occluded
Balance in filtering: Reducing noise while preserving meaningful biomechanical signals
Test procedure complexity: Delays between camera start and actual movement
Camera synchronization: Flash sync is functional but cumbersome
Validation methods: Lack of ground truth for comparison
Data management: Balancing storage needs with version control requirements
Distance measurement accuracy: Current 6% error may impact velocity calculations

Project Management

Scope management: Maintaining focused development without becoming overwhelmed
Long-term direction: Determining optimal path among multiple possibilities
Validation concerns: Establishing clear metrics for success
Solo development: Managing workload and technical complexity
Project documentation: Established journal system for better continuity between development sessions

Next Steps
Short-Term (1-2 weeks)

Fine-tune Savitzky-Golay parameters based on recent test results
Implement joint velocity constraints to complement position filtering
Create simple validation framework with quantifiable error metrics
Test motion smoothing with adjusted parameters to reduce over-correction
Prepare for initial Edgertronic camera integration

Medium-Term (1-2 months)

Complete transition to Edgertronic cameras with genlock
Integrate Extended Kalman Filtering for improved occlusion prediction
Develop specialized modules for specific biomechanical analyses (pitching, jumping)
Test system against controlled movements with known physical properties
Implement subject-specific anatomical constraints

Long-Term Vision

Progress from positional tracking to velocity and acceleration calculation
Validate force and power calculations against force plate measurements
Create specialized analysis modules for different sports applications
Develop a comprehensive biomechanical analysis pipeline from capture to insights
Implement mode-based system architecture for different analysis scenarios:

Full body tracking mode
Pitching analysis mode
Jumping analysis mode
Custom joint tracking mode



Development Notes
Current Implementation Details

Motion smoothing implemented in motion_smoothing.py module
3D pose estimation in 3dpose.py with MediaPipe integration
Camera calibration workflow established in intrinsic.py and extrinsic.py
Video synchronization via flash detection in flash_sync.py
Checkerboard distance validation added to extrinsic.py workflow

Technical Decisions

Using Savitzky-Golay filters for initial smoothing due to signal preservation properties
MediaPipe selected as pose estimator for reliability and cross-platform compatibility
Planning parameter optimization by motion type and camera frame rate
Designing system for human-based validation rather than pendulum-based testing
Maintaining flexibility to support both iPhone and Edgertronic camera systems
Planning to implement validation tests using controlled human movements
6% distance error acceptable for biomechanical tracking; angular measurements less affected than absolute distances

Last Updated: May 19, 2025
Next Review: TBD