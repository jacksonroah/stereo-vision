# Stereo Vision Biomechanical Tracking System

## Project Status Update

### Current Capabilities
- **Calibration System**: Working intrinsic and extrinsic calibration pipeline for both iPhones and Edgertronic cameras
- **Distance Measurement**: System can calculate the distance between cameras and measure distances to objects
- **Basic Object Detection**: Initial experiments with YOLO for ball tracking (with reliability challenges)
- **Frame Synchronization**: Method for flash-synchronizing cameras indoors (primarily for iPhones)
- **Single-Camera Pose Estimation**: Successfully implemented MediaPipe-based pose tracking for static poses

### Hardware Resources
- **Current**: Two iPhones for rapid testing, indoor setup with flash sync capability
- **Incoming**: Larger professional checkerboard for improved calibration accuracy
- **Planned**: GenLock cable for Edgertronic cameras for precise synchronization
- **Available**: Force plates for validation of biomechanical measurements

### Current Challenges
- **Object Tracking Reliability**: YOLO models were inconsistent for tracking during motion
- **Motion Blur**: iPhone cameras introduce motion blur during fast movements
- **Setup Complexity**: Edgertronic cameras offer better quality but require more setup time
- **Pose Tracking Limitations**: Current single-camera implementation struggles with occlusion and side-views
- **Data Continuity**: Gaps in tracking data when confidence drops below threshold
- **Measurement Jitter**: Noise in angle measurements even during static poses

### Recent Progress
- Successfully implemented single-camera pose estimation using MediaPipe
- Validated joint angle measurements for static poses (T-pose showed expected ~90° shoulder angles)
- Identified key limitations with side-view tracking and occlusion
- Developed understanding of temporal filtering requirements for more stable measurements
- Established roadmap for stereo implementation of pose tracking

### Project Direction
The project continues to progress from basic tracking to focused biomechanical analysis:

1. **Refined Calibration**: Improve calibration accuracy with new checkerboard
2. **Pose Estimation Improvements**: Implement temporal filtering and anatomical constraints
3. **Stereo Pose Tracking**: Extend pose estimation to dual cameras with triangulation
4. **Dynamic Movement Validation**: Box jump experiments validated against force plate measurements
5. **Sport-Specific Analysis**: Targeted biomechanical measurements for pitching and batting

### Development Approach
- Structured workflow with defined focus modes to maintain progress and prevent scattered efforts
- Strategic use of iPhones for rapid development and Edgertronics for precision measurements
- Validation at each step against known ground truth measurements
- Focus on producing validated measurements before expanding to more complex motions
- Progressive implementation of filtering techniques inspired by professional motion capture systems

### Near-Term Objectives
- Implement basic temporal filtering to reduce jitter in pose estimation
- Extend pose estimation to two-camera setup with correspondence matching
- Validate joint angle measurements against known reference poses
- Develop visualization tools for 3D joint positions and angles
- Create an integrated pipeline from capture to analysis

### Long-Term Goals
- Create a semi-portable system for baseball biomechanical analysis
- Implement specific metrics like lead leg block angle for pitchers
- Develop trunk rotation, hip-shoulder separation, and arm speed measurements
- Build bat tracking capabilities for swing analysis (angle of approach, bat speed)
- Add advanced features from professional systems like anatomical constraints and predictive modeling