# Stereo Vision Calibration System

This project provides a complete workflow for calibrating a stereo vision system using two cameras, with a focus on making the process repeatable and reducing error rates.

## Directory Structure

The project uses the following directory structure to keep tests organized:

```
stereo_calibration/
├── data/
│   ├── test_001/
│   │   ├── left_camera/
│   │   │   ├── intrinsic_video.mp4
│   │   │   ├── extrinsic_video_001.mp4
│   │   │   ├── extrinsic_video_002.mp4
│   │   │   └── validation_video.mp4
│   │   ├── right_camera/
│   │   │   ├── intrinsic_video.mp4
│   │   │   ├── extrinsic_video_001.mp4
│   │   │   ├── extrinsic_video_002.mp4
│   │   │   └── validation_video.mp4
│   │   └── results/
│   │       ├── intrinsic_params/
│   │       ├── extrinsic_params/
│   │       └── validation_results/
│   ├── test_002/
│   │   └── ...
├── scripts/
│   ├── intrinsic_calibrator_updated.py
│   ├── extrinsic_calibrator_updated.py
│   ├── simple_validator_updated.py
└── README.md
```

## Quick Start Guide

### Step 1: Setting Up a New Test

Create a new test directory structure:

```bash
mkdir -p stereo_calibration/data/test_001/left_camera
mkdir -p stereo_calibration/data/test_001/right_camera
mkdir -p stereo_calibration/data/test_001/results
```

### Step 2: Record Your Videos

For each new test, you need to record three types of videos with both cameras:

#### Intrinsic Calibration Videos
- **Purpose**: Calibrates individual camera settings
- **Record**: Wave the checkerboard in front of the camera, covering the entire field of view
- **Name files**: `intrinsic_video.mp4` in each camera folder

#### Extrinsic Calibration Videos
- **Purpose**: Calculates the relative position between cameras
- **Record**: Place the checkerboard where both cameras can see it
- **Key Improvement**: Record MULTIPLE positions (at least 3-5) of the checkerboard
- **Name files**: 
  - `extrinsic_video_001.mp4`
  - `extrinsic_video_002.mp4`
  - etc.

#### Validation Videos
- **Purpose**: Verifies the accuracy of your calibration
- **Record**: Place a ruler or object of known length visible to both cameras
- **Name files**: `validation_video.mp4` in each camera folder

### Step 3: Run the Calibration Pipeline

```bash
# Run intrinsic calibration
python scripts/intrinsic_calibrator_updated.py --test_dir test_001 --base_dir /path/to/stereo_calibration

# Run extrinsic calibration
python scripts/extrinsic_calibrator_updated.py --test_dir test_001 --base_dir /path/to/stereo_calibration --actual_distance 2420

# Run validation
python scripts/simple_validator_updated.py --test_dir test_001 --base_dir /path/to/stereo_calibration --ruler_length 310
```

## Calibration Steps Explained

### 1. Intrinsic Calibration

The `intrinsic_calibrator_updated.py` script:

- Extracts frames from the intrinsic calibration videos
- Finds checkerboard patterns in the frames
- Calculates camera matrices and distortion coefficients
- Filters out frames with high reprojection error
- Saves the results to the `results/intrinsic_params/` directory

### 2. Extrinsic Calibration

The `extrinsic_calibrator_updated.py` script:

- Finds all extrinsic calibration videos for both cameras
- Matches video pairs between cameras
- Extracts frames from each video pair
- Finds checkerboard patterns in matching frames
- Performs stereo calibration to find the relative position between cameras
- Saves the results to the `results/extrinsic_params/` directory

**Important improvements**:
- Supports multiple extrinsic calibration videos with different checkerboard positions
- This is key to reducing error rates in your extrinsic calibration

### 3. Validation

The `simple_validator_updated.py` script:

- Extracts frames from the validation videos
- Allows you to mark points on a ruler or object of known length
- Triangulates the 3D positions of these points
- Calculates the measured length and distance to the object
- Compares with known values to determine error rates
- Creates visualizations and a validation report

## Tips for Better Results

### For Intrinsic Calibration

1. Move the checkerboard through the entire field of view
2. Include edges and corners of the frame
3. Vary the orientation of the checkerboard (tilt it in different directions)
4. Ensure good lighting for clear pattern detection

### For Extrinsic Calibration

1. **Record multiple positions** - this is the key improvement!
2. Ensure the checkerboard is visible in both cameras for each position
3. Use positions that cover different parts of the shared field of view
4. Keep the checkerboard static in each position (use a tripod if possible)
5. Measure the actual distance between cameras very carefully

### For Validation

1. Use a high-contrast ruler with clear markings
2. Place it at different known distances to test accuracy
3. Make multiple measurements and average the results
4. Try to mark the same physical points in both camera views

## Understanding Your Error Rates

You observed that your camera separation error (23%) is higher than your object distance error (10%). This is expected because:

1. **Error Propagation**: Extrinsic calibration errors don't map linearly to distance measurement errors
2. **Different Measurement Methods**: The baseline is calculated directly from the calibration, while the distance to the object uses triangulation
3. **Averaging Effect**: When measuring an object, errors in the individual camera points can partially cancel out

Using multiple checkerboard positions in extrinsic calibration should help reduce your baseline error significantly.

## Troubleshooting

### Common Issues

1. **"No checkerboard found"**:
   - Check lighting conditions
   - Ensure the checkerboard is flat and fully visible
   - Try adjusting the frame extraction interval

2. **High Reprojection Error**:
   - Make sure the checkerboard dimensions are specified correctly
   - Use a higher quality checkerboard
   - Try recapturing the intrinsic calibration videos

3. **Large Extrinsic Calibration Error**:
   - Add more checkerboard positions in your extrinsic calibration
   - Verify the actual distance between cameras is measured accurately
   - Ensure both cameras remain completely stationary during recording

4. **Validation Errors**:
   - Make sure you're marking the same physical points in both images
   - Try using a ruler with more distinct features
   - Ensure the ruler is visible clearly in both cameras