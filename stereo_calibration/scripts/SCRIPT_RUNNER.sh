#!/bin/bash

# Stereo Camera Calibration Helper Script

# Configuration
BASE_DIR="/path/to/stereo_calibration"  # Change this to your actual path
TEST_DIR="test_001"                     # Change this or make it a parameter
CHECKERBOARD_SIZE="9,7"                 # Change as needed
SQUARE_SIZE=25.0                        # Change as needed
CAMERA_DISTANCE=2420                    # Change as needed
RULER_LENGTH=310                        # Change as needed

# Create directory structure if it doesn't exist
mkdir -p "${BASE_DIR}/data/${TEST_DIR}/left_camera"
mkdir -p "${BASE_DIR}/data/${TEST_DIR}/right_camera"

echo "=========================================="
echo "STEREO VISION CALIBRATION WORKFLOW"
echo "=========================================="
echo "Test directory: ${TEST_DIR}"
echo "Checkerboard: ${CHECKERBOARD_SIZE} corners, ${SQUARE_SIZE}mm squares"
echo "Camera distance: ${CAMERA_DISTANCE}mm"
echo ""

# Show files in directories
echo "Left camera files:"
ls -l "${BASE_DIR}/data/${TEST_DIR}/left_camera/"
echo ""
echo "Right camera files:"
ls -l "${BASE_DIR}/data/${TEST_DIR}/right_camera/"
echo ""

# Menu
PS3="Select an action: "
options=("Run Intrinsic Calibration" "Run Extrinsic Calibration" "Run 3D Pose Estimation" "Exit")
select opt in "${options[@]}"
do
    case $opt in
        "Run Intrinsic Calibration")
            echo "Running intrinsic calibration..."
            python scripts/intrinsic_iphone.py --test_dir "${TEST_DIR}" --base_dir "${BASE_DIR}" --checkerboard_size "${CHECKERBOARD_SIZE}" --square_size "${SQUARE_SIZE}"
            ;;
        "Run Extrinsic Calibration")
            echo "Running extrinsic calibration..."
            python scripts/extrinsic.py --test_dir "${TEST_DIR}" --base_dir "${BASE_DIR}" --actual_distance "${CAMERA_DISTANCE}" 
            ;;
        "Run 3D Pose Estimation")
            echo "Running validation..."
            python scripts/.py --test_dir "${TEST_DIR}" --base_dir "${BASE_DIR}" --ruler_length "${RULER_LENGTH}"
            ;;
        "Exit")
            break
            ;;
        *) 
            echo "Invalid option"
            ;;
    esac
    echo ""
    echo "Done! Press Enter to continue..."
    read
    echo ""
done