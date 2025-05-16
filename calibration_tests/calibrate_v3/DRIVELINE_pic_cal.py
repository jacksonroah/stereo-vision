# Import required libraries
import cv2
import numpy as np

# Initialize variables for calibration
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((7 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:7].T.reshape(-1, 2) * 25  # 100mm squares

# Read the image
image_path = 'test.png'
image = cv2.imread(image_path)

if image is None:
    print("Image not found. Exiting.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray, (9, 7), None)

if ret:
    # Refining corner positions
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    
    # Draw the corners
    cv2.drawChessboardCorners(image, (9, 7), corners2, ret)
    
    # Save or display the image
    cv2.imwrite('checkerboard_detected.png', image)
else:
    print("Could not find chessboard corners. Exiting.")
