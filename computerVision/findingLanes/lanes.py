import cv2
import numpy as np
import matplotlib.pyplot as plt

# Edge detection technique to find lane lines in an image

def canny(image):
    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Step 2: Apply Gaussian Blur - Reduce noise
    # Smooth image by applying a Gaussian Blur with a 5x5 Kernel
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 3: Apply Canny method to find lanes in image
    # Canny performs a derivative in both x and y directions,
    # measuring adjacent changes in intensity
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_interest(image):
    height = image.shape[0]
    # Isolate region of interest
    polygons = np.array([
    [(200, height), (1100, height), (550, 250)]
    ])
    # Project R.O.I.onto black mask of same size
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# Output detected lines
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            # Get coordinates of detected lines
            x1, y1, x2, y2 = line.reshape(4)
            # Draw detected lines
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


# Loading image
image = cv2.imread('test_image.jpg')
# Copy image
lane_image = np.copy(image)

# Calling our functions
canny = canny(lane_image)
cropped_image = region_of_interest(canny)

# Step 4: Use Hough Transform technique to detect straight lines in R.O.I.
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)

# Calling our function
line_image = display_lines(lane_image, lines)
# Draw detected lines onto the original image to show lanes
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
cv2.imshow('result', combo_image)
cv2.waitKey(5000)
