#Opticalflow.py: src/Opticalflow.py
#File Note: This is part of ffframegen via mobile
#Basic Node and confifions

import cv2
import numpy as np

img1 = cv2.imread('vp/frame_0002.png')
img2 = cv2.imread('vp/frame_0003.png')

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Compute dense optical flow using Farneback method
flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

# Create an output image (copy of the first image)
output_img = img1.copy()

# Step size for drawing arrows
step = 16  # You can adjust the step size depending on how dense you want the arrows

# Loop through the image, drawing arrows for the flow
for y in range(0, flow.shape[0], step):
    for x in range(0, flow.shape[1], step):
        # Get the flow vector at each point
        flow_at_point = flow[y, x]
        
        # Get the direction (dx, dy) for the arrow
        dx, dy = flow_at_point[0], flow_at_point[1]

        # Draw the arrow on the image
        end_point = (int(x + dx), int(y + dy))
        color = (0, 255, 0) if dx >= 0 else (0, 0, 255)  # Green for positive x direction, Red for negative
        cv2.arrowedLine(output_img, (x, y), end_point, color, 1, tipLength=0.2)

# Save the resulting image with arrows
cv2.imwrite('optical_flow_with_arrows.png', output_img)