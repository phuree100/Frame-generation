import cv2
import numpy as np

# Load two consecutive images
img1 = cv2.imread('vp/frame_0002.png')
img2 = cv2.imread('vp/frame_0003.png')

# Convert images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Compute dense optical flow using Farneback method
flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

# Function to generate intermediate frame
def generate_intermediate_frame(img1, img2, flow, alpha=0.5):
    h, w = img1.shape[:2]
    
    # Generate a mesh grid of coordinates (x, y)
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    # Get flow for the points (x, y)
    flow_x = flow[..., 0]
    flow_y = flow[..., 1]
    
    # Compute new positions for each pixel based on the flow
    new_x = x + alpha * flow_x
    new_y = y + alpha * flow_y
    
    # Make sure that the new positions are within the bounds of the image
    new_x = np.clip(new_x, 0, w-1)
    new_y = np.clip(new_y, 0, h-1)
    
    # Use remapping to warp img1 to its intermediate position
    map_x = new_x.astype(np.float32)
    map_y = new_y.astype(np.float32)
    
    # Generate the intermediate image
    intermediate_img = cv2.remap(img1, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    
    return intermediate_img

# Generate the intermediate frame (midway between frame 1 and frame 2)
intermediate_frame = generate_intermediate_frame(img1, img2, flow, alpha=0.5)

# Save the intermediate frame
cv2.imwrite('intermediate_frame.png', intermediate_frame)