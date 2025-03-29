import cv2
import numpy as np
import os

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

# Folder containing the images (e.g., 'vp' folder)
folder_path = 'vp2'  # Path to your folder containing images

# Get list of image filenames sorted in order (ensure they are ordered correctly)
image_files = sorted(os.listdir(folder_path))

# Loop through each pair of consecutive images
for i in range(len(image_files) - 1):
    # Read current and next image
    img1_path = os.path.join(folder_path, image_files[i])
    img2_path = os.path.join(folder_path, image_files[i+1])
    
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    # Convert images to grayscale for optical flow calculation
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Compute dense optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Generate the intermediate frame (midway between frame 1 and frame 2)
    intermediate_frame = generate_intermediate_frame(img1, img2, flow, alpha=0.5)
    
    # Generate the frame number from the image filename (assumes filenames are like '0001.jpg', '0002.jpg', etc.)
    frame_number = image_files[i].split('.')[0]
    
    # Save the intermediate frame with the custom naming format: frame_{framenumber}5.png
    intermediate_frame_path = os.path.join(folder_path, f'{frame_number[:10]}5.png')
    cv2.imwrite(intermediate_frame_path, intermediate_frame)

    # Optionally print progress
    print(f"Generated intermediate frame: {intermediate_frame_path}")