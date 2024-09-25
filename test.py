# test.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from data_loader import TableTennisDataset
from unet import UNet

def load_model(model_path, device):
    """Load the trained model."""
    model = UNet(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

def process_image(frame_gray, model, device, threshold=0.5):
    """Process a single image and detect points."""
    frame_gray = frame_gray.to(device).unsqueeze(0)  # Shape: (1, 1, H, W)
    with torch.no_grad():
        # Perform inference
        output = model(frame_gray)  # Output shape: (1, 1, H, W)
        output_np = torch.sigmoid(output).squeeze().cpu().numpy()  # Shape: (H, W)
    
    # Apply thresholding to get a binary mask
    binary_output = (output_np > threshold).astype(np.uint8)
    
    # Detect points
    detected_points = detect_points(binary_output)
    
    return output_np, detected_points

def detect_points(binary_mask):
    """Detect points from the binary mask and assign them to table parts."""
    # Find connected components
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)
    
    # Exclude background label (0)
    component_areas = stats[1:, cv2.CC_STAT_AREA]
    component_centroids = centroids[1:]
    
    # If there are more than 6 components, select the top 6 largest components
    if len(component_areas) >= 6:
        # Get indices that would sort the areas in descending order
        sorted_indices = np.argsort(-component_areas)
        # Get the top 6 components
        top_indices = sorted_indices[:6]
        top_centroids = component_centroids[top_indices]
    else:
        # If fewer than 6 components detected, use all available
        top_centroids = component_centroids
    
    # Assign points to table parts
    assigned_points = assign_points_to_table(top_centroids)
    
    return assigned_points

def assign_points_to_table(centroids):
    """Assign each centroid to a specific part of the table."""
    # Assuming the table is approximately rectangular in the image,
    # we can use the coordinates of the centroids to determine their positions.
    # We'll classify the points into top-left, top-right, bottom-left, bottom-right,
    # top midpoint, and bottom midpoint.

    # Convert centroids to a NumPy array if not already
    centroids = np.array(centroids)
    
    # Sort centroids based on their y-coordinate (vertical position)
    # Lower y-values are at the top of the image
    sorted_by_y = centroids[np.argsort(centroids[:, 1])]
    
    # Separate top and bottom points
    top_points = sorted_by_y[:3]  # First three points (top)
    bottom_points = sorted_by_y[3:]  # Remaining points (bottom)
    
    # Further sort top and bottom points by x-coordinate (horizontal position)
    top_points = top_points[np.argsort(top_points[:, 0])]
    bottom_points = bottom_points[np.argsort(bottom_points[:, 0])]
    
    # Assign points
    assigned_points = {
        'top_left': top_points[0],
        'top_mid': top_points[1],
        'top_right': top_points[2],
        'bottom_left': bottom_points[0],
        'bottom_mid': bottom_points[1],
        'bottom_right': bottom_points[2],
    }
    
    return assigned_points

def visualize_results(frame_gray_np, label_mask_np, output_np, assigned_points):
    """Visualize the input image, ground truth, model output, and detected points."""
    # Prepare to plot the centroids on the image
    frame_with_points = cv2.cvtColor(frame_gray_np, cv2.COLOR_GRAY2BGR)
    point_colors = {
        'top_left': (255, 0, 0),      # Blue
        'top_mid': (0, 255, 0),       # Green
        'top_right': (0, 0, 255),     # Red
        'bottom_left': (255, 255, 0), # Cyan
        'bottom_mid': (255, 0, 255),  # Magenta
        'bottom_right': (0, 255, 255) # Yellow
    }
    for label, centroid in assigned_points.items():
        x, y = int(centroid[0]), int(centroid[1])
        color = point_colors.get(label, (255, 255, 255))  # Default to white
        cv2.circle(frame_with_points, (x, y), 5, color, -1)
        cv2.putText(frame_with_points, label, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Convert BGR to RGB for displaying with Matplotlib
    frame_with_points_rgb = cv2.cvtColor(frame_with_points, cv2.COLOR_BGR2RGB)
    
    # Visualize
    plt.figure(figsize=(20, 5))
    
    # Input Image
    plt.subplot(1, 4, 1)
    plt.imshow(frame_gray_np, cmap='gray')
    plt.title('Input Image')
    plt.axis('off')
    
    # Ground Truth Label
    plt.subplot(1, 4, 2)
    plt.imshow(label_mask_np, cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')
    
    # Model Output
    plt.subplot(1, 4, 3)
    plt.imshow(output_np, cmap='gray')
    plt.title('Model Output')
    plt.axis('off')
    
    # Detected Points
    plt.subplot(1, 4, 4)
    plt.imshow(frame_with_points_rgb)
    plt.title('Detected Points')
    plt.axis('off')
    
    plt.savefig("test_vis.png")
    plt.show()

def main():
    # Set device to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the trained model
    model_path = 'models_corners/best_model.pth'  # Adjust the path if necessary
    model = load_model(model_path, device)
    
    # Instantiate the dataset
    dataset = TableTennisDataset(data_root='data/')
    
    # Number of samples to test
    num_samples = 1
    
    # Randomly select indices for testing
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for idx in indices:
        # Get a sample from the dataset
        frame_gray, label_mask = dataset[idx]
        frame_gray_np = frame_gray.squeeze().numpy()  # Shape: (H, W)
        label_mask_np = label_mask.squeeze().numpy()   # Shape: (H, W)
        
        # Process the image and detect points
        output_np, assigned_points = process_image(frame_gray, model, device, threshold=0.5)
        
        # Visualize the results
        visualize_results(frame_gray_np, label_mask_np, output_np, assigned_points)

if __name__ == '__main__':
    main()
