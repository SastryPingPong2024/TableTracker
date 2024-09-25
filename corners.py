# corners.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from unet import UNet
from test import *

# Set device to GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the trained model
model = UNet(in_channels=1, out_channels=1)
model_path = 'models_corners/best_model.pth'  # Adjust the path if necessary
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()  # Set model to evaluation mode

def refine_corners(frame, bbox):
    frame = frame[bbox[2]:bbox[3], bbox[0]:bbox[1]]
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.resize(frame_gray, (256, 128), interpolation=cv2.INTER_AREA)
    frame_gray = torch.from_numpy(frame_gray).float().to(device).unsqueeze(0)  # Shape: (1, H, W)
    
    output_np, assigned_points = process_image(frame_gray, model, device, threshold=0.25)
    
    corners = np.array([
        assigned_points["top_left"],
        assigned_points["top_right"],
        assigned_points["bottom_left"],
        assigned_points["bottom_right"],
        assigned_points["top_mid"],
        assigned_points["bottom_mid"]
    ])
    corners[:, 1] *= (bbox[3]-bbox[2]) / 128
    corners[:, 0] *= (bbox[1]-bbox[0]) / 256
    corners += np.array([bbox[0], bbox[2]])
    
    return corners.round().astype(int)