import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import csv
from ultralytics import YOLO
from scipy.ndimage import binary_dilation, median_filter
from corners import refine_corners

def median_smoothing(array, window_size):
    """
    Apply median smoothing to an array.

    Args:
        array (np.ndarray): Input array.
        q (int): Smoothing parameter.

    Returns:
        np.ndarray: Smoothed array.
    """
    smoothed_array = median_filter(array, size=(window_size, 1), mode='reflect')
    return smoothed_array.astype(int)

def smooth_data(data):
    data = np.array(data)
    k = 0
    while True:
        if (data[k][1:9] == 0).all():
            k += 1
        else:
            break
    data[:k, 1:] = data[k, 1:]
    data[:, 1:] = median_smoothing(data[:, 1:], 25)
    return data

def extend_mask(mask, k_x, k_y):
    kernel = np.ones((2 * k_y + 1, 2 * k_x + 1), dtype=int)
    extended_mask = binary_dilation(mask, structure=kernel).astype(mask.dtype)
    return extended_mask

def get_detections(img_path, model):
    model_output = model(img_path)[0]
    
    orig_image = model_output.orig_img
    classes = model_output.boxes.cls
    
    table_detections = torch.where(classes == 0)
    if len(table_detections) == 1:        
        x, y, w, h = model_output.boxes.xywh[table_detections[0]].squeeze().cpu().detach().numpy()
        w, h = 1.1*w, 1.3*h
        table_bbox = np.array([x - w / 2, x + w / 2, y - h / 2, y + h / 2])
    else:
        corners = None
    
    base_detections = torch.where(classes == 1)
    if len(base_detections) == 1:
        base_mask = model_output.masks.data[base_detections[0]].squeeze().cpu().detach().numpy()
        fy = orig_image.shape[0] / base_mask.shape[0]
        fx = orig_image.shape[1] / base_mask.shape[1]
        base_mask = extend_mask(base_mask, 0, int(base_mask.sum()/1250))
        base_mask = cv2.resize(base_mask, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
    else:
        base_mask = None
        
    return orig_image, table_bbox.round().astype(int), base_mask

def get_base(base_mask):
    y, x = np.where(base_mask == 1)
    pts = np.array([x, y]).T
    bl = pts[np.argmax(pts @ [-1, +1])]
    br = pts[np.argmax(pts @ [+1, +1])]
    return np.array((bl, br))

def process_frame(frame, model):    
    orig_image, table_bbox, base_mask = get_detections(frame, model)
    corners = refine_corners(orig_image, table_bbox)
    base = get_base(base_mask)
    base_height = (base[0][1] + base[1][1]) // 2
    return corners, base_height

def process_video(input_video_path, output_csv_name, indent=0):
    model = YOLO("models_yolo/yolov8n-seg-finetuned1.pt").cuda()
    
    print(f"{input_video_path}")
    cap = cv2.VideoCapture(input_video_path)

    data = []
    frame_number = 0
    corners = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
    base_height = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            corners, base_height = process_frame(frame, model)
        except Exception as e:
            print(e)
        
        for i in range(len(corners)):
            x, y = corners[i]
            if x == 0 and y == 0 and "prev_corners" in locals():
                corners[i] = prev_corners[i]
        prev_corners = corners 
        
        data.append([
            frame_number,
            *corners.flatten(),
            base_height,
            indent
        ])
    
        frame_number += 1

    cap.release()
    cv2.destroyAllWindows()
    
    data = smooth_data(data)
    with open(f"{output_csv_name}.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["frame","back_left_x","back_left_y","back_right_x","back_right_y","front_left_x","front_left_y","front_right_x","front_right_y","back_mid_x","back_mid_y","front_mid_x","front_mid_y","base_y","indent"])
        for row in data:
            writer.writerow(row)
    
    return data

if __name__ == "__main__":
    process_video("../matches/match406/match406_21.mp4", "temp")

    
    