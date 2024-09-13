from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import csv

def extend_mask(mask, k_x, k_y, flag=False):
    extended_mask = mask.copy()
    
    m, n = mask.shape
    for i in range(m):
        for j in range(n):
            if mask[i][j] == 1:
                a, b, c, d = max(0, i-k_y), min(i+k_y+1, m), max(0, j-k_x), min(j+k_x+1, n)
                if flag: b = min(i+2, m)
                extended_mask[a:b, c:d] = 1
    return extended_mask

def get_detections(img_path, model):
    model_output = model(img_path)[0]
    orig_image = model_output.orig_img
    classes = model_output.boxes.cls
    
    table_detections = torch.where(classes == 0)
    if len(table_detections) == 1:
        table_mask = model_output.masks.data[table_detections[0]].squeeze().cpu().detach().numpy()
        fy = orig_image.shape[0] / table_mask.shape[0]
        fx = orig_image.shape[1] / table_mask.shape[1]
        table_mask = extend_mask(table_mask, 0, int(table_mask.sum()/4500))
        table_mask = cv2.resize(table_mask, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
    else:
        table_mask = None
    
    base_detections = torch.where(classes == 1)
    if len(base_detections) == 1:
        base_mask = model_output.masks.data[base_detections[0]].squeeze().cpu().detach().numpy()
        fy = orig_image.shape[0] / base_mask.shape[0]
        fx = orig_image.shape[1] / base_mask.shape[1]
        base_mask = extend_mask(base_mask, 0, int(base_mask.sum()/1250))
        base_mask = cv2.resize(base_mask, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
    else:
        base_mask = None
        
    return orig_image, table_mask, base_mask

def get_corners(table_mask):
    y, x = np.where(table_mask == 1)
    pts = np.array([x, y]).T
    tl = pts[np.argmax(pts @ [-1, -1])]
    bl = pts[np.argmax(pts @ [-1, +1])]
    tr = pts[np.argmax(pts @ [+1, -1])]
    br = pts[np.argmax(pts @ [+1, +1])]
    return np.array((tl, tr, bl, br))

def get_base(base_mask):
    y, x = np.where(base_mask == 1)
    pts = np.array([x, y]).T
    bl = pts[np.argmax(pts @ [-1, +1])]
    br = pts[np.argmax(pts @ [+1, +1])]
    return np.array((bl, br))

def process_frame(frame, model):    
    orig_image, table_mask, base_mask = get_detections(frame, model)
    corners = get_corners(table_mask)
    base = get_base(base_mask)
    base_height = (base[0][1] + base[1][1]) // 2
    return corners, base_height

def process_video(input_video_path, output_csv_name, indent=0):
    model = YOLO("models/yolov8n-seg-finetuned1.pt").cuda()
    
    print(f"{input_video_path}")
    cap = cv2.VideoCapture(input_video_path)

    data = []
    frame_number = 0
    corners = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
    base_height = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            corners, base_height = process_frame(frame, model)
        except Exception as e:
            pass
        
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
    
    with open(f"{output_csv_name}.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["frame","back_left_x","back_left_y","back_right_x","back_right_y","front_left_x","front_left_y","front_right_x","front_right_y","base_y","indent"])
        for row in data:
            writer.writerow(row)
    
    return data

if __name__ == "__main__":
    process_video("matches/match1/match1_3.mp4", "temp")

    