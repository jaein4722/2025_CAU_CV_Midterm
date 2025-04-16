import yaml
import os
import cv2
import numpy as np
from PIL import Image
import torch
from datetime import datetime

def submission_function(yaml_path, output_json_path):
    ###### can be modified (Only Hyperparameters, which can be modified in demo) ######
    config = load_yaml_config(yaml_path)
    
    ###### can be modified (Only Models, which can't be modified in demo) ######
    test_images = get_test_images(config)
    results_dict = detect_and_save_bboxes('tmp_model.pt', test_images)
    ###### can be modified ######
    save_results_to_file(results_dict, output_json_path)

def load_yaml_config(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_test_images(config):
    test_path = config['test']
    root_path = config['path']

    test_path = os.path.join(root_path, test_path)
    
    if os.path.isdir(test_path):
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        image_paths = []
        for root, _, files in os.walk(test_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(root, file))
        return image_paths
    elif test_path.endswith('.txt'):
        with open(test_path, 'r') as f:
            image_paths = [line.strip() for line in f.readlines()]
        return image_paths
    

def detect_and_save_bboxes(model_path, image_paths):
    from ultralytics import YOLO
    model = YOLO(model_path)
    
    results_dict = {}

    for img_path in image_paths:
        results = model(img_path, verbose=False, task='detect')
        img_results = []
        for result in results:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                bbox = box.xyxy[0].tolist()
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                class_name = result.names[class_id]
                img_results.append({
                    'bbox': bbox,  # [x1, y1, x2, y2]
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name
                })
        results_dict[img_path] = img_results
    return results_dict

def save_results_to_file(results_dict, output_path):
    import json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=4)
    print(f"결과가 {output_path}에 저장되었습니다.")


