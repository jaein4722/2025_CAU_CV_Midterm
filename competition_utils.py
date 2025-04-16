import json
import os
import numpy as np
import cv2
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import yaml
from shapely.geometry import box, Polygon
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import random
import matplotlib.font_manager as fm

def load_yaml_config(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def read_test_image_list(test_list_file):
    image_paths = []
    with open(test_list_file, 'r') as f:
        for line in f:
            image_path = line.strip()
            if image_path:  # 빈 줄 무시
                image_paths.append(image_path)
    return image_paths

def load_detection_results(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        detection_results = json.load(f)
    return detection_results

def load_yolo_labels(label_path, img_width, img_height):
   
    bboxes = []
    classes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:  # 클래스 ID, x_center, y_center, width, height
                # YOLO 형식: 클래스_ID, x_center, y_center, width, height (모두 정규화된 값)
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])
                
                # 정규화된 값을 픽셀 좌표로 변환
                x1 = (x_center - width/2) * img_width
                y1 = (y_center - height/2) * img_height
                x2 = (x_center + width/2) * img_width
                y2 = (y_center + height/2) * img_height
                
                bboxes.append([x1, y1, x2, y2])
                classes.append(class_id)

    return bboxes, classes

def calculate_union_bbox(bboxes):
    """여러 바운딩 박스의 합집합(union)을 계산하는 함수"""
    if not bboxes:
        return None
    
    polygons = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        polygons.append(box(x1, y1, x2, y2))
    
    union_polygon = polygons[0]
    for polygon in polygons[1:]:
        union_polygon = union_polygon.union(polygon)
    
    if isinstance(union_polygon, Polygon):
        bounds = union_polygon.bounds
    else:
        bounds = union_polygon.bounds
    
    return bounds

def calculate_iou(bbox1, bbox2):
    """두 바운딩 박스의 IoU(Intersection over Union)를 계산하는 함수"""
    if bbox1 is None or bbox2 is None:
        return 0.0
    
    x1_1, y1_1, x2_1, y2_1 = bbox1
    
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # 합집합 면적 계산
    union_area = bbox1_area + bbox2_area - intersection_area
    
    # IoU 계산
    iou = intersection_area / union_area if union_area > 0 else 0.0
    
    return iou

def visualize_and_save_comparison(image_path, detection_results, ground_truth_bboxes, 
                                 ground_truth_classes, iou, output_dir, class_names=None):
    """감지 결과와 Ground Truth를 시각화하고 IoU를 표시하여 저장하는 함수"""
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 이미지 로드
    image = cv2.imread(image_path)
    
    # BGR에서 RGB로 변환
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 이미지 크기
    height, width = image.shape[:2]
    
    # 플롯 생성
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    # 색상 설정
    detection_color = 'red'  # 감지 결과는 빨간색
    gt_color = 'green'  # Ground Truth는 녹색색
    
    # 랜덤 색상 생성 함수
    def get_random_color():
        return "#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])
    
    # 클래스별 색상 맵 생성
    if class_names:
        class_colors = {i: get_random_color() for i in range(len(class_names))}
    else:
        class_colors = {}
        
    # 각 감지 결과 바운딩 박스 그리기
    for i, detection in enumerate(detection_results):
        bbox = detection["bbox"]
        confidence = detection["confidence"]
        class_id = detection["class_id"]
        class_name = detection["class_name"] if "class_name" in detection else f"Class {class_id}"
        
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=detection_color, facecolor='none')
        ax.add_patch(rect)
        
        # 클래스 이름과 신뢰도 표시
        label = f"{class_name}: {confidence:.2f}"
        ax.text(x1, y1-5, label, color=detection_color, fontsize=8, backgroundcolor="white")
    
    # 각 Ground Truth 바운딩 박스 그리기
    for i, (bbox, class_id) in enumerate(zip(ground_truth_bboxes, ground_truth_classes)):
        x1, y1, x2, y2 = bbox
        
        # 클래스별 색상 사용 (있는 경우)
        edge_color = class_colors.get(class_id, gt_color)
        
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=edge_color, facecolor='none', linestyle='--')
        ax.add_patch(rect)
        
        # 클래스 이름 표시
        if class_names and class_id < len(class_names):
            class_name = class_names[class_id]
        else:
            class_name = f"Class {class_id}"
        
        ax.text(x1, y2+15, class_name, color=edge_color, fontsize=8, backgroundcolor="white")
    
    # IoU 값 표시
    ax.text(10, 30, f"IoU: {iou:.4f}", color='black', fontsize=12, backgroundcolor="white", weight='bold')
    
    # 범례 추가
    import matplotlib.lines as mlines
    
    det_line = mlines.Line2D([], [], color=detection_color, marker='s', linestyle='-', markersize=10, label='Detection')
    gt_line = mlines.Line2D([], [], color=gt_color, marker='s', linestyle='--', markersize=10, label='Ground Truth')
    
    ax.legend(handles=[det_line, gt_line], loc='upper right')
    
    # 축 제거
    plt.axis('off')
    
    # 파일명 생성 및 저장 (IoU 값을 파일명 앞에 추가)
    # IoU 값을 4자리 숫자로 변환 (예: 0.8546 -> 0854)
    iou_prefix = f"{iou:.4f}".replace(".", "")
    output_filename = f"iou_{iou_prefix}_file_name_{os.path.splitext(os.path.basename(image_path))[0]}.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    return output_path


def eval_and_vis(yaml_path, det_result_path, labels_dir, vis_output_dir, vis=False):
    config = load_yaml_config(yaml_path)
    class_names = config.get('names', None)
    test_path = os.path.join(config.get('path', ''), config.get('test', ''))
    test_image_list = read_test_image_list(test_path)
    
    detection_results = load_detection_results(det_result_path)
    if vis==True:
        os.makedirs(vis_output_dir, exist_ok=True)
    iou_results = {}
    
    for image_path in test_image_list:
        image = cv2.imread(image_path)
        img_height, img_width = image.shape[:2]
        detections = detection_results[image_path]
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(labels_dir, f"{image_name}.txt")
        detected_bboxes = [detection["bbox"] for detection in detections]
        ground_truth_bboxes, ground_truth_classes = load_yolo_labels(label_path, img_width, img_height)
        if not detected_bboxes and not ground_truth_bboxes:
            iou = 1.0  # 둘 다 없으면 완벽한 일치로 간주
        elif not detected_bboxes or not ground_truth_bboxes:
            iou = 0.0  # 하나만 없으면 전혀 일치하지 않음으로 간주
        else:
            detected_union_bbox = calculate_union_bbox(detected_bboxes)
            ground_truth_union_bbox = calculate_union_bbox(ground_truth_bboxes)
            iou = calculate_iou(detected_union_bbox, ground_truth_union_bbox)
        iou_results[image_path] = {
            "IoU": iou,
            "detected_boxes_count": len(detected_bboxes),
            "ground_truth_boxes_count": len(ground_truth_bboxes)
        }
        if vis==True:
            vis_path = visualize_and_save_comparison(
                image_path, 
                detections, 
                ground_truth_bboxes, 
                ground_truth_classes,
                iou, 
                vis_output_dir,
                class_names
            )
    iou_json_path =f"{datetime.now().strftime('%y%m%d_%H%M%S')}_iou_results_per_image.json"
    with open(iou_json_path, 'w', encoding='utf-8') as f:
        json.dump(iou_results, f, ensure_ascii=False, indent=4)
    print(f"이미지별 IoU 결과가 {iou_json_path}에 저장되었습니다.")
    
    if iou_results:
        iou_values = [result["IoU"] for result in iou_results.values()]
        avg_iou = sum(iou_values) / len(iou_values)
        std_dev = (sum((x - avg_iou) ** 2 for x in iou_values) / len(iou_values)) ** 0.5
        print(f"평균 IoU: {avg_iou:.4f}")
        print(f"표준 편차: {std_dev:.4f}")
        print(f"총 이미지 수: {len(iou_values)}")
        return avg_iou, std_dev
    return 0