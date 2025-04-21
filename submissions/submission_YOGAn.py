import gc
import os
import cv2
import yaml
import torch
import random
import numpy as np
from PIL import Image
from datetime import datetime
from models import YOGAn
from utils.ex_dict import update_ex_dict
from models.YOGAn.pkgs.utils.general import non_max_suppression, scale_coords
from utils.offline_augmentation import augment_dataset


def submission_YOGAn(yaml_path, output_json_path, config = None):
    
    ###### can be modified (Only Hyperparameters, which can be modified in demo) ######
    hyperparams = {
        'model_name': 'yogan',
        'epochs': 20,
        'batch': 16,
        'lr0': 0.01,
        'momentum': 0.937,
        'weight_decay': 5e-4,
        'optimizer': 'AdamW',
        'box': 0.05,
        'obj': 1.0, 
        'custom_yaml_path': "models/YOGAn/pkgs/yoga_models/YOGA-n.yaml",
    }
    
    conf = 0.25
    
    if config is None:
        config = YOGAn.ModelConfig()
    config.update_from_dict(hyperparams)
    data_config = load_yaml_config(yaml_path)
    ex_dict = {}
    ex_dict = update_ex_dict(ex_dict, config, initial=True)
    
    ###### can be modified (Only Models, which can't be modified in demo) ######
    from models.YOGAn import Model as YOGAnModel
    ex_dict['Iteration']  = int(yaml_path.split('.yaml')[0][-2:])
    
    Dataset_Name = yaml_path.split('/')[1]
    
    ex_dict['Dataset Name'] = Dataset_Name
    ex_dict['Data Config'] = yaml_path
    ex_dict['Number of Classes'] = data_config['nc']
    ex_dict['Class Names'] = data_config['names']
    
    augment_dataset(Dataset_Name)
    control_random_seed(42)
    
    if config.custom_yaml_path is not None:
        model_yaml_path = config.custom_yaml_path
    else:
        model_yaml_path = f'{config.model_name}.yaml'
    
    model = YOGAnModel(model_yaml_path)
    os.makedirs(config.output_dir, exist_ok=True)
    
    ex_dict['Model Name'] = config.model_name
    ex_dict['Model'] = model
    
    ex_dict = YOGAn.train_model(ex_dict, config)
    
    test_images = get_test_images(data_config)
    results_dict = detect_and_save_bboxes(ex_dict['Model'], test_images, config.imgsz, config.device, conf)
    save_results_to_file(results_dict, output_json_path)
    
    del model
    gc.collect()
    torch.cuda.empty_cache()


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
    
    
def control_random_seed(seed, pytorch=True):
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available()==True:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except:
        pass
        torch.backends.cudnn.benchmark = False 
        

def detect_and_save_bboxes(model, image_paths, imgsz, device, conf_thres=0.25, iou_thres=0.45):
    """
    model     : YOGAn Model 객체 (nn.Module)
    image_paths: List[str] – 입력 이미지 파일 경로 리스트
    imgsz     : int or Tuple[int,int] – 모델에 입력할 정사각(또는 h,w) 크기
    device    : str – 'cuda:0' or 'cpu'
    conf_thres: float – confidence threshold
    iou_thres : float – NMS IoU threshold
    """
    model.to(device).eval()
    results_dict = {}

    # ensure tuple
    if isinstance(imgsz, int):
        imgsz = (imgsz, imgsz)

    for img_path in image_paths:
        # 1) 원본 이미지 로드
        im0 = cv2.imread(img_path)                     # BGR
        assert im0 is not None, f"Image not found: {img_path}"
        h0, w0 = im0.shape[:2]

        # 2) 전처리: BGR→RGB, 리사이즈, [0,255]→[0.0,1.0], CHW, 배치차원
        img = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, imgsz, interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

        # 3) 추론
        with torch.no_grad():
            pred = model(img_tensor)[0]  # [num_preds, 5+classes]

        # 4) NMS
        det = non_max_suppression(pred.unsqueeze(0), conf_thres, iou_thres)[0]

        # 5) 예측 좌표를 원본 크기(h0,w0)로 복원
        if det is not None and len(det):
            det[:, :4] = scale_coords(imgsz, det[:, :4], (h0, w0)).round()

        # 6) 결과 포맷팅
        img_results = []
        if det is not None:
            for *xyxy, conf, cls in det.cpu().numpy():
                x1, y1, x2, y2 = [float(x) for x in xyxy]
                cls   = int(cls)
                img_results.append({
                    'bbox'      : [x1, y1, x2, y2],
                    'confidence': float(conf),
                    'class_id'  : cls,
                    'class_name': model.names[cls],
                })

        results_dict[img_path] = img_results

    return results_dict


def save_results_to_file(results_dict, output_path):
    import json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=4)
    print(f"결과가 {output_path}에 저장되었습니다.")