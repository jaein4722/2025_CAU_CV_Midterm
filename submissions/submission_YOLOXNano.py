import os
import random
import json
from pathlib import Path
from datetime import datetime

import yaml
import cv2
import numpy as np
import torch
from PIL import Image
import gc
import torch

from models import YOLOXNano
from utils.ex_dict import update_ex_dict

def submission_YOLOXNano(yaml_path, output_json_path, config = None):

    ###### can be modified (Only Hyperparameters, which can be modified in demo) ######
    hyperparams = {
        'model_name': 'yoloxnano',
        'depth': 0.56,
        'width': 0.50,
        'depthwise': True,
        'epochs': 20,
        'batch_size': 16,
        'basic_lr_per_img': 0.01 / 16,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'scheduler': 'yoloxwarmcos',
        'no_aug_epochs': 2,
    }
    
    if config is None:
        config = YOLOXNano.ModelConfig()
    config.update_from_dict(hyperparams)
    data_config = load_yaml_config(yaml_path)
    ex_dict = {}
    ex_dict = update_ex_dict(ex_dict, config, initial=True)

    ###### can be modified (Only Models, which can't be modified in demo) ######
    ex_dict['Iteration'] = int(Path(yaml_path).stem.split('_')[-1])          # data_iter_01 → 1

    dataset_name = Path(yaml_path).parts[1]                                  # 예: "airplane"
    ex_dict['Dataset Name'] = dataset_name
    ex_dict['Data Config'] = yaml_path
    ex_dict['Number of Classes'] = data_config['nc']
    ex_dict['Class Names'] = data_config['names']

    control_random_seed(42)

    # ────────── 모델·경로 설정 ──────────
    os.makedirs(config.output_dir, exist_ok=True)
    ex_dict['Model Name'] = config.model_name
    ex_dict['Train Time'] = datetime.now().strftime("%y%m%d_%H%M%S")

    # ────────── 1) 학습 ──────────
    ex_dict = YOLOXNano.train_model(ex_dict, config)     # Model, PT path 포함
    model   = ex_dict['Model']

    # ────────── 2) 추론 ──────────
    test_images = get_test_images(data_config)
    results_dict = detect_and_save_bboxes(model, test_images, output_json_path, input_size=config.input_size, device=config.device)
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    


# ─────────────────────────────────────────────────────────────
#  헬퍼 함수
# ─────────────────────────────────────────────────────────────
def load_yaml_config(yaml_path):
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)


def get_test_images(cfg):
    root = cfg['path']
    test_entry = cfg['test']
    test_path = Path(root) / test_entry

    if test_path.is_dir():
        img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        return [str(p) for p in test_path.rglob('*') if p.suffix.lower() in img_exts]
    elif test_path.suffix == '.txt':
        return [ln.strip() for ln in open(test_path)]
    else:
        raise ValueError(f"지원되지 않는 test 경로: {test_path}")


def control_random_seed(seed, pytorch=True):
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        torch.backends.cudnn.benchmark = False


def detect_and_save_bboxes(model, image_paths, output_json_path, input_size, device='cuda'):
    """
    YOLOX predictor 래퍼를 호출해 기존 파이프라인과 동일한 results_dict를 반환.
    """
    results_dict = YOLOXNano.predict_and_save(model, image_paths, output_json_path, input_size, device=device)
    print(f"결과가 {output_json_path}에 저장되었습니다.")
    return results_dict