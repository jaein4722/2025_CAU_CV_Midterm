import os
import sys
import yaml
import random
import numpy as np
import pandas as pd
from datetime import datetime
import torch
from ultralytics import settings, YOLO

from .model_config import ModelConfig
from .pkgs.train import run as yoga_train
from .pkgs.yoga_models.experimental import attempt_load
from .pkgs.utils.torch_utils import select_device  # 혹은 적절한 device 셀렉터

settings.update({'datasets_dir': './'})

def train_model(ex_dict, config: ModelConfig):
    ex_dict['Train Time'] = datetime.now().strftime("%y%m%d_%H%M%S")
    name = f"{ex_dict['Train Time']}_{ex_dict['Model Name']}_{ex_dict['Dataset Name']}_Iter_{ex_dict['Iteration']}"
    
    yogan_config = config.convert_to_yogan_format(ex_dict, name)

    # YOGA train.py의 run 인터페이스 호출
    sys.argv = [''] # reset jupyter's system argv
    yoga_train(**yogan_config)

    # best.pt 경로를 ex_dict에 기록
    pt_path = f"{ex_dict['Output Dir']}/train/{name}/weights/best.pt"
    ex_dict['PT path'] = pt_path
    model = attempt_load(pt_path, map_location=select_device(config.device))
    ex_dict['Model'] = model
    return ex_dict