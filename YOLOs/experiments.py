import os
import yaml
import timeit
import numpy as np
import pandas as pd
import gc
import torch
from datetime import datetime
from train_utils import *
from config import *

def update_ex_dict(ex_dict, config: BaseConfig, initial = False):
    if initial:
        ex_dict['Experiment Time'] = config.experiment_time
    ex_dict['Epochs'] = config.epochs
    ex_dict['Batch Size'] = config.batch
    ex_dict['Device'] = config.device
    ex_dict['Optimizer'] = config.optimizer
    ex_dict['LR'] = config.lr0
    ex_dict['Weight Decay'] = config.weight_decay
    ex_dict['Momentum'] = config.momentum
    ex_dict['Image Size'] = config.imgsz
    ex_dict['Output Dir'] = config.output_dir
    ex_dict['LRF'] = config.lrf      # Fimal Cosine decay learning rate
    ex_dict['Cos LR'] = config.cos_lr    # Apply Cosine Scheduler

    # Data Augmentation
    ex_dict['hsv_h'] = config.hsv_h
    ex_dict['hsv_s'] = config.hsv_s
    ex_dict['hsv_v'] = config.hsv_v
    ex_dict['degrees'] = config.degrees

    ex_dict['translate'] = config.translate
    ex_dict['scale'] = config.scale
    ex_dict['flipud'] = config.flipud
    ex_dict['fliplr'] = config.fliplr
    ex_dict['mosaic'] = config.mosaic
    ex_dict['mixup'] = config.mixup
    ex_dict['copy_paste'] = config.copy_paste
    
    ex_dict['box'] = config.box
    ex_dict['cls'] = config.cls
    ex_dict['dfl'] = config.dfl
    
    return ex_dict

base_config = BaseConfig()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

configs = [v8Config(), v9Config(), v10Config(), v11Config(), v12Config()]
print('Experiment Start Time:', base_config.experiment_time)
ex_dict = {}
update_ex_dict(ex_dict, base_config, initial=True)

for iteration in range(base_config.iterations[0], base_config.iterations[1]+1):
    print(f'(Iter {iteration})')
    seed = iteration
    ex_dict['Iteration'] = iteration
    
    for j, Dataset_Name in enumerate(base_config.dataset_names):
        print(f'Dataset: {Dataset_Name} ({j+1}/{len(base_config.dataset_names)})'); 
        control_random_seed(seed)
        
        data_yaml_path = f"{base_config.dataset_root}/{Dataset_Name}/data_iter_{iteration:02d}.yaml"
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.load(f, Loader=yaml.FullLoader)
            
        ex_dict['Dataset Name'] = Dataset_Name
        ex_dict['Data Config'] = data_yaml_path
        ex_dict['Number of Classes'] = data_config['nc']
        ex_dict['Class Names'] = data_config['names']
        update_dataset_paths(base_config.dataset_root, Dataset_Name, iteration)
        
        for k, config in enumerate(configs):
            update_ex_dict(ex_dict, config)
            print(f'{config.model_name} ({k+1}/{len(configs)}) (Iter {iteration})', end=' ')
            print(f'Dataset: {Dataset_Name} ({j+1}/{len(base_config.dataset_names)})', end=' ')
            control_random_seed(seed)
            
            if iteration == 1:
                # Load base model config
                temp_model = YOLO(f'{config.model_name}.yaml', verbose=False)
                original_model_dict = temp_model.model.yaml

                # Save original yaml
                os.makedirs("models", exist_ok=True)
                original_yaml_path = os.path.join("models", f"{config.model_name}_original.yaml")
                with open(original_yaml_path, 'w') as f:
                    yaml.dump(original_model_dict, f, sort_keys=False)

                # Customize depth/width and modify corresponding scale value
                custom_model_dict = original_model_dict.copy()
                scale_key = config.model_name.strip()[-1]
                custom_depth = 0.2
                custom_width = 0.25

                # Update scale-specific values
                if 'scales' in custom_model_dict and scale_key in custom_model_dict['scales']:
                    custom_model_dict['scales'][scale_key][0] = custom_depth
                    custom_model_dict['scales'][scale_key][1] = custom_width

                # Also explicitly add depth_multiple and width_multiple
                custom_model_dict['depth_multiple'] = custom_depth
                custom_model_dict['width_multiple'] = custom_width

                # Save customized yaml
                custom_yaml_path = os.path.join("models", f"{config.model_name}_custom.yaml")
                with open(custom_yaml_path, 'w') as f:
                    yaml.dump(custom_model_dict, f, sort_keys=False)
                    
                del temp_model
            
            # Load customized yaml
            custom_yaml_path = os.path.join("models", f"{config.model_name}_custom.yaml")

            # Load modified model
            model = YOLO(custom_yaml_path, verbose=False)
            ex_dict['Model Name'] = config.model_name
            ex_dict['Model']=model
            update_ex_dict(ex_dict, config)
            
            start = timeit.default_timer()
            
            ex_dict = train_model(ex_dict, config)
            ex_dict = evaluate_model(ex_dict)
            
            ex_dict['Train-Test Time'] = timeit.default_timer() - start
            
            eval_dict = format_measures(ex_dict.get('Test Results'))
            output_csv = f"{ex_dict['Experiment Time']}_Results.csv"
            merge_and_update_df(ex_dict, eval_dict, output_csv, exclude_columns=['Model', 'Train Results', 'Test Results'])
            
            print('\n' + '='*50 + '\n')
            
            # Memory optim
            del model
            gc.collect()
            torch.cuda.empty_cache()
            
