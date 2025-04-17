from datetime import datetime
import torch

class BaseConfig:
    def __init__(self, exp_time = datetime.now().strftime("%y%m%d_%H%M%S")):
        self.experiment_time = exp_time
        
        self.project_name = 'Object Detection'
        self.dataset_root = '../Datasets'
        self.output_dir = f'output/{exp_time}'
        self.dataset_names = ['airplane']
        self.model_name = None  # 개별 Config에서 지정
        self.exist_ok = False
        self.dataset_path = None
        
        # Pretrained and AMP
        self.pretrained = False
        self.amp = False

        self.iterations = [1, 10]
        
        self.epochs = 20
        self.patience = 100
        self.batch = 16
        self.imgsz = 640
        self.save = True
        self.save_period = -1
        self.cache = False

        # Optimizer settings
        self.optimizer = 'AdamW'
        self.lr0 = 1e-2
        self.lrf = 1e-2 # Final cosine LR
        self.momentum = 0.937
        self.weight_decay = 1e-4
        self.warmup_epochs = 3.0
        self.warmup_momentum = 0.8
        self.warmup_bias_lr = 0.1

        # Device
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.workers = 8

        # Data splits
        self.train_split = 0.6
        self.val_split = 0.2
        self.test_split = 0.2

        # Data Augmentation
        self.hsv_h = 0.015
        self.hsv_s = 0.7
        self.hsv_v = 0.4
        self.degrees = 10.0
        self.translate = 0.1
        self.scale = 0.5
        self.flipud = 0.0
        self.fliplr = 0.5
        self.mosaic = 1.0
        self.mixup = 0.5
        self.copy_paste = 0.0

        # Loss
        self.box = 5.0
        self.cls = 0.7
        self.dfl = 1.0
        
        # Training strategies
        self.single_cls = False
        self.rect = False
        self.multi_scale = False
        self.cos_lr = False
        self.close_mosaic = 10
        self.resume = False
        self.fraction = 1.0
        self.profile = False
        self.freeze = None
        
        # Segmentation specific
        self.overlap_mask = True
        self.mask_ratio = 4

        # Classification
        self.dropout = 0.0

        # Validation
        self.val = True
        self.plots = True

    def hyperparams(self, allowed_keys=None):
        full_dict = vars(self)
        if allowed_keys is None:
            allowed_keys = {
                'epochs', 'time', 'patience', 'batch', 'imgsz', 'save',
                'save_period', 'cache', 'workers', 'exist_ok',
                'pretrained', 'optimizer', 'deterministic', 'single_cls', 'classes',
                'rect', 'multi_scale', 'cos_lr', 'close_mosaic', 'resume', 'amp', 'fraction',
                'profile', 'freeze', 'lr0', 'lrf', 'momentum', 'weight_decay', 'warmup_epochs',
                'warmup_momentum', 'warmup_bias_lr', 'box', 'cls', 'dfl',
                'overlap_mask', 'mask_ratio', 'dropout', 'val', 'plots'
            }
        return {k: v for k, v in full_dict.items() if k in allowed_keys}


# ---- Individual model configs ----

class v8Config(BaseConfig):
    def __init__(self):
        super().__init__()
        self.model_name = 'yolov8n'


class v9Config(BaseConfig):
    def __init__(self):
        super().__init__()
        self.model_name = 'yolov9t'


class v10Config(BaseConfig):
    def __init__(self):
        super().__init__()
        self.model_name = 'yolov10n'


class v11Config(BaseConfig):
    def __init__(self):
        super().__init__()
        self.model_name = 'yolo11n'


class v12Config(BaseConfig):
    def __init__(self):
        super().__init__()
        self.model_name = 'yolo12n'