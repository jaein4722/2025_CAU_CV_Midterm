# models/YOLOXNano/model_config.py
from models.base_config import BaseConfig

class ModelConfig(BaseConfig):
    """
    YOLOX‑Nano 전용 하이퍼파라미터를 담는다.
    BaseConfig에서 공유하는 experiment_time, output_dir 등은 그대로 상속.
    """
    def __init__(self):
        super().__init__()
        self.model_name = "yoloxnano"
        self.exp_type   = "yolox_nano"      # get_exp(name=...) 인자
        self.depthwise  = True              # Nano 모델 특징
        
        # --- YOLOX 기본 하이퍼파라미터 (Exp 기본값 반영) ---
        # self.depth             = 0.56
        # self.width             = 0.50 -> 4.00M
        
        self.act               = 'silu'
        self.depth             = 0.56
        self.width             = 0.50
        self.input_size        = (self.imgsz, self.imgsz)
        self.test_size         = (self.imgsz, self.imgsz)
        
        # augmentation
        self.random_size       = (10, 20)
        self.multiscale_range  = 0
        self.mosaic_scale      = (0.5, 1.5)
        self.mosaic_prob       = 0.5
        self.enable_mixup      = False
        self.mixup_prob        = 1.0
        self.mixup_scale       = (0.5, 1.5)
        self.hsv_prob          = 1.0
        self.flip_prob         = 0.5
        self.shear             = 2.0
        self.degrees           = 10.0
        self.translate         = 0.1
        
        self.no_aug_epochs     = 2
        
        self.nmsthre           = 0.65
        self.test_conf         = 0.25
        self.min_lr_ratio      = 0.30
        self.scheduler         = 'yoloxwarmcos'
        self.ema               = True
        self.save_history_ckpt = False

        # --- 학습 스케줄・최적화 (BaseConfig 덮어쓰기) ---
        self.epochs            = 20
        self.batch_size        = 16            # exp.train_batch_size
        self.print_interval    = 10  
        self.eval_interval     = 5
        self.data_num_workers  = 8            # exp.data_num_workers

        self.basic_lr_per_img  = 0.01 / self.batch_size   # 이미지당 lr
        self.warmup_epochs     = 2
        self.warmup_lr         = 0
        self.weight_decay      = 5e-4
        self.momentum          = 0.9
        
        self.seed = 42
    
    def apply_to_exp(self, exp, ex_dict: dict):
        """
        ModelConfig의 속성을 YOLOX Exp 객체에 일괄 적용합니다.
        ex_dict를 통해 num_classes, data paths 등도 설정합니다.
        """
        # 데이터·실험 메타정보
        exp.num_classes       = ex_dict.get("Number of Classes", exp.num_classes)
        exp.exp_name          = f"{ex_dict.get('Train Time')}_{self.model_name}_{ex_dict.get('Dataset Name')}_Iter_{ex_dict.get('Iteration')}"
        
        # 모델 구조
        exp.depthwise         = self.depthwise
        exp.act               = self.act
        exp.depth             = self.depth
        exp.width             = self.width

        # 데이터·입력
        exp.input_size        = self.input_size
        exp.test_size         = self.test_size
        exp.data_num_workers  = self.data_num_workers

        # 경로 설정 (COCO JSON은 annotations 폴더에 저장돼야 함)
        idx = ex_dict.get('Iteration')
        config_path = ex_dict.get('Data Config')
        
        # yaml 파일이 위치한 폴더를 data_dir로 사용하여, 파일명 제거
        if config_path:
            from pathlib import Path
            exp.data_dir = str(Path(config_path).parent)
        exp.train_ann         = f"train_data_iter_{idx:02d}.json"
        exp.val_ann           = f"val_data_iter_{idx:02d}.json"
        exp.test_ann          = f"test_data_iter_{idx:02d}.json"

        # 학습 스케줄
        exp.max_epoch         = self.epochs
        exp.warmup_epochs     = self.warmup_epochs
        exp.basic_lr_per_img  = self.basic_lr_per_img
        exp.weight_decay      = self.weight_decay
        exp.momentum          = self.momentum
        exp.eval_interval     = self.eval_interval
        exp.print_interval    = self.print_interval
        exp.no_aug_epochs     = self.no_aug_epochs
        exp.min_lr_ratio      = self.min_lr_ratio
        exp.scheduler         = self.scheduler

        # Augmentation
        exp.random_size       = self.random_size
        exp.multiscale_range  = self.multiscale_range
        exp.mosaic_scale      = self.mosaic_scale
        exp.mosaic_prob       = self.mosaic_prob
        exp.enable_mixup      = self.enable_mixup
        exp.mixup_prob        = self.mixup_prob
        exp.mixup_scale       = self.mixup_scale
        exp.hsv_prob          = self.hsv_prob
        exp.flip_prob         = self.flip_prob
        exp.shear             = self.shear
        exp.degrees           = self.degrees
        exp.translate         = self.translate

        # 기타 설정
        exp.ema               = self.ema
        exp.save_history_ckpt = self.save_history_ckpt
        exp.nmsthre           = self.nmsthre
        exp.test_conf         = self.test_conf
        exp.seed              = self.seed

        return exp