from ..base_config import BaseConfig

class ModelConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.model_name = 'yolov10n'
        self.lr0 = 0.002
        self.dfl = 1.3
        self.cls = 0.2
        self.box = 5.0
        
        self.custom_yaml_path = None # "models/YOLOv10n/yolov10n_custom.yaml"