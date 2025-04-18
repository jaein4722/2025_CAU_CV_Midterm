from ..base_config import BaseConfig

class ModelConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.model_name = 'yolov9t'
        self.lr0 = 0.0025
        self.dfl = 1.3
        self.cls = 0.25
        self.box = 5.0
        
        self.custom_yaml_path = None # "models/YOLOv9t/yolov9t_custom.yaml"