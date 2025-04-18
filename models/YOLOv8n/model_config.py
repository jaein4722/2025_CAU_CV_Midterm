from ..base_config import BaseConfig

class ModelConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.model_name = 'yolov8n'
        self.lr0 = 0.003
        self.dfl = 1.5
        self.cls = 0.3
        self.box = 5.0
        
        self.custom_yaml_path = None # "models/YOLOv8n/yolov8n_custom.yaml"