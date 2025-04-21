from ..base_config import BaseConfig
import os
import yaml

class ModelConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.model_name = 'lightweightob'
        self.lr0 = 0.01
        self.cls = 0.3
        self.box = 5.0
        self.obj = 1.0
        
        self.custom_yaml_path = "models/LightweightOB/pkgs/models/yolov5n_6_0_shufv2_ca.yaml" # "models/YOLO11n/yolo11n_custom.yaml"
        
    def update_from_dict(self, params: dict):
        for k, v in params.items():
            if not hasattr(self, k):
                raise AttributeError(f"ModelConfig has no attribute '{k}'")
            setattr(self, k, v)
        
    def convert_to_lightweightob_format(self, ex_dict: dict, name: str):

        self.data       = ex_dict['Data Config']   
        self.cfg        = self.custom_yaml_path
        self.weights    = ''          # scratch 학습
        self.batch_size = ex_dict['Batch Size']  
        self.epochs     = ex_dict['Epochs'] 
        self.imgsz      = ex_dict['Image Size']     
        self.project    = ex_dict['Output Dir'] + '/train'     
        self.name       = name
        self.hyp        = self.write_hyp_yaml()
        self.device     = ex_dict['Device']
        self.single_cls = True
        
        allowed_keys = {
            'data', 'cfg', 'weights', 'batch_size', 'epochs', 'imgsz', 'project', 'name', 'hyp', 'device', 'single_cls'
        }

        return self.hyperparams(allowed_keys=allowed_keys)
    
    def write_hyp_yaml(self, save_path: str = None) -> str:
        """
        LightweightOB의 기본 hyp.scratch.yaml 기본값을 로드한 뒤, 이 인스턴스의 속성으로 덮어쓴
        하이퍼파라미터 YAML 파일을 생성(덮어쓰기)하고, 경로를 반환합니다.
        Args:
            save_path: 저장할 경로. None이면 BaseConfig.output_dir에
                       'hyp_<model_name>.yaml' 로 저장합니다.
        Returns:
            생성된 (또는 덮어쓴) YAML 파일 경로
        """
        # 1) 기본 YAML 로드
        default_yaml = "models/LightweightOB/pkgs/data/hyps/hyp.scratch.yaml"
        with open(default_yaml, 'r') as f:
            defaults = yaml.safe_load(f)

        # 2) 속성 덮어쓰기 (없으면 기본값 유지)
        hyp_dict = {}
        for key, default_val in defaults.items():
            hyp_dict[key] = getattr(self, key, default_val)

        # 3) 저장 경로 결정
        if save_path is None:
            out_dir = self.get_output_dir()  # BaseConfig 메서드
            filename = f"hyp_{getattr(self, 'model_name', 'LightweightOB')}.yaml"
            save_path = os.path.join(out_dir, filename)

        # 4) 디렉터리 생성 및 YAML 쓰기
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            # sort_keys=False로 기본 순서 유지
            yaml.safe_dump(hyp_dict, f, sort_keys=False)

        return save_path