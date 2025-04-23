# CV submission

### submission_1_20221555.py: FLDet-N

### submission_2_20221555.py: HyperYOLO-T

### submission_3_20221555.py: YOLOv9-tiny


### Execution guideline

> 꼭 새 가상환경을 만들어서 평가해주시면 감사하겠습니다!
> python 버전은 3.10.16입니다.

```
conda create -n CV_jaein4722 python=3.10
conda activate CV_jaein4722
pip install ipykernel
python -m ipykernel install --user --name CV_jaein4722 --dieplay-name CV_jaein4722
pip install -r requirements.txt
```

요청해주신 내용대로 requirements.txt 내용 남겨드립니다!
```
# Base dependencies
torch>=2.6.0
torchvision>=0.21.0
ultralytics==8.3.96
numpy>=1.22.2  # pinned to avoid vulnerabilities
opencv-python>=4.6.0
Pillow>=7.1.2
PyYAML>=6.0
requests>=2.23.0
scipy>=1.4.1

# COCO and Detection Tools
pycocotools>=2.0.6  # COCO mAP

# Utilities
scikit-learn
loguru
rich  # optional for enhanced logging
tqdm>=4.64.0
psutil
py-cpuinfo
natsort
shapely
albumentations>=1.0.3  # training augmentations
ipywidgets>=8.1.5
ninja
tabulate
thop>=0.1.1  # FLOPs computation
tensorboard>=2.13.0

# Visualization
matplotlib>=3.10.1
pandas>=1.1.4
seaborn>=0.11.0

# Export (optional)
# onnx>=1.13.0
# onnx-simplifier==0.4.10
# coremltools>=7.0    # CoreML export
# tensorflow>=2.4.1,<2.14  # TFLite export
# tensorflowjs>=3.9.0  # TF.js export
# nvidia-pyindex        # TensorRT export
# nvidia-tensorrt       # TensorRT export
# openvino-dev>=2023.0  # OpenVINO export

# Interactive and development
ipython  # interactive notebooks
```
