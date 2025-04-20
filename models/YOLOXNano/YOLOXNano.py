# models/YOLOXNano/YOLOXNano.py

import os
import torch
import json, cv2
from types import SimpleNamespace
from pathlib import Path
import shutil
from tqdm import tqdm

from yolox.exp import get_exp               # :contentReference[oaicite:0]{index=0}
from yolox.core.trainer import Trainer       # :contentReference[oaicite:1]{index=1}
from yolox.data.data_augment import preproc
from yolox.utils import postprocess
from utils.yaml2coco import convert_yaml

from .model_config import ModelConfig

CLASS_NAMES = ["airplane"]  # 단일 클래스 프로젝트

def train_model(ex_dict: dict, config: ModelConfig):
    """
    ▶ ex_dict : 실험 메타정보
    ▶ config  : ModelConfig 인스턴스
    리턴값  : ex_dict (PT 경로, model 객체 포함)
    """

    exp = get_exp(exp_file=None, exp_name=config.exp_type)
    exp = config.apply_to_exp(exp=exp, ex_dict=ex_dict)
    
    dataset_name = ex_dict['Dataset Name']
    
    if ex_dict['Iteration'] == 1:
        # symbolic link 설정
        img_src = Path(f"Datasets/{dataset_name}/images")
        for split in ("train2017", "val2017"):
            dst = Path(f"Datasets/{dataset_name}")/split/"images"
            if dst.exists():
                # 이미 폴더나 링크가 있으면 삭제 후 재복사
                if dst.is_symlink() or dst.is_file():
                    dst.unlink()
                else:
                    shutil.rmtree(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(img_src, dst)
            
        root = Path(f"Datasets/{dataset_name}").resolve()
        out  = root / "annotations"
        ymls = sorted(root.glob("data_iter_*.yaml"))
        if not ymls:
            print("⚠️  매칭되는 YAML이 없습니다."); return
        for y in ymls:
            convert_yaml(y, out)

    # 출력 디렉터리 준비
    save_dir = Path(config.output_dir) / "train"
    os.makedirs(save_dir, exist_ok=True)
    exp.output_dir = str(save_dir)

    # ─────────────────────────────────────────────
    # 2) Trainer 인스턴스에 넘길 args 정의
    # ─────────────────────────────────────────────
    args = SimpleNamespace(
        batch_size      = config.batch_size,
        fp16            = False,
        cache           = None,
        logger          = "tensorboard",    # or "wandb"/"mlflow"
        ckpt            = None,
        resume          = False,
        start_epoch     = None,
        exp_file        = None,
        experiment_name = exp.exp_name,
        occupy          = False,
    )

    # ─────────────────────────────────────────────
    # 3) 학습 실행 (단일 GPU)
    # ─────────────────────────────────────────────
    trainer = Trainer(exp, args)
    trainer.train()

    # ─────────────────────────────────────────────
    # 4) best_ckpt 로드 & model 세팅
    # ─────────────────────────────────────────────
    pt_dir = Path(config.output_dir) / "train" / exp.exp_name
    best_pt = pt_dir / "best_ckpt.pth"
    if not best_pt.exists():
        best_pt = pt_dir / "latest_ckpt.pth"

    model = exp.get_model()
    ckpt  = torch.load(best_pt, map_location=config.device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ─────────────────────────────────────────────
    # 5) 결과 저장 후 반환
    # ─────────────────────────────────────────────
    ex_dict["Train Results"] = str(best_pt)
    ex_dict["PT path"]       = str(best_pt)
    ex_dict["Model"]         = model
    return ex_dict


def sigmoid(x):        # YOLOX raw output엔 시그모이드 적용 필요
    return 1.0 / (1.0 + torch.exp(-x))


def detect_and_collect(model, img_path: str, input_size: tuple[int,int], device="cuda"):
    """
    1장 이미지 → bbox dict list 반환 (프로젝트 표준 포맷)
    """
    img_raw = cv2.imread(img_path)
    img, ratio = preproc(img_raw, input_size, swap=(2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float().to(device)
    
    config = ModelConfig()

    with torch.no_grad():
        outputs = model(img)
        outputs = postprocess(outputs, num_classes=1, conf_thre=config.test_conf, nms_thre=config.nmsthre)

    result_list = []
    if outputs[0] is not None:
        outs = outputs[0].cpu()
        bboxes = outs[:, 0:4] / ratio
        scores = sigmoid(outs[:, 4]) * sigmoid(outs[:, 5])   # obj * cls
        for bbox, score in zip(bboxes, scores):
            x1, y1, x2, y2 = bbox.tolist()
            result_list.append({
                "bbox":      [x1, y1, x2, y2],
                "confidence": float(score),
                "class_id":   0,
                "class_name": CLASS_NAMES[0],
            })
    return result_list


def predict_and_save(model, image_paths: list[str], output_json: str,
                     input_size: tuple[int,int], device="cuda"):
    """
    input_size: (height, width) from ModelConfig
    """
    model.to(device).eval()
    results_dict = {}
    for p in tqdm(image_paths, desc="YOLOX inference"):
        results_dict[p] = detect_and_collect(model, p, input_size, device)

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=4)
    print(f"✓ YOLOX 결과 저장 → {output_json}")
    return results_dict