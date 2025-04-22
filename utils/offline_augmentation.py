import os
import glob
import cv2
import random
import numpy as np

def gridmask(img):
    d_min, d_max = 64, 128   # 셀 크기 범위
    ratio = 0.5              # 셀 내 마스크 비율
    rotate = 0               # 회전 각도 (°)
    fill_value = 0           # 마스크된 픽셀 값
    
    h, w = img.shape[:2]
    d = random.randint(d_min, d_max)
    l = int(d * ratio + 0.5)
    st_h = random.randint(0, d)
    st_w = random.randint(0, d)

    # 마스크 생성 (1=유지, 0=가림)
    mask = np.ones((h, w), dtype=np.uint8)
    for i in range(-1, h // d + 1):
        s = i * d + st_h
        e = s + l
        mask[max(s,0):min(e,h), :] = 0
    for j in range(-1, w // d + 1):
        s = j * d + st_w
        e = s + l
        mask[:, max(s,0):min(e,w)] = 0

    # 회전
    if rotate:
        M = cv2.getRotationMatrix2D((w/2, h/2), random.uniform(-rotate, rotate), 1.0)
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)

    # 적용
    if img.ndim == 3:
        mask = mask[:, :, None]
    aug_img = img * mask + fill_value * (1 - mask)
    
    return aug_img

def augment_dataset(dataset_name: str):
    base_dir = os.path.join("Datasets", dataset_name)
    img_dir  = os.path.join(base_dir, "images")
    lbl_dir  = os.path.join(base_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    split_files = glob.glob(os.path.join(base_dir, "train_iter_*.txt"))

    # 이미 증강된 데이터가 하나라도 있으면 스킵
    already = any(
        glob.glob(os.path.join(img_dir, f"*{suffix}{ext}"))
        for suffix in ("_aug",)
        for ext in exts
    )
    if already:
        print("✅ 데이터셋이 이미 증강되어 있습니다. 작업을 건너뜁니다.")
        return

    # gridmask 파라미터
    grid_prob = 0.5          # 적용 확률

    for img_path in glob.glob(os.path.join(img_dir, "*")):
        stem, ext = os.path.splitext(os.path.basename(img_path))
        if ext.lower() not in exts:
            continue

        lbl_path = os.path.join(lbl_dir, f"{stem}.txt")
        if not os.path.isfile(lbl_path):
            print(f"⚠️ 레이블 없음, 스킵: {lbl_path}")
            continue

        # 원본 불러와 좌우 뒤집기
        img      = cv2.imread(img_path)
        aug_img  = cv2.flip(img, 1)

        # 확률적으로 GridMask 적용
        if random.random() < grid_prob:
            aug_img = gridmask(aug_img)

        # 레이블 읽어 x_center 뒤집기
        new_lines = []
        with open(lbl_path, "r") as f:
            for line in f:
                cls, xc, yc, bw, bh = line.strip().split()
                xc = 1.0 - float(xc)
                new_lines.append(f"{cls} {xc:.6f} {yc} {bw} {bh}")

        # 파일명 _aug
        out_img   = os.path.join(img_dir,   f"{stem}_aug{ext}")
        out_lbl   = os.path.join(lbl_dir,   f"{stem}_aug.txt")
        orig_line = os.path.join(base_dir, "images", f"{stem}{ext}")
        aug_line  = os.path.join(base_dir, "images", f"{stem}_aug{ext}")

        # 저장
        cv2.imwrite(out_img, aug_img)
        with open(out_lbl, "w") as wf:
            wf.write("\n".join(new_lines) + "\n")
        print(f"➕ 생성됨: {stem}_aug")

        # train split 파일에만 추가
        for split_file in split_files:
            with open(split_file, "r") as rf:
                lines = {l.strip() for l in rf if l.strip()}
            if orig_line in lines:
                with open(split_file, "a") as af:
                    af.write(aug_line + "\n")

    # ── 요약 ──
    originals = len([p for p in glob.glob(os.path.join(img_dir, "*"))
                     if os.path.splitext(p)[1].lower() in exts and not p.endswith("_aug" + os.path.splitext(p)[1])])
    augs = len([p for p in glob.glob(os.path.join(img_dir, "*_aug*"))
                 if os.path.splitext(p)[1].lower() in exts])
    print(f"\n완료: 원본 {originals}장 → aug {augs}장 (총 {originals + augs}장)")
