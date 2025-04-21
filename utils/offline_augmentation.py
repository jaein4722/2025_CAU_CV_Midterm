import os
import glob
import cv2

def horizontal_flip_dataset(dataset_name: str):
    base_dir = os.path.join("Datasets", dataset_name)
    img_dir  = os.path.join(base_dir, "images")
    lbl_dir  = os.path.join(base_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    # 지원할 이미지 확장자
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    # train/val/test split 텍스트 파일 목록
    split_files = glob.glob(os.path.join(base_dir, "*_iter_*.txt"))

    for img_path in glob.glob(os.path.join(img_dir, "*")):
        stem, ext = os.path.splitext(os.path.basename(img_path))
        if ext.lower() not in exts:
            continue

        lbl_path    = os.path.join(lbl_dir, f"{stem}.txt")
        out_img     = os.path.join(img_dir, f"{stem}_fliplr{ext}")
        out_lbl     = os.path.join(lbl_dir, f"{stem}_fliplr.txt")
        orig_line   = os.path.join(base_dir, "images", f"{stem}{ext}")
        flip_line   = os.path.join(base_dir, "images", f"{stem}_fliplr{ext}")

        # 레이블 없으면 스킵
        if not os.path.isfile(lbl_path):
            print(f"⚠️ 레이블 없음, 스킵: {lbl_path}")
            continue

        # 이미 flip 결과 있으면 스킵
        if os.path.exists(out_img) and os.path.exists(out_lbl):
            print(f"✅ 이미 flip됨: {stem}_fliplr, 스킵")
            # split 파일만 업데이트 필요할 수도 있으니 아래 로직을 그대로 수행해도 OK
        else:
            # 1) 이미지 불러와 flip
            img      = cv2.imread(img_path)
            flip_img = cv2.flip(img, 1)

            # 2) 레이블 읽고 x_center 뒤집기
            new_lines = []
            with open(lbl_path, "r") as f:
                for line in f:
                    cls, xc, yc, bw, bh = line.strip().split()
                    xc = 1.0 - float(xc)
                    new_lines.append(f"{cls} {xc:.6f} {yc} {bw} {bh}")

            # 3) 저장
            cv2.imwrite(out_img, flip_img)
            with open(out_lbl, "w") as wf:
                wf.write("\n".join(new_lines) + "\n")
            print(f"➕ 생성됨: {stem}_fliplr")

        # 4) split 파일 업데이트
        for split_file in split_files:
            # 이미 파일 내에 orig_line 있는지, flip_line 없는지 확인
            with open(split_file, "r") as rf:
                lines = {l.strip() for l in rf if l.strip()}
            if orig_line in lines and flip_line not in lines:
                with open(split_file, "a") as af:
                    af.write(flip_line + "\n")
                print(f"   • {os.path.basename(split_file)} 에 추가: {flip_line}")

    # 요약
    originals = len([p for p in glob.glob(os.path.join(img_dir, "*"))
                     if os.path.splitext(p)[1].lower() in exts and not p.endswith("_fliplr" + os.path.splitext(p)[1])])
    flips = len([p for p in glob.glob(os.path.join(img_dir, "*_fliplr*"))
                 if os.path.splitext(p)[1].lower() in exts])
    print(f"\n완료: 원본 {originals}장 → flip {flips}장 (총 {originals + flips}장)")

if __name__ == "__main__":
    horizontal_flip_dataset("COCO_airplane")
