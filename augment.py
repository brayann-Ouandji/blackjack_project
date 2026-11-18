import os
import cv2
import numpy as np
from pathlib import Path

IMG_DIR = Path("data/dataset/images/train")
LBL_DIR = Path("data/dataset/labels/train")

def rotate_180_labels(labels):
    new_labels = []
    for line in labels:
        parts = line.strip().split()
        if len(parts) == 5:
            cls, x, y, w, h = parts
            new_x = 1.0 - float(x)
            new_y = 1.0 - float(y)
            new_labels.append(f"{cls} {new_x:.6f} {new_y:.6f} {w} {h}")
    return new_labels

def augment():
    images = [f for f in IMG_DIR.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"] and "_aug_" not in f.name]
    print(f"{len(images)} images originales trouvées.")

    for img_path in images:
        lbl_path = LBL_DIR / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue

        img = cv2.imread(str(img_path))
        labels = lbl_path.read_text(encoding="utf-8").splitlines()

        variations = []

        # 1. Plus lumineux (simule un éclairage fort)
        bright = cv2.convertScaleAbs(img, alpha=1.2, beta=30)
        variations.append(("bright", bright, labels))

        # 2. Plus sombre (simule une pièce peu éclairée)
        dark = cv2.convertScaleAbs(img, alpha=0.8, beta=-30)
        variations.append(("dark", dark, labels))

        # 3. Rotation 180° (les cartes restent lisibles)
        rot180 = cv2.rotate(img, cv2.ROTATE_180)
        variations.append(("rot180", rot180, rotate_180_labels(labels)))

        # 4. Rotation 180° + changement de contraste
        rot180_contrast = cv2.convertScaleAbs(rot180, alpha=1.15, beta=10)
        variations.append(("rot180_c", rot180_contrast, rotate_180_labels(labels)))

        # 5. Légère teinte / flou léger (simule une mise au point imparfaite)
        blur = cv2.GaussianBlur(img, (3, 3), 0)
        variations.append(("blur", blur, labels))

        for suffix, aug_img, aug_lbl in variations:
            new_name = f"{img_path.stem}_aug_{suffix}"
            cv2.imwrite(str(IMG_DIR / f"{new_name}{img_path.suffix}"), aug_img)
            (LBL_DIR / f"{new_name}.txt").write_text("\n".join(aug_lbl) + "\n", encoding="utf-8")

    print("Augmentation terminée !")

if __name__ == "__main__":
    augment()
