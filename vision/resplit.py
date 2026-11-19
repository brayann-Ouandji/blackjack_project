# """Re-découpe train/val de façon équilibrée (toutes les classes présentes dans val, mêmes types de scènes).
# python -m vision.resplit            -> simulation (affiche les comptes)
# python -m vision.resplit --apply    -> déplace les fichiers (supprime les images *_aug_* du train : relancez augment.py ensuite)"""
# import argparse, collections, glob, os, random, shutil
# import config

# def main():
#     p = argparse.ArgumentParser(); p.add_argument("--root", default="data/dataset")
#     p.add_argument("--val", type=float, default=0.2); p.add_argument("--seed", type=int, default=0)
#     p.add_argument("--apply", action="store_true"); a = p.parse_args()
#     pairs = []
#     for sp in ("train", "val"):
#         for img in glob.glob(os.path.join(a.root, "images", sp, "*")):
#             stem = os.path.splitext(os.path.basename(img))[0]
#             if "_aug_" in stem: continue
#             lab = os.path.join(a.root, "labels", sp, stem + ".txt")
#             if os.path.isfile(lab):
#                 cls = {int(l.split()[0]) for l in open(lab) if l.strip()}
#                 pairs.append((img, lab, cls))
#     rng = random.Random(a.seed); rng.shuffle(pairs)
#     n_val = max(1, round(a.val * len(pairs))); val = []
#     for c in sorted({c for *_, cl in pairs for c in cl}, key=lambda c: sum(c in cl for *_, cl in pairs)):  # rares d'abord
#         if not any(c in v[2] for v in val):
#             cand = [q for q in pairs if c in q[2] and q not in val]
#             if cand: val.append(cand[0])
#     for q in pairs:
#         if len(val) >= n_val: break
#         if q not in val: val.append(q)
#     train = [q for q in pairs if q not in val]
#     for name, grp in (("train", train), ("val", val)):
#         cnt = collections.Counter(c for q in grp for c in q[2])
#         print(f"[{name}] {len(grp)} images ; images contenant chaque classe :",
#               {config.VISION_CLASSES[c]: cnt[c] for c in sorted(cnt)})
#     if not a.apply: return print("\n(simulation) ajoutez --apply pour déplacer les fichiers")
#     for sp in ("train", "val"):
#         for f in glob.glob(os.path.join(a.root, "*", sp, "*_aug_*")): os.remove(f)
#     for sp, grp in (("train", train), ("val", val)):
#         for img, lab, _ in grp:
#             for src, kind in ((img, "images"), (lab, "labels")):
#                 dst = os.path.join(a.root, kind, sp, os.path.basename(src))
#                 if os.path.abspath(src) != os.path.abspath(dst):
#                     os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.move(src, dst)
#     for f in glob.glob(os.path.join(a.root, "labels", "*.cache")): os.remove(f)
#     print("OK")

# if __name__ == "__main__":
#     main()

import os
import random
import shutil
from pathlib import Path

# Chemins vers vos dossiers (à adapter si besoin)
base_dir = Path(r"C:\Users\braya\Downloads\blackjack_project\data\dataset")
train_images_dir = base_dir / "images" / "train"
train_labels_dir = base_dir / "labels" / "train"
val_images_dir = base_dir / "images" / "val"
val_labels_dir = base_dir / "labels" / "val"

# Nombre d'images à déplacer de train vers val
NUM_TO_MOVE = 100

def move_files():
    # Lister toutes les images d'entraînement
    all_train_images = list(train_images_dir.glob("*.jpg"))

    if len(all_train_images) < NUM_TO_MOVE:
        print("Pas assez d'images dans train.")
        return

    # Sélectionner aléatoirement 100 images
    images_to_move = random.sample(all_train_images, NUM_TO_MOVE)

    moved_count = 0
    for img_path in images_to_move:
        # Trouver le label .txt correspondant
        label_name = img_path.stem + ".txt"
        label_path = train_labels_dir / label_name

        # Déplacer l'image
        shutil.move(str(img_path), str(val_images_dir / img_path.name))

        # Déplacer le label s'il existe
        if label_path.exists():
            shutil.move(str(label_path), str(val_labels_dir / label_name))

        moved_count += 1

    print(f"{moved_count} images et labels ont été déplacés de 'train' vers 'val'.")

if __name__ == "__main__":
    move_files()