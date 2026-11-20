import os
import random
import shutil
from pathlib import Path

# Chemins vers vos dossiers
base_dir = Path(r"C:\Users\braya\Downloads\blackjack_project\data\dataset")
train_images_dir = base_dir / "images" / "train"
train_labels_dir = base_dir / "labels" / "train"
val_images_dir = base_dir / "images" / "val"
val_labels_dir = base_dir / "labels" / "val"

# Nombre d'images à déplacer de train vers val
NUM_TO_MOVE = 100

def move_files():
    # Lister  les images d'entraînement
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