# """Fine-tuning YOLO sur VOTRE dataset (même base que lab05/scripts/train_custom_yolo.py : yolo11n, CPU, batch 4).
# Le data.yaml est généré depuis config.VISION_CLASSES -> l'ordre des classes est toujours le bon.
# Run 1 : python -m vision.train_yolo --name yolo_run1 --model yolo11n.pt --epochs 30 --imgsz 416
# Run 2 : python -m vision.train_yolo --name yolo_run2 --model yolo11s.pt --epochs 60 --imgsz 640   (changements à noter dans le rapport)
# Poids : runs/detect/<name>/weights/best.pt -> copier vers vision/weights/best.pt"""
# import argparse, json, os
# from pathlib import Path
# import config

# def write_yaml(root, train, val):
#     root = Path(root).resolve()
#     lines = [f"path: {root.as_posix()}", f"train: {train}", f"val: {val}", "names:"]
#     lines += [f"  {i}: '{n}'" for i, n in enumerate(config.VISION_CLASSES)]
#     os.makedirs("runs", exist_ok=True)
#     out = Path("runs/data_generated.yaml"); out.write_text("\n".join(lines) + "\n", encoding="utf-8")
#     return str(out)

# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument("--name", required=True); p.add_argument("--model", default="yolo11n.pt")
#     p.add_argument("--epochs", type=int, default=30); p.add_argument("--imgsz", type=int, default=416); p.add_argument("--mosaic", type=float, default=0.0)
#     p.add_argument("--batch", type=int, default=4); p.add_argument("--device", default="cpu")
#     p.add_argument("--root", default="data/dataset"); p.add_argument("--train", default="images/train")
#     p.add_argument("--val", default="images/val")
#     a = p.parse_args()
#     from ultralytics import YOLO
#     YOLO(a.model).train(data=write_yaml(a.root, a.train, a.val), epochs=a.epochs, imgsz=a.imgsz,
#                         mosaic=a.mosaic, fliplr=0.0, batch=a.batch, device=a.device, name=a.name, project="runs/vision_logs")
#     os.makedirs("runs/vision_logs", exist_ok=True)
#     json.dump(vars(a), open(f"runs/vision_logs/{a.name}.json", "w"), indent=2)

# if __name__ == "__main__":
#     main()


"""Fine-tuning YOLO sur VOTRE dataset (même base que lab05/scripts/train_custom_yolo.py : yolo11n, CPU, batch 4).
Le data.yaml est généré depuis config.VISION_CLASSES -> l'ordre des classes est toujours le bon.
Run 1 : python -m vision.train_yolo --name yolo_run1 --model yolo11n.pt --epochs 30 --imgsz 416
Run 2 : python -m vision.train_yolo --name yolo_run2 --model yolo11s.pt --epochs 60 --imgsz 640   (changements à noter dans le rapport)
Poids : runs/detect/<name>/weights/best.pt -> copier vers vision/weights/best.pt"""
import argparse, json, os
from pathlib import Path
import config

def write_yaml(root, train, val):
    root = Path(root).resolve()
    lines = [f"path: {root.as_posix()}", f"train: {train}", f"val: {val}", "names:"]
    lines += [f"  {i}: '{n}'" for i, n in enumerate(config.VISION_CLASSES)]
    os.makedirs("runs", exist_ok=True)
    out = Path("runs/data_generated.yaml"); out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(out)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True); p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--epochs", type=int, default=80); p.add_argument("--imgsz", type=int, default=640); p.add_argument("--mosaic", type=float, default=0.0)
    p.add_argument("--batch", type=int, default=4); p.add_argument("--device", default="cpu")
    p.add_argument("--root", default="data/dataset"); p.add_argument("--train", default="images/train")
    p.add_argument("--val", default="images/val")
    a = p.parse_args()
    from ultralytics import YOLO
    YOLO(a.model).train(data=write_yaml(a.root, a.train, a.val), epochs=a.epochs, imgsz=a.imgsz,
                        mosaic=a.mosaic, fliplr=0.0, batch=a.batch, device=a.device, name=a.name, project="runs/vision_logs")
    os.makedirs("runs/vision_logs", exist_ok=True)
    json.dump(vars(a), open(f"runs/vision_logs/{a.name}.json", "w"), indent=2)

if __name__ == "__main__":
    main()
