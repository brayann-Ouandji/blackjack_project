# """Enveloppe YOLO : image -> liste de Det. Seul fichier qui importe ultralytics."""
# from vision.occlusion import Det

# class CardDetector:
#     def __init__(self, weights="vision/weights/best.pt", imgsz=416, device="cpu"):
#         from ultralytics import YOLO          # import tardif : le reste du projet marche sans
#         self.model, self.imgsz, self.device = YOLO(weights), imgsz, device

#     def predict(self, frame):
#         r = self.model.predict(frame, imgsz=self.imgsz, device=self.device, verbose=False)[0]
#         names = self.model.names
#         return [Det(names[int(c)], float(p), tuple(map(float, b)))
#                 for b, p, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]


"""Enveloppe YOLO : image -> liste de Det. Seul fichier qui importe ultralytics."""
import config
from vision.occlusion import Det

class CardDetector:
    def __init__(self, weights="vision/weights/best.pt", imgsz=640, device="cpu"):
        from ultralytics import YOLO          # import tardif : le reste du projet marche sans
        self.model, self.imgsz, self.device = YOLO(weights), imgsz, device

    def predict(self, frame):
        r = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=config.MIN_CONF,
            device=self.device,
            verbose=False,
        )[0]
        names = self.model.names
        return [Det(names[int(c)], float(p), tuple(map(float, b)))
                for b, p, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]
