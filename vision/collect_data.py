"""Capture d'images webcam pour le dataset. ESPACE = sauver, ESC = quitter.
Ex : python -m vision.collect_data --out data/raw
Étiquetez ensuite avec Roboflow / Label Studio / CVAT (boîte = carte entière, classe = rang, 'hidden' si couverte)."""
import argparse, os, time
import cv2

def main():
    p = argparse.ArgumentParser(); p.add_argument("--out", default="data/raw"); p.add_argument("--source", default="0")
    a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
    cap, n = cv2.VideoCapture(int(a.source), cv2.CAP_DSHOW if a.source.isdigit() else a.source), 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        view = frame.copy()
        cv2.putText(view, f"{n} images | ESPACE=sauver ESC=quitter", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)
        cv2.imshow("collecte", view)
        k = cv2.waitKey(1) & 0xFF
        if k == 27: break
        if k == 32:
            cv2.imwrite(os.path.join(a.out, f"img_{int(time.time()*1000)}.jpg"), frame); n += 1
    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
