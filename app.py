"""python app.py --source 0 --weights vision/weights/best.pt --q runs/run2/q.pkl      (r = nouvelle manche, q = quitter)"""
import argparse
import cv2
import config
from vision.detector import CardDetector
from vision.occlusion import analyse
from game.tracker import CardTracker
from game.session import decide, split_zones
from rl.agent import QAgent

COL = {"hidden": (0, 0, 255), "wait": (200, 200, 200), "playing": (0, 200, 0), "over": (0, 165, 255)}

def draw_box(frame, d, color, text):
    x1, y1, x2, y2 = map(int, d.box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, text, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="vision/weights/best.pt"); p.add_argument("--q", required=True)
    p.add_argument("--source", default="0", help="index caméra (0, 1…) ou fichier vidéo de secours")
    a = p.parse_args()
    det, agent, tracker = CardDetector(a.weights), QAgent.load(a.q), CardTracker()
    cap = cv2.VideoCapture(int(a.source) if a.source.isdigit() else a.source)
    while True:
        ok, frame = cap.read()
        if not ok: break
        h = frame.shape[0]
        an = analyse(det.predict(frame), frame.shape)
        if config.DEALER_MODE:
            dealer, player = split_zones(an.valid, h)
            tracker.update([d.label for d in player]); dl = sorted(d.label for d in dealer)
        else:
            tracker.update([d.label for d in an.valid]); dl = None
        dec = decide(agent, an, tracker.confirmed(), h, dl)
        for d in an.valid:  draw_box(frame, d, (0, 200, 0), f"{d.label} {d.conf:.2f}")
        for d in an.hidden: draw_box(frame, d, (0, 0, 255), "CACHEE")
        cv2.putText(frame, dec.text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COL[dec.status], 3)
        cv2.imshow("Blackjack RL", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"): break
        if k == ord("r"): tracker.reset()
    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
