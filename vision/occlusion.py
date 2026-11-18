# """Détection des cartes cachées / mal séparées. Fonctions pures, testables sans caméra."""
# from dataclasses import dataclass, field
# import config

# @dataclass
# class Det:
#     label: str
#     conf: float
#     box: tuple          # (x1, y1, x2, y2)

# @dataclass
# class Analysis:
#     valid: list = field(default_factory=list)    # cartes lisibles et isolées
#     hidden: list = field(default_factory=list)   # cartes cachées / illisibles / chevauchées
#     message: str = None                          # texte à afficher si problème

# def _area(b): return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

# def overlap_ratio(a, b):
#     """Intersection / aire de la plus petite boîte (1.0 = une carte entièrement sous l'autre)."""
#     ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
#     iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
#     m = min(_area(a), _area(b))
#     return (ix * iy) / m if m else 0.0

# def analyse(dets, frame_shape=None):
#     """Classe chaque détection : valide ou cachée. Règles :
#     1) classe 'hidden' prédite par YOLO ; 2) confiance < MIN_CONF ;
#     3) chevauchement avec une autre carte ; 4) carte coupée par le bord de l'image."""
#     bad = set()
#     for i, d in enumerate(dets):
#         if d.label == config.HIDDEN or d.conf < config.MIN_CONF:
#             bad.add(i)
#         if frame_shape:
#             h, w = frame_shape[:2]; m = config.EDGE_MARGIN
#             if d.box[0] <= m or d.box[1] <= m or d.box[2] >= w - m or d.box[3] >= h - m:
#                 bad.add(i)
#         for j in range(i + 1, len(dets)):
#             if overlap_ratio(d.box, dets[j].box) > config.OVERLAP_THR:
#                 bad.update((i, j))
#     res = Analysis(
#         valid=[d for i, d in enumerate(dets) if i not in bad],
#         hidden=[d for i, d in enumerate(dets) if i in bad])
#     if res.hidden:
#         res.message = config.MSG_HIDDEN
#     return res


"""Détection des cartes cachées / mal séparées. Fonctions pures, testables sans caméra."""
from dataclasses import dataclass, field
import config

@dataclass
class Det:
    label: str
    conf: float
    box: tuple          # (x1, y1, x2, y2)

@dataclass
class Analysis:
    valid: list = field(default_factory=list)    # cartes lisibles et isolées
    hidden: list = field(default_factory=list)   # cartes cachées / illisibles / chevauchées
    message: str = None                          # texte à afficher si problème

def _area(b): return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

def overlap_ratio(a, b):
    """Intersection / aire de la plus petite boîte (1.0 = une carte entièrement sous l'autre)."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    m = min(_area(a), _area(b))
    return (ix * iy) / m if m else 0.0

def analyse(dets, frame_shape=None):
    """Classe chaque détection : valide ou cachée. Règles :
    1) classe 'hidden' prédite par YOLO ; 2) confiance < MIN_CONF ;
    3) chevauchement avec une autre carte ; 4) carte coupée par le bord de l'image."""
    bad = set()
    for i, d in enumerate(dets):
        if d.label == config.HIDDEN or d.conf < config.MIN_CONF:
            bad.add(i)
        if frame_shape:
            h, w = frame_shape[:2]; m = config.EDGE_MARGIN
            if d.box[0] <= m or d.box[1] <= m or d.box[2] >= w - m or d.box[3] >= h - m:
                bad.add(i)
        if not config.CORNER_MODE:                # en mode coins, des coins voisins ne sont pas un problème
            for j in range(i + 1, len(dets)):
                if overlap_ratio(d.box, dets[j].box) > config.OVERLAP_THR:
                    bad.update((i, j))
    res = Analysis(
        valid=[d for i, d in enumerate(dets) if i not in bad],
        hidden=[d for i, d in enumerate(dets) if i in bad])
    if res.hidden:
        res.message = config.MSG_HIDDEN
    return res
