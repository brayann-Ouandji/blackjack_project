"""Relie détections -> état de jeu -> décision de l'agent RL."""
from collections import Counter
from dataclasses import dataclass
from rl.env import card_value, hand_total, outcome_solo, HIT
import config

@dataclass
class Decision:
    status: str            # 'hidden' | 'wait' | 'playing' | 'over'
    text: str              # message principal (verdict / conseil)
    total: int = 0
    faces: str = ""        # cartes AFFICHÉES telles quelles : "A K" (jamais 1 ou 10)

def split_zones(valid, frame_h):
    """Banquier = moitié haute de l'image, joueur = moitié basse (si DEALER_MODE)."""
    top = [d for d in valid if (d.box[1] + d.box[3]) / 2 < frame_h / 2]
    return top, [d for d in valid if d not in top]

def cards_from_corners(corner_labels):
    """Coins détectés -> cartes. Une carte = CORNERS_PER_CARD coins du même rang.
    Retourne (liste de rangs, incomplet) ; incomplet=True si un rang n'a pas un nombre de coins multiple de 4
    (coin couvert, carte cachée ou coin manqué par YOLO)."""
    k, cards, incomplete = config.CORNERS_PER_CARD, [], False
    for lab, n in Counter(corner_labels).items():
        cards += [lab] * (n // k)
        incomplete |= n % k != 0
    return cards, incomplete

def decide(agent, analysis, confirmed_labels, frame_h=None, dealer_labels=None):
    if analysis.message:
        return Decision("hidden", analysis.message)
    if config.CORNER_MODE:
        confirmed_labels, incomplete = cards_from_corners(confirmed_labels or [])
        if dealer_labels:
            dealer_labels = cards_from_corners(dealer_labels)[0]
        if incomplete:
            return Decision("hidden", config.MSG_HIDDEN)
    if not confirmed_labels:
        return Decision("wait", "Posez vos cartes")
    values = [card_value(l) for l in confirmed_labels]   # valeurs : UNIQUEMENT pour la logique
    total, soft = hand_total(values)
    faces = " ".join(sorted(confirmed_labels, key=config.RANKS.index))   # affichage : rangs réels
    res = outcome_solo(values)
    if res:
        return Decision("over", res, total, faces)
    if config.DEALER_MODE:
        if not dealer_labels:
            return Decision("wait", "Posez la carte du banquier (moitié haute)", total, faces)
        state = (total, soft, card_value(dealer_labels[0]))
    else:
        state = (total, soft)
    action = "TIRER" if agent.act(state, greedy=True) == HIT else "RESTER"
    return Decision("playing", f"Total {total} -> {action}", total, faces)
