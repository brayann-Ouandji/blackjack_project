"""Relie détections -> état de jeu -> décision de l'agent RL."""
from collections import Counter
from dataclasses import dataclass
from rl.env import card_value, hand_total, outcome_solo, HIT
import config
import math

@dataclass
class Decision:
    status: str            # 'hidden' | 'wait' | 'playing' | 'over'
    text: str              # verdict / conseil
    total: int = 0
    faces: str = ""        # cartes AFFICHÉES telles quelles : "A K" (jamais 1 ou 10)

def split_zones(valid, frame_h):
    """Banquier = moitié haute de l'image, joueur = moitié basse (si DEALER_MODE)."""
    top = [d for d in valid if (d.box[1] + d.box[3]) / 2 < frame_h / 2]
    return top, [d for d in valid if d not in top]

def cards_from_corners(corner_labels):
    """Compte 1 carte dès 2 coins détectés (1 coin isolé = ignoré)."""
    k, cards = config.CORNERS_PER_CARD, []
    for lab, n in Counter(corner_labels).items():
        if n >= 2:
            cards += [lab] * math.ceil(n / k)
    return cards, False

def decide(agent, analysis, confirmed_labels, frame_h=None, dealer_labels=None,
           unresolved=False):
    """Calcule la décision à partir de l'analyse visuelle et des étiquettes confirmées."""
    if analysis and analysis.message:
        return Decision("hidden", analysis.message)

    raw_labels = list(confirmed_labels or [])
    if config.CORNER_MODE:
        labels, incomplete = cards_from_corners(raw_labels)
    else:
        labels, incomplete = raw_labels, False

    if not labels:
        return Decision("hidden" if unresolved else "wait",
                        config.MSG_HIDDEN if unresolved else "Posez vos cartes")

    values = [card_value(l) for l in labels]
    total, soft = hand_total(values)
    faces = " ".join(sorted(labels, key=config.RANKS.index))

    if unresolved or incomplete:
        return Decision("hidden", f"Total partiel {total} - {config.MSG_HIDDEN}", total, faces)

    outcome = outcome_solo(values)
    if outcome:
        return Decision("over", outcome, total, faces)

    if config.DEALER_MODE:
        d_labels = cards_from_corners(dealer_labels)[0] if (config.CORNER_MODE and dealer_labels) else dealer_labels
        if not d_labels:
            return Decision("wait", "Posez la carte du banquier", total, faces)
        state = (total, soft, card_value(d_labels[0]))
    else:
        state = (total, soft)

    action = "TIRER" if agent.act(state, greedy=True) == HIT else "RESTER"
    return Decision("playing", f"Total {total} -> {action}", total, faces)