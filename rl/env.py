"""Environnement Blackjack : règles uniquement, aucune dépendance caméra."""
import random

STAND, HIT = 0, 1

def card_value(rank: str) -> int:
    """'A'->1 (11 géré par hand_total), '2'..'10', J/Q/K->10."""
    return 1 if rank == "A" else 10 if rank in ("J", "Q", "K") else int(rank)

def hand_total(values):
    """Retourne (total, as_utilisable). L'as vaut 11 si cela ne dépasse pas 21."""
    t = sum(values)
    return (t + 10, True) if 1 in values and t + 10 <= 21 else (t, False)

def outcome_solo(values):
    """Verdict utilisable aussi par l'app caméra. None = partie en cours."""
    t, _ = hand_total(values)
    if t > 21:  return "perdu (trop pioché)"
    if t == 21: return "gagné (21)"
    return None

class Blackjack:
    def __init__(self, dealer=False, seed=None):
        self.dealer = dealer
        self.rng = random.Random(seed)

    def draw(self):                      # paquet infini : 10/V/D/R = 4/13
        return min(self.rng.randint(1, 13), 10)

    def reset(self):
        self.hand = [self.draw(), self.draw()]
        self.house = [self.draw()]
        return self.state()

    def state(self):
        t, soft = hand_total(self.hand)
        return (t, soft, self.house[0]) if self.dealer else (t, soft)

    def step(self, action):
        """-> (état, récompense, fini, résultat)."""
        if action == HIT:
            self.hand.append(self.draw())
            res = outcome_solo(self.hand)
            if res: return self.state(), (-1 if res.startswith("perdu") else 1), True, res
            return self.state(), 0, False, None
        t, _ = hand_total(self.hand)     # STAND
        if not self.dealer:
            return self.state(), 0, True, "arrêt sous 21 (non gagné)"
        while hand_total(self.house)[0] < 17:
            self.house.append(self.draw())
        h, _ = hand_total(self.house)
        if h > 21 or t > h: return self.state(), 1, True, "gagné"
        if t == h:          return self.state(), 0, True, "égalité"
        return self.state(), -1, True, "perdu"
