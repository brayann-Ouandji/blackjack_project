"""Stabilisation temporelle : évite qu'une détection qui clignote change la main."""
from collections import Counter, deque
import config

class CardTracker:
    def __init__(self, window=config.STABLE_FRAMES, ratio=config.STABLE_RATIO):
        self.hist, self.window, self.ratio = deque(maxlen=window), window, ratio

    def update(self, labels):
        self.hist.append(Counter(labels))

    def reset(self):
        self.hist.clear()

    def confirmed(self):
        """Pour chaque rang, le nombre de cartes vu de façon stable (>= ratio des frames)."""
        if len(self.hist) < self.window:
            return None
        out = []
        for lab in set().union(*self.hist):
            n, freq = Counter(h.get(lab, 0) for h in self.hist).most_common(1)[0]
            if n > 0 and freq / self.window >= self.ratio:
                out += [lab] * n
        return sorted(out)
