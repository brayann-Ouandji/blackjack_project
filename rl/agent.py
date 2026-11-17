"""Agent Q-learning tabulaire (epsilon-greedy)."""
import pickle, random
from collections import defaultdict
from rl.env import STAND, HIT

class QAgent:
    def __init__(self, alpha=0.05, gamma=1.0, eps=1.0, seed=None):
        self.Q = defaultdict(lambda: [0.0, 0.0])
        self.alpha, self.gamma, self.eps = alpha, gamma, eps
        self.rng = random.Random(seed)

    def act(self, s, greedy=False):
        if not greedy and self.rng.random() < self.eps:
            return self.rng.choice((STAND, HIT))
        q = self.Q[s]
        return HIT if q[HIT] > q[STAND] else STAND

    def learn(self, s, a, r, s2, done):
        target = r if done else r + self.gamma * max(self.Q[s2])
        self.Q[s][a] += self.alpha * (target - self.Q[s][a])

    def save(self, path):
        with open(path, "wb") as f: pickle.dump(dict(self.Q), f)

    @classmethod
    def load(cls, path):
        a = cls(eps=0.0)
        with open(path, "rb") as f: a.Q.update(pickle.load(f))
        return a
