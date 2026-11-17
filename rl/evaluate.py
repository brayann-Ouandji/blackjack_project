"""Évaluation d'une politique (greedy ou aléatoire) : pourcentages de résultats."""
import random
from collections import defaultdict
from rl.env import STAND, HIT

def evaluate(env, agent=None, n=20_000):
    res, rng = defaultdict(int), random.Random(123)
    for _ in range(n):
        s, done = env.reset(), False
        while not done:
            a = rng.choice((STAND, HIT)) if agent is None else agent.act(s, greedy=True)
            s, _, done, out = env.step(a)
        res[out.split(" ")[0]] += 1
    return {k: 100 * v / n for k, v in res.items()}

def policy_table(agent, dealer=False):
    """total -> 'TIRER'/'RESTER' (sans as utilisable) pour le rapport."""
    rows = {}
    for t in range(4, 21):
        s = (t, False, 10) if dealer else (t, False)
        rows[t] = "TIRER" if agent.act(s, greedy=True) == HIT else "RESTER"
    return rows
