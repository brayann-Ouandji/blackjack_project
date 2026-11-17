"""Entraînement d'un run, journal CSV + modèle sauvegardés dans runs/<nom>/.
Ex : python -m rl.train --name run1 --episodes 5000 --eps 0.1 --eps-decay 1
     python -m rl.train --name run2 --episodes 200000 --eps 1 --eps-decay 0.99995
     python -m rl.train --name dealer --dealer --episodes 500000 --eps-decay 0.99999
"""
import argparse, csv, json, os
from rl.env import Blackjack
from rl.agent import QAgent
from rl.evaluate import evaluate

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--episodes", type=int, default=200_000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--eps", type=float, default=1.0)
    p.add_argument("--eps-decay", type=float, default=0.99995)
    p.add_argument("--eps-min", type=float, default=0.01)
    p.add_argument("--block", type=int, default=5000, help="évaluation tous les N épisodes")
    p.add_argument("--dealer", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    out = os.path.join("runs", a.name); os.makedirs(out, exist_ok=True)
    env, agent = Blackjack(a.dealer, seed=a.seed), QAgent(a.alpha, 1.0, a.eps, seed=a.seed)
    eval_env = Blackjack(a.dealer, seed=999)
    rows = []
    for ep in range(1, a.episodes + 1):
        s, done = env.reset(), False
        while not done:
            act = agent.act(s)
            s2, r, done, _ = env.step(act)
            agent.learn(s, act, r, s2, done)
            s = s2
        agent.eps = max(a.eps_min, agent.eps * a.eps_decay)
        if ep % a.block == 0:
            m = evaluate(eval_env, agent, 5000)
            w, l = m.get("gagné", 0), m.get("perdu", 0)
            rows.append({"episode": ep, "eps": round(agent.eps, 4), "win": w, "loss": l, "other": 100 - w - l})
    with open(os.path.join(out, "log.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    json.dump(vars(a), open(os.path.join(out, "params.json"), "w"), indent=2)
    agent.save(os.path.join(out, "q.pkl"))
    print(a.name, "final:", rows[-1])

if __name__ == "__main__":
    main()
