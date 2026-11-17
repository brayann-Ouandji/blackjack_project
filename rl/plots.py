"""Courbes d'apprentissage comparant plusieurs runs : python -m rl.plots run1 run2 dealer"""
import csv, sys, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main(names):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for n in names:
        rows = list(csv.DictReader(open(os.path.join("runs", n, "log.csv"))))
        x = [int(r["episode"]) for r in rows]
        ax[0].plot(x, [float(r["win"]) for r in rows], label=n)
        ax[1].plot(x, [float(r["loss"]) for r in rows], label=n)
    ax[0].set_title("Taux de victoire (%)"); ax[1].set_title("Taux de défaite (%)")
    for a in ax: a.set_xlabel("épisodes d'entraînement"); a.legend(); a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig("runs/comparaison.png", dpi=130)
    print("-> runs/comparaison.png")

if __name__ == "__main__":
    main(sys.argv[1:])
