"""Vérifie le dataset YOLO : python -m vision.check_dataset [--root data/dataset] [--min 80]
Compte les boîtes par classe pour chaque sous-dossier de labels/ (train, val, Train…),
signale classes rares, fichiers vides, ids invalides, et un data.yaml dont l'ordre des classes diffère de config.py."""
import argparse, glob, os
from collections import Counter
import config

def check_yaml(root):
    f = os.path.join(root, "data.yaml")
    if not os.path.isfile(f): return
    import yaml
    names = yaml.safe_load(open(f, encoding="utf-8")).get("names", {})
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    names = [str(n) for n in names]
    print("data.yaml :", "ordre des classes OK" if names == config.VISION_CLASSES
          else f"ATTENTION ordre/noms différents !\n  yaml  = {names}\n  config = {config.VISION_CLASSES}")

def main():
    p = argparse.ArgumentParser(); p.add_argument("--root", default="data/dataset"); p.add_argument("--min", type=int, default=80)
    a = p.parse_args(); n_cls = len(config.VISION_CLASSES)
    check_yaml(a.root)
    total = Counter()
    splits = sorted(d for d in glob.glob(os.path.join(a.root, "labels", "*")) if os.path.isdir(d))
    if not splits: print("Aucun dossier labels/* trouvé dans", a.root)
    for d in splits:
        files = glob.glob(os.path.join(d, "*.txt"))
        c, empty = Counter(), 0
        for f in files:
            lines = [l.split() for l in open(f) if l.strip()]
            empty += not lines
            for l in lines:
                cid = int(l[0])
                if not 0 <= cid < n_cls: print(f"ERREUR classe {cid} invalide dans {f}")
                c[cid] += 1
        print(f"\n[{os.path.basename(d)}] {len(files)} images, {empty} sans étiquette, {sum(c.values())} boîtes")
        for i, name in enumerate(config.VISION_CLASSES):
            print(f"  {name:>6}: {c[i]}")
        total += c
    print("\nClasses sous le seuil :", [config.VISION_CLASSES[i] for i in range(n_cls) if total[i] < a.min] or "aucune")

if __name__ == "__main__":
    main()
