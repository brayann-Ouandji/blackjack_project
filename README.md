# Blackjack – reconnaissance de cartes (webcam + YOLO) + apprentissage par renforcement
Aligné sur le cours (dépôt WYTan-THMA/AIWithPythonESIGELEC, lab05) : yolo11n, CPU, CVAT, webcam, démo live.

## Architecture
```
config.py            seuils et classes partagés
rl/       env.py (règles) · agent.py (Q-learning) · train.py · evaluate.py · plots.py
vision/   collect_data.py · check_dataset.py · train_yolo.py · detector.py · occlusion.py (carte cachée)
game/     tracker.py (stabilisation sur N frames) · session.py (détections -> décision RL)
app.py    démo live (--source 0 = webcam, ou un fichier vidéo de secours)
tests/    python -m tests.test_all   (sans caméra)
```
Installation : même `.venv` que le cours, `pip install -r requirements.txt` (mêmes versions que `lab05/requirements-lab05.txt`).

## Phase A – Règles + RL sans caméra (FAIT)
1. `python -m tests.test_all`
2. `python -m rl.train --name run1 --episodes 5000 --eps 0.1 --eps-decay 1 --block 500`
3. `python -m rl.train --name run2 --episodes 200000`
4. `python -m rl.train --name dealer --dealer --episodes 300000 --eps-decay 0.99998`
5. `python -m rl.plots run1 run2 dealer` -> `runs/comparaison.png`
Réussite : solo = 0 % de défaite ; avec banquier ~43 % gagné / ~48 % perdu.

## Phase B – Données (CVAT, comme au lab05)
**B1 Matériel.** Jeu de 52 cartes, table à fond uni, lumière stable, webcam fixe (même point de vue que la démo).
**B2 Quantité.** ~100 boîtes par rang (pas par image : 5 cartes/image = 5 boîtes) -> 250 à 400 images. Variez cartes (4 couleurs par rang), disposition, lumière, 2 fonds minimum.
Répartition : 70-80 % cartes séparées et lisibles, 20-30 % cartes chevauchées/couvertes/coupées (classe `hidden`).
**B3 Capture.** `python -m vision.collect_data --out data/raw --source 0` (ESPACE = sauver, ESC = quitter). Changez la scène entre deux photos. (`lab05/scripts/webcam_capture.py` fait la même chose.) Gardez les images originales : le compte CVAT gratuit peut ne pas les réexporter.
**B4 Annotation dans CVAT Online (app.cvat.ai).**
1. Se connecter (GitHub/Google). 2. Tasks -> Create new task `Blackjack-Cartes`.
3. C'est le vrai projet de groupe : **là, utilisez le Project** (le prof demande de ne pas le consommer pour le lab, pas pour le projet final).
4. Labels, dans cet ordre exact : `A, 2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K, hidden`.
5. Rectangles : une boîte par carte **entière**, classe = rang (couleur ignorée). `hidden` = rang non lisible (couvert, doigt, coupé) : boîte sur la partie visible. Une carte posée dessus et lisible garde son rang. Aucune carte visible sans boîte.
6. Règles écrites dans un doc partagé, découpage du travail à 3 par lots d'images.
**B5 Export.** Export annotations en **Ultralytics YOLO Detection**. Chaque `.txt` : `classe x_centre y_centre largeur hauteur` (0 à 1). Associez les `.txt` aux images originales (mêmes noms).
**B6 Organisation.** Rangez en `data/dataset/images/{train,val}` et `labels/{train,val}` (~80/20, **par scène** : photos quasi identiques dans le même groupe) et copiez le `data.yaml` exporté dans `data/dataset/`. Si CVAT nomme les dossiers autrement (Train…), adaptez ou passez `--train/--val` à `train_yolo`.
**B7 Contrôle.** `python -m vision.check_dataset --root data/dataset` : compte par classe, images vides, ids invalides, et **alerte si l'ordre des classes du data.yaml diffère de config.py**.
Réussite : ≥ 80 boîtes par classe (hidden inclus), aucune erreur, 20 images relues à l'œil, aucune scène partagée entre train et val.
Conseil : faites d'abord 30 images, un mini-entraînement de 5 min, et vérifiez que le pipeline marche.

## Phase C – Reconnaissance (≥ 2 entraînements documentés)
Le `data.yaml` d'entraînement est généré depuis `config.py` : l'ordre des classes est toujours bon.
1. Run 1 (base du prof) : `python -m vision.train_yolo --name yolo_run1 --model yolo11n.pt --epochs 30 --imgsz 416`
2. Analysez la matrice de confusion (6/9, J/Q…), notez le problème.
3. Run 2 (changement documenté) : `python -m vision.train_yolo --name yolo_run2 --model yolo11s.pt --epochs 60 --imgsz 640` (ou plus d'images/augmentations). Les paramètres sont sauvés dans `runs/vision_logs/`.
4. Copiez le meilleur `best.pt` vers `vision/weights/best.pt`.
Note : l'entraînement CPU est lent (batch 4 par défaut) ; commencez petit.

## Phase D – Cartes cachées / séparées
Message « Carte cachée - séparez les cartes » si : classe `hidden` · confiance < 0,50 · chevauchement > 15 % · carte coupée par le bord. Réglez `MIN_CONF` et `OVERLAP_THR` dans `config.py`.
Réussite : aucune main calculée tant que le message s'affiche.

## Phase E – Démo live
`python app.py --source 0 --weights vision/weights/best.pt --q runs/run2/q.pkl` (r = nouvelle manche, q = quitter).
Sans webcam : `--source chemin/video.mp4`. Si la caméra 0 échoue : `--source 1`.

## Phase F – Extensions
Banquier : `DEALER_MODE = True` + modèle `--q runs/dealer/q.pkl` (moitié haute de l'image = banquier). 2 joueurs : une zone par joueur + `Blackjack2P` dans `rl/env.py`.
