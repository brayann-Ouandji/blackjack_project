RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
# Index 0 = 'none' (inutilisé), 1 = 'A', 2 = '2', ..., 13 = 'K', 14 = 'hidden
# Classes YOLO : 13 rangs + 'hidden' (carte vue mais illisible/couverte, étiquetée à la main)
VISION_CLASSES = ["none"] + RANKS + ["hidden"]
HIDDEN = "hidden"


# Vision
MIN_CONF = 0.50          # en dessous : carte considérée illisible -> "carte cachée"
OVERLAP_THR = 0.15       # part de la plus petite boîte recouverte par une autre
EDGE_MARGIN = 5          # px : carte collée au bord de l'image = coupée
STABLE_FRAMES = 8        # fenêtre de vote temporel
STABLE_RATIO = 0.7       # part des frames où le même compte doit apparaître
MSG_HIDDEN = "Carte cachée - séparez les cartes"

# Jeu
DEALER_MODE = False      # False = joueur seul (étape 1) ; True = avec banquier
