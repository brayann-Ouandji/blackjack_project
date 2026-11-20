RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
# Classes YOLO : 13 rangs + 'hidden' (carte vue mais illisible/couverte, étiquetée à la main)
VISION_CLASSES = RANKS + ["hidden"]   # index 0 = A ... 12 = K, 13 = hidden (ordre exporté par CVAT)
HIDDEN = "hidden"


# Vision
CORNER_MODE = True        # dataset étiqueté sur les COINS (4 par carte) : une carte = 4 détections du même rang
CORNERS_PER_CARD = 4
MIN_CONF = 0.65          # en dessous : carte considérée illisible -> "carte cachée"
OVERLAP_THR = 0.15       # part de la plus petite boîte recouverte par une autre
EDGE_MARGIN = 0          # px : carte collée au bord de l'image = coupée
STABLE_FRAMES = 10        # fenêtre de vote temporel
STABLE_RATIO = 0.8       # part des frames où le même compte doit apparaître
MSG_HIDDEN = "Carte cachée - séparez les cartes"

# Jeu
DEALER_MODE = False      # False = joueur seul (étape 1) ; True = avec banquier
