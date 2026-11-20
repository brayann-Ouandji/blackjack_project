"""Lancer : python -m tests.test_all"""
import config
from rl.env import Blackjack, hand_total, card_value, outcome_solo, HIT, STAND
from rl.agent import QAgent
from vision.occlusion import Det, analyse, overlap_ratio
from game.tracker import CardTracker
from game.session import decide, cards_from_corners

def test_rules():
    assert hand_total([1, 10]) == (21, True)
    assert hand_total([1, 1, 9]) == (21, True)
    assert hand_total([10, 6, 8]) == (24, False)
    assert card_value("K") == 10 and card_value("A") == 1
    assert outcome_solo([10, 6, 8]).startswith("perdu")
    assert outcome_solo([1, 10]).startswith("gagné")
    assert outcome_solo([10, 5]) is None

def test_env_stand_and_bust():
    env = Blackjack(seed=1); env.reset(); env.hand = [10, 5]
    assert env.step(STAND)[2] is True
    env.reset(); env.hand = [10, 6]; env.draw = lambda: 10
    assert env.step(HIT)[1] == -1

def test_occlusion():
    ok = [Det("7", .9, (100, 100, 200, 250)), Det("K", .9, (300, 100, 400, 250))]
    assert analyse(ok, (480, 640)).message is None
    cover = ok + [Det("3", .9, (320, 120, 420, 270))]
    assert analyse(cover, (480, 640)).message is None                  # mode coins : le chevauchement ne compte pas
    config.CORNER_MODE = False
    a = analyse(cover, (480, 640)); assert a.message == config.MSG_HIDDEN and len(a.valid) == 1
    config.CORNER_MODE = True
    assert analyse([Det("7", .3, (100, 100, 200, 250))], (480, 640)).message
    assert analyse([Det("hidden", .9, (100, 100, 200, 250))], (480, 640)).message
    assert analyse([Det("7", .9, (0, 100, 100, 250))], (480, 640)).message
    assert overlap_ratio((0, 0, 10, 10), (20, 20, 30, 30)) == 0

def test_tracker_stability():
    t = CardTracker(window=5, ratio=0.6)
    for i in range(5): t.update(["7", "K"] if i != 2 else ["7"])
    assert t.confirmed() == ["7", "K"]

def test_decision_flow():
    ag = QAgent(eps=0.0)
    ag.Q[(12, False)] = [0.0, 0.5]
    ok = analyse([Det("2", .9, (10, 10, 60, 90)), Det("K", .9, (200, 10, 260, 90))], (480, 640))
    C = lambda *ranks: [r for r in ranks for _ in range(4)]          # 4 coins par carte
    assert "TIRER" in decide(ag, ok, C("2", "K")).text
    bad = analyse([Det("2", .2, (10, 10, 60, 90))], (480, 640))
    assert decide(ag, bad, C("2")).status == "hidden"
    assert decide(ag, ok, C("10", "6", "8")).status == "over"
    assert decide(ag, ok, C("2") + ["K", "K", "K"]).status == "playing"
    d = decide(ag, ok, C("K", "A"))                                  # affichage = rangs, logique = valeurs
    assert d.faces == "A K" and d.total == 21 and d.status == "over"

# # def test_cards_from_corners():
# #     assert cards_from_corners(["7"] * 4 + ["K"] * 8) == (["7", "K", "K"], False)
# #     assert cards_from_corners(["7"] * 3)[1] is True and cards_from_corners(["7"] * 5)[1] is True

# # def test_cards_from_corners():
# #     assert cards_from_corners(["7"] * 4 + ["K"] * 8) == (["7", "K", "K"], False)
# #     assert cards_from_corners(["7"]) == (["7"], False)
# #     assert cards_from_corners(["7"] * 3) == (["7"], False)
# #     assert cards_from_corners(["7"] * 5) == (["7", "7"], False)


def test_cards_from_corners():
    assert cards_from_corners(["7"] * 4 + ["K"] * 8) == (["7", "K", "K"], False)
    assert cards_from_corners(["7"]) == ([], False)          # 1 coin isolé = ignoré
    assert cards_from_corners(["7"] * 2) == (["7"], False)   # dès 2 coins = 1 carte
    assert cards_from_corners(["7"] * 3) == (["7"], False)
    assert cards_from_corners(["7"] * 5) == (["7", "7"], False)

# def test_cards_from_corners():
#     assert cards_from_corners(["7"] * 4 + ["K"] * 8) == (["7", "K", "K"], False)
#     assert cards_from_corners(["7"]) == ([], False)              # 1 coin isolé = ignoré
#     assert cards_from_corners(["7"] * 2) == (["7"], False)       # 2 à 5 coins = 1 carte
#     assert cards_from_corners(["7"] * 5) == (["7"], False)       # 5e coin (faux positif) = toujours 1 carte
#     assert cards_from_corners(["7"] * 6) == (["7", "7"], False)  # dès 6 coins = 2 cartes

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("OK", n)
