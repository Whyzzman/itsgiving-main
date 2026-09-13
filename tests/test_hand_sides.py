from pathlib import Path
import sys
from types import SimpleNamespace as S
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import its_giving_v3 as app


def rock_hand(label=None, mirrored=True, score=.95, shift=0):
    points = [S(x=.5 + shift, y=.75, z=0) for _ in range(21)]
    for j, tip in enumerate((8, 12, 16, 20)):
        x = .5 + shift + (j - 1.5) * .06
        points[tip - 3] = S(x=x, y=.63, z=0)
        points[tip - 2] = S(x=x, y=.5, z=0)
        points[tip] = S(x=x, y=.32 if tip in (8, 20) else .6, z=0)
    categories = [S(category_name=label, score=score)] if label else []
    return app.Hand(points, 640, 480, categories, mirrored=mirrored)


class HandSideTests(unittest.TestCase):
    def test_mirrored_input(self):
        for label, pose in (("Right", "rock"), ("Left", "rock_left")):
            hand = rock_hand(label)
            self.assertTrue(hand.rock)
            self.assertEqual(app.decide(None, [hand])[0], pose)

    def test_unmirrored_input(self):
        self.assertEqual(app.decide(None, [rock_hand("Left", mirrored=False)])[0], "rock")
        self.assertEqual(app.decide(None, [rock_hand("Right", mirrored=False)])[0], "rock_left")

    def test_unknown_or_uncertain_hand_is_not_guessed(self):
        for hand in (rock_hand(), rock_hand("Right", score=.55), rock_hand("Other")):
            self.assertEqual(hand.side, "Unknown")
            self.assertIsNone(app.decide(None, [hand])[0])

    def test_screen_position_does_not_choose_the_reaction(self):
        for shift in (-.25, .25):
            self.assertEqual(app.decide(None, [rock_hand("Right", shift=shift)])[0], "rock")
            self.assertEqual(app.decide(None, [rock_hand("Left", shift=shift)])[0], "rock_left")

    def test_two_hands_have_stable_priority(self):
        left, right = rock_hand("Left"), rock_hand("Right")
        for hands in ([left, right], [right, left]):
            self.assertEqual(app.decide(None, hands)[0], "rock")

    def test_keys_and_asset(self):
        self.assertEqual(app.TEST_KEYS["r"], "rock")
        self.assertEqual(app.TEST_KEYS["e"], "rock_left")
        self.assertEqual(set(app.TEST_KEYS.values()), set(app.POSES))
        self.assertTrue(Path(app.find_asset_file("rock_left")).is_file())


if __name__ == "__main__":
    unittest.main()
