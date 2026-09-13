import unittest
from types import SimpleNamespace as S

import numpy as np
import its_giving_v3 as app


def hand(x, y):
    return S(palm=np.array([x, y]), open_palm=False, point_camera=False,
             rock=False, thumbs_up=False, fist=False)


class LogCarryTests(unittest.TestCase):
    def test_reference_pose_mirrored_scaled_and_reordered(self):
        # Approximate face box and palm centers from the user's HUD screenshot.
        for scale in (0.5, 1, 2):
            face = S(center=np.array([917, 558]) * scale,
                     w=400 * scale, h=484 * scale, sigma=False)
            for side in (-1, 1):
                hands = [hand(*(face.center + np.array([side * x, y]) * scale))
                         for x, y in ((158, 239), (665, 356))]
                for ordered in (hands, hands[::-1]):
                    self.assertEqual(app.decide(face, ordered)[0], 'log_carry')

    def test_unrelated_positions_do_not_trigger(self):
        face = S(center=(0, 0), w=100, h=100, sigma=False)
        for positions in (
                [(40, 50)],                         # Only one hand.
                [(40, 50), (-165, 74)],              # Opposite shoulders.
                [(40, -50), (165, -25)],             # Above the head.
                [(40, 150), (165, 174)],             # Too low.
                [(40, 50), (60, 74)],                # Hands together.
                [(40, 50), (300, 74)]):              # Too far apart.
            self.assertNotEqual(app.decide(face, [hand(*p) for p in positions])[0],
                                'log_carry')

    def test_pointing_keeps_priority(self):
        face = S(center=(0, 0), w=100, h=100, sigma=False)
        hands = [hand(40, 50), hand(165, 74)]
        hands[0].point_camera = True
        self.assertEqual(app.decide(face, hands)[0], 'point_camera')
