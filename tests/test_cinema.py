import unittest
from types import SimpleNamespace as S
import numpy as np
import its_giving_v3 as app


def palm(x, y=300, opened=True):
    return S(palm=np.array([x,y]), open_palm=opened, point_camera=False,
             rock=False, thumbs_up=False, fist=False)

class CinemaTests(unittest.TestCase):
    def test_two_palms_on_opposite_sides(self):
        face=S(center=(400,300), w=200, h=250, sigma=False)
        for hands in ([palm(170),palm(630)], [palm(630),palm(170)]):
            self.assertEqual(app.decide(face,hands)[0], 'cinema')

    def test_one_closed_or_same_side_is_not_cinema(self):
        face=S(center=(400,300), w=200, h=250, sigma=False)
        for hands in ([palm(170)], [palm(170),palm(630,opened=False)],
                      [palm(150),palm(180)], [palm(170,700),palm(630,700)]):
            self.assertNotEqual(app.decide(face,hands)[0], 'cinema')
