"""Regression checks using approximate landmark positions from the user's HUD."""
from pathlib import Path
import sys
from types import SimpleNamespace as S
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import its_giving_v3 as app

# Screenshot coordinates: extended index/thumb, three folded fingers.
POINTS = np.array([
    (1700,1070), (1582,930), (1440,810), (1330,752), (1258,692),
    (1558,715), (1490,645), (1448,575), (1423,515),
    (1598,796), (1325,790), (1311,857), (1362,884),
    (1594,915), (1325,922), (1345,964), (1424,974),
    (1576,1037), (1347,1030), (1373,1040), (1442,1040)
], dtype=float)


def hand(points):
    return app.Hand([S(x=x/2048, y=y/1192, z=0) for x,y in points], 2048, 1192)


class PointCameraTests(unittest.TestCase):
    def test_new_screenshot_index_can_be_labelled_bent(self):
        points = np.array([
            (1545,788), (1450,791), (1355,784), (1338,781), (1410,764),
            (1320,545), (1268,519), (1232,498), (1212,474),
            (1415,485), (1392,682), (1427,753), (1447,714),
            (1526,501), (1497,712), (1509,750), (1521,715),
            (1626,552), (1595,729), (1594,758), (1608,725)
        ], dtype=float)
        for mirror in (-1, 1):
            h = hand((points - points[0]) * [mirror, 1] + [900, 600])
            self.assertFalse(h.finger_extended[1])
            self.assertTrue(h.point_camera)
            self.assertEqual(app.decide(None, [h])[0], 'point_camera')

    def test_screenshot_pose_with_rotation_scale_and_mirroring(self):
        for mirror in (-1, 1):
            for angle in (0, .7, -1.2):
                for scale in (.5, 1.2):
                    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                                         [np.sin(angle), np.cos(angle)]])
                    points = (POINTS-POINTS[0]) * [mirror, 1]
                    h = hand(points @ rotation * scale + [900,600])
                    self.assertTrue(h.point_camera)
                    self.assertEqual(app.decide(None, [h])[0], 'point_camera')

    def test_thumb_position_is_unrestricted(self):
        points=POINTS.copy()
        points[4]=points[5]
        self.assertTrue(hand(points).point_camera)

    def test_extended_other_fingers_do_not_trigger(self):
        for tip in (12,16,20):
            points=POINTS.copy()
            points[tip]=points[tip-2]+1.5*(points[tip-2]-points[tip-3])
            self.assertFalse(hand(points).point_camera)

    def test_folded_index_does_not_trigger(self):
        points=POINTS.copy()
        points[8]=points[5]
        self.assertFalse(hand(points).point_camera)


if __name__ == '__main__':
    unittest.main()
