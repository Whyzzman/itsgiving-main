import unittest
from types import SimpleNamespace as S
import its_giving_v3 as app
import numpy as np

class SigmaTests(unittest.TestCase):
    def test_overlay_follows_face_and_hides_without_tracking(self):
        points = np.zeros((478, 2), np.float32)
        angles = np.linspace(0, 2 * np.pi, len(app.FACE_OVAL), endpoint=False)
        points[app.FACE_OVAL] = np.column_stack((50 + 30 * np.cos(angles),
                                               50 + 40 * np.sin(angles)))
        points[app.FACE_ANCHORS] = [[35, 35], [65, 35], [50, 90]]
        source = np.full((100, 100, 4), 255, np.uint8)
        source[:, :, 3] = (255 * app.face_alpha(points, source.shape)).astype(np.uint8)
        effect = app.FaceOverlay.__new__(app.FaceOverlay)
        effect.frames, effect.anchors = [source], [points[app.FACE_ANCHORS]]
        target = S(points=points + [70, 20])
        target.points = target.points.astype(np.float32)
        rendered = effect.render(0, target, (180, 220, 3))
        self.assertGreater(rendered[70, 120, 3], 240)
        self.assertEqual(rendered[50, 50, 3], 0)
        self.assertEqual(rendered[0, 0, 3], 0)
        self.assertFalse(effect.render(0, None, (180, 220, 3)).any())

    def test_pursed_lips_and_squint(self):
        self.assertTrue(app.sigma_expression(dict(mouthPucker=.65, eyeSquintLeft=.3, eyeSquintRight=.3)))
        self.assertEqual(app.decide(S(sigma=True), [])[0], 'sigma')

    def test_neutral_speech_and_blinks_are_rejected(self):
        for scores in ({}, dict(mouthPucker=.7), dict(eyeSquintLeft=.8),
                       dict(mouthPucker=.7, browDownLeft=.5, jawOpen=.6),
                       dict(mouthPucker=.7, eyeSquintLeft=.5, eyeBlinkLeft=.9)):
            self.assertFalse(app.sigma_expression(scores))
