import unittest
from types import SimpleNamespace as S
import its_giving_v3 as app

class SigmaTests(unittest.TestCase):
    def test_pursed_lips_and_squint(self):
        self.assertTrue(app.sigma_expression(dict(mouthPucker=.65, eyeSquintLeft=.3, eyeSquintRight=.3)))
        self.assertEqual(app.decide(S(sigma=True), [])[0], 'sigma')

    def test_neutral_speech_and_blinks_are_rejected(self):
        for scores in ({}, dict(mouthPucker=.7), dict(eyeSquintLeft=.8),
                       dict(mouthPucker=.7, browDownLeft=.5, jawOpen=.6),
                       dict(mouthPucker=.7, eyeSquintLeft=.5, eyeBlinkLeft=.9)):
            self.assertFalse(app.sigma_expression(scores))
