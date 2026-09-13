import unittest
from types import SimpleNamespace as S

import numpy as np

import its_giving_v3 as app


def face(scale=1, center=(320, 180)):
    return S(center=np.asarray(center), w=160 * scale, h=200 * scale,
             sigma=False)


def fist(f, x=0.4, y=1.2, side='Right', size=0.4, closed=True):
    palm = f.center + np.array([x * f.w, y * f.h])
    points = np.tile(palm, (21, 1))
    points[9] += [0, size * f.w]
    return S(palm=palm, points=points, chest_fist=closed, side=side)


class ChestBeatTests(unittest.TestCase):
    def test_two_beats_from_either_hand_at_different_camera_distances(self):
        for scale in (0.5, 1, 2):
            for side, sign in (('Right', 1), ('Left', -1), ('Unknown', 1)):
                f, detector = face(scale), app.ChestBeatDetector()
                results = [detector.update(f, [fist(f, x * sign, side=side)], i * 0.1)
                           for i, x in enumerate((0.55, 0.3, 0.55, 0.3))]
                self.assertEqual(results, [False, False, False, True])

    def test_vertical_and_depth_beats(self):
        for axis, values in (('y', (1.4, 1.1, 1.4, 1.1)),
                             ('size', (0.5, 0.4, 0.5, 0.4))):
            f, detector = face(), app.ChestBeatDetector()
            results = [detector.update(f, [fist(f, **{axis: v})], i * 0.1)
                       for i, v in enumerate(values)]
            self.assertEqual(results, [False, False, False, True])

    def test_resting_fist_jitter_slow_movement_and_single_tap_stay_idle(self):
        cases = [([0.4] * 80, 0.03), ([0.4, 0.43, 0.38, 0.4] * 20, 0.03),
                 ([0.55, 0.3, 0.55] + [0.55] * 50, 0.1),
                 (np.linspace(0.8, 0.1, 20), 0.04),
                 ([0.4 + 0.25 * np.cos(i * 0.1) for i in range(100)], 0.1)]
        for positions, dt in cases:
            f, detector = face(), app.ChestBeatDetector()
            self.assertFalse(any(detector.update(f, [fist(f, x)], i * dt)
                                 for i, x in enumerate(positions)))

    def test_open_hands_and_motion_outside_chest_stay_idle(self):
        for kwargs in ({'closed': False}, {'y': 0.2}, {'y': 2.8}):
            f, detector = face(), app.ChestBeatDetector()
            self.assertFalse(any(detector.update(f, [fist(f, x, **kwargs)], i * 0.1)
                                 for i, x in enumerate((0.55, 0.3, 0.55, 0.3))))

    def test_face_and_camera_scale_movement_do_not_count_as_beats(self):
        detector = app.ChestBeatDetector()
        for i, scale in enumerate((1, 1.3, 1, 1.3)):
            f = face(scale, (320 + i * 10, 180 + i * 10))
            self.assertFalse(detector.update(f, [fist(f)], i * 0.1))

    def test_missing_face_hand_or_frame_gap_resets_partial_motion(self):
        for loss in ('face', 'hand', 'gap'):
            f, detector = face(), app.ChestBeatDetector()
            for i, x in enumerate((0.55, 0.3, 0.55)):
                self.assertFalse(detector.update(f, [fist(f, x)], i * 0.1))
            if loss == 'face':
                self.assertFalse(detector.update(None, [fist(f)], 0.25))
            elif loss == 'hand':
                self.assertFalse(detector.update(f, [], 0.25))
            self.assertFalse(detector.update(f, [fist(f, 0.3)], 0.8 if loss == 'gap' else 0.3))

    def test_hand_order_changes_do_not_mix_tracks(self):
        f, detector = face(), app.ChestBeatDetector()
        for i, x in enumerate((0.55, 0.3, 0.55, 0.3)):
            hands = [fist(f, x, side='Right'), fist(f, -0.4, side='Left')]
            if i % 2:
                hands.reverse()
            self.assertEqual(detector.update(f, hands, i * 0.1), i == 3)

    def test_switching_hands_cannot_complete_a_partial_beat(self):
        f, detector = face(), app.ChestBeatDetector()
        for i, (x, side) in enumerate(((0.55, 'Right'), (0.3, 'Right'),
                                       (0.55, 'Left'), (0.3, 'Left'))):
            self.assertFalse(detector.update(f, [fist(f, x, side=side)], i * 0.1))

    def test_cooldown_and_playback_reset_require_fresh_beats(self):
        f, detector = face(), app.ChestBeatDetector()
        for start in (0, 2):
            for i, x in enumerate((0.55, 0.3, 0.55, 0.3)):
                self.assertEqual(detector.update(f, [fist(f, x)], start + i * 0.1), i == 3)
            self.assertFalse(detector.update(f, [fist(f, 0.55)], start + 0.4))
        detector = app.ChestBeatDetector()
        for i, x in enumerate((0.55, 0.3, 0.55)):
            detector.update(f, [fist(f, x)], i * 0.1)
        detector.reset()  # Playback clears any in-progress gesture.
        self.assertFalse(detector.update(f, [fist(f, 0.3)], 0.3))

    def test_green_key_keeps_neutral_foreground_and_original_alpha(self):
        source = np.array([[[0, 255, 0, 255], [90, 90, 90, 255],
                            [30, 30, 30, 100], [20, 80, 20, 255]]], np.uint8)
        keyed = app.key_green(source)
        self.assertEqual(keyed[0, 0, 3], 0)
        np.testing.assert_array_equal(keyed[0, 1:3], source[0, 1:3])
        self.assertTrue(0 < keyed[0, 3, 3] < 255)
        self.assertEqual(source[0, 0, 3], 255)
