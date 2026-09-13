from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voice_trigger import PhraseMatcher, VoiceTrigger


class PhraseTests(unittest.TestCase):
    def test_word_boundaries(self):
        for text in ("Україна", "букет", "думаю", "музика", "", "в моєму"):
            self.assertFalse(PhraseMatcher("у").accept(text), text)
        for text in ("У", "у моєму", "а у тебе", "У, моєму!"):
            self.assertTrue(PhraseMatcher("у").accept(text), text)

    def test_phrase_and_partial_deduplication(self):
        matcher = PhraseMatcher("у моєму")
        self.assertFalse(matcher.accept("у"))
        self.assertTrue(matcher.accept("у моєму"))
        self.assertFalse(matcher.accept("у моєму серці"))
        self.assertFalse(matcher.accept("у моєму серці", final=True))
        self.assertTrue(matcher.accept("у моєму"))

    def test_final_only_and_reset(self):
        matcher = PhraseMatcher("у")
        self.assertTrue(matcher.accept("у моєму", final=True))
        self.assertTrue(matcher.accept("у"))
        matcher.reset()
        self.assertTrue(matcher.accept("у"))

    def test_empty_phrase_rejected(self):
        with self.assertRaises(ValueError):
            PhraseMatcher("  ! ")


class VoiceQueueTests(unittest.TestCase):
    def test_multiple_phrases_route_and_deduplicate(self):
        voice = VoiceTrigger("unused", phrase="у моєму", extra_phrases=("що",))
        self.assertIsNone(voice.match_phrase("якщо щось"))
        self.assertEqual(voice.match_phrase("що"), "що")
        self.assertIsNone(voice.match_phrase("що таке", final=True))
        self.assertEqual(voice.match_phrase("у моєму"), "у моєму")
        self.assertIsNone(voice.match_phrase("у моєму", final=True))
        self.assertEqual(voice.match_phrase("що", final=True), "що")

    def test_new_phrase_has_priority_when_both_match(self):
        voice = VoiceTrigger("unused", extra_phrases=("що",))
        self.assertEqual(voice.match_phrase("що у тебе", final=True), "що")
        self.assertEqual(voice.match_phrase("у"), "у")

    def setUp(self):
        self.voice = VoiceTrigger("unused", enabled=True)

    def test_audio_queue_bounded_and_suppression(self):
        for _ in range(30):
            self.voice._callback(bytes(20), 10, None, None)
        self.assertEqual(self.voice.audio.qsize(), 8)
        self.voice.events.put((self.voice.generation, time.monotonic(), "у"))
        self.voice.set_suppressed(True)
        self.assertTrue(self.voice.audio.empty())
        self.assertIsNone(self.voice.poll())
        self.voice._callback(bytes(20), 10, None, None)
        self.assertTrue(self.voice.audio.empty())

    def test_cooldown_and_generation(self):
        with patch("voice_trigger.time.monotonic", return_value=10):
            self.voice.set_suppressed(True)
            self.voice.set_suppressed(False)
            self.assertFalse(self.voice._listening())
        with patch("voice_trigger.time.monotonic", return_value=11):
            self.assertTrue(self.voice._listening())
            self.voice.events.put((self.voice.generation - 1, 11, "у"))
            self.assertIsNone(self.voice.poll())
            self.voice.events.put((self.voice.generation, 11, "у"))
            self.assertEqual(self.voice.poll(), "у")

    def test_stale_event_ignored(self):
        self.voice.events.put((self.voice.generation, time.monotonic() - 2, "у"))
        self.assertIsNone(self.voice.poll())

    def test_switch_off_and_close_drop_events(self):
        self.voice.events.put((self.voice.generation, time.monotonic(), "у"))
        self.voice.toggle()
        self.assertEqual(self.voice.status, "OFF")
        self.assertIsNone(self.voice.poll())
        self.voice.close()
        self.assertTrue(self.voice.stop_event.is_set())

    def test_silent_microphone_is_visible(self):
        self.voice.state = "LISTENING"
        self.voice.last_voice_at = 10
        with patch("voice_trigger.time.monotonic", return_value=14):
            self.voice._callback(bytes(200), 100, None, None)
            self.assertEqual(self.voice.level_db, -120)
            self.assertEqual(self.voice.status, "QUIET INPUT")
        with patch("voice_trigger.time.monotonic", return_value=17):
            self.assertEqual(self.voice.status, "NO AUDIO DATA")

    def test_input_level_recovers_when_sound_arrives(self):
        self.voice.state = "LISTENING"
        signal = np.full(100, 3277, dtype=np.int16).tobytes()
        with patch("voice_trigger.time.monotonic", return_value=10):
            self.voice._callback(signal, 100, None, None)
            self.assertAlmostEqual(self.voice.level_db, -20, places=1)
            self.assertEqual(self.voice.status, "LISTENING")

    def test_model_error_can_be_retried(self):
        self.voice.state = "MODEL ERROR"
        with patch.object(self.voice, "start") as start:
            self.voice.toggle()
        start.assert_called_once()
        self.assertTrue(self.voice.enabled)


if __name__ == "__main__":
    unittest.main()
