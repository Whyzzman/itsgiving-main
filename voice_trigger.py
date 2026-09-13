"""Local Ukrainian speech triggers. Audio is processed in memory, never uploaded."""
import json
from pathlib import Path
import queue
import re
import threading
import time
import numpy as np


def words(text):
    return re.findall(r"[^\W\d_]+(?:['’][^\W\d_]+)*", text.casefold(), re.UNICODE)


class PhraseMatcher:
    """Match whole words, once per utterance, including partial recognition."""

    def __init__(self, phrase):
        self.target = words(phrase)
        if not self.target:
            raise ValueError("Voice phrase must contain a word")
        self.fired = False

    def reset(self):
        self.fired = False

    def accept(self, text, final=False):
        tokens = words(text)
        match = any(tokens[i:i + len(self.target)] == self.target
                    for i in range(len(tokens) - len(self.target) + 1))
        fire = match and not self.fired
        self.fired = self.fired or fire
        if final:
            self.reset()
        return fire


class VoiceTrigger:
    """Bounded microphone queue and a background Vosk recognizer."""

    def __init__(self, model_path, phrase="у", device=None, enabled=True, extra_phrases=()):
        self.model_path = Path(model_path)
        self.phrase, self.device = phrase, device
        self.matchers = {p: PhraseMatcher(p) for p in (*extra_phrases, phrase)}
        self.enabled, self.suppressed = enabled, False
        self.resume_at = 0.0
        self.generation = 0
        self.audio = queue.Queue(maxsize=8)
        self.events = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.thread = None
        self.stream = None
        self.state = "OFF" if not enabled else "LOADING"
        self.heard = ""
        self.device_name = "--"
        self.level_db = -120.0
        self.last_audio_at = 0.0
        self.last_voice_at = 0.0
        self.error = ""
        self.dropped_blocks = 0

    def match_phrase(self, text, final=False):
        # Evaluate every matcher so final results reset all utterance state.
        matches = [p for p, matcher in self.matchers.items() if matcher.accept(text, final)]
        return matches[0] if matches else None

    @property
    def status(self):
        if not self.enabled:
            return "OFF"
        if self.state != "LISTENING":
            return self.state
        if self.suppressed or time.monotonic() < self.resume_at:
            return "PLAYBACK PAUSE"
        if self.last_audio_at and time.monotonic() - self.last_audio_at > 2:
            return "NO AUDIO DATA"
        if self.last_audio_at and time.monotonic() - self.last_voice_at > 3:
            return "QUIET INPUT"
        return "LISTENING"

    def _listening(self):
        return self.enabled and not self.suppressed and time.monotonic() >= self.resume_at

    def _invalidate(self):
        self.generation += 1
        for q in (self.audio, self.events):
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    def set_suppressed(self, suppressed):
        if suppressed != self.suppressed:
            self.suppressed = suppressed
            self.resume_at = float("inf") if suppressed else time.monotonic() + 0.6
            self._invalidate()

    def toggle(self):
        if self.enabled and self.state.endswith("ERROR"):
            self.start()
            return
        self.enabled = not self.enabled
        self._invalidate()
        if self.enabled and (self.thread is None or not self.thread.is_alive()):
            self.start()

    def start(self):
        if self.thread is not None and self.thread.is_alive():
            return
        if not self.enabled:
            return
        self.state = "LOADING"
        self.error = ""
        self.thread = threading.Thread(target=self._run, name="ukrainian-voice", daemon=True)
        self.thread.start()

    def _callback(self, indata, frames, timing, status):
        if self.enabled and not self.stop_event.is_set():
            samples = np.frombuffer(indata, dtype=np.int16).astype(np.float32)
            rms = float(np.sqrt(np.mean(samples * samples))) / 32768 if samples.size else 0.0
            self.level_db = 20 * float(np.log10(max(rms, 1e-6)))
            self.last_audio_at = time.monotonic()
            if self.level_db > -50:
                self.last_voice_at = self.last_audio_at
        if not self._listening() or self.stop_event.is_set():
            return
        item = (self.generation, time.monotonic(), bytes(indata))
        try:
            self.audio.put_nowait(item)
        except queue.Full:
            self.dropped_blocks += 1
            # Drop the oldest block instead of delaying the camera or microphone.
            try:
                self.audio.get_nowait()
            except queue.Empty:
                pass
            try:
                self.audio.put_nowait(item)
            except queue.Full:
                pass

    def _run(self):
        stage = "MODEL"
        try:
            import sounddevice as sd
            from vosk import Model, KaldiRecognizer, SetLogLevel

            if not (self.model_path / "am" / "final.mdl").is_file():
                raise RuntimeError(f"Missing Ukrainian speech model: {self.model_path}")
            SetLogLevel(-1)
            model = Model(str(self.model_path))
            if self.stop_event.is_set():
                return
            stage = "MIC"
            info = sd.query_devices(self.device, "input")
            self.device_name = str(info["name"])
            rate = int(info["default_samplerate"])
            recognizer = KaldiRecognizer(model, rate)
            generation = -1
            with sd.RawInputStream(device=self.device, samplerate=rate, channels=1,
                                   dtype="int16", blocksize=max(1, rate // 10),
                                   callback=self._callback) as stream:
                self.stream = stream
                self.state = "LISTENING"
                self.last_audio_at = self.last_voice_at = time.monotonic()
                print(f"Voice: listening for '{self.phrase}' on {info['name']} (local Ukrainian model).")
                while not self.stop_event.is_set():
                    try:
                        item_generation, captured, data = self.audio.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if not self._listening() or item_generation != self.generation:
                        continue
                    if generation != item_generation or time.monotonic() - captured > 1.5:
                        recognizer.Reset()
                        for matcher in self.matchers.values():
                            matcher.reset()
                        generation = item_generation
                        if time.monotonic() - captured > 1.5:
                            continue
                    final = bool(recognizer.AcceptWaveform(data))
                    result = json.loads(recognizer.Result() if final else recognizer.PartialResult())
                    text = result.get("text" if final else "partial", "")
                    if text:
                        self.heard = text
                    matched = self.match_phrase(text, final)
                    if matched is not None and self._listening() and generation == self.generation:
                        try:
                            self.events.put_nowait((generation, time.monotonic(), matched))
                        except queue.Full:
                            pass
        except Exception as e:
            self.state = f"{stage} ERROR"
            self.error = str(e)
            print(f"Voice {stage.lower()} error: {e}")
            if stage == "MIC":
                print("Check the selected input and macOS microphone permission; press v to retry.")
            else:
                print("Check the Vosk package and Ukrainian model folder; press v to retry.")
        finally:
            self.stream = None

    def poll(self):
        try:
            generation, detected, text = self.events.get_nowait()
        except queue.Empty:
            return None
        if self._listening() and generation == self.generation and time.monotonic() - detected < 1.5:
            return text
        return None

    def close(self):
        self.stop_event.set()
        self._invalidate()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
