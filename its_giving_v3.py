#!/usr/bin/env python3
"""
its_giving_v3.py — meme reactions on top of your face, live in Zoom / Meet.

Hand gestures plus a local Ukrainian voice trigger for the guitar video.

  python its_giving_v3.py [--camera 1] [--no-vcam] [--size 640x480] [--no-flip]

Keys: q quit, d toggle HUD, 
      r rock, t thumbs_up, l log_carry, b bouquet, p point_camera (video).
      e rock_left. Rock horns: right hand -> rock, left hand -> rock_left.
      g guitar video, j Vlad video (say "що"), v toggle microphone. Say "у" to play the guitar clip.
      --voice-phrase "у моєму" narrows the trigger; --no-voice disables listening.
"""
import argparse
import os
import platform
import subprocess
import sys
import time
import urllib.request
import wave
from functools import lru_cache

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from voice_trigger import VoiceTrigger

POSES = ["rock", "rock_left", "thumbs_up", "log_carry", "bouquet", "point_camera", "guitar_voice", "vlad_voice", "sigma", "cinema"]
TEST_KEYS = {"r": "rock", "e": "rock_left", "t": "thumbs_up", "l": "log_carry", "b": "bouquet", "p": "point_camera", "g": "guitar_voice", "j": "vlad_voice", "s": "sigma", "c": "cinema"}

FACE_SCALE = 2.0
ASSET_SCALES = {"log_carry": 1.2, "bouquet": 1.3}
# Zoom after fitting to the frame, so 2x remains visible even near the size cap.
ASSET_ZOOM = {"point_camera": 2.0}
ASSET_MIRRORED = {"log_carry", "rock", "rock_left"}
# Rightward shift as a fraction of image width, aligning the mirrored face.
ASSET_X_OFFSETS = {"log_carry": 0.09}
# Additional downward shift as a fraction of the displayed image height.
ASSET_Y_OFFSETS = {"thumbs_up": 0.15, "bouquet": 0.35}
HOLD_FRAMES = 10
ARM = {"rock": 5, "rock_left": 5, "thumbs_up": 5, "log_carry": 6, "bouquet": 7, "point_camera": 2, "sigma": 4, "cinema": 4}

MODELS = {
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
}
HERE = os.path.dirname(os.path.abspath(__file__))
VOICE_MODEL = os.path.join(HERE, "models", "vosk-model-small-uk-v3-nano")


def ensure_models():
    mdir = os.path.join(HERE, "models")
    os.makedirs(mdir, exist_ok=True)
    paths = {}
    for name, url in MODELS.items():
        path = os.path.join(mdir, name)
        if not os.path.exists(path):
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(url, path)
        paths[name] = path
    return paths


def preflight(model_path):
    """Open a detector in a throwaway subprocess: bad macOS builds abort() uncatchably."""
    code = (
        "import sys\n"
        "from mediapipe.tasks import python as t\n"
        "from mediapipe.tasks.python import vision\n"
        "vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(\n"
        "    base_options=t.BaseOptions(model_asset_path=sys.argv[1]),\n"
        "    running_mode=vision.RunningMode.VIDEO, num_faces=1,\n"
        "    output_face_blendshapes=True)).close()\n"
    )
    proc = subprocess.run([sys.executable, "-c", code, model_path], capture_output=True, text=True)
    if proc.returncode == 0:
        return
    err = (proc.stderr or "") + (proc.stdout or "")
    print(f"\nMediaPipe cannot start a detector here (python {platform.python_version()}, "
          f"mediapipe {getattr(mp, '__version__', '?')}, exit {proc.returncode}).\n")
    if "Service is unavailable" in err or "MetalHelper" in err or proc.returncode == -6:
        print("Cause: mediapipe 0.10.30+ ships macOS wheels that abort on startup.\n"
              "Fix (Python 3.11 or 3.12) — install the pinned set:\n"
              "  pip install -r requirements.txt\n"
              "If you already installed something newer by hand, force it back:\n"
              '  pip install "mediapipe==0.10.21" "numpy<2" "opencv-python<5" "opencv-contrib-python<5"\n')
    else:
        print(err[-1500:])
    sys.exit(1)


def build_detectors(model_paths):
    face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_paths["face_landmarker.task"]),
        running_mode=vision.RunningMode.VIDEO, num_faces=1, output_face_blendshapes=True))
    hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_paths["hand_landmarker.task"]),
        running_mode=vision.RunningMode.VIDEO, num_hands=2))
    return face, hand


class Clock:
    """Strictly increasing timestamps for the life of a detector."""

    def __init__(self):
        self.t0, self.last = time.monotonic(), -1

    def next(self):
        self.last = max(int((time.monotonic() - self.t0) * 1000), self.last + 1)
        return self.last


class Asset:
    """One reaction: a list of BGRA frames plus per-frame durations (ms) for GIFs."""

    def __init__(self, frames, durations):
        self.frames = frames
        self.durations = durations
        self.cum = np.cumsum(durations)
        self.total = int(self.cum[-1])
        h, w = frames[0].shape[:2]
        self.aspect = w / h
        self._cache = {}

    def frame_at(self, ms):
        if len(self.frames) == 1:
            return 0
        return int(np.searchsorted(self.cum, ms % self.total, side="right"))

    def scaled(self, idx, height):
        key = (idx, height)
        if key not in self._cache:
            if len(self._cache) > 64:
                self._cache.clear()
            w = max(1, int(round(height * self.aspect)))
            self._cache[key] = cv2.resize(self.frames[idx], (w, height), interpolation=cv2.INTER_AREA)
        return self._cache[key]

    def close(self):
        pass


FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
FACE_ANCHORS = [33, 263, 152]  # Outer eye corners and chin.


def face_alpha(points, shape):
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.rint(points[FACE_OVAL]).astype(np.int32)], 255)
    # Feather inward so the background outside the facial contour stays hidden.
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    feather = max(1.0, np.ptp(points[FACE_OVAL, 1]) * 0.025)
    return np.clip(distance / feather, 0, 1)


class FaceOverlay:
    """Keep GIF timing, masking and aligning each frame to the live face."""

    def __init__(self, asset):
        self.frames, self.anchors = [], []
        options = vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(
                model_asset_path=os.path.join(HERE, "models", "face_landmarker.task")),
            running_mode=vision.RunningMode.IMAGE, num_faces=1)
        with vision.FaceLandmarker.create_from_options(options) as detector:
            for frame in asset.frames:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                if not result.face_landmarks:
                    self.frames.append(None)
                    self.anchors.append(None)
                    continue
                h, w = frame.shape[:2]
                points = np.array([[lm.x * w, lm.y * h]
                                   for lm in result.face_landmarks[0]], np.float32)
                cutout = frame.copy()
                cutout[:, :, 3] = (frame[:, :, 3] * face_alpha(points, frame.shape)).astype(np.uint8)
                self.frames.append(cutout)
                self.anchors.append(points[FACE_ANCHORS])
        valid = [i for i, frame in enumerate(self.frames) if frame is not None]
        if not valid:
            raise ValueError("No face detected in sigma asset")
        for i, frame in enumerate(self.frames):
            if frame is None:
                nearest = min(valid, key=lambda j: abs(j - i))
                self.frames[i] = self.frames[nearest]
                self.anchors[i] = self.anchors[nearest]

    def render(self, idx, face, shape):
        h, w = shape[:2]
        if face is None:
            return np.zeros((h, w, 4), np.uint8)
        transform = cv2.getAffineTransform(self.anchors[idx], face.points[FACE_ANCHORS])
        sprite = cv2.warpAffine(self.frames[idx], transform, (w, h), flags=cv2.INTER_LINEAR)
        sprite[:, :, 3] = (sprite[:, :, 3] * face_alpha(face.points, shape)).astype(np.uint8)
        return sprite


class ReactionAudio:
    """Play a reaction's WAV through macOS audio, once per loop."""

    def __init__(self, asset_path, duration_ms=None, loop=True):
        self.loop = loop
        self.path = os.path.splitext(asset_path)[0] + ".wav"
        self.available = os.path.isfile(self.path) and os.path.isfile("/usr/bin/afplay")
        if duration_ms is None and self.available:
            with wave.open(self.path, "rb") as audio:
                duration_ms = audio.getnframes() * 1000 / audio.getframerate()
        self.duration_ms = max(duration_ms or 1.0, 1.0)
        self.process = None
        self.cycle, self.last_ms = None, -1
        self.warned = False

    def stop(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.process.communicate()
            self.process = None
        self.cycle, self.last_ms = None, -1

    def sync(self, ms):
        if not self.available:
            if not self.warned:
                print("Reaction audio unavailable: requires the matching .wav file and macOS afplay.")
                self.warned = True
            return
        cycle = int(max(ms, 0) // self.duration_ms) if self.loop else 0
        if cycle != self.cycle or ms < self.last_ms:
            self.stop()
            try:
                self.process = subprocess.Popen(
                    ["/usr/bin/afplay", self.path], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            except OSError as e:
                print(f"Could not start reaction audio: {e}")
                self.available, self.warned = False, True
                return
            self.cycle = cycle
        self.last_ms = ms
        if self.process is not None and self.process.poll() not in (None, 0):
            error = self.process.communicate()[1].decode(errors="replace").strip()
            print(f"Reaction audio playback failed: {error}")
            self.stop()
            self.available, self.warned = False, True


class VideoAsset:
    """Time-based looping video with audio and only the current frame in memory."""

    def __init__(self, path, mirrored=False):
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        ok, frame = self.cap.read()
        if not ok or not np.isfinite(self.fps) or self.fps <= 0 or not np.isfinite(count) or count < 1:
            self.cap.release()
            raise ValueError(f"Cannot decode video: {path}")
        self.count = int(count)
        self.total = self.count * 1000.0 / self.fps
        self.audio = ReactionAudio(path, self.total)
        self.aspect = frame.shape[1] / frame.shape[0]
        self.mirrored = mirrored
        self._frame, self._index = frame, 0
        self._scaled_key, self._scaled = None, None
        self._warned = False

    def frame_at(self, ms):
        return int((max(ms, 0) % self.total) * self.fps / 1000) % self.count

    def scaled(self, idx, height):
        if idx != self._index:
            # Decode small advances sequentially; seek on restart or long skips.
            if idx < self._index or idx - self._index > 8:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                self._index = idx - 1
            while self._index < idx:
                ok, frame = self.cap.read()
                if not ok:
                    if not self._warned:
                        print("Video frame unavailable; holding the last decoded frame.")
                        self._warned = True
                    break
                self._frame = frame
                self._index += 1
                self._scaled_key = None
        key = (self._index, height)
        if key != self._scaled_key:
            width = max(1, int(round(height * self.aspect)))
            frame = cv2.resize(self._frame, (width, height), interpolation=cv2.INTER_AREA)
            if self.mirrored:
                frame = cv2.flip(frame, 1)
            self._scaled = to_bgra(frame)
            self._scaled_key = key
        return self._scaled

    def close(self):
        self.audio.stop()
        self.cap.release()


def draw_fissure(frame, photo_box, elapsed):
    """Deterministic branching cracks and a shockwave, once per appearance."""
    if not 0 <= elapsed < 2.0:
        return
    x, y, w, h = photo_box
    center = np.array([x + w / 2, y + h / 2])
    scale = min(frame.shape[:2]) / 720
    layer = frame.copy()
    rng = np.random.default_rng(47)
    growth = min(elapsed / 0.45, 1.0)
    for angle in np.linspace(0, 2 * np.pi, 14, endpoint=False):
        direction = np.array([np.cos(angle), np.sin(angle)])
        normal = np.array([-direction[1], direction[0]])
        edge = min(w / (2 * max(abs(direction[0]), 1e-6)),
                   h / (2 * max(abs(direction[1]), 1e-6)))
        start = center + direction * edge
        length = rng.uniform(100, 240) * scale
        points = [start]
        for step in range(1, 7):
            points.append(start + direction * length * step / 6 * growth
                          + normal * rng.uniform(-18, 18) * scale * growth)
        points = np.asarray(points, np.int32)
        cv2.polylines(layer, [points], False, (25, 20, 15), max(2, int(6 * scale)), cv2.LINE_AA)
        cv2.polylines(layer, [points], False, (100, 190, 255), max(1, int(2 * scale)), cv2.LINE_AA)
        branch = points[3] + (direction * 45 + normal * rng.choice([-40, 40])) * scale * growth
        cv2.line(layer, tuple(points[3]), tuple(branch.astype(int)), (100, 190, 255), max(1, int(scale)), cv2.LINE_AA)
    if elapsed < 0.7:
        radius = 1 + elapsed * 2.5
        cv2.ellipse(layer, tuple(center.astype(int)),
                    (int(w * radius / 2), int(h * radius / 2)), 0, 0, 360,
                    (170, 220, 255), max(1, int(5 * scale * (1 - elapsed / 0.7))), cv2.LINE_AA)
    opacity = min(1.0, elapsed / 0.05, (2.0 - elapsed) / 0.6)
    cv2.addWeighted(layer, opacity, frame, 1 - opacity, 0, dst=frame)


def bouquet_fire_sprite(effect, elapsed_ms, photo_box):
    """One-shot expanding fire, keyed from green and rendered behind the photo."""
    if elapsed_ms < 0 or elapsed_ms >= effect.total:
        return None
    x, y, w, h = photo_box
    progress = elapsed_ms / effect.total
    height = max(8, int(h * (1.05 + 0.65 * progress)))
    sprite = effect.scaled(effect.frame_at(elapsed_ms), height).copy()
    bgr = sprite[:, :, :3].astype(np.float32)
    blue, green, red = cv2.split(bgr)
    alpha = np.clip((np.maximum(red, blue) - 0.15 * green) / 150, 0, 1)
    fh, fw = alpha.shape
    edge_x = np.minimum(np.linspace(0, 1, fw), np.linspace(1, 0, fw))
    edge_y = np.minimum(np.linspace(0, 1, fh), np.linspace(1, 0, fh))
    alpha *= np.clip(edge_y[:, None] / 0.12, 0, 1) * np.clip(edge_x[None, :] / 0.12, 0, 1)
    # Suppress green spill on translucent flames without darkening the background.
    sprite[:, :, 1] = np.minimum(green, red * 0.8).astype(np.uint8)
    fade = min(1.0, elapsed_ms / 100, (effect.total - elapsed_ms) / 300)
    sprite[:, :, 3] = (alpha * 255 * max(fade, 0)).astype(np.uint8)
    sh, sw = sprite.shape[:2]
    return sprite, int(x + (w - sw) / 2), int(y + (h - sh) / 2)


def to_bgra(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def placeholder(label):
    img = np.zeros((300, 300, 4), np.uint8)
    cv2.circle(img, (150, 150), 140, (0, 0, 255, 220), -1)
    cv2.putText(img, label, (12, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255, 255), 2)
    return Asset([img], [100])


def find_asset_file(pose):
    adir = os.path.join(HERE, "assets")
    if not os.path.isdir(adir):
        return None
    exts = (".gif", ".png", ".jpg", ".jpeg", ".mp4", ".mov")
    for fn in sorted(os.listdir(adir)):
        stem, ext = os.path.splitext(fn)
        if ext.lower() in exts and (stem == pose or stem.endswith("_" + pose)):
            return os.path.join(adir, fn)
    return None


def load_asset(pose):
    path = find_asset_file(pose)
    if path is None:
        print(f"  {pose:16s} missing -> placeholder")
        return placeholder(pose)
    if path.lower().endswith((".mp4", ".mov")):
        try:
            asset = VideoAsset(path, mirrored=pose in ASSET_MIRRORED)
        except ValueError as e:
            print(f"  {pose:16s} {e} -> placeholder")
            return placeholder(pose)
        sound = "with audio" if asset.audio.available else "audio unavailable"
        print(f"  {pose:16s} {os.path.basename(path)} ({asset.total / 1000:.1f}s, {asset.fps:g} fps, {sound})")
        return asset
    frames, durations = [], []
    if path.lower().endswith(".gif"):
        from PIL import Image, ImageSequence
        with Image.open(path) as im:
            for f in ImageSequence.Iterator(im):
                frames.append(cv2.cvtColor(np.array(f.convert("RGBA")), cv2.COLOR_RGBA2BGRA))
                durations.append(max(20, int(f.info.get("duration", 100))))
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            frames, durations = [to_bgra(img)], [100]
    if not frames:
        print(f"  {pose:16s} could not read {os.path.basename(path)} -> placeholder")
        return placeholder(pose)
    if pose in ASSET_MIRRORED:
        frames = [cv2.flip(frame, 1) for frame in frames]
    print(f"  {pose:16s} {os.path.basename(path)}  ({len(frames)} frame{'s' if len(frames) > 1 else ''})")
    return Asset(frames, durations)


def overlay(frame, sprite, x, y):
    """Alpha-composite BGRA sprite onto BGR frame at top-left (x, y), clipped to the frame."""
    H, W = frame.shape[:2]
    h, w = sprite.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return frame
    s = sprite[y0 - y:y1 - y, x0 - x:x1 - x]
    a = s[:, :, 3:4].astype(np.float32) / 255.0
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (a * s[:, :, :3] + (1 - a) * roi).astype(np.uint8)
    return frame


def swag_badge_scale(width, height):
    return min(1.15, (width - 16) / 470, height * 0.24 / 150)


def draw_swag_badge(frame, bounds, elapsed):
    """Animated reaction sticker; included in both preview and virtual camera."""
    H, W = frame.shape[:2]
    x, y, sw, sh = bounds
    badge = np.zeros((150, 470, 4), np.uint8)
    ink, pink, lime = (24, 15, 28, 255), (185, 65, 255, 255), (80, 255, 220, 255)
    cv2.rectangle(badge, (12, 14), (467, 147), ink, -1)
    cv2.rectangle(badge, (3, 3), (457, 137), lime, -1)
    cv2.rectangle(badge, (7, 7), (453, 133), ink, -1)
    cv2.rectangle(badge, (7, 7), (453, 32), pink, -1)
    cv2.putText(badge, 'SWAG.EXE  //  VIBE CHECK PASSED', (19, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, ink, 1, cv2.LINE_AA)
    cv2.putText(badge, 'SWAG DETECTED', (94, 76),
                cv2.FONT_HERSHEY_DUPLEX, 1.02, (255, 255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(badge, 'TOO COOL TO DEBUG', (96, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.56, pink, 1, cv2.LINE_AA)
    # Draw a sunglasses face directly; OpenCV's text font has no emoji glyphs.
    cv2.circle(badge, (49, 77), 30, lime, -1, cv2.LINE_AA)
    for left in (23, 51):
        cv2.rectangle(badge, (left, 63), (left + 23, 77), ink, -1)
        cv2.line(badge, (left + 4, 66), (left + 10, 66), (255, 255, 255, 255), 2)
    cv2.line(badge, (44, 67), (55, 67), ink, 3)
    cv2.ellipse(badge, (50, 83), (13, 10), 0, 10, 165, ink, 2, cv2.LINE_AA)
    cv2.putText(badge, 'SWAG LEVEL', (20, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.35, lime, 1, cv2.LINE_AA)
    for i in range(15):
        color = pink if i >= 12 else lime
        cv2.rectangle(badge, (110 + 22*i, 113), (126 + 22*i, 124), color, -1)

    # A small pop-in followed by a gentle pulse, without flashing.
    pop = 0.9 + 0.1 * min(max(elapsed, 0.0) / 0.2, 1.0)
    pulse = 1.0 + 0.015 * np.sin(elapsed * 4.0)
    scale = swag_badge_scale(W, H) * pop * pulse
    bw, bh = max(1, int(470 * scale)), max(1, int(150 * scale))
    badge = cv2.resize(badge, (bw, bh), interpolation=cv2.INTER_AREA)
    bx = int(np.clip(x + sw / 2 - bw / 2, 4, max(4, W - bw - 4)))
    # The rock layout reserves space for the badge above the photo.
    by = y - bh - 12
    overlay(frame, badge, bx, by)


def dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def sigma_expression(scores):
    """Pursed lips plus narrowed eyes or lowered brows; reject open-mouth speech."""
    pucker = scores.get("mouthPucker", 0)
    funnel = scores.get("mouthFunnel", 0)
    squint = (scores.get("eyeSquintLeft", 0) + scores.get("eyeSquintRight", 0)) / 2
    brow = (scores.get("browDownLeft", 0) + scores.get("browDownRight", 0)) / 2
    blink = max(scores.get("eyeBlinkLeft", 0), scores.get("eyeBlinkRight", 0))
    return bool((pucker > 0.35 or funnel > 0.4)
                and (squint > 0.16 or brow > 0.16)
                and scores.get("jawOpen", 0) < 0.25 and blink < 0.7)


class Face:
    def __init__(self, lms, W, H, blendshapes=()):
        p = np.array([[l.x * W, l.y * H] for l in lms], np.float32)
        self.points = p
        x0, y0 = p.min(0)
        x1, y1 = p.max(0)
        self.box = (int(x0), int(y0), int(x1), int(y1))
        self.w, self.h = float(x1 - x0), float(y1 - y0)
        self.center = ((x0 + x1) / 2, (y0 + y1) / 2)
        scores = {c.category_name: float(c.score) for c in blendshapes}
        self.sigma = sigma_expression(scores)


class Hand:
    def __init__(self, lms, W, H, handedness=None, mirrored=True):
        self.side, self.side_score = "Unknown", 0.0
        if handedness:
            category = max(handedness, key=lambda c: c.score)
            self.side_score = float(category.score)
            if category.category_name in ("Left", "Right") and self.side_score >= 0.6:
                self.side = category.category_name
                # MediaPipe handedness assumes a mirrored selfie image.
                if not mirrored:
                    self.side = "Left" if self.side == "Right" else "Right"
        p = np.array([[l.x * W, l.y * H] for l in lms], np.float32)
        self.points = p
        self.palm = p[[0, 5, 9, 13, 17]].mean(0)
        ext = [dist(p[0], p[t]) > 1.2 * dist(p[0], p[t - 2]) for t in (8, 12, 16, 20)]
        self.finger_extended = [dist(p[2], p[4]) > 1.25 * dist(p[2], p[3]), *ext]
        # Joint angles are diagnostic measurements, separate from trigger rules.
        self.finger_angles = []
        for a, b, c in ((2, 3, 4), (5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)):
            u, v = p[a] - p[b], p[c] - p[b]
            denom = float(np.linalg.norm(u) * np.linalg.norm(v))
            angle = float(np.degrees(np.arccos(np.clip(np.dot(u, v) / denom, -1, 1)))) if denom > 1e-6 else None
            self.finger_angles.append(angle)
        self.open_palm = bool(all(ext) and all(
            angle is not None and angle > 145 for angle in self.finger_angles[1:]))
        # Rock horns: index and pinky extended, middle and ring folded.
        # Wrist-relative distances work for either hand, regardless of rotation.
        # The thumb is deliberately unrestricted.
        self.rock = bool(ext[0] and ext[3]
                         and dist(p[0], p[12]) < 1.05 * dist(p[0], p[10])
                         and dist(p[0], p[16]) < 1.05 * dist(p[0], p[14]))
        # A closed fist with an extended thumb pointing up in the camera frame.
        # Normalize by palm size so either hand works at different distances.
        palm_size = max(dist(p[0], p[9]), dist(p[5], p[17]), 1.0)
        folded = all(dist(p[0], p[t]) < 1.15 * dist(p[0], p[t - 2])
                     for t in (8, 12, 16, 20))
        thumb = p[4] - p[2]
        self.thumbs_up = bool(
            folded
            and dist(p[2], p[4]) > 1.25 * dist(p[2], p[3])
            and -thumb[1] > 0.4 * palm_size
            and -thumb[1] > 1.2 * abs(thumb[0])
            and p[4][1] < p[[5, 9, 13, 17], 1].min() - 0.15 * palm_size
        )
        # Tucked thumb distinguishes holding a bouquet from giving a thumbs-up.
        self.fist = bool(folded and not self.thumbs_up
                         and dist(p[4], p[5]) < 0.75 * palm_size)
        # MediaPipe depth uses the same scale as normalized x; negative is closer.
        xyz = np.array([[l.x * W, l.y * H, getattr(l, "z", 0.0) * W] for l in lms], np.float32)

        def angle3(a, b, c):
            u, v = xyz[a] - xyz[b], xyz[c] - xyz[b]
            denom = float(np.linalg.norm(u) * np.linalg.norm(v))
            return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / denom, -1, 1)))) if denom > 1e-6 else 0.0

        direction = xyz[8] - xyz[5]
        palm3 = max(float(np.linalg.norm(xyz[9] - xyz[0])), 1.0)
        pointing_at_lens = bool(
            angle3(5, 6, 7) > 145 and angle3(6, 7, 8) > 145
            and all(angle3(t - 3, t - 2, t) < 140 for t in (12, 16, 20))
            and -direction[2] > 0.4 * palm3
            and -direction[2] > 0.9 * float(np.linalg.norm(direction[:2]))
        )
        # A straight index can point across the palm and fail the wrist-distance
        # EXT estimate. Use its angle instead; the thumb stays unrestricted.
        relaxed_point = bool(
            self.finger_angles[1] is not None
            and self.finger_angles[1] > 145
            and dist(p[5], p[8]) > 0.2 * palm_size
            and all(not extended and angle is not None and angle < 130
                    for extended, angle in zip(ext[1:], self.finger_angles[2:]))
        )
        self.point_camera = pointing_at_lens or relaxed_point


def decide(face, hands):
    """Return the first matching gesture and its HUD measurements."""
    d = {"hands": len(hands)}
    if face is not None and len(hands) == 2 and all(h.open_palm for h in hands):
        rel = [(h.palm - np.asarray(face.center)) / [max(face.w, 1.0), max(face.h, 1.0)]
               for h in hands]
        if rel[0][0] * rel[1][0] < 0 and all(
                0.4 < abs(x) < 3.0 and -1.0 < y < 1.2 for x, y in rel):
            return "cinema", d
    if any(h.point_camera for h in hands):
        return "point_camera", d
    # Keep the custom gesture available even when the face is briefly lost.
    if any(h.rock and h.side == "Right" for h in hands):
        return "rock", d
    if any(h.rock and h.side == "Left" for h in hands):
        return "rock_left", d
    if any(h.thumbs_up for h in hands):
        return "thumbs_up", d
    if face is None:
        return None, d

    if len(hands) == 1 and hands[0].fist:
        x, y = (hands[0].palm - np.asarray(face.center)) / [max(face.w, 1.0), max(face.h, 1.0)]
        # One closed hand in front of the chest, below the chin.
        if abs(x) < 0.9 and 0.65 < y < 2.4:
            return "bouquet", d
    if len(hands) >= 2:
        a, b = hands[0], hands[1]
        # Carry pose: inner palm beside the neck, outer palm farther along
        # the same shoulder, level with or slightly below the inner palm.
        # Normalize to the face so mirroring and camera distance do not matter.
        rel = [(h.palm - np.asarray(face.center)) / [max(face.w, 1.0), max(face.h, 1.0)]
               for h in (a, b)]
        same_side = rel[0][0] * rel[1][0] > 0
        inner, outer = sorted(rel, key=lambda p: abs(p[0]))
        inner_x, inner_y = abs(inner[0]), inner[1]
        outer_x, outer_y = abs(outer[0]), outer[1]
        if (same_side
                and 0.2 < inner_x < 0.95 and 0.2 < inner_y < 0.95
                and 0.95 < outer_x < 2.4 and 0.25 < outer_y < 1.25
                and 0.55 < outer_x - inner_x < 1.9
                and -0.15 < outer_y - inner_y < 0.65):
            return "log_carry", d
    if face.sigma:
        return "sigma", d
    return None, d


# BGR colors, one for each finger from thumb to pinky.
FINGER_COLORS = [(60, 195, 255), (100, 255, 100), (255, 220, 60), (255, 120, 210), (150, 100, 255)]
FINGER_NAMES = ("THUMB", "INDEX", "MIDDLE", "RING", "PINKY")


@lru_cache(maxsize=64)
def voice_text_sprite(text, size):
    """Unicode text for Ukrainian speech diagnostics (OpenCV fonts are ASCII)."""
    from PIL import Image, ImageDraw, ImageFont
    font = None
    for path in ("/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "C:/Windows/Fonts/arial.ttf"):
        if os.path.isfile(path):
            font = ImageFont.truetype(path, size)
            break
    if font is None:
        font = ImageFont.load_default(size=size)
    bounds = font.getbbox(text or " ")
    tile = Image.new("RGBA", (max(1, bounds[2] - bounds[0] + 4), max(size + 4, bounds[3] - bounds[1] + 4)))
    ImageDraw.Draw(tile).text((2 - bounds[0], 2 - bounds[1]), text, font=font,
                             fill=(210, 255, 230, 255), stroke_width=1, stroke_fill=(10, 15, 12, 255))
    return cv2.cvtColor(np.array(tile), cv2.COLOR_RGBA2BGRA)


def draw_hud(img, shown, raw, d, face, hands):
    H, W = img.shape[:2]
    unit = max(0.55, min(W / 960, H / 720, 1.5))

    def label(text, x, y, color=(120, 255, 190), size=0.45):
        cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                    size * unit, (5, 10, 15), 3, cv2.LINE_AA)
        cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                    size * unit, color, 1, cv2.LINE_AA)

    def panel(x, y, w, h):
        x, y, w, h = map(int, (x, y, w, h))
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x+w), min(H, y+h)
        if x1 <= x0 or y1 <= y0:
            return
        roi = img[y0:y1, x0:x1]
        tint = np.full_like(roi, (22, 18, 12))
        cv2.addWeighted(roi, 0.25, tint, 0.75, 0, dst=roi)
        cv2.rectangle(img, (x0, y0), (x1-1, y1-1), (95, 135, 90), 1)

    if face is not None:
        x0, y0, x1, y1 = face.box
        cv2.rectangle(img, (x0, y0), (x1, y1), (80, 180, 80), 1)
        center = tuple(map(int, face.center))
        cv2.drawMarker(img, center, (130, 255, 180), cv2.MARKER_CROSS, 16, 1)
        label("FACE / TRACKED", x0, y1 + 16 * unit)

    for hand_id, hand in enumerate(hands, 1):
        points = np.rint(hand.points).astype(int)
        for a, b in ((0, 5), (5, 9), (9, 13), (13, 17), (17, 0)):
            cv2.line(img, tuple(points[a]), tuple(points[b]), (150, 170, 150), 1, cv2.LINE_AA)
        for finger, color in enumerate(FINGER_COLORS):
            chain = [0, *range(1 + finger * 4, 5 + finger * 4)]
            for a, b in zip(chain, chain[1:]):
                cv2.line(img, tuple(points[a]), tuple(points[b]), color, 2, cv2.LINE_AA)
            for idx in chain[1:]:
                pt = tuple(points[idx])
                cv2.circle(img, pt, 4 if idx % 4 == 0 else 3, color, -1, cv2.LINE_AA)
                if idx % 4 == 0:
                    cv2.circle(img, pt, 8, color, 1, cv2.LINE_AA)
                label(str(idx), pt[0] + 6, pt[1] - 5, color, 0.32)
        cv2.circle(img, tuple(points[0]), 5, (230, 240, 240), -1, cv2.LINE_AA)
        label("0 / WRIST", points[0][0] + 8, points[0][1] + 14 * unit, size=0.32)
        palm = tuple(map(int, hand.palm))
        cv2.drawMarker(img, palm, (240, 240, 240), cv2.MARKER_CROSS, 18, 1)
        label(f"H{hand_id:02} {hand.side.upper()} X:{palm[0]} Y:{palm[1]}", palm[0]+10, palm[1]+18*unit, size=0.35)

    panel(8 * unit, 8 * unit, W - 16 * unit, 166 * unit)
    label("[ V3 / LANDMARK DEBUG ]", 18 * unit, 28 * unit)
    label(f"FPS {d.get('fps', 0):04.1f}  DETECT {d.get('detect_ms', 0):05.1f}ms  "
          f"HANDS {len(hands)}/2  FACE {'LOCK' if face is not None else 'LOST'}",
          18 * unit, 48 * unit, size=0.4)
    label(f"RAW {raw or '--'}  >  SHOW {shown or '--'}", 18 * unit, 68 * unit, size=0.4)
    label(f"MIC {d.get('voice', 'OFF')} | v toggle mic | local Ukrainian speech",
          18 * unit, 102 * unit, (180, 220, 190), 0.33)
    level = d.get("voice_db", -120.0)
    label(f"INPUT {level:.0f} dB   DROP {d.get('voice_drops', 0)}", 18 * unit, 120 * unit, size=0.33)
    bx, by, bw, bh = int(220 * unit), int(111 * unit), int(130 * unit), max(3, int(8 * unit))
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (55, 65, 55), -1)
    fill = int(bw * np.clip((level + 60) / 60, 0, 1))
    if fill:
        cv2.rectangle(img, (bx, by), (bx + fill, by + bh), (80, 230, 130), -1)
    details = [f"Microphone: {d.get('voice_device', '--')}"]
    if d.get("voice_error"):
        details.append("Помилка: " + d["voice_error"][:95])
    for i, text in enumerate(details):
        overlay(img, voice_text_sprite(text[:110], max(9, int(14 * unit))),
                int(18 * unit), int((128 + 21*i) * unit))

    for hand_id, hand in enumerate(hands, 1):
        x, y = W - 242 * unit, (184 + (hand_id - 1) * 162) * unit
        panel(x, y, 234 * unit, 152 * unit)
        label(f"H{hand_id:02} {hand.side.upper()} {hand.side_score:.0%}", x+10*unit, y+20*unit)
        label("2D JOINT ANGLE / EXT ESTIMATE", x+10*unit, y+37*unit, (170, 180, 180), 0.3)
        for i, (name, color, angle, extended) in enumerate(zip(
                FINGER_NAMES, FINGER_COLORS, hand.finger_angles, hand.finger_extended)):
            value = " --" if angle is None else f"{angle:3.0f}"
            label(f"{name:6} {value}deg  {'EXT' if extended else 'BENT'}",
                  x+10*unit, y+(57+18*i)*unit, color, 0.38)
        label(f"ROCK {int(hand.rock)} UP {int(hand.thumbs_up)} FIST {int(hand.fist)} POINT {int(hand.point_camera)}",
              x+10*unit, y+145*unit, size=0.32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="webcam index (try 1 if 0 is your iPhone)")
    ap.add_argument("--no-vcam", action="store_true", help="preview only; don't start the virtual camera")
    ap.add_argument("--skip-check", action="store_true", help="skip the MediaPipe startup check")
    ap.add_argument("--size", default="1280x720", help="capture size, e.g. 1280x720 or 640x480 (lower = faster)")
    ap.add_argument("--no-flip", action="store_true", help="don't mirror the image")
    ap.add_argument("--no-voice", action="store_true", help="start with microphone recognition off; v toggles it")
    ap.add_argument("--voice-phrase", default="у", help="Ukrainian trigger: у, or у моєму for fewer accidental triggers")
    ap.add_argument("--mic", type=int, default=None, help="microphone device index; default is the system input")
    args = ap.parse_args()
    voice = VoiceTrigger(VOICE_MODEL, phrase=args.voice_phrase, device=args.mic, enabled=not args.no_voice, extra_phrases=("що",))

    model_paths = ensure_models()
    if not args.skip_check:
        preflight(model_paths["face_landmarker.task"])

    cap = cv2.VideoCapture(args.camera)
    if cap.isOpened() and "x" in args.size:
        w, h = args.size.lower().split("x")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(w))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h))
    ok, frame = False, None
    if cap.isOpened():
        for _ in range(5):
            ok, frame = cap.read()
            if not ok:
                break
    if not ok:
        sys.exit(f"Could not read from camera {args.camera}.\n"
                 "  - try --camera 1\n"
                 "  - System Settings > Privacy & Security > Camera: allow your terminal app, then re-run")
    H, W = frame.shape[:2]
    print(f"Camera {args.camera}: {W}x{H}")

    clock = Clock()
    window = "it's giving v3 (q quit, d HUD, v mic, r / e / t / l / b / p / g / j / s / c test)"

    print("Assets:")
    assets = {pose: load_asset(pose) for pose in POSES}
    sigma_overlay = FaceOverlay(assets["sigma"])
    bouquet_fire = VideoAsset(os.path.join(HERE, "assets", "bouquet_fire.mp4"))
    sounds = {pose: asset.audio for pose, asset in assets.items() if isinstance(asset, VideoAsset)}
    sounds["rock"] = ReactionAudio(os.path.join(HERE, "assets", "rock.wav"))
    sounds["rock_left"] = ReactionAudio(os.path.join(HERE, "assets", "rock_left.wav"))
    sounds["bouquet"] = ReactionAudio(os.path.join(HERE, "assets", "bouquet.wav"), loop=False)
    sounds["thumbs_up"] = ReactionAudio(os.path.join(HERE, "assets", "thumbs_up.wav"), loop=False)
    sounds["log_carry"] = ReactionAudio(os.path.join(HERE, "assets", "log_carry.wav"), loop=False)
    sounds["sigma"] = ReactionAudio(os.path.join(HERE, "assets", "sigma.wav"))

    vcam = None
    if not args.no_vcam:
        try:
            import pyvirtualcam
            vcam = pyvirtualcam.Camera(width=W, height=H, fps=30, fmt=pyvirtualcam.PixelFormat.BGR)
            print(f"Virtual camera: '{vcam.device}'  <- pick this camera in Zoom / Meet")
        except Exception as e:
            print(f"Virtual camera unavailable ({e}). Preview-only.")

    face_det, hand_det = build_detectors(model_paths)
    shown, hold, show_hud = None, 0, True
    arm = {p: 0 for p in POSES}
    shown_since = 0.0
    forced, forced_until = None, 0.0
    sm_center, sm_h = np.array([W / 2, H / 2], np.float32), H * 0.45
    print("Running. Focus the preview window: q quit, d HUD, v mic,  r right rock, e left rock, t thumbs_up, l log_carry, b bouquet, p video, g guitar, j vlad, s sigma, c cinema")

    last_frame_time, fps = time.monotonic(), 0.0
    try:
        voice.start()
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera stopped returning frames.")
                break
            if frame.shape[0] != H or frame.shape[1] != W:
                frame = cv2.resize(frame, (W, H))
            if not args.no_flip:
                frame = cv2.flip(frame, 1)

            ts = clock.next()
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            detect_start = time.monotonic()
            fr = face_det.detect_for_video(mp_img, ts)
            hr = hand_det.detect_for_video(mp_img, ts)
            detect_ms = (time.monotonic() - detect_start) * 1000
            face = Face(fr.face_landmarks[0], W, H, fr.face_blendshapes[0] if fr.face_blendshapes else ()) if fr.face_landmarks else None
            hands = [Hand(h, W, H, hr.handedness[i] if i < len(hr.handedness) else [],
                          mirrored=not args.no_flip) for i, h in enumerate(hr.hand_landmarks)]
            raw, dbg = decide(face, hands)
            frame_time = time.monotonic()
            instant_fps = 1.0 / max(frame_time - last_frame_time, 1e-6)
            fps = instant_fps if fps == 0 else 0.9 * fps + 0.1 * instant_fps
            last_frame_time = frame_time
            dbg.update(fps=fps, detect_ms=detect_ms)

            fired = None
            for p in POSES:
                arm[p] = arm[p] + 1 if raw == p else 0
                if raw == p and arm[p] >= ARM.get(p, 3):
                    fired = p
            now = time.monotonic()
            if forced and now >= forced_until:
                if shown == forced and isinstance(assets[forced], VideoAsset):
                    shown, hold = None, 0
                forced = None
            spoken = voice.poll()
            voice_pose = "vlad_voice" if spoken == "що" else "guitar_voice"
            if spoken is not None and isinstance(assets.get(voice_pose), VideoAsset):
                forced = voice_pose
                forced_until = now + assets[forced].total / 1000
                assets[forced].audio.stop()
                shown, hold = None, 0
                print(f"Voice trigger: {spoken} -> {forced}")
            if forced and now < forced_until:
                fired = forced
            if fired:
                if fired != shown:
                    shown_since = now
                shown, hold = fired, 3 if fired == "rock_left" else HOLD_FRAMES
            elif hold > 0:
                hold -= 1
            else:
                shown = None

            voice.set_suppressed(isinstance(assets.get(shown), VideoAsset)
                                 or (shown in sounds and sounds[shown].available))
            dbg["voice"] = voice.status
            dbg.update(voice_db=voice.level_db, voice_heard=voice.heard,
                       voice_device=voice.device_name, voice_phrase=f"{voice.phrase} / що",
                       voice_error=voice.error, voice_drops=voice.dropped_blocks)

            if face is not None:
                sm_center = 0.7 * sm_center + 0.3 * np.array(face.center, np.float32)
                sm_h = 0.7 * sm_h + 0.3 * face.h * FACE_SCALE

            # Draw diagnostics on the camera preview first; memes sit above them.
            preview = frame
            if show_hud:
                preview = frame.copy()
                draw_hud(preview, shown, raw, dbg, face, hands)

            if shown:
                asset = assets[shown]
                idx = asset.frame_at(int((now - shown_since) * 1000))
                target_h = sm_h * ASSET_SCALES.get(shown, 1.0)
                max_h = H * 0.98
                if shown == "rock":
                    # Reserve the badge's largest pulse, gap and frame margins.
                    rock_top = int(np.ceil(150 * swag_badge_scale(W, H) * 1.015)) + 16
                    max_h = H - rock_top - 4
                fitted_h = int(min(target_h, max_h, (W * 0.98) / asset.aspect)) // 8 * 8
                h = int(max(fitted_h, 8) * ASSET_ZOOM.get(shown, 1.0))
                if shown == "vlad_voice":
                    # Fill the output frame, preserving aspect ratio and cropping
                    # excess edges, independent of the tracked face position.
                    h = int(np.ceil(max(H, W / asset.aspect)))
                sprite = asset.scaled(idx, max(h, 8))
                sh, sw = sprite.shape[:2]
                y_offset = ASSET_Y_OFFSETS.get(shown, 0.0) * sh
                x_offset = ASSET_X_OFFSETS.get(shown, 0.0) * sw
                x = int(sm_center[0] - sw / 2 + x_offset)
                y = int(sm_center[1] - sh / 2 - 0.05 * sh + y_offset)
                if shown == "vlad_voice":
                    x, y = (W - sw) // 2, (H - sh) // 2
                if shown == "rock":
                    y = int(np.clip(y, rock_top, H - sh - 4))
                if shown == "log_carry":
                    draw_fissure(frame, (x, y, sw, sh), now - shown_since)
                    if show_hud:
                        draw_fissure(preview, (x, y, sw, sh), now - shown_since)
                if shown == "bouquet":
                    fire = bouquet_fire_sprite(bouquet_fire, (now - shown_since) * 1000,
                                               (x, y, sw, sh))
                    if fire is not None:
                        flames, fx, fy = fire
                        overlay(frame, flames, fx, fy)
                        if show_hud:
                            overlay(preview, flames, fx, fy)
                if shown == "sigma":
                    sprite = sigma_overlay.render(idx, face, frame.shape)
                    x, y = 0, 0
                overlay(frame, sprite, x, y)
                if show_hud:
                    overlay(preview, sprite, x, y)
                if shown == "rock":
                    draw_swag_badge(frame, (x, y, sw, sh), now - shown_since)
                    if show_hud:
                        draw_swag_badge(preview, (x, y, sw, sh), now - shown_since)

            for pose, sound in sounds.items():
                if pose != shown:
                    sound.stop()
            if shown in sounds:
                sounds[shown].sync((now - shown_since) * 1000)

            if vcam:
                vcam.send(frame)
                vcam.sleep_until_next_frame()

            cv2.imshow(window, preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                show_hud = not show_hud
            elif key == ord("v"):
                voice.toggle()
            elif 0 < key < 256 and chr(key) in TEST_KEYS:
                forced = TEST_KEYS[chr(key)]
                duration = assets[forced].total / 1000 if isinstance(assets[forced], VideoAsset) else 2.0
                forced_until = now + duration
                if isinstance(assets[forced], VideoAsset):
                    assets[forced].audio.stop()
                    shown = None  # A manual replay always starts at the beginning.
    finally:
        voice.close()
        cap.release()
        for sound in sounds.values():
            sound.stop()
        for asset in assets.values():
            asset.close()
        bouquet_fire.close()
        face_det.close()
        hand_det.close()
        if vcam:
            vcam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
