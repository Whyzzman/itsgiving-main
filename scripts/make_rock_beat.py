"""Generate an original low-register, bass-heavy beat using synthesis only."""
from pathlib import Path
import wave
import numpy as np

RATE = 44100
BPM = 110
BEAT = 60 / BPM
LENGTH = 16 * BEAT
rng = np.random.default_rng(20260913)
mix = np.zeros((round(LENGTH * RATE), 2), np.float64)


def add(signal, beat, gain=1.0, pan=0.0):
    start = round(beat * BEAT * RATE)
    indices = (start + np.arange(len(signal))) % len(mix)
    stereo = signal[:, None] * np.array([np.sqrt((1-pan)/2), np.sqrt((1+pan)/2)])
    np.add.at(mix, indices, stereo * gain)


def timeline(seconds):
    return np.arange(round(seconds * RATE)) / RATE


def fade(signal, seconds=.005):
    n = min(round(seconds * RATE), len(signal)//2)
    signal[:n] *= np.linspace(0, 1, n)
    signal[-n:] *= np.linspace(1, 0, n)
    return signal


# Slow, deep kick without the sharp high-frequency click.
for beat in (0, 1.75, 3.5, 4, 5.5, 7.25, 8, 9.75, 11.5, 12, 13.5, 15.25):
    t = timeline(.42)
    pitch = 39 + 72 * np.exp(-t * 32)
    kick = np.sin(2*np.pi*np.cumsum(pitch)/RATE) * np.exp(-t*10)
    add(fade(kick,.008), beat, .85)
for beat in (2, 6, 10, 14):
    t = timeline(.17)
    noise = rng.normal(size=len(t))
    noise = np.convolve(noise, np.ones(18)/18, mode='same')
    clap = noise * (np.exp(-t*40) + .6*np.exp(-np.maximum(t-.015, 0)*60)*(t>.015))
    clap += .5*np.sin(2*np.pi*145*t)*np.exp(-t*24)
    add(fade(np.tanh(clap*1.7)), beat, .18, -.08)
for i in range(32):
    t = timeline(.06 if i%4 else .11)
    noise = rng.normal(size=len(t))
    hat = (noise-np.convolve(noise,np.ones(9)/9,mode='same'))*np.exp(-t*70)
    add(fade(hat), i*.5 + (.06 if i%2 else 0), .022 if i%2 else .016, .3 if i%2 else -.3)

# Sustained sub-bass at 41–62 Hz, centered in stereo, with gentle pitch slides.
for beat, midi, beats in [(0,28,1.65),(1.75,28,1.4),(3.5,31,.45),(4,28,1.4),(5.5,35,1.3),(7.25,31,.65),
                          (8,28,1.65),(9.75,28,1.4),(11.5,35,.45),(12,31,1.4),(13.5,28,1.3),(15.25,28,.65)]:
    t = timeline(beats*BEAT)
    freq = 440*2**((midi-69)/12)
    phase = 2*np.pi*np.cumsum(freq*(1+.10*np.exp(-t*22)))/RATE
    bass = .9*np.sin(phase) + .22*np.sin(2*phase) + .09*np.sin(3*phase)
    bass = np.tanh(bass*1.3)*np.exp(-t*.65)
    duck = 1 - .65*np.exp(-t*16)
    add(fade(bass*duck,.015), beat, .82)

# Low, muted synth motif replaces the bright cowbell entirely.
notes = [(0,52),(1.75,55),(3.5,50),(4,52),(5.5,47),(7.25,50)]
for bar in (0,8):
    for beat,midi in notes:
        t = timeline(.50)
        freq=440*2**((midi-69)/12)
        lead=(np.sin(2*np.pi*freq*t) + .18*np.sin(2*np.pi*freq*2*t))*np.exp(-t*7)
        lead=fade(lead,.018)
        add(lead,bar+beat,.13,-.1)
        add(lead,bar+beat+.75,.025,.35)

# Gentle saturation and headroom; preserve the downbeat for instant triggering.
mix=np.tanh(mix*1.15)
mix *= .78/max(np.max(np.abs(mix)),1e-9)
edge=round(.003*RATE)
mix[:edge] *= np.linspace(0,1,edge)[:,None]
mix[-edge:] *= np.linspace(1,0,edge)[:,None]
output=Path(__file__).resolve().parents[1]/'assets'/'rock.wav'
with wave.open(str(output),'wb') as wav:
    wav.setnchannels(2); wav.setsampwidth(2); wav.setframerate(RATE)
    wav.writeframes(np.rint(mix*32767).astype('<i2').tobytes())
print(f'{output.name}: {LENGTH:.2f}s, {BPM} BPM, original synthesized stereo beat')
