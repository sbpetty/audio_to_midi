from pathlib import Path
import argparse

import librosa
import numpy as np


parser = argparse.ArgumentParser(
    description="Detect the pitch of a single piano note."
)
parser.add_argument("audio_path", type=Path)

args = parser.parse_args()

if not args.audio_path.exists():
    raise SystemExit(f"File not found: {args.audio_path}")

audio, sample_rate = librosa.load(
    args.audio_path,
    sr=None,
    mono=True
)

# Analyze the more stable middle portion of the piano note.
start = int(0.15 * sample_rate)
end = int(0.75 * sample_rate)
segment = audio[start:end]

frequencies = librosa.yin(
    segment,
    fmin=27.5,
    fmax=4186.0,
    sr=sample_rate,
    frame_length=4096,
    hop_length=256
)

frequencies = frequencies[np.isfinite(frequencies)]

if len(frequencies) == 0:
    raise SystemExit("Could not detect a pitch.")

frequency = float(np.median(frequencies))
midi_pitch = round(69 + 12 * np.log2(frequency / 440.0))
note_name = librosa.midi_to_note(midi_pitch)

print(f"Estimated frequency: {frequency:.2f} Hz")
print(f"Detected note: {note_name}")
print(f"MIDI pitch: {midi_pitch}")