from pathlib import Path
import argparse

import librosa
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(
    description="Load an audio file and display its waveform."
)

parser.add_argument(
    "audio_path",
    type=Path,
    help="Path to the audio file"
)

args = parser.parse_args()
audio_path = args.audio_path

if not audio_path.exists():
    raise SystemExit(f"Audio file not found: {audio_path}")

print(f"Loading audio from: {audio_path}")

y, sample_rate = librosa.load(
    audio_path,
    sr=None,
    mono=True
)

duration = len(y) / sample_rate

print(f"Sample rate: {sample_rate} samples per second")
print(f"Number of samples: {len(y)}")
print(f"Duration: {duration:.2f} seconds")

time = librosa.times_like(y, sr=sample_rate)

plt.figure(figsize=(12, 4))
plt.plot(time, y)
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Audio waveform")
plt.tight_layout()
plt.show()
