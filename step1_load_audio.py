from pathlib import Path

import librosa
import matplotlib.pyplot as plt


audio_path = Path(__file__).parent / "melody.wav"

print(f"Looking for audio at: {audio_path}")
print(f"File exists: {audio_path.exists()}")

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
