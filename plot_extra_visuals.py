from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------
# Paths
# ------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "Dataset"

MACHINE = "fan"

normal_dir = DATASET_DIR / MACHINE / "test" / "normal"
anomaly_dir = DATASET_DIR / MACHINE / "test" / "anomaly"

normal_path = next(normal_dir.glob("*.wav"))
anomaly_path = next(anomaly_dir.glob("*.wav"))

print("Normal file:", normal_path)
print("Anomaly file:", anomaly_path)


# ------------------------------------------------
# Load audio
# ------------------------------------------------
def load_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    return y, sr


# ------------------------------------------------
# 1. RMS Energy
# ------------------------------------------------
# RMS shows loudness / energy over time
def plot_rms(audio_path, label):
    y, sr = load_audio(audio_path)

    rms = librosa.feature.rms(y=y)[0]
    times = librosa.frames_to_time(range(len(rms)), sr=sr)

    plt.figure(figsize=(12, 4))
    plt.plot(times, rms)

    plt.title(f"{MACHINE.upper()} - {label} RMS Energy")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy")

    plt.tight_layout()
    plt.savefig(f"{MACHINE}_{label}_rms_energy.png", dpi=300)
    plt.show()


# ------------------------------------------------
# 2. Spectral Centroid
# ------------------------------------------------
# Spectral centroid shows where the "center of mass"
# of the frequency spectrum is.
# Higher value = brighter / sharper sound
def plot_spectral_centroid(audio_path, label):
    y, sr = load_audio(audio_path)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    times = librosa.frames_to_time(range(len(centroid)), sr=sr)

    plt.figure(figsize=(12, 4))
    plt.plot(times, centroid)

    plt.title(f"{MACHINE.upper()} - {label} Spectral Centroid")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")

    plt.tight_layout()
    plt.savefig(f"{MACHINE}_{label}_spectral_centroid.png", dpi=300)
    plt.show()


# ------------------------------------------------
# 3. Zero Crossing Rate
# ------------------------------------------------
# ZCR counts how often the signal crosses zero.
# It can indicate noisiness or roughness of the signal.
def plot_zero_crossing_rate(audio_path, label):
    y, sr = load_audio(audio_path)

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    times = librosa.frames_to_time(range(len(zcr)), sr=sr)

    plt.figure(figsize=(12, 4))
    plt.plot(times, zcr)

    plt.title(f"{MACHINE.upper()} - {label} Zero Crossing Rate")
    plt.xlabel("Time (s)")
    plt.ylabel("ZCR")

    plt.tight_layout()
    plt.savefig(f"{MACHINE}_{label}_zero_crossing_rate.png", dpi=300)
    plt.show()


# ------------------------------------------------
# Run
# ------------------------------------------------
plot_rms(normal_path, "normal")
plot_rms(anomaly_path, "anomaly")

plot_spectral_centroid(normal_path, "normal")
plot_spectral_centroid(anomaly_path, "anomaly")

plot_zero_crossing_rate(normal_path, "normal")
plot_zero_crossing_rate(anomaly_path, "anomaly")