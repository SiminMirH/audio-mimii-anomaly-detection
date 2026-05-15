from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------
# Paths
# ------------------------------------------------
# Path to the main project folder
BASE_DIR = Path(__file__).resolve().parents[1]

# Path to dataset folder
DATASET_DIR = BASE_DIR / "Dataset"

# Select machine category
# Change this later if needed:
# "fan", "pump", "valve"
MACHINE = "fan"


# ------------------------------------------------
# Folder-based loading
# ------------------------------------------------
# Path to normal test samples
normal_dir = DATASET_DIR / MACHINE / "test" / "normal"

# Path to anomaly test samples
anomaly_dir = DATASET_DIR / MACHINE / "test" / "anomaly"


# ------------------------------------------------
# Automatically select first audio file
# ------------------------------------------------
# next(...) takes the first matching wav file

normal_path = next(normal_dir.glob("*.wav"))

anomaly_path = next(anomaly_dir.glob("*.wav"))


print("Normal file:", normal_path)
print("Anomaly file:", anomaly_path)


# ------------------------------------------------
# Function for visualization
# ------------------------------------------------
def plot_audio(audio_path, label):

    # --------------------------------------------
    # Load audio file
    # --------------------------------------------
    # y  = audio signal (waveform values)
    # sr = sample rate
    #
    # sr=None keeps original dataset sample rate
    #
    y, sr = librosa.load(audio_path, sr=None)

    print(f"\nLoaded {label} audio")
    print("Sample rate:", sr)
    print("Audio shape:", y.shape)

    # ============================================
    # 1. WAVEFORM
    # ============================================
    # Waveform shows:
    # amplitude over time
    #
    # x-axis = time
    # y-axis = loudness/amplitude
    #
    plt.figure(figsize=(12, 4))

    librosa.display.waveshow(y, sr=sr)

    plt.title(f"{MACHINE.upper()} - {label} Waveform")

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.tight_layout()

    # Save image
    plt.savefig(
        f"{MACHINE}_{label}_waveform.png",
        dpi=300
    )

    plt.show()

    # ============================================
    # 2. STFT SPECTROGRAM
    # ============================================
    #
    # STFT = Short-Time Fourier Transform
    #
    # Converts audio from:
    # time domain -> frequency domain
    #
    # We can now see:
    # frequencies changing over time
    #
    # D = complex-valued frequency representation
    #
    D = librosa.stft(y)

    # Convert amplitudes to decibel scale (dB)
    #
    # Easier for visualization
    #
    S_db = librosa.amplitude_to_db(
        np.abs(D),
        ref=np.max
    )

    plt.figure(figsize=(12, 4))

    librosa.display.specshow(
        S_db,
        sr=sr,
        x_axis="time",
        y_axis="log"   # logarithmic frequency axis
    )

    plt.colorbar(format="%+2.0f dB")

    plt.title(f"{MACHINE.upper()} - {label} STFT Spectrogram")

    plt.tight_layout()

    plt.savefig(
        f"{MACHINE}_{label}_spectrogram.png",
        dpi=300
    )

    plt.show()

    # ============================================
    # 3. LOG-MEL SPECTROGRAM
    # ============================================
    #
    # Mel scale approximates
    # how humans perceive sound frequencies
    #
    # Frequently used in:
    # - CNNs
    # - Transformers
    # - Audio classification
    # - Speech recognition
    #
    # Better aligned with human hearing
    #
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128
    )

    # Convert Mel spectrogram to dB scale
    log_mel_spec = librosa.power_to_db(
        mel_spec,
        ref=np.max
    )

    plt.figure(figsize=(12, 4))

    librosa.display.specshow(
        log_mel_spec,
        sr=sr,
        x_axis="time",
        y_axis="mel"
    )

    plt.colorbar(format="%+2.0f dB")

    plt.title(f"{MACHINE.upper()} - {label} Log-Mel Spectrogram")

    plt.tight_layout()

    plt.savefig(
        f"{MACHINE}_{label}_logmel.png",
        dpi=300
    )

    plt.show()


# ------------------------------------------------
# Run visualizations
# ------------------------------------------------
plot_audio(normal_path, "normal")
plot_audio(anomaly_path, "anomaly")