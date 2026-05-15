from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt

# ------------------------------------------------
# Paths
# ------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "Dataset"

MACHINE = "fan"

# ------------------------------------------------
# Folder-based loading
# ------------------------------------------------
normal_dir = DATASET_DIR / MACHINE / "test" / "normal"
anomaly_dir = DATASET_DIR / MACHINE / "test" / "anomaly"

normal_path = next(normal_dir.glob("*.wav"))
anomaly_path = next(anomaly_dir.glob("*.wav"))

print("Normal file:", normal_path)
print("Anomaly file:", anomaly_path)


def plot_mfcc(audio_path, label):
    y, sr = librosa.load(audio_path, sr=None)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    plt.figure(figsize=(12, 4))

    librosa.display.specshow(
        mfcc,
        sr=sr,
        x_axis="time"
    )

    plt.colorbar()
    plt.title(f"{MACHINE.upper()} - {label} MFCC Heatmap")
    plt.xlabel("Time (s)")
    plt.ylabel("MFCC Coefficients")

    plt.tight_layout()

    plt.savefig(
        f"{MACHINE}_{label}_mfcc_heatmap.png",
        dpi=300
    )

    plt.show()


plot_mfcc(normal_path, "normal")
plot_mfcc(anomaly_path, "anomaly")