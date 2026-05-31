from pathlib import Path

import librosa
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


# ------------------------------------------------
# Paths
# ------------------------------------------------
# Project root:
# AD_Benchmark_MIMII/
BASE_DIR = Path(__file__).resolve().parents[2]

# Dataset folder
DATASET_DIR = BASE_DIR / "Dataset"

# Machine category
# Change this to "pump" or "valve" later
MACHINE = "fan"


# ------------------------------------------------
# Dataset paths
# ------------------------------------------------
train_normal_dir = DATASET_DIR / MACHINE / "train" / "normal"
test_normal_dir = DATASET_DIR / MACHINE / "test" / "normal"
test_anomaly_dir = DATASET_DIR / MACHINE / "test" / "anomaly"


# ------------------------------------------------
# MFCC feature extraction
# ------------------------------------------------
def extract_mfcc_features(audio_path, n_mfcc=13):
    """
    Extract MFCC features from one audio file.

    Pipeline:
    audio file
    -> load waveform
    -> compute MFCCs
    -> mean/std pooling
    -> fixed-size feature vector

    Why pooling?
    MFCC output has shape:
        n_mfcc x time_frames

    Machine learning models like SVM need a fixed-size vector.
    Therefore, we summarize each MFCC coefficient over time
    using mean and standard deviation.
    """

    # Load audio
    # y = waveform
    # sr = sample rate
    y, sr = librosa.load(audio_path, sr=None)

    # Extract MFCCs
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc
    )

    # Mean pooling over time
    mfcc_mean = np.mean(mfcc, axis=1)

    # Standard deviation pooling over time
    mfcc_std = np.std(mfcc, axis=1)

    # Combine mean and std into one feature vector
    feature_vector = np.concatenate([
        mfcc_mean,
        mfcc_std
    ])

    return feature_vector


# ------------------------------------------------
# Load dataset
# ------------------------------------------------
def load_features_from_folder(folder_path, label):
    """
    Load all wav files from a folder and extract MFCC features.

    label:
        0 = normal
        1 = anomaly
    """

    features = []
    labels = []
    file_paths = []

    wav_files = sorted(folder_path.glob("*.wav"))

    for wav_file in wav_files:
        feature_vector = extract_mfcc_features(wav_file)

        features.append(feature_vector)
        labels.append(label)
        file_paths.append(wav_file)

    return np.array(features), np.array(labels), file_paths


# ------------------------------------------------
# Main pipeline
# ------------------------------------------------
def main():
    print("Machine:", MACHINE)
    print("Train normal folder:", train_normal_dir)
    print("Test normal folder:", test_normal_dir)
    print("Test anomaly folder:", test_anomaly_dir)

    # --------------------------------------------
    # 1. Load training data
    # --------------------------------------------
    # One-Class SVM is trained only on normal samples.
    X_train, y_train, train_files = load_features_from_folder(
        train_normal_dir,
        label=0
    )

    # --------------------------------------------
    # 2. Load test data
    # --------------------------------------------
    X_test_normal, y_test_normal, normal_files = load_features_from_folder(
        test_normal_dir,
        label=0
    )

    X_test_anomaly, y_test_anomaly, anomaly_files = load_features_from_folder(
        test_anomaly_dir,
        label=1
    )

    # Combine normal and anomaly test samples
    X_test = np.vstack([
        X_test_normal,
        X_test_anomaly
    ])

    y_test = np.concatenate([
        y_test_normal,
        y_test_anomaly
    ])

    print("\nDataset sizes:")
    print("Train normal:", X_train.shape)
    print("Test normal:", X_test_normal.shape)
    print("Test anomaly:", X_test_anomaly.shape)

    # --------------------------------------------
    # 3. Feature scaling
    # --------------------------------------------
    # SVM is sensitive to feature scale.
    # We fit the scaler only on training normal data.
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    # --------------------------------------------
    # 4. Train One-Class SVM
    # --------------------------------------------
    # One-Class SVM learns the boundary of normal data.
    #
    # nu:
    # approximate fraction of expected outliers
    #
    # kernel="rbf":
    # nonlinear boundary
    #
    model = OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=0.1
    )

    model.fit(X_train_scaled)

    # --------------------------------------------
    # 5. Predict
    # --------------------------------------------
    # sklearn One-Class SVM output:
    # +1 = normal
    # -1 = anomaly
    #
    raw_predictions = model.predict(X_test_scaled)

    # Convert to our labels:
    # 0 = normal
    # 1 = anomaly
    y_pred = np.where(raw_predictions == 1, 0, 1)

    # --------------------------------------------
    # 6. Anomaly scores
    # --------------------------------------------
    # decision_function:
    # higher = more normal
    # lower = more anomalous
    #
    # We multiply by -1 so:
    # higher score = more anomalous
    anomaly_scores = -model.decision_function(X_test_scaled)

    # --------------------------------------------
    # 7. Evaluation
    # --------------------------------------------
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["normal", "anomaly"]
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    auc = roc_auc_score(y_test, anomaly_scores)
    print("\nROC-AUC:", round(auc, 4))

    # --------------------------------------------
    # 8. Show most anomalous files
    # --------------------------------------------
    all_test_files = normal_files + anomaly_files

    sorted_indices = np.argsort(anomaly_scores)[::-1]

    print("\nTop 10 most anomalous test files:")

    for idx in sorted_indices[:10]:
        print(
            f"score={anomaly_scores[idx]:.4f}",
            all_test_files[idx].name
        )


if __name__ == "__main__":
    main()