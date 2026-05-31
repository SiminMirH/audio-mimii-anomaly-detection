from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import layers, models


# ------------------------------------------------
# Paths
# ------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "Dataset"

MACHINES = ["fan", "pump", "valve"]


# ------------------------------------------------
# Log-Mel feature extraction
# ------------------------------------------------
def extract_logmel_features(audio_path, n_mels=64):
    """
    Convert one audio file into one fixed-size Log-Mel feature vector.

    Pipeline:
    audio
    -> Mel spectrogram
    -> log scale
    -> mean/std pooling
    -> fixed-size vector
    """

    y, sr = librosa.load(audio_path, sr=None)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels
    )

    logmel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # Mean and std over time
    logmel_mean = np.mean(logmel, axis=1)
    logmel_std = np.std(logmel, axis=1)

    feature_vector = np.concatenate([
        logmel_mean,
        logmel_std
    ])

    return feature_vector


# ------------------------------------------------
# Load features
# ------------------------------------------------
def load_features_from_folder(folder_path, label):
    features = []
    labels = []

    wav_files = sorted(folder_path.glob("*.wav"))

    for wav_file in wav_files:
        features.append(extract_logmel_features(wav_file))
        labels.append(label)

    return np.array(features), np.array(labels)


# ------------------------------------------------
# Build Dense Autoencoder
# ------------------------------------------------
def build_autoencoder(input_dim):
    """
    Dense Autoencoder:
    input -> encoder -> bottleneck -> decoder -> reconstructed input

    The model learns to reconstruct normal sounds.
    Higher reconstruction error = more anomalous.
    """

    input_layer = layers.Input(shape=(input_dim,))

    # Encoder
    x = layers.Dense(64, activation="relu")(input_layer)
    x = layers.Dense(32, activation="relu")(x)

    # Bottleneck
    bottleneck = layers.Dense(16, activation="relu")(x)

    # Decoder
    x = layers.Dense(32, activation="relu")(bottleneck)
    x = layers.Dense(64, activation="relu")(x)

    output_layer = layers.Dense(input_dim, activation="linear")(x)

    model = models.Model(
        inputs=input_layer,
        outputs=output_layer
    )

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model


# ------------------------------------------------
# Run one machine
# ------------------------------------------------
def run_one_machine(machine):
    print("\n" + "=" * 60)
    print(f"Machine: {machine}")
    print("=" * 60)

    train_normal_dir = DATASET_DIR / machine / "train" / "normal"
    test_normal_dir = DATASET_DIR / machine / "test" / "normal"
    test_anomaly_dir = DATASET_DIR / machine / "test" / "anomaly"

    # Train only on normal sounds
    X_train, _ = load_features_from_folder(
        train_normal_dir,
        label=0
    )

    # Test on normal + anomaly
    X_test_normal, y_test_normal = load_features_from_folder(
        test_normal_dir,
        label=0
    )

    X_test_anomaly, y_test_anomaly = load_features_from_folder(
        test_anomaly_dir,
        label=1
    )

    X_test = np.vstack([
        X_test_normal,
        X_test_anomaly
    ])

    y_test = np.concatenate([
        y_test_normal,
        y_test_anomaly
    ])

    print("Train normal:", X_train.shape)
    print("Test normal:", X_test_normal.shape)
    print("Test anomaly:", X_test_anomaly.shape)

    # Scale features
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    input_dim = X_train_scaled.shape[1]

    model = build_autoencoder(input_dim)

    # Train autoencoder only on normal data
    model.fit(
        X_train_scaled,
        X_train_scaled,
        epochs=50,
        batch_size=128,
        validation_split=0.1,
        shuffle=True,
        verbose=1
    )

    # Reconstruct test samples
    X_test_reconstructed = model.predict(X_test_scaled)

    # Reconstruction error = anomaly score
    reconstruction_errors = np.mean(
        np.square(X_test_scaled - X_test_reconstructed),
        axis=1
    )

    # ROC-AUC
    auc = roc_auc_score(
        y_test,
        reconstruction_errors
    )

    # Simple threshold:
    # 95th percentile of training reconstruction error
    X_train_reconstructed = model.predict(X_train_scaled)

    train_errors = np.mean(
        np.square(X_train_scaled - X_train_reconstructed),
        axis=1
    )

    threshold = np.percentile(train_errors, 95)

    y_pred = np.where(
        reconstruction_errors > threshold,
        1,
        0
    )

    cm = confusion_matrix(y_test, y_pred)

    report = classification_report(
        y_test,
        y_pred,
        target_names=["normal", "anomaly"],
        output_dict=True,
        zero_division=0
    )

    print("Threshold:", threshold)
    print("Confusion matrix:")
    print(cm)
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Anomaly Recall: {report['anomaly']['recall']:.4f}")
    print(f"Anomaly F1: {report['anomaly']['f1-score']:.4f}")

    return {
        "machine": machine,
        "train_normal_files": len(X_train),
        "test_normal_files": len(X_test_normal),
        "test_anomaly_files": len(X_test_anomaly),
        "roc_auc": auc,
        "accuracy": report["accuracy"],
        "normal_precision": report["normal"]["precision"],
        "normal_recall": report["normal"]["recall"],
        "anomaly_precision": report["anomaly"]["precision"],
        "anomaly_recall": report["anomaly"]["recall"],
        "anomaly_f1": report["anomaly"]["f1-score"],
        "tn": cm[0, 0],
        "fp": cm[0, 1],
        "fn": cm[1, 0],
        "tp": cm[1, 1],
        "threshold": threshold,
    }


# ------------------------------------------------
# Main
# ------------------------------------------------
def main():
    results = []

    for machine in MACHINES:
        result = run_one_machine(machine)
        results.append(result)

    results_df = pd.DataFrame(results)

    print("\nFinal Autoencoder Results:")
    print(results_df)

    output_path = BASE_DIR / "Autoencoder" / "dense_autoencoder_results.csv"

    results_df.to_csv(
        output_path,
        index=False
    )

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()