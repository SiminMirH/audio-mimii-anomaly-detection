from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models


# ------------------------------------------------
# Paths
# ------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "Dataset"

MACHINES = ["fan", "pump", "valve"]


# ------------------------------------------------
# Log-Mel Sequence Extraction
# ------------------------------------------------
def extract_logmel_sequence(audio_path, n_mels=64, max_frames=313):
    """
    Convert audio into a Log-Mel sequence.

    Shape:
    original logmel: (mel_bands, time_frames)
    transposed:      (time_frames, mel_bands)

    Transformer reads the audio as a sequence over time.
    Each time frame has 64 Mel-frequency features.
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

    # Normalize to 0-1
    logmel = (logmel - logmel.min()) / (logmel.max() - logmel.min() + 1e-8)

    # Pad or cut to fixed time length
    if logmel.shape[1] < max_frames:
        pad_width = max_frames - logmel.shape[1]
        logmel = np.pad(
            logmel,
            ((0, 0), (0, pad_width)),
            mode="constant"
        )
    else:
        logmel = logmel[:, :max_frames]

    # Transformer expects sequence:
    # (time_frames, features)
    sequence = logmel.T

    return sequence.astype(np.float32)


# ------------------------------------------------
# Load Dataset
# ------------------------------------------------
def load_sequences_from_folder(folder_path, label):
    sequences = []
    labels = []

    wav_files = sorted(folder_path.glob("*.wav"))

    for wav_file in wav_files:
        sequence = extract_logmel_sequence(wav_file)
        sequences.append(sequence)
        labels.append(label)

    return np.array(sequences), np.array(labels)


# ------------------------------------------------
# Transformer Block
# ------------------------------------------------
def transformer_block(x, num_heads=4, key_dim=32, ff_dim=128, dropout=0.1):
    """
    Basic Transformer encoder block.

    Multi-head attention:
    learns relationships between time frames.

    Feed-forward network:
    processes each time frame representation.
    """

    # Self-attention
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim
    )(x, x)

    attention_output = layers.Dropout(dropout)(attention_output)

    # Residual connection + normalization
    x = layers.LayerNormalization(epsilon=1e-6)(x + attention_output)

    # Feed-forward network
    ff_output = layers.Dense(ff_dim, activation="relu")(x)
    ff_output = layers.Dense(x.shape[-1])(ff_output)
    ff_output = layers.Dropout(dropout)(ff_output)

    # Residual connection + normalization
    x = layers.LayerNormalization(epsilon=1e-6)(x + ff_output)

    return x


# ------------------------------------------------
# Build Transformer Autoencoder
# ------------------------------------------------
def build_transformer_autoencoder(input_shape):
    """
    Transformer Autoencoder.

    Input:
    Log-Mel sequence

    The model learns to reconstruct normal Log-Mel sequences.
    High reconstruction error = anomaly.
    """

    input_layer = layers.Input(shape=input_shape)

    # Project input features to embedding dimension
    x = layers.Dense(64)(input_layer)

    # Transformer encoder blocks
    x = transformer_block(x)
    x = transformer_block(x)

    # Bottleneck-like compressed representation
    x = layers.Dense(32, activation="relu")(x)

    # Decoder projection
    x = layers.Dense(64, activation="relu")(x)

    # Reconstruct original Mel features
    output_layer = layers.Dense(input_shape[-1], activation="sigmoid")(x)

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

    X_train, _ = load_sequences_from_folder(train_normal_dir, label=0)
    X_test_normal, y_test_normal = load_sequences_from_folder(test_normal_dir, label=0)
    X_test_anomaly, y_test_anomaly = load_sequences_from_folder(test_anomaly_dir, label=1)

    X_test = np.concatenate([X_test_normal, X_test_anomaly], axis=0)
    y_test = np.concatenate([y_test_normal, y_test_anomaly], axis=0)

    print("Train normal:", X_train.shape)
    print("Test normal:", X_test_normal.shape)
    print("Test anomaly:", X_test_anomaly.shape)

    input_shape = X_train.shape[1:]

    model = build_transformer_autoencoder(input_shape)

    model.summary()

    # Train only on normal sounds
    model.fit(
        X_train,
        X_train,
        epochs=20,
        batch_size=32,
        validation_split=0.1,
        shuffle=True,
        verbose=1
    )

    # Reconstruct test sequences
    X_test_reconstructed = model.predict(X_test)

    # Reconstruction error
    reconstruction_errors = np.mean(
        np.square(X_test - X_test_reconstructed),
        axis=(1, 2)
    )

    auc = roc_auc_score(
        y_test,
        reconstruction_errors
    )

    # Threshold from training reconstruction error
    X_train_reconstructed = model.predict(X_train)

    train_errors = np.mean(
        np.square(X_train - X_train_reconstructed),
        axis=(1, 2)
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

    print("\nFinal Transformer Autoencoder Results:")
    print(results_df)

    output_path = BASE_DIR / "Autoencoder" / "transformer_autoencoder_results.csv"

    results_df.to_csv(
        output_path,
        index=False
    )

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()