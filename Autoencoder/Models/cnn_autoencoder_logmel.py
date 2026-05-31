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
# Log-Mel Spectrogram Extraction
# ------------------------------------------------
def extract_logmel_image(audio_path, n_mels=64, max_frames=313):
    """
    Convert one audio file into a fixed-size Log-Mel spectrogram image.

    Output shape:
    (n_mels, max_frames, 1)

    CNNs need fixed-size 2D input.
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

    # Normalize values roughly to 0-1 range
    logmel = (logmel - logmel.min()) / (logmel.max() - logmel.min() + 1e-8)

    # Pad or cut time frames to fixed length
    if logmel.shape[1] < max_frames:
        pad_width = max_frames - logmel.shape[1]
        logmel = np.pad(
            logmel,
            ((0, 0), (0, pad_width)),
            mode="constant"
        )
    else:
        logmel = logmel[:, :max_frames]

    # Add channel dimension for CNN
    logmel = logmel[..., np.newaxis]

    return logmel.astype(np.float32)


# ------------------------------------------------
# Load Dataset
# ------------------------------------------------
def load_images_from_folder(folder_path, label):
    images = []
    labels = []

    wav_files = sorted(folder_path.glob("*.wav"))

    for wav_file in wav_files:
        image = extract_logmel_image(wav_file)
        images.append(image)
        labels.append(label)

    return np.array(images), np.array(labels)


# ------------------------------------------------
# Build CNN Autoencoder
# ------------------------------------------------
def build_cnn_autoencoder(input_shape):
    """
    CNN Autoencoder:

    Input Log-Mel image
    -> CNN Encoder
    -> Bottleneck representation
    -> CNN Decoder
    -> Reconstructed Log-Mel image

    The model learns to reconstruct normal sounds.
    Higher reconstruction error indicates anomaly.
    """

    input_layer = layers.Input(shape=input_shape)

    # -------------------------
    # Encoder
    # -------------------------
    x = layers.Conv2D(
        16,
        kernel_size=(3, 3),
        activation="relu",
        padding="same"
    )(input_layer)

    x = layers.MaxPooling2D(
        pool_size=(2, 2),
        padding="same"
    )(x)

    x = layers.Conv2D(
        32,
        kernel_size=(3, 3),
        activation="relu",
        padding="same"
    )(x)

    x = layers.MaxPooling2D(
        pool_size=(2, 2),
        padding="same"
    )(x)

    x = layers.Conv2D(
        64,
        kernel_size=(3, 3),
        activation="relu",
        padding="same"
    )(x)

    encoded = layers.MaxPooling2D(
        pool_size=(2, 2),
        padding="same"
    )(x)

    # -------------------------
    # Decoder
    # -------------------------
    x = layers.Conv2D(
        64,
        kernel_size=(3, 3),
        activation="relu",
        padding="same"
    )(encoded)

    x = layers.UpSampling2D(
        size=(2, 2)
    )(x)

    x = layers.Conv2D(
        32,
        kernel_size=(3, 3),
        activation="relu",
        padding="same"
    )(x)

    x = layers.UpSampling2D(
        size=(2, 2)
    )(x)

    x = layers.Conv2D(
        16,
        kernel_size=(3, 3),
        activation="relu",
        padding="same"
    )(x)

    x = layers.UpSampling2D(
        size=(2, 2)
    )(x)

    decoded = layers.Conv2D(
        1,
        kernel_size=(3, 3),
        activation="sigmoid",
        padding="same"
    )(x)

    # Crop output to match input size
    decoded = layers.Cropping2D(
        cropping=((0, 0), (0, decoded.shape[2] - input_shape[1]))
    )(decoded)

    model = models.Model(
        inputs=input_layer,
        outputs=decoded
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

    X_train, _ = load_images_from_folder(train_normal_dir, label=0)
    X_test_normal, y_test_normal = load_images_from_folder(test_normal_dir, label=0)
    X_test_anomaly, y_test_anomaly = load_images_from_folder(test_anomaly_dir, label=1)

    X_test = np.concatenate([X_test_normal, X_test_anomaly], axis=0)
    y_test = np.concatenate([y_test_normal, y_test_anomaly], axis=0)

    print("Train normal:", X_train.shape)
    print("Test normal:", X_test_normal.shape)
    print("Test anomaly:", X_test_anomaly.shape)

    input_shape = X_train.shape[1:]

    model = build_cnn_autoencoder(input_shape)

    model.summary()

    # Train only on normal spectrograms
    model.fit(
        X_train,
        X_train,
        epochs=30,
        batch_size=64,
        validation_split=0.1,
        shuffle=True,
        verbose=1
    )

    # Reconstruct test spectrograms
    X_test_reconstructed = model.predict(X_test)

    # Reconstruction error per sample
    reconstruction_errors = np.mean(
        np.square(X_test - X_test_reconstructed),
        axis=(1, 2, 3)
    )

    auc = roc_auc_score(y_test, reconstruction_errors)

    # Threshold from train reconstruction error
    X_train_reconstructed = model.predict(X_train)

    train_errors = np.mean(
        np.square(X_train - X_train_reconstructed),
        axis=(1, 2, 3)
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

    print("\nFinal CNN Autoencoder Results:")
    print(results_df)

    output_path = BASE_DIR / "Autoencoder" / "cnn_autoencoder_results.csv"

    results_df.to_csv(
        output_path,
        index=False
    )

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()