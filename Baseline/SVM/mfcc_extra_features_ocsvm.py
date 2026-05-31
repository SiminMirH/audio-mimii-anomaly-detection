from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


# ------------------------------------------------
# Paths
# ------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = BASE_DIR / "Dataset"

MACHINES = ["fan", "pump", "valve"]
NU_VALUES = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3]


# ------------------------------------------------
# Improved feature extraction
# ------------------------------------------------
def extract_features(audio_path, n_mfcc=20):
    """
    Improved handcrafted feature vector.

    Features:
    - MFCC mean/std
    - Delta MFCC mean/std
    - Delta-Delta MFCC mean/std
    - RMS mean/std
    - Spectral Centroid mean/std
    - Zero Crossing Rate mean/std
    """

    # Load audio
    y, sr = librosa.load(audio_path, sr=None)

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc
    )

    # Delta MFCC = first-order change over time
    delta_mfcc = librosa.feature.delta(mfcc)

    # Delta-Delta MFCC = second-order change over time
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    # RMS Energy
    rms = librosa.feature.rms(y=y)

    # Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)

    # Helper: summarize feature over time
    def summarize(feature):
        mean = np.mean(feature, axis=1)
        std = np.std(feature, axis=1)
        return np.concatenate([mean, std])

    feature_vector = np.concatenate([
        summarize(mfcc),
        summarize(delta_mfcc),
        summarize(delta2_mfcc),
        summarize(rms),
        summarize(spectral_centroid),
        summarize(zcr),
    ])

    return feature_vector


# ------------------------------------------------
# Load features from folder
# ------------------------------------------------
def load_features_from_folder(folder_path, label):
    features = []
    labels = []
    files = []

    wav_files = sorted(folder_path.glob("*.wav"))

    for wav_file in wav_files:
        feature_vector = extract_features(wav_file)

        features.append(feature_vector)
        labels.append(label)
        files.append(wav_file)

    return np.array(features), np.array(labels), files


# ------------------------------------------------
# Run one experiment
# ------------------------------------------------
def run_one_machine(machine, nu):
    train_normal_dir = DATASET_DIR / machine / "train" / "normal"
    test_normal_dir = DATASET_DIR / machine / "test" / "normal"
    test_anomaly_dir = DATASET_DIR / machine / "test" / "anomaly"

    print("\n" + "=" * 60)
    print(f"Machine: {machine} | nu = {nu}")
    print("=" * 60)

    # Train data: normal only
    X_train, _, _ = load_features_from_folder(
        train_normal_dir,
        label=0
    )

    # Test data: normal + anomaly
    X_test_normal, y_test_normal, _ = load_features_from_folder(
        test_normal_dir,
        label=0
    )

    X_test_anomaly, y_test_anomaly, _ = load_features_from_folder(
        test_anomaly_dir,
        label=1
    )

    X_test = np.vstack([X_test_normal, X_test_anomaly])
    y_test = np.concatenate([y_test_normal, y_test_anomaly])

    print("Train normal:", X_train.shape)
    print("Test normal:", X_test_normal.shape)
    print("Test anomaly:", X_test_anomaly.shape)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train One-Class SVM
    model = OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=nu
    )

    model.fit(X_train_scaled)

    # Predict
    raw_pred = model.predict(X_test_scaled)

    # sklearn:
    # +1 = normal
    # -1 = anomaly
    #
    # convert to:
    # 0 = normal
    # 1 = anomaly
    y_pred = np.where(raw_pred == 1, 0, 1)

    # Anomaly score
    anomaly_scores = -model.decision_function(X_test_scaled)

    # Evaluation
    cm = confusion_matrix(y_test, y_pred)

    report = classification_report(
        y_test,
        y_pred,
        target_names=["normal", "anomaly"],
        output_dict=True,
        zero_division=0
    )

    auc = roc_auc_score(y_test, anomaly_scores)

    print("Confusion matrix:")
    print(cm)
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Anomaly Recall: {report['anomaly']['recall']:.4f}")
    print(f"Anomaly F1: {report['anomaly']['f1-score']:.4f}")

    return {
        "machine": machine,
        "nu": nu,
        "train_normal_files": len(X_train),
        "test_normal_files": len(X_test_normal),
        "test_anomaly_files": len(X_test_anomaly),
        "accuracy": report["accuracy"],
        "roc_auc": auc,
        "normal_precision": report["normal"]["precision"],
        "normal_recall": report["normal"]["recall"],
        "anomaly_precision": report["anomaly"]["precision"],
        "anomaly_recall": report["anomaly"]["recall"],
        "anomaly_f1": report["anomaly"]["f1-score"],
        "tn": cm[0, 0],
        "fp": cm[0, 1],
        "fn": cm[1, 0],
        "tp": cm[1, 1],
    }


# ------------------------------------------------
# Main
# ------------------------------------------------
def main():
    all_results = []

    for machine in MACHINES:
        for nu in NU_VALUES:
            result = run_one_machine(machine, nu)
            all_results.append(result)

    results_df = pd.DataFrame(all_results)

    print("\nFinal results:")
    print(results_df)

    output_path = BASE_DIR / "Baseline" / "SVM" / "ocsvm_improved_features_results.csv"
    results_df.to_csv(output_path, index=False)

    print(f"\nSaved results to: {output_path}")

    best_results = results_df.loc[
        results_df.groupby("machine")["roc_auc"].idxmax()
    ]

    print("\nBest result per machine based on ROC-AUC:")
    print(
        best_results[
            ["machine", "nu", "roc_auc", "anomaly_recall", "anomaly_f1"]
        ]
    )


if __name__ == "__main__":
    main()