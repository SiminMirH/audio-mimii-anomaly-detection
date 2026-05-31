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
# MFCC feature extraction
# ------------------------------------------------
def extract_mfcc_features(audio_path, n_mfcc=13):
    """
    Convert one audio file into one fixed-size feature vector.

    Steps:
    1. Load audio waveform
    2. Extract MFCCs
    3. Compute mean and standard deviation over time
    4. Concatenate them into one vector

    Output size:
    n_mfcc=13 -> 13 means + 13 stds = 26 features
    """

    y, sr = librosa.load(audio_path, sr=None)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    feature_vector = np.concatenate([mfcc_mean, mfcc_std])

    return feature_vector


# ------------------------------------------------
# Load folder
# ------------------------------------------------
def load_features_from_folder(folder_path, label):
    """
    Load all .wav files from a folder.

    label:
    0 = normal
    1 = anomaly
    """

    features = []
    labels = []
    files = []

    wav_files = sorted(folder_path.glob("*.wav"))

    for wav_file in wav_files:
        features.append(extract_mfcc_features(wav_file))
        labels.append(label)
        files.append(wav_file)

    return np.array(features), np.array(labels), files


# ------------------------------------------------
# Run One-Class SVM for one machine and one nu
# ------------------------------------------------
def run_one_machine(machine, nu):
    """
    Train and evaluate One-Class SVM for one machine type.
    """

    train_normal_dir = DATASET_DIR / machine / "train" / "normal"
    test_normal_dir = DATASET_DIR / machine / "test" / "normal"
    test_anomaly_dir = DATASET_DIR / machine / "test" / "anomaly"

    print("\n" + "=" * 60)
    print(f"Machine: {machine} | nu = {nu}")
    print("=" * 60)

    # Load training data: only normal files
    X_train, _, _ = load_features_from_folder(
        train_normal_dir,
        label=0
    )

    # Load test normal files
    X_test_normal, y_test_normal, normal_files = load_features_from_folder(
        test_normal_dir,
        label=0
    )

    # Load test anomaly files
    X_test_anomaly, y_test_anomaly, anomaly_files = load_features_from_folder(
        test_anomaly_dir,
        label=1
    )

    X_test = np.vstack([X_test_normal, X_test_anomaly])
    y_test = np.concatenate([y_test_normal, y_test_anomaly])

    # Scale features
    # Fit scaler only on train data to avoid data leakage
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
    # sklearn output:
    # +1 = normal
    # -1 = anomaly
    raw_pred = model.predict(X_test_scaled)

    # Convert:
    # 0 = normal
    # 1 = anomaly
    y_pred = np.where(raw_pred == 1, 0, 1)

    # Anomaly score
    # decision_function: higher means more normal
    # multiply by -1 so higher means more anomalous
    anomaly_scores = -model.decision_function(X_test_scaled)

    # Metrics
    cm = confusion_matrix(y_test, y_pred)

    report = classification_report(
        y_test,
        y_pred,
        target_names=["normal", "anomaly"],
        output_dict=True,
        zero_division=0
    )

    auc = roc_auc_score(y_test, anomaly_scores)

    normal_precision = report["normal"]["precision"]
    normal_recall = report["normal"]["recall"]
    anomaly_precision = report["anomaly"]["precision"]
    anomaly_recall = report["anomaly"]["recall"]
    anomaly_f1 = report["anomaly"]["f1-score"]
    accuracy = report["accuracy"]

    print("Confusion matrix:")
    print(cm)

    print(f"ROC-AUC: {auc:.4f}")
    print(f"Anomaly Recall: {anomaly_recall:.4f}")
    print(f"Anomaly F1: {anomaly_f1:.4f}")

    return {
        "machine": machine,
        "nu": nu,
        "train_normal_files": len(X_train),
        "test_normal_files": len(X_test_normal),
        "test_anomaly_files": len(X_test_anomaly),
        "accuracy": accuracy,
        "roc_auc": auc,
        "normal_precision": normal_precision,
        "normal_recall": normal_recall,
        "anomaly_precision": anomaly_precision,
        "anomaly_recall": anomaly_recall,
        "anomaly_f1": anomaly_f1,
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

    print("\n\nFinal results:")
    print(results_df)

    # Save all results
    output_path = BASE_DIR / "Baseline" / "SVM" / "ocsvm_tuning_results.csv"
    results_df.to_csv(output_path, index=False)

    print(f"\nSaved results to: {output_path}")

    # Best result per machine based on ROC-AUC
    best_results = results_df.loc[
        results_df.groupby("machine")["roc_auc"].idxmax()
    ]

    print("\nBest result per machine based on ROC-AUC:")
    print(best_results[["machine", "nu", "roc_auc", "anomaly_recall", "anomaly_f1"]])


if __name__ == "__main__":
    main()