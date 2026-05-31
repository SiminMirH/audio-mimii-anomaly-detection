from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------
# Paths
# ------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = BASE_DIR / "Dataset"

MACHINES = ["fan", "pump", "valve"]

# Similar idea to nu:
# expected fraction of anomalies/outliers
CONTAMINATION_VALUES = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3]


# ------------------------------------------------
# Feature extraction
# ------------------------------------------------
def extract_features(audio_path, n_mfcc=20):
    """
    Convert one audio file into one fixed-size feature vector.

    Features:
    - MFCC mean/std
    - Delta MFCC mean/std
    - Delta-Delta MFCC mean/std
    - RMS mean/std
    - Spectral Centroid mean/std
    - Zero Crossing Rate mean/std

    Output:
    126-dimensional feature vector
    """

    y, sr = librosa.load(audio_path, sr=None)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    rms = librosa.feature.rms(y=y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=y)

    def summarize(feature):
        mean = np.mean(feature, axis=1)
        std = np.std(feature, axis=1)
        return np.concatenate([mean, std])

    return np.concatenate([
        summarize(mfcc),
        summarize(delta_mfcc),
        summarize(delta2_mfcc),
        summarize(rms),
        summarize(spectral_centroid),
        summarize(zcr),
    ])


# ------------------------------------------------
# Load data
# ------------------------------------------------
def load_features_from_folder(folder_path, label):
    features = []
    labels = []

    wav_files = sorted(folder_path.glob("*.wav"))

    for wav_file in wav_files:
        features.append(extract_features(wav_file))
        labels.append(label)

    return np.array(features), np.array(labels)


# ------------------------------------------------
# Run one experiment
# ------------------------------------------------
def run_one_machine(machine, contamination):
    train_normal_dir = DATASET_DIR / machine / "train" / "normal"
    test_normal_dir = DATASET_DIR / machine / "test" / "normal"
    test_anomaly_dir = DATASET_DIR / machine / "test" / "anomaly"

    print("\n" + "=" * 60)
    print(f"Machine: {machine} | contamination = {contamination}")
    print("=" * 60)

    # Train only on normal sounds
    X_train, _ = load_features_from_folder(train_normal_dir, label=0)

    # Test on normal + anomaly
    X_test_normal, y_test_normal = load_features_from_folder(test_normal_dir, label=0)
    X_test_anomaly, y_test_anomaly = load_features_from_folder(test_anomaly_dir, label=1)

    X_test = np.vstack([X_test_normal, X_test_anomaly])
    y_test = np.concatenate([y_test_normal, y_test_anomaly])

    print("Train normal:", X_train.shape)
    print("Test normal:", X_test_normal.shape)
    print("Test anomaly:", X_test_anomaly.shape)

    # Scaling is not always required for tree models,
    # but we keep it for consistency with SVM comparison.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Isolation Forest learns what normal data looks like
    # and isolates unusual samples faster.
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_scaled)

    # sklearn output:
    # +1 = normal
    # -1 = anomaly
    raw_pred = model.predict(X_test_scaled)

    # Convert to:
    # 0 = normal
    # 1 = anomaly
    y_pred = np.where(raw_pred == 1, 0, 1)

    # decision_function:
    # higher = more normal
    # We multiply by -1:
    # higher = more anomalous
    anomaly_scores = -model.decision_function(X_test_scaled)

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
        "contamination": contamination,
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
        for contamination in CONTAMINATION_VALUES:
            result = run_one_machine(machine, contamination)
            all_results.append(result)

    results_df = pd.DataFrame(all_results)

    print("\nFinal results:")
    print(results_df)

    output_path = (
        BASE_DIR
        / "Baseline"
        / "IsolationForest"
        / "isolation_forest_results.csv"
    )

    results_df.to_csv(output_path, index=False)

    print(f"\nSaved results to: {output_path}")

    best_results = results_df.loc[
        results_df.groupby("machine")["roc_auc"].idxmax()
    ]

    print("\nBest result per machine based on ROC-AUC:")
    print(
        best_results[
            ["machine", "contamination", "roc_auc", "anomaly_recall", "anomaly_f1"]
        ]
    )


if __name__ == "__main__":
    main()