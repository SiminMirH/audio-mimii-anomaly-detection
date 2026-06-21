from pathlib import Path
import pickle

import librosa
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, RobustScaler


# ------------------------------------------------
# Notes on feature design
# ------------------------------------------------
# This experiment is kept FILE-LEVEL:
# one 10-second audio file -> one feature vector -> one anomaly score.
#
# For a fair comparison with One-Class SVM, we use the SAME feature set:
# MFCC, Delta MFCC, Delta-Delta MFCC, RMS, Spectral Centroid,
# Spectral Bandwidth, Spectral Rolloff, and Zero Crossing Rate.
#
# For each time-varying feature, we only use mean and standard deviation.
# We could add more statistics such as median, min, max, and percentiles,
# but that would increase the feature dimension a lot.
# With limited industrial audio data, this may increase the risk of overfitting.
#
# Therefore, the main comparison uses a compact common feature set
# for both OCSVM and Isolation Forest.
#
# Efficiency improvement:
# Feature extraction is expensive.
# Therefore, this script extracts features ONCE per machine,
# saves them as .pkl files, and then reuses the cached features
# for all Isolation Forest parameter combinations.


# ------------------------------------------------
# Paths
# ------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = BASE_DIR / "Dataset"

OUTPUT_DIR = BASE_DIR / "Baseline" / "IsolationForest"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = OUTPUT_DIR / "feature_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MACHINES = ["fan", "pump", "valve"]

CONTAMINATION_VALUES = [0.01, 0.03, 0.05, 0.1]
MAX_SAMPLES_VALUES = [64, 128, 256, "auto"]

SCALERS = {
    "standard": StandardScaler,
    "robust": RobustScaler
}


# ------------------------------------------------
# Feature extraction
# ------------------------------------------------
def summarize(feature):
    """
    Summarize a time-varying feature into one fixed-size vector.

    We use only mean and standard deviation to keep the feature set
    consistent with the OCSVM experiment.
    """
    mean = np.mean(feature, axis=1)
    std = np.std(feature, axis=1)

    return np.concatenate([mean, std])


def extract_features(audio_path, n_mfcc=20):
    """
    File-level handcrafted feature vector.

    Features:
    - MFCC mean/std
    - Delta MFCC mean/std
    - Delta-Delta MFCC mean/std
    - RMS mean/std
    - Spectral Centroid mean/std
    - Spectral Bandwidth mean/std
    - Spectral Rolloff mean/std
    - Zero Crossing Rate mean/std

    Total dimension:
    - MFCC: 20 * 2 = 40
    - Delta MFCC: 20 * 2 = 40
    - Delta-Delta MFCC: 20 * 2 = 40
    - RMS: 2
    - Spectral Centroid: 2
    - Spectral Bandwidth: 2
    - Spectral Rolloff: 2
    - ZCR: 2

    Total = 130 features per audio file.
    """

    y, sr = librosa.load(audio_path, sr=None)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc
    )

    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    rms = librosa.feature.rms(y=y)

    spectral_centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr
    )

    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr
    )

    zcr = librosa.feature.zero_crossing_rate(y=y)

    feature_vector = np.concatenate([
        summarize(mfcc),
        summarize(delta_mfcc),
        summarize(delta2_mfcc),
        summarize(rms),
        summarize(spectral_centroid),
        summarize(spectral_bandwidth),
        summarize(spectral_rolloff),
        summarize(zcr),
    ])

    return feature_vector.astype(np.float32)


# ------------------------------------------------
# Load features from folder
# ------------------------------------------------
def load_features_from_folder(folder_path, label):
    features = []
    labels = []

    wav_files = sorted(folder_path.glob("*.wav"))

    for i, wav_file in enumerate(wav_files, start=1):
        if i % 500 == 0:
            print(f"Extracted {i}/{len(wav_files)} files from {folder_path}")

        features.append(extract_features(wav_file))
        labels.append(label)

    return np.array(features), np.array(labels)


# ------------------------------------------------
# Cache features per machine
# ------------------------------------------------
def get_cache_path(machine):
    return CACHE_DIR / f"{machine}_common_features.pkl"


def extract_or_load_features(machine, force_reextract=False):
    cache_path = get_cache_path(machine)

    if cache_path.exists() and not force_reextract:
        print(f"\nLoading cached features for {machine}: {cache_path}")

        with open(cache_path, "rb") as f:
            data = pickle.load(f)

        print("Train normal:", data["X_train"].shape)
        print("Test normal:", data["X_test_normal"].shape)
        print("Test anomaly:", data["X_test_anomaly"].shape)

        return data

    print("\n" + "=" * 80)
    print(f"Extracting features for machine: {machine}")
    print("=" * 80)

    train_normal_dir = DATASET_DIR / machine / "train" / "normal"
    test_normal_dir = DATASET_DIR / machine / "test" / "normal"
    test_anomaly_dir = DATASET_DIR / machine / "test" / "anomaly"

    X_train, y_train = load_features_from_folder(
        train_normal_dir,
        label=0
    )

    X_test_normal, y_test_normal = load_features_from_folder(
        test_normal_dir,
        label=0
    )

    X_test_anomaly, y_test_anomaly = load_features_from_folder(
        test_anomaly_dir,
        label=1
    )

    X_test = np.vstack([X_test_normal, X_test_anomaly])
    y_test = np.concatenate([y_test_normal, y_test_anomaly])

    data = {
        "machine": machine,
        "X_train": X_train,
        "y_train": y_train,
        "X_test_normal": X_test_normal,
        "y_test_normal": y_test_normal,
        "X_test_anomaly": X_test_anomaly,
        "y_test_anomaly": y_test_anomaly,
        "X_test": X_test,
        "y_test": y_test,
    }

    with open(cache_path, "wb") as f:
        pickle.dump(data, f)

    print(f"\nSaved cached features to: {cache_path}")
    print("Train normal:", X_train.shape)
    print("Test normal:", X_test_normal.shape)
    print("Test anomaly:", X_test_anomaly.shape)

    return data


# ------------------------------------------------
# Run one experiment from cached features
# ------------------------------------------------
def run_one_experiment(data, contamination, max_samples, scaler_name):
    machine = data["machine"]

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    X_test_normal = data["X_test_normal"]
    X_test_anomaly = data["X_test_anomaly"]

    print("\n" + "=" * 80)
    print(
        f"Machine: {machine} | "
        f"contamination={contamination} | "
        f"max_samples={max_samples} | "
        f"scaler={scaler_name}"
    )
    print("=" * 80)

    print("Train normal:", X_train.shape)
    print("Test normal:", X_test_normal.shape)
    print("Test anomaly:", X_test_anomaly.shape)

    scaler = SCALERS[scaler_name]()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = IsolationForest(
        n_estimators=500,
        contamination=contamination,
        max_samples=max_samples,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_scaled)

    raw_pred = model.predict(X_test_scaled)

    y_pred = np.where(raw_pred == 1, 0, 1)

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
        "max_samples": max_samples,
        "scaler": scaler_name,
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

    # Set to True only if you changed the feature extraction code
    # and want to rebuild the cached .pkl files.
    FORCE_REEXTRACT = False

    for machine in MACHINES:
        data = extract_or_load_features(
            machine=machine,
            force_reextract=FORCE_REEXTRACT
        )

        for contamination in CONTAMINATION_VALUES:
            for max_samples in MAX_SAMPLES_VALUES:
                for scaler_name in SCALERS.keys():
                    result = run_one_experiment(
                        data=data,
                        contamination=contamination,
                        max_samples=max_samples,
                        scaler_name=scaler_name
                    )
                    all_results.append(result)

    results_df = pd.DataFrame(all_results)

    print("\nFinal results:")
    print(results_df)

    output_path = OUTPUT_DIR / "isolation_forest_common_features_cached_results.csv"
    results_df.to_csv(output_path, index=False)

    print(f"\nSaved results to: {output_path}")

    best_results = results_df.loc[
        results_df.groupby("machine")["roc_auc"].idxmax()
    ]

    print("\nBest result per machine based on ROC-AUC:")
    print(
        best_results[
            [
                "machine",
                "contamination",
                "max_samples",
                "scaler",
                "roc_auc",
                "anomaly_recall",
                "anomaly_f1"
            ]
        ]
    )


if __name__ == "__main__":
    main()