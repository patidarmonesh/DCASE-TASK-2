#!/usr/bin/env python3
import os
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score, roc_curve, auc

# ───── Shared Validation Utility ─────
def validate_inputs(file_path, expected_type=dict):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Critical file missing: {file_path}")
    data = np.load(file_path, allow_pickle=True)
    if not isinstance(data.item(), expected_type):
        raise ValueError(f"Expected a dict in {file_path}, got {type(data.item())}")

# ───── Paths & Constants ─────
INPUT_BASE = "/kaggle/working/processed/file_scores"
machines   = ["ToyCar","ToyTrain","bearing","fan","gearbox","slider","valve"]

def load_numpy(path):
    return np.load(path, allow_pickle=True).item()

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def get_label_from_filename(fname: str) -> int:
    low = fname.lower()
    if "normal" in low:
        return 0
    elif "anomaly" in low or "anomalous" in low:
        return 1
    else:
        raise ValueError(f"Cannot determine label for {fname}")

def compute_pauc(y_true, y_score, max_fpr=0.1):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.searchsorted(fpr, max_fpr, side="right")
    return auc(fpr[:idx], tpr[:idx]) / max_fpr

if __name__ == "__main__":
    all_scores = []
    all_labels = []

    for machine in machines:
        test_dir = os.path.join(INPUT_BASE, machine, "test")
        scores_p = os.path.join(test_dir, "file_scores.npy")
        fnames_p = os.path.join(test_dir, "original_filenames.pkl")

        if not (os.path.exists(scores_p) and os.path.exists(fnames_p)):
            print(f"[SKIP] Missing {machine}/test")
            continue

        validate_inputs(scores_p, expected_type=dict)
        scores_dict = load_numpy(scores_p)        # dict: {orig.wav: score}
        fnames = load_pickle(fnames_p)            # list of orig.wav

        for fname in fnames:
            try:
                label = get_label_from_filename(fname)
                score = scores_dict.get(fname, None)
                if score is None:
                    print(f"[ERROR] Missing score for {fname}")
                    continue
                all_scores.append(float(score))
                all_labels.append(int(label))
            except ValueError as e:
                print(f"[ERROR] {e}")

    scores = np.array(all_scores)
    labels = np.array(all_labels)

    if len(np.unique(labels)) >= 2:
        auc_val  = roc_auc_score(labels, scores)
        pauc_val = compute_pauc(labels, scores, max_fpr=0.1)
        print("=== Final Results ===")
        print(f"AUC: {auc_val:.4f}")
        print(f"pAUC@10%: {pauc_val:.4f}")
        print(f"Total files: {len(labels)}")
        print(f"Normal files: {np.sum(labels == 0)}")
        print(f"Anomalous files: {np.sum(labels == 1)}")
    else:
        print("Insufficient variety of labels to compute AUC/pAUC.")
