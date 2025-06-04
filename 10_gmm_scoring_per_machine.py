#!/usr/bin/env python3
import os
import pickle
import numpy as np
from sklearn.mixture import GaussianMixture

# ───── Shared Validation Utility ─────
def validate_inputs(file_path, expected_shape=None):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Critical file missing: {file_path}")
    if expected_shape is not None:
        arr = np.load(file_path, allow_pickle=True)
        if arr.ndim != 2 or arr.shape[1] != expected_shape:
            raise ValueError(f"Shape mismatch in {file_path}: found {arr.shape}, expected (*, {expected_shape})")

# ───── Paths & Constants ─────
INPUT_BASE  = "/kaggle/working/processed/vqvae"
CKPT_BASE   = "/kaggle/working/checkpoints/gmm"
OUTPUT_BASE = "/kaggle/working/processed/gmm"
machines    = ["ToyCar","ToyTrain","bearing","fan","gearbox","slider","valve"]
splits      = ["train","test","supplemental"]

def load_numpy(path):
    return np.load(path, allow_pickle=True)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_numpy(arr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)

def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

if __name__ == "__main__":
    os.makedirs(CKPT_BASE, exist_ok=True)
    for machine in machines:
        print(f"\n=== (10) GMM Scoring for {machine} ===")
        train_zq_p = os.path.join(INPUT_BASE, machine, "train", "z_q.npy")
        train_fn_p = os.path.join(INPUT_BASE, machine, "train", "filenames.pkl")
        if not (os.path.exists(train_zq_p) and os.path.exists(train_fn_p)):
            print(f"[SKIP] Missing train VQ data for {machine}")
            continue

        # Validate shape (N_train, 32)
        validate_inputs(train_zq_p, expected_shape=32)
        z_q_train = load_numpy(train_zq_p)       # (N_train,32)
        fnames_train = load_pickle(train_fn_p)   # list length N_train
        if z_q_train.shape[0] != len(fnames_train):
            print(f"[ERROR] Mismatch in {machine}/train: {len(fnames_train)} filenames vs {z_q_train.shape[0]} samples")
            continue

        # STEP 1: Fit GMM on train z_q
        gmm = GaussianMixture(n_components=5, covariance_type="full", random_state=42)
        gmm.fit(z_q_train)
        ckpt_path = os.path.join(CKPT_BASE, machine, "gmm.pkl")
        save_pickle(gmm, ckpt_path)
        print(f"  ✔ Trained GMM for {machine}: {z_q_train.shape}")

        mean_s, std_s = None, None
        # STEP 2: Score each split
        for split in splits:
            print(f"  → {machine}/{split}")
            zq_p = os.path.join(INPUT_BASE, machine, split, "z_q.npy")
            fn_p = os.path.join(INPUT_BASE, machine, split, "filenames.pkl")
            if not (os.path.exists(zq_p) and os.path.exists(fn_p)):
                print(f"    [SKIP] Missing VQ data for {machine}/{split}")
                continue

            validate_inputs(zq_p, expected_shape=32)
            z_q_split = load_numpy(zq_p)             # (N_split,32)
            fnames_split = load_pickle(fn_p)         # list length N_split
            if z_q_split.shape[0] != len(fnames_split):
                print(f"    [ERROR] Mismatch in {machine}/{split}")
                continue

            raw_scores = -gmm.score_samples(z_q_split)  # (N_split,)
            # Normalize using train stats (once)
            if split == "train":
                mean_s = raw_scores.mean()
                std_s  = raw_scores.std() if raw_scores.std() > 0 else 1.0

            norm_scores = (raw_scores - mean_s) / std_s

            # Save raw + norm + filenames
            out_dir = os.path.join(OUTPUT_BASE, machine, split)
            os.makedirs(out_dir, exist_ok=True)
            save_numpy(raw_scores,    os.path.join(out_dir, "s_GMM.npy"))
            save_numpy(norm_scores,   os.path.join(out_dir, "s_GMM_norm.npy"))
            save_pickle({'scores': raw_scores, 'scores_norm': norm_scores, 'filenames': fnames_split},
                        os.path.join(out_dir, "s_GMM.pkl"))
            print(f"    [SAVED] {machine}/{split}: {raw_scores.shape}")

    print("\n✅ 10: GMM scoring complete.")
