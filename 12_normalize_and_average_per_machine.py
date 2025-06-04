#!/usr/bin/env python3
import os
import numpy as np
import pickle

# ───── Shared Validation Utility ─────
def validate_inputs(file_path, expected_shape=None):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Critical file missing: {file_path}")
    if expected_shape is not None:
        arr = np.load(file_path, allow_pickle=True)
        if arr.ndim != 1 or arr.shape[0] != expected_shape:
            raise ValueError(
                f"Shape mismatch in {file_path}: found {arr.shape}, expected ({expected_shape},)"
            )

# ───── Paths & Constants ─────
AE_BASE   = "/kaggle/working/processed/ae"
GMM_BASE  = "/kaggle/working/processed/gmm"
OUT_BASE  = "/kaggle/working/processed/final_scores"

machines = ["ToyCar", "ToyTrain", "bearing", "fan", "gearbox", "slider", "valve"]
splits   = ["train", "test", "supplemental"]

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
    pickle.dump(obj, open(path, "wb"))

if __name__ == "__main__":
    for machine in machines:
        print(f"\n=== (12) Normalize & Average {machine} ===")
        # Look for train-split AE + GMM to compute min/max
        ae_train_p  = os.path.join(AE_BASE,  machine, "train", "s_AE.npy")
        gmm_train_p = os.path.join(GMM_BASE, machine, "train", "s_GMM.npy")
        if not (os.path.exists(ae_train_p) and os.path.exists(gmm_train_p)):
            print(f"[SKIP] Missing train AE/GMM for {machine}")
            continue

        s_AE_train  = load_numpy(ae_train_p)
        s_GMM_train = load_numpy(gmm_train_p)
        if s_AE_train.size == 0 or s_GMM_train.size == 0:
            print(f"[ERROR] Zero-length train for {machine}")
            continue

        min_AE, max_AE   = s_AE_train.min(),  s_AE_train.max()
        min_GMM, max_GMM = s_GMM_train.min(), s_GMM_train.max()

        for split in splits:
            print(f"  → {machine}/{split}")
            ae_p  = os.path.join(AE_BASE,  machine, split, "s_AE.npy")
            gmm_p = os.path.join(GMM_BASE, machine, split, "s_GMM.npy")
            fn_p  = os.path.join(GMM_BASE, machine, split, "s_GMM.pkl")

            if not (os.path.exists(ae_p) and os.path.exists(gmm_p) and os.path.exists(fn_p)):
                print(f"    [SKIP] Missing files in {machine}/{split}")
                continue

            s_AE   = load_numpy(ae_p)                         # shape (N,)
            s_GMM  = load_numpy(gmm_p)                        # shape (N,)
            gmm_dict = load_pickle(fn_p)                      # dict with keys "scores", "scores_norm", "filenames"
            fnames = gmm_dict["filenames"]                    # list of length N

            if not (s_AE.shape[0] == s_GMM.shape[0] == len(fnames)):
                print(f"    [ERROR] Length mismatch {s_AE.shape[0]}, {s_GMM.shape[0]}, {len(fnames)}")
                continue

            denom_AE  = (max_AE - min_AE) if (max_AE - min_AE) > 1e-8 else 1.0
            denom_GMM = (max_GMM - min_GMM) if (max_GMM - min_GMM) > 1e-8 else 1.0

            norm_AE   = np.clip((s_AE - min_AE) / denom_AE,   0, 1)
            norm_GMM  = np.clip((s_GMM - min_GMM) / denom_GMM, 0, 1)
            s_tilde_A = 0.5 * (norm_AE + norm_GMM)

            out_dir = os.path.join(OUT_BASE, machine, split)
            os.makedirs(out_dir, exist_ok=True)
            save_numpy(norm_AE,    os.path.join(out_dir, "norm_AE.npy"))
            save_numpy(norm_GMM,   os.path.join(out_dir, "norm_GMM.npy"))
            save_numpy(s_tilde_A,  os.path.join(out_dir, "s_tilde.npy"))
            save_pickle(fnames,    os.path.join(out_dir, "filenames.pkl"))
            print(f"    [SAVED] {machine}/{split} → norm_AE, norm_GMM, s_tilde")

    print("\n✅ 12: Normalization & averaging complete.")
