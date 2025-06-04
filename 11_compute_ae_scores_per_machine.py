#!/usr/bin/env python3
import os
import pickle
import numpy as np

# ───── Shared Validation Utility ─────
def validate_inputs(file_path, expected_shape=None):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Critical file missing: {file_path}")
    if expected_shape is not None:
        arr = np.load(file_path, allow_pickle=True)
        if arr.ndim != 2 or arr.shape[1] != expected_shape:
            raise ValueError(
                f"Shape mismatch in {file_path}: found {arr.shape}, expected (*, {expected_shape})"
            )

# ───── Paths & Constants ─────
PCA_BASE    = "/kaggle/working/data/dcase2025t2/dev_data/pca_128"
VQVAE_BASE  = "/kaggle/working/processed/vqvae"
AE_OUTPUT   = "/kaggle/working/processed/ae"
machines    = ["ToyCar", "ToyTrain", "bearing", "fan", "gearbox", "slider", "valve"]
splits      = ["train", "test", "supplemental"]

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
        print(f"\n=== (11) AE Score {machine} ===")
        for split in splits:
            print(f"  → {machine}/{split}")
            z_pca_p = os.path.join(PCA_BASE, machine, split, "z_pca.pickle")
            zhat_p  = os.path.join(VQVAE_BASE, machine, split, "z_hat.npy")
            fn_p    = os.path.join(VQVAE_BASE, machine, split, "filenames.pkl")

            if not (os.path.exists(z_pca_p) and os.path.exists(zhat_p) and os.path.exists(fn_p)):
                print(f"    [SKIP] Missing files for {machine}/{split}")
                continue

            data_pca = load_pickle(z_pca_p)
            Zpca = data_pca["features"]          # (N, 1024)
            fnames = data_pca["filenames"]
            Zhat  = np.load(zhat_p, allow_pickle=True)  # (N, 1024)
            if Zhat.shape != Zpca.shape:
                print(f"    [ERROR] Shape mismatch {Zhat.shape} vs {Zpca.shape}")
                continue

            s_AE = np.sum((Zpca - Zhat) ** 2, axis=1)  # (N,)
            out_dir = os.path.join(AE_OUTPUT, machine, split)
            os.makedirs(out_dir, exist_ok=True)
            save_numpy(s_AE, os.path.join(out_dir, "s_AE.npy"))
            save_pickle({"scores": s_AE, "filenames": fnames}, os.path.join(out_dir, "s_AE.pkl"))
            print(f"    [SAVED] {machine}/{split}/s_AE.npy → {s_AE.shape}")

    print("\n✅ 11: AE scoring complete.")
