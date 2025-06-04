#!/usr/bin/env python3
import os
import random
import pickle
import numpy as np
from sklearn.decomposition import PCA

# ───── Reproducibility Seed ─────
import torch
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ───── Shared Validation Utility ─────
def validate_inputs(file_path, expected_shape=None):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Critical file missing: {file_path}")
    if expected_shape is not None:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        feats = data.get("features", None)
        if feats is None or feats.ndim != 2 or feats.shape[1] != expected_shape:
            raise ValueError(f"Shape mismatch in {file_path}: found {feats.shape if feats is not None else None}, expected (*, {expected_shape})")

# ───── Paths & Constants ─────
BASE_INPUT     = "/kaggle/working/data/dcase2025t2/dev_data/processed"
PCA_PARAMS_DIR = "/kaggle/working/checkpoints/pca_params"
PCA_OUTPUT_DIR = "/kaggle/working/data/dcase2025t2/dev_data/pca_128"  # Keep original name for compatibility

machines = ["ToyCar","ToyTrain","bearing","fan","gearbox","slider","valve"]
splits   = ["train","test","supplemental"]

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

if __name__ == "__main__":
    for machine in machines:
        print(f"\n=== PCA for {machine} ===")
        train_path = os.path.join(BASE_INPUT, machine, "train", "mpef_embeddings.pickle")
        validate_inputs(train_path)  # ensure (N_train, 2575)
        train_data = load_pickle(train_path)
        X_train = train_data['features']  # (N_train,  D_total)
        if X_train.ndim != 2:
            raise ValueError(f"Expected 2D array at {train_path}, got {X_train.shape}")
        D_total = X_train.shape[1]

        # 1) Fit PCA on train only (changed to 1024 components)
        mean_vec = np.mean(X_train, axis=0)     # (D_total,)
        X_centered = X_train - mean_vec
        pca = PCA(n_components=1024, svd_solver="randomized", whiten=False)
        pca.fit(X_centered)                    # (N_train, D_total)
        comps = pca.components_                # (1024, D_total)

        # Save PCA params with variance info
        os.makedirs(os.path.join(PCA_PARAMS_DIR, machine), exist_ok=True)
        save_pickle({
            'mean': mean_vec,
            'components': comps,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'explained_variance': pca.explained_variance_
        }, os.path.join(PCA_PARAMS_DIR, machine, "pca_params.pkl"))
        print(f"  ✔ PCA params saved for {machine} → comps (1024×{D_total})")

        # 2) Project each split
        for split in splits:
            print(f"  → Projecting {machine}/{split}")
            split_path = os.path.join(BASE_INPUT, machine, split, "mpef_embeddings.pickle")
            if not os.path.isfile(split_path):
                print(f"    [SKIP] Missing {split_path}")
                continue
            validate_inputs(split_path, expected_shape=D_total)
            data_split = load_pickle(split_path)
            X_split = data_split['features']   # (N_split, D_total)
            fnames  = data_split['filenames']
            if X_split.shape[1] != D_total:
                raise ValueError(f"Dim mismatch in {machine}/{split}: {X_split.shape[1]} vs {D_total}")

            Xc = X_split - mean_vec                        # center using train mean
            Zpca = np.dot(Xc, comps.T)                     # (N_split, 1024)

            out_dir = os.path.join(PCA_OUTPUT_DIR, machine, split)
            os.makedirs(out_dir, exist_ok=True)
            save_pickle({'features': Zpca, 'filenames': fnames},
                        os.path.join(out_dir, "z_pca.pickle"))
            print(f"    [SAVED] {machine}/{split}/z_pca.pickle → {Zpca.shape}")

    print("\n✅ 07: PCA per machine complete.")
