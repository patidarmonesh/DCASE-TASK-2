#!/usr/bin/env python3
import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import random
import torch

# ───── Reproducibility Seed ─────
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
            raise ValueError(
                f"Shape mismatch in {file_path}: found {feats.shape if feats is not None else None}, expected (*, {expected_shape})"
            )

# ───── Constants & Paths ─────
EMB_TYPES = ["beats", "clap", "htsat", "wav2vec2"]

BASE_INPUTS = {
    "beats": "/kaggle/input/beats-embeddings/dcase2025t2/dev_data/processed",
    "clap": "/kaggle/input/claps-embeddings/dcase2025t2/dev_data/processed",
    "htsat": "/kaggle/input/htsat-embeddings/dcase2025t2/dev_data/processed/htsat",
    "wav2vec2": "/kaggle/input/wav2vec2-embeddings/dcase2025t2/dev_data/processed/wav2vec2"
}

BASE_OUTPUT = "/kaggle/working/data/dcase2025t2/dev_data/processed"
CKPT_DIR = "/kaggle/working/checkpoints/emb_scalers"
BATCH_SIZE = 10000  # For memory efficiency

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def process_embeddings(features: np.ndarray, mask=None, scaler=None, fit_scaler=False):
    """
    When fit_scaler=True:
      - Drop constant dims
      - Fit a StandardScaler (in batches if large)
      - Return clipped normalized features, mask, and scaler object

    When fit_scaler=False:
      - Apply existing mask & scaler to features
      - Return clipped normalized features, mask, and scaler
    """
    if fit_scaler:
        std_devs = np.std(features, axis=0)
        non_const = std_devs > 1e-6
        feats_clean = features[:, non_const]
        if feats_clean.shape[1] == 0:
            return None, None, None
        scaler_ = StandardScaler()
        N, D = feats_clean.shape
        if N > BATCH_SIZE:
            scaler_.partial_fit(feats_clean[:BATCH_SIZE, :])
            for start in range(BATCH_SIZE, N, BATCH_SIZE):
                end = min(start + BATCH_SIZE, N)
                scaler_.partial_fit(feats_clean[start:end, :])
            feats_norm = scaler_.transform(feats_clean)
        else:
            feats_norm = scaler_.fit_transform(feats_clean)
        feats_clip = np.clip(feats_norm, -5, 5)
        return feats_clip, non_const, scaler_
    else:
        feats_clean = features[:, mask]
        feats_norm = scaler.transform(feats_clean)
        feats_clip = np.clip(feats_norm, -5, 5)
        return feats_clip, mask, scaler

if __name__ == "__main__":
    machines = ["ToyCar", "ToyTrain", "bearing", "fan", "gearbox", "slider", "valve"]
    splits = ["train", "test", "supplemental"]
    os.makedirs(CKPT_DIR, exist_ok=True)

    for machine in machines:
        print(f"\n=== (06) CLEAN & MERGE for {machine} ===")

        # 1) LOAD TRAIN EMBEDDINGS FOR ALL TYPES
        train_data = {}
        for emb in EMB_TYPES:
            emb_file = f"{emb}_embeddings.pickle"
            p = os.path.join(BASE_INPUTS[emb], machine, "train", emb_file)
            validate_inputs(p)
            train_data[emb] = load_pickle(p)

        # 1a) INTERSECT FILENAMES AMONG ALL 4 TYPES (TRAIN ONLY)
        train_fnames_sets = [set(train_data[emb]["filenames"]) for emb in EMB_TYPES]
        common_train = sorted(set.intersection(*train_fnames_sets))
        if len(common_train) == 0:
            raise ValueError(f"No common train filenames for {machine}")

        # 2) FIT SCALERS AND MASKS ON THESE COMMON TRAIN FEATURES
        scaler_info = {}
        for emb in EMB_TYPES:
            feats = train_data[emb]["features"]       # shape: (N_train, D_emb)
            fnames = train_data[emb]["filenames"]
            idxs = [fnames.index(fn) for fn in common_train]
            feats_common = feats[idxs, :]             # (len(common_train), D_emb)
            cleaned, mask, scaler = process_embeddings(feats_common, fit_scaler=True)
            if cleaned is None:
                raise ValueError(f"All dims dropped for {emb}/{machine}/train")
            scaler_info[emb] = {"mask": mask, "scaler": scaler}
            save_pickle(
                scaler_info[emb],
                os.path.join(CKPT_DIR, machine, f"{emb}_scaler.pkl")
            )
            print(f"  ✔ {emb}/train cleaned: {cleaned.shape} (mask dims={mask.sum()})")

        # 3) CLEAN & MERGE FOR EACH SPLIT (TRAIN, TEST, SUPPLEMENTAL)
        for split in splits:
            print(f"  → Merging {machine}/{split}")
            split_data = {}
            for emb in EMB_TYPES:
                emb_file = f"{emb}_embeddings.pickle"
                p = os.path.join(BASE_INPUTS[emb], machine, split, emb_file)
                if not os.path.isfile(p):
                    split_data = None
                    break
                split_data[emb] = load_pickle(p)
            if split_data is None:
                print(f"    [SKIP] Missing embeddings for {machine}/{split}")
                continue

            # INTERSECT FILENAMES ACROSS EMBEDDING TYPES FOR THIS SPLIT ONLY
            sets_split = [set(split_data[emb]["filenames"]) for emb in EMB_TYPES]
            common_split = set.intersection(*sets_split)
            common_final = sorted(common_split)
            if len(common_final) == 0:
                print(f"    [ERROR] No overlapping filenames for {machine}/{split}")
                continue

            per_type_clean = []
            feature_dims = {}
            for emb in EMB_TYPES:
                feats = split_data[emb]["features"]      # shape: (N_split, D_emb)
                fnames = split_data[emb]["filenames"]
                idxs = [fnames.index(fn) for fn in common_final]
                feats_sub = feats[idxs, :]               # (len(common_final), D_emb)

                info = load_pickle(os.path.join(CKPT_DIR, machine, f"{emb}_scaler.pkl"))
                cleaned, _, _ = process_embeddings(
                    feats_sub,
                    mask=info["mask"],
                    scaler=info["scaler"],
                    fit_scaler=False
                )
                if cleaned is None:
                    print(f"    [ERROR] {emb}/{machine}/{split} cleaned empty.")
                    per_type_clean = None
                    break

                per_type_clean.append(cleaned)
                feature_dims[emb] = cleaned.shape[1]
                print(f"    • {emb}/{split} → {cleaned.shape}")

            if per_type_clean is None:
                continue

            merged = np.concatenate(per_type_clean, axis=1)  # (len(common_final), sum_of_dims)
            out_dict = {
                "features": merged,
                "filenames": common_final,
                "feature_dims": feature_dims
            }
            out_path = os.path.join(BASE_OUTPUT, machine, split, "mpef_embeddings.pickle")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "wb") as f:
                pickle.dump(out_dict, f)
            print(f"    [SAVED] {out_path}: {merged.shape}, dims={feature_dims}")

    print("\n✅ 06: Embedding fusion (clean & merge) complete.")
