#!/usr/bin/env python3
import os
import numpy as np
import pickle
from collections import defaultdict
from tqdm import tqdm

# ───── Shared Validation Utility ─────
def validate_inputs(file_path, expected_shape=None):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Critical file missing: {file_path}")
    if expected_shape is not None:
        arr = np.load(file_path, allow_pickle=True)
        if arr.ndim != 1:
            raise ValueError(f"Shape mismatch in {file_path}: found {arr.shape}, expected 1D")

# ───── Paths & Constants ─────
INPUT_BASE  = "/kaggle/working/processed/final_scores"
OUTPUT_BASE = "/kaggle/working/processed/file_scores"
machines    = ["ToyCar","ToyTrain","bearing","fan","gearbox","slider","valve"]
splits      = ["train","test","supplemental"]

def load_numpy(path):
    return np.load(path, allow_pickle=True)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_numpy(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, obj)

def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pickle.dump(obj, open(path, "wb"))

if __name__ == "__main__":
    for machine in machines:
        print(f"\n=== (13) File‐Level Pooling for {machine} ===")
        for split in splits:
            print(f"  → {machine}/{split}")
            s_p = os.path.join(INPUT_BASE, machine, split, "s_tilde.npy")
            fn_p= os.path.join(INPUT_BASE, machine, split, "filenames.pkl")
            if not (os.path.exists(s_p) and os.path.exists(fn_p)):
                print(f"    [SKIP] Missing {machine}/{split}")
                continue

            validate_inputs(s_p)
            s_tilde = load_numpy(s_p)          # (N_seg,)
            seg_fns  = load_pickle(fn_p)       # length N_seg
            if s_tilde.shape[0] != len(seg_fns):
                print(f"    [ERROR] Length mismatch {s_tilde.shape[0]} vs {len(seg_fns)}")
                continue

            file_scores = defaultdict(list)
            for seg_name, score in zip(seg_fns, s_tilde):
                if "_seg" in seg_name:
                    orig = seg_name.split("_seg")[0] + ".wav"
                elif "_segment" in seg_name:
                    orig = seg_name.split("_segment")[0] + ".wav"
                else:
                    orig = seg_name.rsplit("_", 1)[0] + ".wav"
                file_scores[orig].append(float(score))

            pooled = {fname: float(np.max(scores)) for fname, scores in file_scores.items()}
            all_files = sorted(pooled.keys())

            out_dir = os.path.join(OUTPUT_BASE, machine, split)
            os.makedirs(out_dir, exist_ok=True)
            save_numpy(pooled, os.path.join(out_dir, "file_scores.npy"))
            save_pickle(all_files, os.path.join(out_dir, "original_filenames.pkl"))
            print(f"    [SAVED] {machine}/{split} → {len(all_files)} files")

    print("\n✅ 13: File‐level pooling complete.")
