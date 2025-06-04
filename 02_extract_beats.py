# ==================== SETUP BEATs EMBEDDING EXTRACTION ====================

# 1. Clone the UNILM repository (only if missing)
import os
import sys
import random
import numpy as np
import soundfile as sf
import pickle
import torch
import torchaudio
from tqdm import tqdm

# ---------- Reproducibility Seed ----------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ---------- Clone repo if needed ----------
if not os.path.isdir("/kaggle/working/unilm"):
    !git clone https://github.com/microsoft/unilm.git /kaggle/working/unilm
else:
    print("⏩ /kaggle/working/unilm already exists – skipping clone.")

# ---------- Add 'beats' to sys.path ----------
sys.path.append("/kaggle/working/unilm/beats")

# ---------- Prepare checkpoints ----------
!mkdir -p /kaggle/working/unilm/beats/checkpoints
!cp /kaggle/input/checkpoint/BEATs_iter3_plus_AS2M.pt \
      /kaggle/working/unilm/beats/checkpoints/BEATs_iter3_plus_AS2M.pt

# ---------- (Optional) Upgrade torch/torchaudio ----------
!pip install --upgrade torch torchaudio

# ==================== BEGIN BEATs EMBEDDING EXTRACTION SCRIPT ====================

# ---------- Validation Utility ----------
def validate_inputs(file_path, expected_shape=None):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Critical file missing: {file_path}")
    if expected_shape is not None:
        data = sf.read(file_path)[0]
        if data.ndim != expected_shape:
            raise ValueError(f"Dimension mismatch in {file_path}: found {data.ndim}, expected {expected_shape}")

# ---------- Import BEATs ----------
from BEATs import BEATs, BEATsConfig  # make sure unilm/beats is in PYTHONPATH

# ---------- Embedding Extraction Helper ----------
def extract_beats_embedding(wav: np.ndarray, sr: int = 16000, device="cpu") -> np.ndarray:
    """
    Input:
      wav: 1D NumPy array of length 16000 (1 second at 16 kHz)
      sr: sampling rate of wav
    Output:
      768-D NumPy array from BEATs model
    """
    # Resample to 16 kHz if needed
    if sr != 16000:
        wav_tensor = torch.from_numpy(wav).unsqueeze(0)  # (1, L)
        wav_tensor = torchaudio.functional.resample(wav_tensor, sr, 16000)
        wav = wav_tensor.squeeze(0).cpu().numpy()

    # Convert to float32 and send to device
    wav_tensor = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0).to(device)  # (1, 16000)

    with torch.no_grad():
        # BEATs.extract_features returns a tuple; first element is hidden states (1, T', 768)
        hidden_states, _ = model.extract_features(wav_tensor, None)
        # Mean-pool over time dimension -> shape (1, 768)
        emb = hidden_states.mean(dim=1).cpu().numpy().squeeze(0)  # (768,)
    # Extra shape check (from Code 2)
    if emb.ndim != 1 or emb.shape[0] != 768:
        raise ValueError(f"BEATs embedding wrong shape: {emb.shape}")
    return emb

# ---------- Batch Processing ----------
def process_machine_split(machine: str, split: str, in_root: str, out_root: str, device="cpu"):
    """
    Reads all 1-second WAVs under in_root/<machine>/<split>/raw_segments/,
    extracts a 768-D BEATs embedding for each, and saves the stack as:
      out_root/<machine>/<split>/beats_embeddings.pickle
    Stores corresponding filenames for traceability.
    """
    seg_dir = os.path.join(in_root, machine, split, "raw_segments")
    if not os.path.isdir(seg_dir):
        print(f"[SKIP] Missing folder: {seg_dir}")
        return

    save_dir = os.path.join(out_root, machine, split)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "beats_embeddings.pickle")

    embeddings = []
    filenames = []

    for fname in tqdm(sorted(os.listdir(seg_dir)), desc=f"{machine}/{split}"):
        if not fname.lower().endswith(".wav"):
            continue
        wav_path = os.path.join(seg_dir, fname)
        try:
            # Validate file exists
            validate_inputs(wav_path)
            wav, sr = sf.read(wav_path)
            # Skip any clip shorter than 1 second (after resample)
            if len(wav) < 16000:
                print(f"[SKIP] Too short: {fname}")
                continue
            emb = extract_beats_embedding(wav, sr, device=device)  # shape: (768,)
            embeddings.append(emb)
            filenames.append(fname)
        except Exception as e:
            print(f"[ERROR] {machine}/{split}/{fname}: {e}")

    if embeddings:
        arr = np.stack(embeddings, axis=0)
        out_data = {
            'features': arr,
            'filenames': filenames
        }
        with open(save_path, "wb") as f:
            pickle.dump(out_data, f)
        print(f"[SAVED] {save_path}: {arr.shape} with {len(filenames)} filenames")
    else:
        print(f"[EMPTY] No embeddings generated for {machine}/{split}")

# ---------- Main Loop ----------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load BEATs model
    checkpoint_path = "/kaggle/working/unilm/beats/checkpoints/BEATs_iter3_plus_AS2M.pt"
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = BEATsConfig(ckpt["cfg"])
    global model
    model = BEATs(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    IN_ROOT = "/kaggle/input/processed-1sec-clip/dcase2025t2/dev_data/processed"
    OUT_ROOT = "/kaggle/working/data/dcase2025t2/dev_data/processed"

    machine_types = ["ToyCar", "ToyTrain", "bearing", "fan", "gearbox", "slider", "valve"]
    splits = ["train", "test", "supplemental"]

    for machine in machine_types:
        for split in splits:
            process_machine_split(machine, split, IN_ROOT, OUT_ROOT, device=device)

    print("✅ BEATs embedding extraction complete.")
