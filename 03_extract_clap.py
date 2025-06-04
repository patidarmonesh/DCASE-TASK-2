# ===================== INSTALLATION & SETUP =====================

# 1. Remove preinstalled torch to avoid conflicts
!pip uninstall -y torch torchvision torchaudio

# 2. Install compatible versions (CUDA 11.8)
!pip install --quiet torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --extra-index-url https://download.pytorch.org/whl/cu118

# 3. Install dependencies
!pip install --quiet timm transformers librosa soundfile h5py torchlibrosa

# 4. Clone & install CLAP
!rm -rf /kaggle/working/CLAP
!git clone https://github.com/LAION-AI/CLAP.git /kaggle/working/CLAP
%cd /kaggle/working/CLAP
!pip install -e .
%cd -
# ===================== INSTALLATION & SETUP =====================


# ===================== IMPORTS & CONFIG =====================
import os
import random
import numpy as np
import soundfile as sf
import pickle
import torch
import torchaudio
from tqdm import tqdm
import laion_clap

# ───── Reproducibility ─────
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ───── Device Setup ─────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===================== MODEL INITIALIZATION =====================
# PyTorch 2.6+ safe globals fix
from numpy.core.multiarray import scalar, _reconstruct
from numpy import dtype
from numpy.dtypes import Float64DType, Float32DType
from torch.serialization import add_safe_globals
add_safe_globals([scalar, dtype, _reconstruct, Float64DType, Float32DType])

model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
model.load_ckpt()

# ===================== CORE FUNCTIONALITY =====================
def validate_inputs(file_path):
    """From Code 2: Essential file validation"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Critical file missing: {file_path}")

def resample_to_48k(wav: np.ndarray, sr: int) -> np.ndarray:
    """Enhanced from Code 1 with type safety"""
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 48000:
        wav_tensor = torch.from_numpy(wav).float().unsqueeze(0)
        wav_tensor = torchaudio.functional.resample(wav_tensor, sr, 48000)
        wav = wav_tensor.squeeze(0).cpu().numpy()
    return wav.astype(np.float32)

def extract_clap_embedding(wav: np.ndarray, sr: int) -> np.ndarray:
    """Combined best of both codes"""
    wav_48k = resample_to_48k(wav, sr)
    tmp_path = "/kaggle/working/_clap_temp.wav"
    with sf.SoundFile(tmp_path, "w", samplerate=48000, channels=1, format="WAV") as f:
        f.write(wav_48k)
    embed = model.get_audio_embedding_from_filelist([tmp_path], use_tensor=False)
    os.remove(tmp_path)
    emb = embed[0]
    # Shape check from Code 2
    if emb.ndim != 1 or emb.shape[0] not in (512, 768):
        raise ValueError(f"CLAP embedding wrong shape: {emb.shape}")
    return emb

def process_machine_split(machine: str, split: str, in_root: str, out_root: str):
    """Enhanced version with traceability from Code 1 + validation from Code 2"""
    seg_dir = os.path.join(in_root, machine, split, "raw_segments")
    if not os.path.isdir(seg_dir):
        print(f"[SKIP] Missing directory: {seg_dir}")
        return

    save_dir = os.path.join(out_root, machine, split)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "clap_embeddings.pickle")

    embeddings = []
    filenames = []
    
    for fname in tqdm(sorted(os.listdir(seg_dir)), desc=f"{machine}/{split}"):
        if not fname.lower().endswith(".wav"):
            continue
            
        wav_path = os.path.join(seg_dir, fname)
        try:
            validate_inputs(wav_path)  # From Code 2
            wav, sr = sf.read(wav_path)
            
            if len(wav) < sr:
                print(f"[SKIP] Too short: {fname}")
                continue
                
            emb = extract_clap_embedding(wav, sr)
            embeddings.append(emb)
            filenames.append(fname)  # Traceability from Code 1
            
        except Exception as e:
            print(f"[ERROR] {machine}/{split}/{fname}: {e}")

    if embeddings:
        arr = np.stack(embeddings)
        with open(save_path, "wb") as f:
            pickle.dump({
                "features": arr,
                "filenames": filenames  # Preserved structure
            }, f)
        print(f"[SAVED] {save_path}: {arr.shape}, {len(filenames)} files")
    else:
        print(f"[EMPTY] No embeddings for {machine}/{split}")

# ===================== EXECUTION =====================
if __name__ == "__main__":
    IN_ROOT = "/kaggle/input/processed-1sec-clip/dcase2025t2/dev_data/processed"
    OUT_ROOT = "/kaggle/working/data/dcase2025t2/dev_data/processed"
    
    machine_types = ["ToyCar", "ToyTrain", "bearing", "fan", "gearbox", "slider", "valve"]
    splits = ["train", "test", "supplemental"]

    for machine in machine_types:
        for split in splits:
            process_machine_split(machine, split, IN_ROOT, OUT_ROOT)

    print("✅ CLAP embedding extraction complete with full traceability!")
