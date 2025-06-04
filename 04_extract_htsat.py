#!/usr/bin/env python3
# =============================================================================
# 04_extract_htsat.py
#
# HTSat embedding extraction for DCASE2025 Task 2.
# Removes strict 768‐dim check and adapts to HTSat model’s output (e.g., 527‐dim).
# =============================================================================

import os
import sys
import random
import pickle
import numpy as np
import soundfile as sf
import librosa
import torch
from tqdm import tqdm

# ─── Reproducibility Seed ───
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ─── Shared Validation Utility ───
def validate_inputs(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Critical file missing: {file_path}")

# ─── Clone & Install Dependencies (if running in notebook/Colab) ───
# (Skip if already done in your environment)
# !pip install torchlibrosa museval
# !pip install -U torch==2.0.1 torchaudio==2.0.2
# !pip install librosa==0.10.0 sox tqdm soundfile
# !apt-get install -y sox ffmpeg
# !git clone https://github.com/RetroCirce/HTS-Audio-Transformer.git /kaggle/working/HTS-Audio-Transformer

# ─── HTSat Setup ───
sys.path.append('/kaggle/working/HTS-Audio-Transformer')  # Adjust if your repo path differs
from model.htsat import HTSAT_Swin_Transformer
import config as htsat_config

class HTSATExtractor:
    def __init__(self, ckpt_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        self.model = HTSAT_Swin_Transformer(
            spec_size=256,
            patch_size=4,
            in_chans=1,
            num_classes=527,      # Matches the provided checkpoint
            window_size=8,
            config=htsat_config,
            depths=[2, 2, 6, 2],
            embed_dim=96,
            patch_stride=(4, 4),
            num_heads=[4, 8, 16, 32]
        )
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        self.model.eval()
        self.model.to(self.device)
        self.sample_rate = 32000

    def _load_audio(self, path: str) -> torch.Tensor:
        wav, sr = sf.read(path)
        if sr != self.sample_rate:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sample_rate)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        return torch.from_numpy(wav).float()

    def extract_embedding(self, audio_path: str) -> np.ndarray:
        waveform = self._load_audio(audio_path).unsqueeze(0).to(self.device)  # (1, L)
        with torch.no_grad():
            output_dict = self.model(waveform, None, True)
            framewise_emb = output_dict['framewise_output'].cpu().numpy()[0]  # (T, C) e.g. (T, 527)
            emb = framewise_emb.mean(axis=0)  # (C,)
        # No longer enforce a fixed dimension—just return whatever length C is
        return emb

def process_machine_split(machine: str, split: str, in_root: str, out_root: str, extractor: HTSATExtractor):
    seg_dir = os.path.join(in_root, machine, split, "raw_segments")
    if not os.path.isdir(seg_dir):
        print(f"[SKIP] Missing folder: {seg_dir}")
        return

    save_dir = os.path.join(out_root, "htsat", machine, split)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "htsat_embeddings.pickle")

    embeddings = []
    filenames = []

    for fname in tqdm(sorted(os.listdir(seg_dir)), desc=f"HTSat {machine}/{split}"):
        if not fname.lower().endswith(".wav"):
            continue
        wav_path = os.path.join(seg_dir, fname)
        validate_inputs(wav_path)
        try:
            emb = extractor.extract_embedding(wav_path)
            # Verify it’s 1D
            if emb.ndim != 1:
                raise ValueError(f"HTSat embedding not 1D: {emb.shape}")
            embeddings.append(emb)
            filenames.append(fname)
        except Exception as e:
            print(f"[ERROR] {machine}/{split}/{fname}: {e}")

    if embeddings:
        feat_array = np.stack(embeddings, axis=0)  # (N, C)
        out_dict = {'features': feat_array, 'filenames': filenames}
        with open(save_path, "wb") as f:
            pickle.dump(out_dict, f)
        print(f"[SAVED] {save_path}: {feat_array.shape} features, {len(filenames)} names")
    else:
        print(f"[EMPTY] No HTSat embeddings for {machine}/{split}")

if __name__ == "__main__":
    # ─── Configuration ───
    CKPT_PATH = "/kaggle/input/hts-audio-transformer/HTSAT_AudioSet_Saved_1.ckpt"
    IN_ROOT   = "/kaggle/input/processed-1sec-clip/dcase2025t2/dev_data/processed"
    OUT_ROOT  = "/kaggle/working/data/dcase2025t2/dev_data/processed"
    machine_types = ["ToyCar", "ToyTrain", "bearing", "fan", "gearbox", "slider", "valve"]
    splits        = ["train", "test", "supplemental"]

    extractor = HTSATExtractor(CKPT_PATH)

    for machine in machine_types:
        for split in splits:
            print(f"\nProcessing {machine}/{split}")
            process_machine_split(machine, split, IN_ROOT, OUT_ROOT, extractor)

    print("\n✅ HTSat embedding extraction complete.")
