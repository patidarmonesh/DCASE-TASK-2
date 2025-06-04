# =================== SETUP & INSTALLATION ===================
!pip uninstall -y torch torchaudio transformers numpy
!pip install torch==2.3.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
!pip install transformers==4.41.1 numpy==1.26.4

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"  # Stability fix from new code

# =================== IMPORTS & CONFIG ===================
import random
import numpy as np
import soundfile as sf
import pickle
import torch
import torchaudio
from tqdm.auto import tqdm  # Better progress bar from new code
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

# ───── Reproducibility Seed (from purana) ─────
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ───── Validation Utility (from purana) ─────
def validate_inputs(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Critical file missing: {file_path}")

# =================== CORE FUNCTIONALITY ===================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model with newer version (from new code)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device).eval()

def extract_wav2vec2_embedding(wav: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Combined best of both versions"""
    # Resampling logic from purana with type safety
    if sr != 16000:
        wav_tensor = torch.from_numpy(wav).unsqueeze(0)
        wav = torchaudio.functional.resample(wav_tensor, orig_freq=sr, new_freq=16000).squeeze(0).numpy()
    
    # Processing from new code with validation
    inputs = feature_extractor(wav, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    
    # Shape check from purana
    if emb.ndim != 1 or emb.shape[0] != 768:
        raise ValueError(f"Wav2Vec2 embedding wrong shape: {emb.shape}")
    return emb

def process_machine_split(machine, split, in_root, out_root):
    """Hybrid approach maintaining purana's structure"""
    seg_dir = os.path.join(in_root, machine, split, "raw_segments")
    if not os.path.exists(seg_dir):
        print(f"[SKIP] Missing directory: {seg_dir}")
        return

    # Preserve directory structure from purana
    save_dir = os.path.join(out_root, "wav2vec2", machine, split)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "wav2vec2_embeddings.pickle")

    embeddings = []
    filenames = []

    for fname in tqdm(sorted(os.listdir(seg_dir)), desc=f"{machine}/{split}"):
        if not fname.endswith(".wav"):
            continue
            
        wav_path = os.path.join(seg_dir, fname)
        try:
            # Combined validation from both codes
            validate_inputs(wav_path)  # From purana
            wav, sr = sf.read(wav_path)
            
            if len(wav) < 16000:  # Length check from new code
                print(f"[SKIP] Too short: {fname}")
                continue
                
            embeddings.append(extract_wav2vec2_embedding(wav, sr))
            filenames.append(fname)
            
        except Exception as e:
            print(f"[ERROR] {machine}/{split}/{fname}: {str(e)}")  # Detailed error from new code

    if embeddings:
        with open(save_path, "wb") as f:
            pickle.dump({
                'features': np.stack(embeddings),
                'filenames': filenames
            }, f)
        print(f"[SAVED] {save_path}: {len(embeddings)} embeddings")

# =================== EXECUTION ===================
if __name__ == "__main__":
    IN_ROOT = "/kaggle/input/processed-1sec-clip/dcase2025t2/dev_data/processed"
    OUT_ROOT = "/kaggle/working/data/dcase2025t2/dev_data/processed"
    machines = ["ToyCar", "ToyTrain", "bearing", "fan", "gearbox", "slider", "valve"]
    splits = ["train", "test", "supplemental"]
    
    for machine in machines:
        for split in splits:
            process_machine_split(machine, split, IN_ROOT, OUT_ROOT)
    
    print("✅ Wav2Vec2 embedding extraction complete with traceability!")
