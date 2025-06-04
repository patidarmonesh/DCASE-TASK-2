#!/usr/bin/env python3
import os
import librosa
import soundfile as sf

# ───── Shared Validation Utility ─────
def validate_inputs(file_path, expected_shape=None):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Critical file missing: {file_path}")
    if expected_shape is not None:
        data = sf.read(file_path)[0]
        # We only know waveform length, not shape; skip shape check for raw .wav
        # But do verify sampling rate:
        _, sr = sf.read(file_path, always_2d=True)
        if sr != expected_shape:
            raise ValueError(f"Sample rate mismatch in {file_path}: found {sr}, expected {expected_shape}")

# ───── Script Begins ─────
def segment_and_resample(
    raw_root: str,
    processed_root: str,
    sr: int = 16000,
    segment_duration: float = 1.0
):
    """
    For each machine / split under raw_root:
      - Load each .wav at original SR, verify SR if desired.
      - Resample to `sr` (16 kHz).
      - Split into non-overlapping `segment_duration`‐second segments.
      - Save under: processed_root/<machine>/<split>/raw_segments/
    NOTE: any “tail” shorter than 1 second is dropped.
    """
    seg_len = int(sr * segment_duration)

    if not os.path.isdir(raw_root):
        raise FileNotFoundError(f"Raw root not found: {raw_root}")

    machines = [m for m in os.listdir(raw_root)
                if os.path.isdir(os.path.join(raw_root, m))]
    for machine in machines:
        machine_raw = os.path.join(raw_root, machine)
        for split in os.listdir(machine_raw):
            split_in = os.path.join(machine_raw, split)
            if not os.path.isdir(split_in):
                continue

            split_out = os.path.join(processed_root, machine, split, "raw_segments")
            os.makedirs(split_out, exist_ok=True)

            for fname in sorted(os.listdir(split_in)):
                if not fname.lower().endswith(".wav"):
                    continue
                path = os.path.join(split_in, fname)
                # Validate sample rate
                try:
                    validate_inputs(path, expected_shape=sr)  # warns if SR != 16000
                except ValueError:
                    # If found different SR, we still proceed (we will resample).
                    pass

                audio, orig_sr = librosa.load(path, sr=None)
                # Resample if needed
                if orig_sr != sr:
                    audio = librosa.resample(audio, orig_sr, sr)

                total = len(audio)
                n_segs = total // seg_len  # drop leftover < 1s
                for idx in range(n_segs):
                    start = idx * seg_len
                    end = start + seg_len
                    segment = audio[start:end]
                    out_fname = f"{os.path.splitext(fname)[0]}_seg{idx:02d}.wav"
                    out_path = os.path.join(split_out, out_fname)
                    with sf.SoundFile(out_path, "w", samplerate=sr, channels=1, format="WAV") as f:
                        f.write(segment)

if __name__ == "__main__":
    RAW_ROOT  = "/kaggle/working/data/dcase2025t2/dev_data/raw"
    PROC_ROOT = "/kaggle/working/data/dcase2025t2/dev_data/processed"
    os.makedirs(PROC_ROOT, exist_ok=True)
    segment_and_resample(RAW_ROOT, PROC_ROOT, sr=16000, segment_duration=1.0)
    print("✅ 01: Segmentation & resampling complete.")
