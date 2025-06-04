import os
import pickle
import torch
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm

def compute_multi_res_mel(
    seg_root: str,
    out_root: str,
    sr: int = 16000,
    device: str = 'cuda'
):
    configs = [
        {'name': '64ms',  'n_fft': 1024,  'hop_length': 512},
        {'name': '256ms', 'n_fft': 4096,  'hop_length': 2048},
        {'name': '1000ms','n_fft': 16000, 'hop_length': 8000}
    ]
    transforms = {}
    for cfg in configs:
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=cfg['n_fft'],
            hop_length=cfg['hop_length'],
            n_mels=128,
            power=2.0
        ).to(device)
        db_transform = torchaudio.transforms.AmplitudeToDB(
            stype='power',
            top_db=80.0
        ).to(device)
        transforms[cfg['name']] = (mel_spec, db_transform)

    for machine in tqdm(os.listdir(seg_root), desc="Machines"):
        mdir = os.path.join(seg_root, machine)
        if not os.path.isdir(mdir):
            continue

        for split in ['train', 'test', 'supplemental']:
            seg_dir = os.path.join(mdir, split, 'raw_segments')
            if not os.path.isdir(seg_dir):
                continue

            save_dir = os.path.join(out_root, machine, split)
            os.makedirs(save_dir, exist_ok=True)
            output_data = {}

            for fname in tqdm(os.listdir(seg_dir), desc=f"{machine}/{split}", leave=False):
                if not fname.endswith('.wav'):
                    continue
                wav_path = os.path.join(seg_dir, fname)
                waveform, orig_sr = torchaudio.load(wav_path)
                if orig_sr != sr:
                    waveform = torchaudio.functional.resample(waveform, orig_sr, sr)
                waveform = waveform.to(device)

                clip_mels = {}
                for name, (mel_spec, db_transform) in transforms.items():
                    mels = mel_spec(waveform)            # [1,128,T]
                    log_mels = db_transform(mels)        # [1,128,T]
                    # Resize to [1,128,32]
                    resized = F.interpolate(
                        log_mels.unsqueeze(0),         # [1,1,128,T]
                        size=(128, 32),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)                       # [1,128,32]
                    arr = resized.squeeze(0).transpose(0, 1).cpu().numpy()  # [32,128]
                    clip_mels[name] = arr

                clip_id = os.path.splitext(fname)[0]
                output_data[clip_id] = clip_mels

            if output_data:
                out_path = os.path.join(save_dir, 'mels_multires.pickle')
                with open(out_path, 'wb') as f:
                    pickle.dump(output_data, f)
                print(f"Saved {len(output_data)} clips to {out_path}")


if __name__ == "__main__":
    SEG_ROOT = "/kaggle/input/d/moneshpatidar/dcase2025/data/dcase2025t2/dev_data/processed"
    OUT_ROOT = "/kaggle/working/dcase2025t2/dev_data/processed"
    compute_multi_res_mel(SEG_ROOT, OUT_ROOT)
