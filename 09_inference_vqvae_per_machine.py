#!/usr/bin/env python3
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ───── Shared Validation Utility ─────
def validate_inputs(file_path, expected_shape=None):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Critical file missing: {file_path}")
    if expected_shape is not None:
        data = pickle.load(open(file_path, "rb"))
        feats = data.get("features", None)
        if feats is None or feats.ndim != 2 or feats.shape[1] != expected_shape:
            raise ValueError(
                f"Shape mismatch in {file_path}: found {feats.shape if feats is not None else None}, expected (*, {expected_shape})"
            )

# ───── VQ‐VAE Model Definition (Matches Training: input_dim=1024) ─────
class VectorQuantizer(nn.Module):
    def __init__(self, num_codes=128, code_dim=32, beta=1.0):
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(num_codes, code_dim))
        self.beta = beta

    def forward(self, z):
        dist = torch.cdist(z, self.codebook)
        indices = torch.argmin(dist, dim=1)
        z_q = self.codebook[indices]
        codebook_loss = F.mse_loss(z_q.detach(), z)
        commitment_loss = F.mse_loss(z_q, z.detach())
        vq_loss = codebook_loss + self.beta * commitment_loss
        z_q_st = z + (z_q - z).detach()
        return z_q_st, vq_loss, indices, z_q

class VQVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256),  nn.ReLU(),
            nn.Linear(256, 64),   nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.quantizer = VectorQuantizer(num_codes=128, code_dim=32, beta=1.0)
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),   nn.ReLU(),
            nn.Linear(64, 256),  nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 1024)
        )

    def forward(self, x):
        z = self.encoder(x)
        z_q_st, vq_loss, indices, z_q_vec = self.quantizer(z)
        z_hat = self.decoder(z_q_st)
        return z_hat, z_q_vec, indices

# ───── Paths & Constants ─────
PCA_BASE   = "/kaggle/working/data/dcase2025t2/dev_data/pca_128"
CKPT_DIR   = "/kaggle/working/checkpoints/vqvae"
VQVAE_BASE = "/kaggle/working/processed/vqvae"
BATCH_SIZE = 10000

machines = ["ToyCar", "ToyTrain", "bearing", "fan", "gearbox", "slider", "valve"]
splits   = ["train", "test", "supplemental"]

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_numpy(arr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for machine in machines:
        print(f"\n=== (09) Inference VQ‐VAE for {machine} ===")
        ckpt_path = os.path.join(CKPT_DIR, machine, "vqvae_model.pth")
        if not os.path.isfile(ckpt_path):
            print(f"[SKIP] No VQ‐VAE checkpoint for {machine}")
            continue

        model = VQVAE().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        for split in splits:
            print(f"  → {machine}/{split}")
            pca_path = os.path.join(PCA_BASE, machine, split, "z_pca.pickle")
            validate_inputs(pca_path, expected_shape=1024)
            data = load_pickle(pca_path)
            Zpca = data["features"]  # (N, 1024)
            fnames = data["filenames"]
            N = Zpca.shape[0]

            z_hat_list = []
            z_q_list   = []
            idx_list   = []

            for start in range(0, N, BATCH_SIZE):
                end = min(start + BATCH_SIZE, N)
                batch = torch.tensor(Zpca[start:end], dtype=torch.float32).to(device)
                with torch.no_grad():
                    z_hat_b, z_q_b, indices = model(batch)
                z_hat_list.append(z_hat_b.cpu().numpy())   # (batch_size, 1024)
                z_q_list.append(z_q_b.cpu().numpy())       # (batch_size, 32)
                idx_list.append(indices.cpu().numpy())      # (batch_size,)

            Z_hat = np.concatenate(z_hat_list, axis=0)  # (N, 1024)
            Z_q   = np.concatenate(z_q_list,   axis=0)  # (N, 32)
            Inds  = np.concatenate(idx_list,   axis=0)  # (N,)

            out_dir = os.path.join(VQVAE_BASE, machine, split)
            os.makedirs(out_dir, exist_ok=True)
            save_numpy(Z_hat,      os.path.join(out_dir, "z_hat.npy"))
            save_numpy(Z_q,        os.path.join(out_dir, "z_q.npy"))
            save_numpy(Inds,       os.path.join(out_dir, "vq_codes.npy"))
            with open(os.path.join(out_dir, "filenames.pkl"), "wb") as f:
                pickle.dump(fnames, f)

            print(f"    [SAVED] {machine}/{split}: z_hat, z_q, vq_codes, filenames")
    print("\n✅ 09: All VQ‐VAE inference done.")
