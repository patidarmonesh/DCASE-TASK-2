#!/usr/bin/env python3
import os
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import MiniBatchKMeans

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
        data = pickle.load(open(file_path, "rb"))
        feats = data.get("features", None)
        if feats is None or feats.ndim != 2 or feats.shape[1] != expected_shape:
            raise ValueError(
                f"Shape mismatch in {file_path}: found {feats.shape if feats is not None else None}, expected (*, {expected_shape})"
            )

# ───── VQ‐VAE Definition (input_dim=1024 now) ─────
class MetaDataset(Dataset):
    def __init__(self, features, filenames):
        self.features = features  # torch.Tensor (N, 1024)
        self.filenames = filenames

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.filenames[idx]

class VectorQuantizer(nn.Module):
    def __init__(self, num_codes=128, code_dim=32, beta=1.0):
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(num_codes, code_dim))
        self.beta = beta

    def forward(self, z):
        # z: (B, code_dim)
        dist = torch.cdist(z, self.codebook)  # (B, 128)
        indices = torch.argmin(dist, dim=1)   # (B,)
        z_q = self.codebook[indices]          # (B, 32)
        codebook_loss = F.mse_loss(z_q.detach(), z)
        commitment_loss = F.mse_loss(z_q, z.detach())
        vq_loss = codebook_loss + self.beta * commitment_loss
        z_q_st = z + (z_q - z).detach()
        return z_q_st, vq_loss, indices

class VQVAE(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder now takes 1024 → 512 → 256 → 64 → 32
        self.encoder = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256),  nn.ReLU(),
            nn.Linear(256, 64),   nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.quantizer = VectorQuantizer(num_codes=128, code_dim=32, beta=1.0)
        # Decoder maps 32 → 64 → 256 → 512 → 1024
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),   nn.ReLU(),
            nn.Linear(64, 256),  nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 1024)
        )

    def init_codebook(self, data_sample):
        with torch.no_grad():
            z_latent = self.encoder(data_sample)  # (M, 32)
        kmeans = MiniBatchKMeans(n_clusters=128, batch_size=256, n_init=10)
        kmeans.fit(z_latent.cpu().numpy())
        centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        with torch.no_grad():
            self.quantizer.codebook.copy_(centers)

    def forward(self, x):
        z = self.encoder(x)                     # (B, 32)
        z_q, vq_loss, indices = self.quantizer(z)
        x_hat = self.decoder(z_q)               # (B, 1024)
        rec_loss = F.mse_loss(x_hat, x)
        return x_hat, vq_loss, rec_loss, indices

# ───── Paths & Constants ─────
PCA_BASE  = "/kaggle/working/data/dcase2025t2/dev_data/pca_128"
CKPT_DIR  = "/kaggle/working/checkpoints/vqvae"
machines  = ["ToyCar", "ToyTrain", "bearing", "fan", "gearbox", "slider", "valve"]

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

if __name__ == "__main__":
    os.makedirs(CKPT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for machine in machines:
        print(f"\n=== (08a) Train VQ‐VAE for {machine} ===")
        # Now expect 1024-D PCA
        train_pca_path = os.path.join(PCA_BASE, machine, "train", "z_pca.pickle")
        validate_inputs(train_pca_path, expected_shape=1024)

        data = load_pickle(train_pca_path)
        feats = data["features"]   # (N_train, 1024)
        fnames = data["filenames"] # list length N_train

        # Convert to torch.Tensor
        X = torch.tensor(feats, dtype=torch.float32).to(device)
        dataset = MetaDataset(X, fnames)
        loader  = DataLoader(dataset, batch_size=512, shuffle=True)

        model = VQVAE().to(device)
        subset = X[:min(10000, X.shape[0])]
        model.init_codebook(subset)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        filename_to_code = {}

        best_loss = float("inf")
        patience = 10
        wait = 0

        for epoch in range(200):
            model.train()
            total_loss = total_rec = total_vq = 0.0
            code_counts = torch.zeros(
                model.quantizer.codebook.shape[0], device=device
            )

            for feats_batch, fnames_batch in loader:
                feats_batch = feats_batch.to(device)
                optimizer.zero_grad()
                x_hat, vq_loss, rec_loss, indices = model(feats_batch)
                loss = rec_loss + vq_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item() * feats_batch.size(0)
                total_rec  += rec_loss.item() * feats_batch.size(0)
                total_vq   += vq_loss.item() * feats_batch.size(0)
                code_counts += torch.bincount(
                    indices, minlength=code_counts.shape[0]
                )

                for nm, idx in zip(fnames_batch, indices.cpu().tolist()):
                    filename_to_code[nm] = idx

            epoch_loss = total_loss / len(dataset)
            epoch_rec  = total_rec / len(dataset)
            epoch_vq   = total_vq / len(dataset)
            usage = (code_counts > 0).sum().item() / code_counts.shape[0]
            print(
                f" Epoch {epoch+1:03d} | Loss {epoch_loss:.4f} |"
                f" Rec {epoch_rec:.4f} | VQ {epoch_vq:.4f} |"
                f" CodesUsed {usage*100:.1f}%"
            )

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                wait = 0
                tmp_path = os.path.join(CKPT_DIR, machine, "vqvae_best.pth")
                save_model(model, tmp_path)
            else:
                wait += 1
                if wait >= patience:
                    print(f"  ↺ Early stopping at epoch {epoch+1}")
                    break

            dead = set(range(model.quantizer.codebook.shape[0])) - set(
                torch.nonzero(code_counts).cpu().numpy().flatten().tolist()
            )
            if dead:
                print(f"  ↺ Resetting {len(dead)} dead codes")
                with torch.no_grad():
                    z_latent = model.encoder(subset)
                    for i, code_idx in enumerate(dead):
                        model.quantizer.codebook.data[code_idx] = z_latent[
                            i % z_latent.shape[0]
                        ]

        best_ckpt = os.path.join(CKPT_DIR, machine, "vqvae_best.pth")
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        final_ckpt = os.path.join(CKPT_DIR, machine, "vqvae_model.pth")
        save_model(model, final_ckpt)
        torch.save(
            model.quantizer.codebook.detach().cpu().numpy(),
            os.path.join(CKPT_DIR, machine, "codebook.npy")
        )
        with open(os.path.join(CKPT_DIR, machine, "filename_to_code.pkl"), "wb") as f:
            pickle.dump(filename_to_code, f)
        print(f"✅ VQ‐VAE trained & saved for {machine}")

    print("\n✅ 08a: All VQ‐VAE models trained.")
