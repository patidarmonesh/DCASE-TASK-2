import os
import pickle
import torch
import torch.nn as nn
from tqdm import tqdm

class MLPHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(32 * 128, 256),
            nn.LeakyReLU(0.1),  # Changed from ReLU to prevent dead neurons
            nn.Linear(256, 128)
        )
        # Proper weight initialization for LeakyReLU
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.layers(x.flatten(start_dim=1))

class BranchBMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp_64ms = MLPHead()
        self.mlp_256ms = MLPHead()
        self.mlp_1000ms = MLPHead()

    def forward(self, x64, x256, x1000):
        return (self.mlp_64ms(x64), 
                self.mlp_256ms(x256), 
                self.mlp_1000ms(x1000))

def process_mlp_embeddings(input_root, output_root, batch_size=64, device='cuda'):
    model = BranchBMLP().to(device)
    model.eval()

    for machine in tqdm(os.listdir(input_root), desc="Machines"):
        machine_path = os.path.join(input_root, machine)
        if not os.path.isdir(machine_path):
            continue

        for split in ['train', 'test', 'supplemental']:
            pickle_path = os.path.join(machine_path, split, 'mels_multires.pickle')
            if not os.path.exists(pickle_path):
                continue

            # Load data with enhanced error handling
            try:
                with open(pickle_path, 'rb') as f:
                    mel_data = pickle.load(f)
            except Exception as e:
                print(f"Error loading {pickle_path}: {str(e)}")
                continue

            embeddings = {}
            clip_ids = list(mel_data.keys())
            
            for i in tqdm(range(0, len(clip_ids), batch_size), 
                        desc=f"{machine}/{split}", 
                        leave=False):
                batch_ids = clip_ids[i:i+batch_size]
                batch_data = [mel_data[cid] for cid in batch_ids]

                # Prepare tensors without clamping
                x64 = torch.stack([torch.tensor(d['64ms'], dtype=torch.float32) for d in batch_data]).to(device)
                x256 = torch.stack([torch.tensor(d['256ms'], dtype=torch.float32) for d in batch_data]).to(device)
                x1000 = torch.stack([torch.tensor(d['1000ms'], dtype=torch.float32) for d in batch_data]).to(device)

                # Debug prints
                print(f"\nBatch {i//batch_size} Input Stats:")
                print(f"64ms: μ={x64.mean().item():.3f} ± {x64.std().item():.3f}")
                print(f"256ms: μ={x256.mean().item():.3f} ± {x256.std().item():.3f}")
                print(f"1000ms: μ={x1000.mean().item():.3f} ± {x1000.std().item():.3f}")

                with torch.no_grad():
                    h64, h256, h1000 = model(x64, x256, x1000)

                # Debug prints
                print(f"Batch {i//batch_size} Output Stats:")
                print(f"64ms: μ={h64.mean().item():.3f} ± {h64.std().item():.3f}")
                print(f"256ms: μ={h256.mean().item():.3f} ± {h256.std().item():.3f}")
                print(f"1000ms: μ={h1000.mean().item():.3f} ± {h1000.std().item():.3f}")

                # Store results
                for j, cid in enumerate(batch_ids):
                    embeddings[cid] = {
                        '64ms': h64[j].cpu().numpy(),
                        '256ms': h256[j].cpu().numpy(),
                        '1000ms': h1000[j].cpu().numpy()
                    }

            # Save embeddings
            output_dir = os.path.join(output_root, machine, split)
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, 'mlp_embeddings.pickle'), 'wb') as f:
                pickle.dump(embeddings, f)
            print(f"Saved {len(embeddings)} embeddings to {output_dir}")

if __name__ == "__main__":
    # Configuration
    INPUT_ROOT = "/kaggle/input/mels-multries/dev_data/processed"
    OUTPUT_ROOT = "/kaggle/working/dcase2025t2/dev_data/embeddings_branchb"
    
    process_mlp_embeddings(
        input_root=INPUT_ROOT,
        output_root=OUTPUT_ROOT,
        batch_size=64,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
