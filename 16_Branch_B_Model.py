# scripts/branch_b_pipeline.py
import os, pickle, random
import numpy as np
from glob import glob
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score, roc_curve

from src.models.branch_b_model import BranchBModel, GradientReversalLayer
from src.scoring.gmm_scorer import BranchBScorer

# ------------------------------
# Dataset to load precomputed multires mel
# ------------------------------
class MelSegmentDataset(Dataset):
    def __init__(self, root_dir, split, machines=None, labels_csv=None):
        self.samples = []
        # enumerate machines
        for m in os.listdir(root_dir):
            if machines and m not in machines: continue
            pkl = os.path.join(root_dir, m, split, 'mels_multires.pickle')
            if not os.path.exists(pkl): continue
            data = pickle.load(open(pkl, 'rb'))
            for seg_id, feats in data.items():
                # feats: dict of {'64ms': np.array, ...}
                fname = f"{m}/{split}/{seg_id}"
                label = None
                if labels_csv:
                    # parse from CSV if available
                    label = labels_csv.get(f"{m}/{seg_id}.wav", None)
                self.samples.append((feats, m, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feats, machine, label = self.samples[idx]
        # stack and flatten
        x64   = torch.tensor(feats['64ms'], dtype=torch.float).flatten()
        x256  = torch.tensor(feats['256ms'], dtype=torch.float).flatten()
        x1000 = torch.tensor(feats['1000ms'], dtype=torch.float).flatten()
        return x64, x256, x1000, machine, label

# ------------------------------
# Training on normals only
# ------------------------------
def train_branch_b(train_dir, machines, device='cuda'):
    ds = MelSegmentDataset(train_dir, 'train', machines)
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    model = BranchBModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(50):
        model.train()
        for x64,x256,x1000,m,_ in loader:
            x64,x256,x1000 = [t.to(device) for t in (x64,x256,x1000)]
            z, dlogits = model(x64,x256,x1000)
            # domain labels not used here; unsupervised
            loss = dlogits.norm() * 0.0
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    return model

# ------------------------------
# Extract embeddings for GMM
# ------------------------------
def extract_embeddings(model, data_dir, machines, split, device='cuda'):
    ds = MelSegmentDataset(data_dir, split, machines)
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    embeddings, mids, labels = [], [], []
    model.eval()
    with torch.no_grad():
        for x64,x256,x1000,m,lab in loader:
            x64,x256,x1000 = [t.to(device) for t in (x64,x256,x1000)]
            z,_ = model(x64,x256,x1000)
            embeddings.append(z.cpu().numpy())
            mids.extend(m)
            labels.extend(lab)
    return np.vstack(embeddings), np.array(mids), np.array(labels)

# ------------------------------
# Main
# ------------------------------
if __name__ == '__main__':
    DATA_ROOT = '/kaggle/working/dcase2025t2/dev_data/processed'
    machines = os.listdir(DATA_ROOT)

    # 1. Train on development normals
    model = train_branch_b(DATA_ROOT, machines)

    # 2. Extract train embeddings
    emb_train, mids_train, _ = extract_embeddings(model, DATA_ROOT, machines, 'train')

    # 3. Fit GMM
    scorer = BranchBScorer(n_components=8)
    scorer.fit(emb_train, mids_train)

    # 4. Infer on dev-test, compute file-level scores
    emb_test, mids_test, labels_test = extract_embeddings(model, DATA_ROOT, machines, 'test')
    scores = scorer.score(emb_test, mids_test)

    # map segment to file and take max
    file_scores, file_labels = {}, {}
    for (emb, mid, lab, sc) in zip(emb_test, mids_test, labels_test, scores):
        fid = mid  # include segment id parsing if needed
        file_scores.setdefault(fid, []).append(sc)
        file_labels[fid] = lab

    y_true, y_pred = [], []
    for fid, scs in file_scores.items():
        y_true.append(file_labels[fid])
        y_pred.append(max(scs))

    auc = roc_auc_score(y_true, y_pred)
    fpr,tpr,_ = roc_curve(y_true, y_pred)
    p_auc = np.trapz(tpr[fpr<=0.1], fpr[fpr<=0.1]) / 0.1
    print(f"Dev AUC={auc:.4f}, pAUC={p_auc:.4f}")

    # 5. Save model and GMM
    torch.save(model.state_dict(), 'branch_b.pt')
    pickle.dump(scorer, open('branch_b_scorer.pkl','wb'))

    # 6. (Optional) inference on evaluation set analogous to above
