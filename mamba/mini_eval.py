import torch
import torch.nn as nn
import pickle
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error
from mamba_ssm import Mamba

# --- CONFIGURATION ---
DATA_FILE = 'mamba_mini_dataset.pkl'
CODE_MAP_FILE = '../med2vec/required_inputs/code_map.pkl'
MODEL_PATH = 'mamba_amd_mini_v2.pth'

EMB_DIM = 100
BATCH_SIZE = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- LOAD DATA (SAME AS BEFORE) ---
with open(DATA_FILE, 'rb') as f:
    raw_data = pickle.load(f)
with open(CODE_MAP_FILE, 'rb') as f:
    code_map = pickle.load(f, encoding='latin1')

test_data = raw_data[int(len(raw_data)*0.8):]

class MambaDataset(Dataset):
    def __init__(self, data, mapping):
        self.data = data
        self.mapping = mapping
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        row = self.data[idx]
        numeric_seq = []
        for visit in row['codes']:
            ids = [self.mapping.get(c, 0) for c in visit if c in self.mapping]
            if not ids: ids = [0]
            numeric_seq.append(ids)
        return {'seq': numeric_seq, 'deltas': row['deltas'], 
                'y_class': row['y_class'], 'y_reg': row['y_reg']}

def collate_fn(batch):
    batch_emb_seqs, batch_deltas, batch_y_class, batch_y_reg = [], [], [], []
    MAX_CODES = 10 
    for item in batch:
        visit_tensors = []
        for v in item['seq']:
            v_proc = v[:MAX_CODES] + [0]*(MAX_CODES - len(v))
            visit_tensors.append(v_proc)
        batch_emb_seqs.append(torch.tensor(visit_tensors))
        batch_deltas.append(torch.tensor(item['deltas']))
        batch_y_class.append(item['y_class'])
        batch_y_reg.append(item['y_reg'])
    padded_x = pad_sequence(batch_emb_seqs, batch_first=True, padding_value=0)
    padded_t = pad_sequence(batch_deltas, batch_first=True, padding_value=0)
    return padded_x, padded_t, torch.tensor(batch_y_class), torch.tensor(batch_y_reg)

test_loader = DataLoader(MambaDataset(test_data, code_map), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

class MedMamba(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.time_proj = nn.Linear(1, emb_dim)
        self.mamba = Mamba(d_model=emb_dim, d_state=16, d_conv=4, expand=2)
        self.norm = nn.LayerNorm(emb_dim)
        self.head_class = nn.Linear(emb_dim, 1)
        self.head_reg = nn.Linear(emb_dim, 1)

    def forward(self, x, deltas):
        emb = self.embedding(x) 
        visit_emb = torch.sum(emb, dim=2)
        d = torch.log1p(deltas.unsqueeze(-1).float())
        t_emb = self.time_proj(d)
        out = self.mamba(visit_emb + t_emb)
        out = self.norm(out)
        final = out[:, -1, :] 
        return self.head_class(final), self.head_reg(final)

VOCAB_SIZE = len(code_map) + 1
model = MedMamba(VOCAB_SIZE, EMB_DIM).to(device)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("Model loaded.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

model.eval()

# --- EVALUATION ---
preds_cls, labels_cls = [], []
preds_reg, labels_reg = [], []

with torch.no_grad():
    for x, t, y_c, y_r in test_loader:
        x, t = x.to(device), t.to(device)
        logits, reg_log = model(x, t)
        
        # Classification
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        preds_cls.extend(probs)
        labels_cls.extend(y_c.numpy())
        
        # Regression (Convert Log -> Real Days)
        reg_days = torch.expm1(reg_log).cpu().numpy().flatten() # exp(x) - 1
        y_r_np = y_r.numpy()
        
        for i in range(len(y_c)):
            if y_c[i] == 1:
                preds_reg.append(reg_days[i])
                labels_reg.append(y_r_np[i])

print("\n--- RESULTS V2 (Log-Scaled) ---")
try:
    auc = roc_auc_score(labels_cls, preds_cls)
    ap = average_precision_score(labels_cls, preds_cls)
    print(f"AUC: {auc:.4f} | AP: {ap:.4f}")
except:
    print("Error calculating AUC.")

if len(labels_reg) > 0:
    mae = mean_absolute_error(labels_reg, preds_reg)
    print(f"MAE (Regression): {mae:.2f} days")
    print(f"Example: True {labels_reg[0]:.1f} days vs Pred {preds_reg[0]:.1f} days")
else:
    print("No AMD cases for regression test.")