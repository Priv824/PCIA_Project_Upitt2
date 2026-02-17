import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from mamba_ssm import Mamba

# --- CONFIGURATION ---
DATA_FILE = 'mamba_mini_dataset.pkl'
CODE_MAP_FILE = '../med2vec/required_inputs/code_map.pkl'
MED2VEC_MODEL = '../med2vec/model/med2vec_final.pt'
SAVE_MODEL_NAME = 'mamba_amd_mini_v2.pth'

EMB_DIM = 100     
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device} | Emb Dim: {EMB_DIM}")

# --- 1. LOAD DATA ---
with open(DATA_FILE, 'rb') as f:
    raw_data = pickle.load(f)

with open(CODE_MAP_FILE, 'rb') as f:
    code_map = pickle.load(f, encoding='latin1')
VOCAB_SIZE = len(code_map) + 1 

# --- 2. DATASET ---
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
    batch_emb_seqs = []
    batch_deltas = []
    batch_y_class = []
    batch_y_reg = []
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

train_data, val_data = train_test_split(raw_data, test_size=0.2, random_state=42)
train_loader = DataLoader(MambaDataset(train_data, code_map), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(MambaDataset(val_data, code_map), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# --- 3. MODEL ---
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

model = MedMamba(VOCAB_SIZE, EMB_DIM).to(device)

# --- 4. LOAD WEIGHTS (FIXED) ---
print("Loading Med2Vec weights...")
try:
    state = torch.load(MED2VEC_MODEL, map_location=device)
    # Check both W_emb.weight and direct state dict
    if isinstance(state, dict) and 'W_emb.weight' in state:
        weights = state['W_emb.weight']
    else:
        # Sometimes saved as dict without keys if just the tensor
        weights = state
    
    # FIX: Check for Transpose
    if weights.shape[0] == EMB_DIM and weights.shape[1] != EMB_DIM:
        print(f"  -> Transposing weights from {weights.shape} to {(weights.shape[1], weights.shape[0])}")
        weights = weights.t()
        
    # Load safe portion
    rows = min(model.embedding.weight.shape[0], weights.shape[0])
    model.embedding.weight.data[:rows] = weights[:rows]
    print(f"Success: Loaded {rows} embeddings.")
except Exception as e:
    print(f"Skipping weight load: {e}")

# --- 5. TRAIN ---
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
crit_cls = nn.BCEWithLogitsLoss()
crit_reg = nn.MSELoss()

print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x, t, y_c, y_r in train_loader:
        x, t = x.to(device), t.to(device)
        y_c, y_r = y_c.to(device).float(), y_r.to(device).float()
        
        optimizer.zero_grad()
        out_c, out_r = model(x, t)
        
        loss_c = crit_cls(out_c.squeeze(), y_c)
        
        # FIX: Train on Log-Days to stabilize loss
        mask = (y_c == 1)
        loss_r = 0.0
        if mask.sum() > 0:
            target_log = torch.log1p(y_r[mask]) # log(days + 1)
            loss_r = crit_reg(out_r[mask].squeeze(), target_log)
            
        loss = loss_c + (0.5 * loss_r) # Now we can weight it more heavily
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), SAVE_MODEL_NAME)
print(f"Saved model to {SAVE_MODEL_NAME}")