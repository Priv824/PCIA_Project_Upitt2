import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import time
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error

# --- CONFIG ---
TRAIN_FILE = 'data_3k_mamba_train.pkl'
TEST_FILE  = 'data_1k_mamba_test.pkl'
CODE_MAP   = 'code_map_3k.pkl'
MED2VEC    = '../med2vec/med2vec_3k.pt'

EMB_DIM = 100
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- LSTM Training on {device} ---")

# --- DATA ---
with open(CODE_MAP, 'rb') as f: vocab_size = len(pickle.load(f)) + 1
with open(TRAIN_FILE, 'rb') as f: train_data = pickle.load(f)
with open(TEST_FILE, 'rb') as f: test_data = pickle.load(f)

class AmdDataset(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def collate(batch):
    seqs, deltas, y_c, y_r, lens = [], [], [], [], []
    for item in batch:
        visit_tensors = [torch.tensor(v[:10] + [0]*(10-len(v))) for v in item['seq']]
        seqs.append(torch.stack(visit_tensors))
        deltas.append(torch.tensor(item['deltas']))
        y_c.append(item['y_class'])
        y_r.append(item['y_reg'])
        lens.append(len(item['seq']))
    return pad_sequence(seqs, batch_first=True), pad_sequence(deltas, batch_first=True), torch.tensor(y_c), torch.tensor(y_r), torch.tensor(lens)

train_loader = DataLoader(AmdDataset(train_data), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
test_loader = DataLoader(AmdDataset(test_data), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

# --- MODEL ---
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMB_DIM, padding_idx=0)
        self.time_proj = nn.Linear(1, EMB_DIM)
        self.lstm = nn.LSTM(EMB_DIM, 128, batch_first=True)
        self.head = nn.Linear(128, 1)
        self.reg = nn.Linear(128, 1)

    def forward(self, x, t, lens):
        emb = torch.sum(self.embedding(x), dim=2) 
        t_emb = self.time_proj(torch.log1p(t.unsqueeze(-1).float()))
        packed = pack_padded_sequence(emb + t_emb, lens.cpu(), batch_first=True, enforce_sorted=False)
        _, (ht, _) = self.lstm(packed)
        out = ht[-1]
        return self.head(out), self.reg(out)

model = LSTMModel().to(device)

try:
    state = torch.load(MED2VEC)
    model.embedding.weight.data.copy_(state['W_emb.weight'])
    print("Loaded Med2Vec weights.")
except:
    print("WARNING: Could not load Med2Vec weights.")

# --- TRAIN ---
opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)
crit_c, crit_r = nn.BCEWithLogitsLoss(), nn.MSELoss()

print("Starting Training...")
start = time.time()
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x, t, yc, yr, lens in train_loader:
        x, t, yc, yr = x.to(device), t.to(device), yc.to(device).float(), yr.to(device).float()
        opt.zero_grad()
        l, r = model(x, t, lens)
        loss = crit_c(l.view(-1), yc)
        if (yc==1).sum() > 0:
            loss += 0.5 * crit_r(r[yc==1].view(-1), torch.log1p(yr[yc==1]))
        loss.backward()
        opt.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

# --- EVAL ---
model.eval()
preds, labels, reg_p, reg_l = [], [], [], []
with torch.no_grad():
    for x, t, yc, yr, lens in test_loader:
        x, t = x.to(device), t.to(device)
        l, r = model(x, t, lens)
        preds.extend(torch.sigmoid(l).cpu().numpy())
        labels.extend(yc.numpy())
        mask = yc.numpy() == 1
        if mask.any():
            reg_p.extend(torch.expm1(r.cpu()).numpy()[mask])
            reg_l.extend(yr.numpy()[mask])

print(f"\n=== LSTM RESULTS ===")
print(f"AUC: {roc_auc_score(labels, preds):.4f}")
print(f"AP : {average_precision_score(labels, preds):.4f}")
if reg_l: print(f"MAE: {mean_absolute_error(reg_l, reg_p):.2f} days")