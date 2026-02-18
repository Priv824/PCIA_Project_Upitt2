import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

# --- CONFIGURATION ---
DATA_FILE = '../pred&regre/data_3k_med2vec.pkl'
CODE_MAP_FILE = '../pred&regre/code_map_3k.pkl'
SAVE_FILE = 'med2vec_3k.pt'

# Med2Vec Hyperparameters (Paper Default-ish)
EMB_DIM = 100       # Size of code embeddings (W_c)
HIDDEN_DIM = 100    # Size of visit embeddings (W_v)
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training Exact Med2Vec Architecture on {device}")

# --- 1. DATA PREPARATION ---
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Missing {DATA_FILE}. Run 0_process_splits.py first.")

print("Loading data...")
with open(DATA_FILE, 'rb') as f: raw_visits = pickle.load(f)
with open(CODE_MAP_FILE, 'rb') as f: vocab = pickle.load(f)
VOCAB_SIZE = len(vocab) + 1

# Reconstruct Patient Trajectories from [-1] delimiters
# We need pairs of (Visit_t, Visit_t+1)
patient_seqs = []
current_patient = []
for v in raw_visits:
    if v == [-1]:
        if len(current_patient) > 1:
            patient_seqs.append(current_patient)
        current_patient = []
    else:
        current_patient.append(v)

print(f"Reconstructed {len(patient_seqs)} patient trajectories.")

# Create Training Pairs (Input Visit -> Target: Next Visit + Current Visit)
train_pairs = []
for p in patient_seqs:
    for i in range(len(p) - 1):
        train_pairs.append((p[i], p[i+1])) # (Current, Next)

print(f"Generated {len(train_pairs)} training pairs (t, t+1).")

class Med2VecDataset(Dataset):
    def __init__(self, pairs, vocab_size):
        self.pairs = pairs
        self.vocab_size = vocab_size
        
    def __len__(self): return len(self.pairs)
    
    def __getitem__(self, idx):
        # Med2Vec takes multi-hot input in theory, but for efficiency we use indices
        # and sum them in the model.
        v_curr, v_next = self.pairs[idx]
        
        # We need fixed length for batching indices
        # But for targets (multi-label prediction), we need Multi-Hot vectors
        target_curr = torch.zeros(self.vocab_size)
        target_curr[v_curr] = 1.0
        
        target_next = torch.zeros(self.vocab_size)
        target_next[v_next] = 1.0
        
        # Pad input to max length 15 for embedding lookup
        inp = v_curr[:15] + [0]*(15-len(v_curr))
        
        return torch.tensor(inp), target_curr, target_next

dataset = Med2VecDataset(train_pairs, VOCAB_SIZE)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- 2. THE EXACT MED2VEC ARCHITECTURE ---
class Med2VecOriginal(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim):
        super(Med2VecOriginal, self).__init__()
        
        # 1. Code Embedding Layer (W_c)
        self.W_c = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.b_c = nn.Parameter(torch.zeros(emb_dim))
        
        # 2. Visit Embedding Layer (W_v) - The interaction layer
        self.W_v = nn.Linear(emb_dim, hidden_dim)
        self.b_v = nn.Parameter(torch.zeros(hidden_dim))
        
        # 3. Output Decoder (Predicting codes)
        self.W_out = nn.Linear(hidden_dim, vocab_size)
        
        # Activations
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: [Batch, Max_Codes]
        
        # --- Eq 1: Code Level Representation (u_t) ---
        # "Sum of code vectors + bias -> ReLU"
        # We handle the "x_t" binary multiplication by looking up embeddings and summing
        embedded = self.W_c(x)             # [Batch, Max_Codes, Emb_Dim]
        u_t = torch.sum(embedded, dim=1)   # Sum over codes in visit
        u_t = self.relu(u_t + self.b_c)    # Add bias and ReLU
        
        # --- Eq 2: Visit Level Representation (v_t) ---
        # "Linear transformation of u_t -> ReLU"
        v_t = self.relu(self.W_v(u_t) + self.b_v)
        
        # --- Eq 3: Prediction (Logits) ---
        # Used for both current and next visit prediction
        logits = self.W_out(v_t)
        
        return logits

model = Med2VecOriginal(VOCAB_SIZE, EMB_DIM, HIDDEN_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss() # Multi-label classification loss

# --- 3. TRAIN ---
print("Starting Training...")

for epoch in range(EPOCHS):
    total_loss = 0
    model.train()
    
    for x_idx, y_curr, y_next in loader:
        x_idx = x_idx.to(device)
        y_curr = y_curr.to(device)
        y_next = y_next.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(x_idx)
        
        # LOSS 1: Predict Current Visit (Autoencoder)
        loss_curr = criterion(logits, y_curr)
        
        # LOSS 2: Predict Next Visit (Temporal)
        # In exact Med2Vec, we use the *same* v_t to predict neighbors
        loss_next = criterion(logits, y_next)
        
        # Combine losses (Paper typically treats them equally)
        loss = loss_curr + loss_next
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(loader):.4f}")

# --- 4. SAVE ---
# We save the W_c (Code Embeddings) for Mamba/LSTM initialization
torch.save({'W_emb.weight': model.W_c.weight.data}, SAVE_FILE)
print(f"Training Complete. Saved W_c weights to {SAVE_FILE}")