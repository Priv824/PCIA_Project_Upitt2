import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import os
import time
import gc
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --- 1. CONFIGURATION ---
INPUT_FILE = 'required_inputs/icd10_grouped.seqs'
MODEL_SAVE_PATH = 'model/med2vec_full_best.pt'
os.makedirs('model', exist_ok=True)

# Hyperparameters
NUM_X_CODES = 19882   # Total unique codes
NUM_Y_CODES = 19882   # Output dimension
EMB_SIZE = 200        # Embedding dimension
HIDDEN_SIZE = 200     # Hidden layer dimension
BATCH_SIZE = 64       # Batch size (optimized for stability)
EPOCHS = 20           # Max epochs
PATIENCE = 3          # Early stopping patience
NUM_WORKERS = 0       # 0 = Main process only (Prevents WSL RAM crashes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Configuration ---")
print(f"Device       : {device}")
print(f"Batch Size   : {BATCH_SIZE}")
print(f"Embedding Dim: {EMB_SIZE}")
print(f"Workers      : {NUM_WORKERS} (Safe Mode)")
print(f"---------------------")

# --- 2. DATASET DEFINITION ---
class MedicalVisitDataset(Dataset):
    def __init__(self, patient_data):
        self.samples = []
        # Flatten patient history into (current_visit, next_visit) pairs
        for patient in patient_data:
            if len(patient) < 2:
                continue
            for i in range(len(patient) - 1):
                self.samples.append((patient[i], patient[i+1]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        curr, next_visit = self.samples[idx]
        
        # Create Multi-Hot Vectors on the fly
        x = torch.zeros(NUM_X_CODES)
        x[curr] = 1.0
        
        y = torch.zeros(NUM_Y_CODES)
        y[next_visit] = 1.0
        
        return x, y

# --- 3. DATA LOADING ---
print("\n[1/4] Loading dataset...")
with open(INPUT_FILE, 'rb') as f:
    full_data = pickle.load(f, encoding='latin1')

# Split by Patient ID to prevent data leakage
train_patients, val_patients = train_test_split(full_data, test_size=0.2, random_state=42)

# Memory Cleanup: Delete raw data to free RAM
del full_data
gc.collect()

print(f"      Training Patients  : {len(train_patients)}")
print(f"      Validation Patients: {len(val_patients)}")

train_dataset = MedicalVisitDataset(train_patients)
val_dataset = MedicalVisitDataset(val_patients)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

print(f"      Total Training Samples: {len(train_dataset)}")

# --- 4. MODEL ARCHITECTURE ---
class Med2Vec(nn.Module):
    def __init__(self, num_codes, emb_size, hidden_size):
        super(Med2Vec, self).__init__()
        self.W_emb = nn.Linear(num_codes, emb_size)
        self.W_hidden = nn.Linear(emb_size, hidden_size)
        self.W_out = nn.Linear(hidden_size, num_codes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        emb = self.relu(self.W_emb(x))
        visit_rep = self.relu(self.W_hidden(emb))
        out = self.W_out(visit_rep)
        return out

model = Med2Vec(NUM_X_CODES, EMB_SIZE, HIDDEN_SIZE).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

# --- 5. TRAINING LOOP ---
print("\n[2/4] Starting Training...")
best_val_loss = float('inf')
patience_counter = 0
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0
    
    # Training Phase
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
    for x_batch, y_batch in loop:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        loop.set_postfix(loss=loss.item()) # Live update in bar
        
    avg_train_loss = total_train_loss / len(train_loader)
    
    # Validation Phase
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            val_out = model(x_val)
            v_loss = criterion(val_out, y_val)
            total_val_loss += v_loss.item()
            
    avg_val_loss = total_val_loss / len(val_loader)
    
    # Print Stats (Permanent Record)
    print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}", end="")
    
    # Early Stopping & Saving
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f" | [SAVED BEST]")
    else:
        patience_counter += 1
        print(f" | [No Improv {patience_counter}/{PATIENCE}]")
        
    if patience_counter >= PATIENCE:
        print(f"\n[!] Early stopping triggered at Epoch {epoch+1}.")
        break

total_time = (time.time() - start_time) / 60
print(f"\n[3/4] Training Finished in {total_time:.2f} minutes.")
print(f"[4/4] Best Model Saved to: {MODEL_SAVE_PATH}")