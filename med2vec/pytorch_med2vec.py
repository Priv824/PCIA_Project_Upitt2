import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import time

class Med2VecPyTorch(nn.Module):
    def __init__(self, num_codes, emb_size, hidden_size, num_labels, demo_size=0):
        super(Med2VecPyTorch, self).__init__()
        # EXACT MATCH: params['W_emb'] and params['b_emb'] from original
        self.W_emb = nn.Linear(num_codes, emb_size)
        
        # EXACT MATCH: params['W_hidden'] with optional demoSize
        self.W_hidden = nn.Linear(emb_size + demo_size, hidden_size)
        
        # EXACT MATCH: params['W_output'] for the predictive task
        self.W_out = nn.Linear(hidden_size, num_labels if num_labels > 0 else num_codes)
        
        # Original uses T.maximum(..., 0) which is ReLU
        self.activation = nn.ReLU()

    def forward(self, x, d=None):
        # Step 1: Code Embedding
        emb = self.activation(self.W_emb(x))
        
        # Step 2: Concatenate demographics if they exist
        if d is not None:
            emb = torch.cat((emb, d), dim=1)
            
        # Step 3: Visit Representation
        visit_rep = self.activation(self.W_hidden(emb))
        
        # Step 4: Output Logits (Softmax is handled by CrossEntropyLoss)
        logits = self.W_out(visit_rep)
        return logits, visit_rep

def train():
    # Setup matching your previous run
    NUM_X_CODES = 19882
    NUM_Y_CODES = 1518
    EMB_SIZE = 100 # cr_size
    HIDDEN_SIZE = 100 # vr_size
    BATCH_SIZE = 1024 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Med2VecPyTorch(NUM_X_CODES, EMB_SIZE, HIDDEN_SIZE, NUM_Y_CODES).to(DEVICE)
    
    # Original used Adadelta, but Adam is significantly faster on modern GPUs
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Using Binary Cross Entropy with Logits to match the multi-label nature of the data
    criterion = nn.BCEWithLogitsLoss()

    print("Loading Data...")
    with open('mini_icd10.seqs', 'rb') as f:
        seqs = pickle.load(f, encoding='latin1')
    with open('mini_icd10_grouped.seqs', 'rb') as f:
        labels = pickle.load(f, encoding='latin1')

    print("Training on: {}".format(DEVICE))
    for epoch in range(20):
        model.train()
        epoch_loss = 0
        start = time.time()
        
        for i in range(0, len(seqs), BATCH_SIZE):
            batch_x = seqs[i:i+BATCH_SIZE]
            batch_y = labels[i:i+BATCH_SIZE]
            
            # Clean data: Remove the [-1] patient separators for the batch
            valid = [j for j, v in enumerate(batch_x) if v != [-1]]
            if not valid: continue
            
            # Convert to Multi-hot Tensors (matches padMatrix logic)
            x_tensor = torch.zeros(len(valid), NUM_X_CODES).to(DEVICE)
            y_tensor = torch.zeros(len(valid), NUM_Y_CODES).to(DEVICE)
            
            for idx, real_idx in enumerate(valid):
                x_tensor[idx, batch_x[real_idx]] = 1.0
                y_tensor[idx, batch_y[real_idx]] = 1.0
            
            optimizer.zero_grad()
            logits, _ = model(x_tensor)
            loss = criterion(logits, y_tensor)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print("Epoch {} | Loss: {:.4f} | Time: {:.2f}s".format(epoch, epoch_loss, time.time()-start))

    # Save weights in a format we can use for AMD prediction
    torch.save(model.state_dict(), "med2vec_final.pt")

if __name__ == "__main__":
    train()