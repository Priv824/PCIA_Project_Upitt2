import pandas as pd
import numpy as np
import pickle
import os
import random
from tqdm import tqdm

# --- CONFIGURATION ---
CSV_PATH = '../Dataset/sorted_ICD_Data.csv'
OUTPUT_FILE = 'mamba_mini_dataset.pkl'
MINI_SAMPLE_SIZE = 2000 

# Keywords to identify AMD in 'DX_NAME'
AMD_KEYWORDS = ['macular degeneration', 'age-related macular', 'h35.3']

print(f"Loading CSV from {CSV_PATH}...")
# Using the specific columns you provided
df = pd.read_csv(CSV_PATH, usecols=['Pseudo_id', 'CONTACT_DATE', 'DX_NAME', 'CURRENT_ICD10_LIST'])

# 1. Random Sample (Mini Set)
all_patients = df['Pseudo_id'].unique()
if len(all_patients) > MINI_SAMPLE_SIZE:
    print(f"Sampling {MINI_SAMPLE_SIZE} random patients from {len(all_patients)} total...")
    selected_patients = np.random.choice(all_patients, MINI_SAMPLE_SIZE, replace=False)
    df = df[df['Pseudo_id'].isin(selected_patients)]
else:
    print(f"Dataset has fewer than {MINI_SAMPLE_SIZE} patients. Using all.")

print("Parsing Dates...")
df['CONTACT_DATE'] = pd.to_datetime(df['CONTACT_DATE'])

# 2. Identify AMD Cases
print("Identifying AMD cases...")
def is_amd(text):
    text = str(text).lower()
    return any(k in text for k in AMD_KEYWORDS)

df['IS_AMD'] = df['DX_NAME'].apply(is_amd)

# Find Diagnosis Date (First occurrence of AMD)
amd_dates = df[df['IS_AMD'] == True].groupby('Pseudo_id')['CONTACT_DATE'].min().to_dict()
print(f"Found {len(amd_dates)} patients with AMD in this subset.")

# 3. Build Sequences
print("Building patient sequences...")
df = df.sort_values(['Pseudo_id', 'CONTACT_DATE'])
grouped = df.groupby('Pseudo_id')

dataset = []

for pid, group in tqdm(grouped):
    diagnosis_date = amd_dates.get(pid, None)
    
    # Group by Date (Handle multiple codes per visit)
    visits = group.groupby('CONTACT_DATE')
    
    seq_codes = []
    seq_deltas = []
    prev_date = None
    
    for date, visit_rows in visits:
        # Stop if we reach the diagnosis date
        if diagnosis_date and date >= diagnosis_date:
            break
            
        # Collect codes for this visit
        codes_this_visit = []
        for raw_val in visit_rows['CURRENT_ICD10_LIST']:
            # Handle comma-separated strings if present
            if isinstance(raw_val, str):
                codes_this_visit.extend([c.strip() for c in raw_val.split(',')])
            else:
                codes_this_visit.append(str(raw_val))
        
        # Deduplicate codes in the visit
        codes_this_visit = list(set(codes_this_visit))
        seq_codes.append(codes_this_visit)
        
        # Time Delta
        if prev_date is None:
            seq_deltas.append(0)
        else:
            days = (date - prev_date).days
            seq_deltas.append(days)
        prev_date = date
        
    # Need at least 2 visits to be useful
    if len(seq_codes) < 2:
        continue
        
    # Labels
    y_class = 1.0 if diagnosis_date else 0.0
    y_reg = 0.0
    if diagnosis_date:
        y_reg = (diagnosis_date - prev_date).days
        
    dataset.append({
        'pid': pid,
        'codes': seq_codes,
        'deltas': seq_deltas,
        'y_class': y_class,
        'y_reg': y_reg
    })

print(f"Saved {len(dataset)} sequences to {OUTPUT_FILE}")
with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(dataset, f)