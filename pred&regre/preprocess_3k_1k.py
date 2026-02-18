import pandas as pd
import numpy as np
import pickle
import os
from tqdm import tqdm

# --- CONFIGURATION ---
# We are inside 'pred&regre', so we look up to 'Dataset'
TRAIN_CSV = '../Dataset/train_split.csv'
TEST_CSV  = '../Dataset/test_split.csv'

# Output Files (Local to this folder)
MED2VEC_INPUT = 'data_3k_med2vec.pkl'
MAMBA_TRAIN   = 'data_3k_mamba_train.pkl'
MAMBA_TEST    = 'data_1k_mamba_test.pkl'
CODE_MAP      = 'code_map_3k.pkl'

TRAIN_COUNT = 3000
TEST_COUNT = 1000

AMD_KEYWORDS = ['macular degeneration', 'age-related macular', 'h35.3', '362.5']

def process_split(csv_path, sample_size, is_train=True, existing_map=None):
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path, usecols=['Pseudo_id', 'CONTACT_DATE', 'DX_NAME', 'CURRENT_ICD10_LIST'])
    
    # 1. SORTING
    print("  -> Sorting by Patient and Date...")
    df['CONTACT_DATE'] = pd.to_datetime(df['CONTACT_DATE'])
    df = df.sort_values(['Pseudo_id', 'CONTACT_DATE'])
    
    # 2. SAMPLING
    unique_pids = df['Pseudo_id'].unique()
    if len(unique_pids) >= sample_size:
        # Fixed seed for reproducibility
        selected_pids = np.random.RandomState(42).choice(unique_pids, sample_size, replace=False)
        df = df[df['Pseudo_id'].isin(selected_pids)]
        print(f"  -> Sampled {sample_size} patients.")
    else:
        print(f"  -> Warning: Requested {sample_size}, found {len(unique_pids)}. Using all.")

    # 3. LABELING AMD
    print("  -> Identifying AMD labels...")
    def is_amd(text): return any(k in str(text).lower() for k in AMD_KEYWORDS)
    df['IS_AMD'] = df['DX_NAME'].apply(is_amd)
    amd_dates = df[df['IS_AMD'] == True].groupby('Pseudo_id')['CONTACT_DATE'].min().to_dict()

    # 4. BUILDING VOCAB (Train Only)
    if is_train and existing_map is None:
        print("  -> Building Vocabulary...")
        code_counts = {}
        for raw in df['CURRENT_ICD10_LIST']:
            codes = str(raw).split(',') if isinstance(raw, str) else [str(raw)]
            for c in codes:
                c = c.strip()
                code_counts[c] = code_counts.get(c, 0) + 1
        sorted_codes = sorted(code_counts.keys())
        code_map = {code: i+1 for i, code in enumerate(sorted_codes)}
    else:
        code_map = existing_map

    # 5. GENERATING SEQUENCES
    print("  -> Generating sequences...")
    med2vec_visits = []
    model_data = []
    
    grouped = df.groupby('Pseudo_id')
    for pid, group in tqdm(grouped):
        diagnosis_date = amd_dates.get(pid, None)
        visits = group.groupby('CONTACT_DATE')
        
        seq_codes = []
        seq_deltas = []
        prev_date = None
        
        for date, visit_rows in visits:
            if diagnosis_date and date >= diagnosis_date: break
            
            visit_codes = []
            for raw in visit_rows['CURRENT_ICD10_LIST']:
                raw_list = str(raw).split(',') if isinstance(raw, str) else [str(raw)]
                visit_codes.extend([c.strip() for c in raw_list])
            
            ids = [code_map.get(c, 0) for c in set(visit_codes) if c in code_map]
            if not ids: continue
            
            seq_codes.append(ids)
            med2vec_visits.append(ids)
            
            if prev_date is None: seq_deltas.append(0)
            else: seq_deltas.append((date - prev_date).days)
            prev_date = date
            
        if len(seq_codes) < 2: continue
        
        y_class = 1.0 if diagnosis_date else 0.0
        y_reg = (diagnosis_date - prev_date).days if diagnosis_date else 0.0
        
        model_data.append({
            'seq': seq_codes,
            'deltas': seq_deltas,
            'y_class': y_class,
            'y_reg': y_reg
        })
        med2vec_visits.append([-1]) # Delimiter

    return med2vec_visits, model_data, code_map

# --- EXECUTION ---
print("\n--- PROCESSING TRAIN SPLIT ---")
m2v_train, mamba_train, vocab = process_split(TRAIN_CSV, TRAIN_COUNT, is_train=True)

print("\n--- PROCESSING TEST SPLIT ---")
_, mamba_test, _ = process_split(TEST_CSV, TEST_COUNT, is_train=False, existing_map=vocab)

print("\n--- SAVING FILES ---")
with open(CODE_MAP, 'wb') as f: pickle.dump(vocab, f)
with open(MED2VEC_INPUT, 'wb') as f: pickle.dump(m2v_train, f)
with open(MAMBA_TRAIN, 'wb') as f: pickle.dump(mamba_train, f)
with open(MAMBA_TEST, 'wb') as f: pickle.dump(mamba_test, f)

print("Done. Files ready.")