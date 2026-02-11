#!/usr/bin/env python2
import pandas as pd
import cPickle as pickle
import sys

def process_icd10(input_csv, visit_output, label_output):
    print "Reading CSV..."
    df = pd.read_csv(input_csv)

    # --- PART 1: MAPPING CODES ---
    print "Mapping ICD-10 codes..."
    unique_codes = df['CURRENT_ICD10_LIST'].unique()
    code_map = {code: i for i, code in enumerate(unique_codes)}
    
    # Create 3-digit groups (Label)
    df['group_code'] = df['CURRENT_ICD10_LIST'].astype(str).str[:3]
    unique_groups = df['group_code'].unique()
    group_map = {code: i for i, code in enumerate(unique_groups)}

    print "Found {} unique ICD-10 codes.".format(len(unique_codes))
    print "Found {} unique 3-digit groups.".format(len(unique_groups))

    # --- PART 2: BUILDING LISTS ---
    print "Constructing visit lists..."
    
    # CRITICAL FIX: Convert Date to string YYYY-MM-DD to remove time components
    # This ensures all events on the same day are grouped into ONE visit
    df['DATE_ONLY'] = pd.to_datetime(df['CONTACT_DATE']).dt.strftime('%Y-%m-%d')

    visit_file_list = []
    label_file_list = []
    
    # Group by Patient
    grouped_patients = df.groupby('Pseudo_id', sort=False)

    for patient_id, patient_data in grouped_patients:
        # Group by the CLEAN DATE string
        grouped_visits = patient_data.groupby('DATE_ONLY', sort=False)
        
        for date, visit_data in grouped_visits:
            codes_raw = visit_data['CURRENT_ICD10_LIST'].tolist()
            groups_raw = visit_data['group_code'].tolist()
            
            visit_ints = [code_map[c] for c in codes_raw]
            group_ints = [group_map[g] for g in groups_raw]
            
            visit_file_list.append(visit_ints)
            label_file_list.append(group_ints)
        
        visit_file_list.append([-1])
        label_file_list.append([-1])

    # --- PART 3: SAVING ---
    print "Saving files..."
    with open(visit_output, 'wb') as f:
        pickle.dump(visit_file_list, f)
    with open(label_output, 'wb') as f:
        pickle.dump(label_file_list, f)
    with open("code_map.pkl", 'wb') as f:
        pickle.dump(code_map, f)
        
    print "------------------------------------------------"
    print "Complete!"
    print "1. Visit File: {}".format(visit_output)
    print "2. Label File: {}".format(label_output)
    print "3. Total Codes: {}".format(len(unique_codes))
    print "4. Total Groups: {}".format(len(unique_groups))
    print "------------------------------------------------"

if __name__ == "__main__":
    process_icd10('sorted_ICD_Data.csv', 'icd10.seqs', 'icd10_grouped.seqs')