import cPickle as pickle
import random
import sys

def create_mini_dataset(input_visit, input_label, ratio=0.1):
    print "Loading full datasets..."
    with open(input_visit, 'rb') as f:
        full_visits = pickle.load(f)
    with open(input_label, 'rb') as f:
        full_labels = pickle.load(f)
        
    print "Full dataset size: {:,} visits".format(len(full_visits))
    
    # The data is a flat list of visits separated by [-1].
    # We need to reconstruct "Patient" chunks to sample correctly.
    
    patients_visits = []
    patients_labels = []
    
    current_p_visits = []
    current_p_labels = []
    
    for v, l in zip(full_visits, full_labels):
        if v == [-1]: # End of patient
            patients_visits.append(current_p_visits)
            patients_labels.append(current_p_labels)
            current_p_visits = []
            current_p_labels = []
        else:
            current_p_visits.append(v)
            current_p_labels.append(l)
            
    total_patients = len(patients_visits)
    print "Total Patients found: {:,}".format(total_patients)
    
    # Sample random patients
    sample_size = int(total_patients * ratio)
    print "Selecting random {:,} patients ({}%)...".format(sample_size, ratio*100)
    
    indices = range(total_patients)
    random.shuffle(indices)
    selected_indices = indices[:sample_size]
    
    # Flatten back to Med2Vec format
    mini_visits = []
    mini_labels = []
    
    for idx in selected_indices:
        # Add all visits for this patient
        for v in patients_visits[idx]:
            mini_visits.append(v)
        # Add delimiter
        mini_visits.append([-1])
        
        for l in patients_labels[idx]:
            mini_labels.append(l)
        mini_labels.append([-1])
        
    # Save
    print "Saving mini datasets..."
    with open('mini_icd10.seqs', 'wb') as f:
        pickle.dump(mini_visits, f)
    with open('mini_icd10_grouped.seqs', 'wb') as f:
        pickle.dump(mini_labels, f)
        
    print "Done! Created 'mini_icd10.seqs' with {:,} visits.".format(len(mini_visits))

if __name__ == "__main__":
    create_mini_dataset('icd10.seqs', 'icd10_grouped.seqs', ratio=0.1)