import pandas as pd
import cPickle as pickle
import sys

def debug_files():
    # 1. Check CSV Row Count
    csv_file = 'sorted_ICD_Data.csv'
    print "--- Checking CSV: {} ---".format(csv_file)
    try:
        df = pd.read_csv(csv_file)
        print "Total Rows in CSV: {:,}".format(len(df))
        print "Unique Patients: {:,}".format(df['Pseudo_id'].nunique())
        print "Unique Dates: {:,}".format(df['CONTACT_DATE'].nunique())
    except Exception as e:
        print "Error reading CSV: {}".format(e)

    # 2. Check Pickle File Content
    pkl_file = 'icd10.seqs'
    print "\n--- Checking Pickle: {} ---".format(pkl_file)
    try:
        with open(pkl_file, 'rb') as f:
            seqs = pickle.load(f)
        
        print "Total 'Visits' in Pickle: {:,}".format(len(seqs))
        
        # Analyze the first few items
        print "\nFirst 5 items in Pickle:"
        for i in range(min(5, len(seqs))):
            print "Item {}: {}".format(i, seqs[i])
            
        # Check distribution
        lengths = [len(x) for x in seqs]
        avg_len = sum(lengths) / float(len(lengths))
        print "\nAverage codes per visit: {:.2f}".format(avg_len)
        print "Max codes in a visit: {}".format(max(lengths))
        
    except Exception as e:
        print "Error reading Pickle: {}".format(e)

if __name__ == "__main__":
    debug_files()
    