#!/usr/bin/env python2
"""
Arrange rows so records sharing the same pseudo_id are consecutive,
and ordered by date within each ID.
Compatible with Python 2.7.
"""

import argparse
import pandas as pd

def reorder_rows(input_path, output_path, id_col, date_col):
    df = pd.read_csv(input_path)

    # Check if columns exist
    if id_col not in df.columns:
        raise ValueError("Column '{}' not found in dataset.".format(id_col))
    if date_col not in df.columns:
        raise ValueError("Column '{}' not found in dataset.".format(date_col))

    # Convert date column to datetime objects
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    # Sort by ID first, then by Date
    print "Sorting by {} and {}...".format(id_col, date_col)
    
    # sort_values is available in pandas >= 0.17 (common in Py2.7)
    df = df.sort_values(by=[id_col, date_col], ascending=[True, True])

    # Write to CSV
    df.to_csv(output_path, index=False)
    print "Success! Wrote reordered rows to {}".format(output_path)

def main():
    parser = argparse.ArgumentParser(description="Group by ID and sort by Date.")
    parser.add_argument("input_csv", help="Path to source CSV")
    parser.add_argument("output_csv", help="Path to output CSV")
    
    # Arguments for column names
    parser.add_argument("--id_col", default="Pseudo_id", help="Name of the ID column")
    parser.add_argument("--date_col", default="CONTACT_DATE", help="Name of the Date column")

    args = parser.parse_args()
    reorder_rows(args.input_csv, args.output_csv, args.id_col, args.date_col)

if __name__ == "__main__":
    main()