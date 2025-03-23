#!/usr/bin/env python3

import argparse
import glob
import os
import sys
import h5py

def merge_hdf5_files(output_file, input_files):
    """
    Merge datasets from multiple HDF5 files into a single output file.
    If duplicate dataset names occur, a suffix based on the input file name is appended.
    """
    with h5py.File(output_file, 'w') as fout:
        for infile in input_files:
            print(f"Merging file: {infile}")
            with h5py.File(infile, 'r') as fin:
                for key in fin:
                    # If dataset already exists, append the file's basename (without extension)
                    if key in fout:
                        base = os.path.splitext(os.path.basename(infile))[0]
                        new_key = f"{key}_{base}"
                        print(f"  Duplicate key '{key}' found. Using '{new_key}'.")
                        fin.copy(key, fout, name=new_key)
                    else:
                        fin.copy(key, fout, name=key)
    print(f"Merge complete! Output file: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple HDF5 files (with distinct datasets) into one output file."
    )
    parser.add_argument("output", help="Name of the output merged HDF5 file.")
    parser.add_argument("inputs", nargs="+",
                        help="Input HDF5 file(s) or glob pattern(s).")
    args = parser.parse_args()

    # Expand any glob patterns in the input arguments
    expanded_files = []
    for pattern in args.inputs:
        files = glob.glob(pattern)
        if not files:
            print(f"Warning: No files found for pattern '{pattern}'.", file=sys.stderr)
        expanded_files.extend(files)

    if not expanded_files:
        print("Error: No input files found. Exiting.", file=sys.stderr)
        sys.exit(1)

    merge_hdf5_files(args.output, expanded_files)

if __name__ == "__main__":
    main()