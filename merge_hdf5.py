#!/usr/bin/env python3

import argparse
import tqdm
import h5py

def merge_hdf5_files(output_file, input_files):
    """
    Merge datasets from multiple HDF5 files into a single output file.
    If duplicate dataset names occur, a suffix based on the input file name is appended.
    """

    elts_per_group = 10000
    group_idx = 0
    idx = 0

    with h5py.File(output_file, "w") as fout:
        group = fout.create_group(f"{group_idx}")

        for infile in input_files:
            print(f"Merging file: {infile}")
            with h5py.File(infile, "r") as fin:


                with tqdm.tqdm(total=len(fin) // 2) as pbar:
                    for key in fin:
                        # we will add numbered keys later
                        if not key.startswith("UniRef50"):
                            continue

                        fin.copy(key, group, name=key)

                        # add hardlink to dataset with index
                        group[str(idx)] = group[key]

                        pbar.update(1)
                        idx += 1

                        if len(group) >= elts_per_group:
                            group_idx += 1
                            group = fout.create_group(f"{group_idx}")
                            idx = 0

    print(f"Merge complete! Output file: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple HDF5 files (with distinct datasets) into one output file."
    )
    parser.add_argument("output", help="Name of the output merged HDF5 file.")
    parser.add_argument("inputs", nargs="+",
                        help="Input HDF5 file(s).")
    args = parser.parse_args()

    merge_hdf5_files(args.output, args.inputs)

if __name__ == "__main__":
    main()