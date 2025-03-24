"""
Creates an HDF5 file with keys as UniRef50 clusters (for diversity) and values of UniRef90 sequences (for quantity).
"""

from typing import List
from pathlib import Path
import h5py
import numpy as np
import json
import sqlite3
import tqdm
import os
import argparse

def main(frac):
    dataset_dir = Path("/state/partition1/user/mattfeng/")

    dataset_type = "train"

    sequences_db = dataset_dir / f"{dataset_type}.db"
    clusters_jsonl = dataset_dir / f"{dataset_type}_clusters_{frac}.jsonl"

    with \
        h5py.File(dataset_dir / f"esm2_pretrain_{frac}.h5", "w") as fout, \
        open(clusters_jsonl) as fin, \
        sqlite3.connect(sequences_db) as conn:


        with tqdm.tqdm(total=os.path.getsize(clusters_jsonl), unit="B", unit_scale=True) as pbar:

            for idx, line in enumerate(fin):
                pbar.update(len(line))

                cursor = conn.cursor()

                cluster = json.loads(line.strip())
                cluster_id: str = cluster["ur50_id"]
                seq_ids: List[str] = cluster["ur90_id"]
                seqs: List[str] = []

                for seq_id in seq_ids:
                    query = "SELECT sequence FROM protein WHERE id = ?"
                    cursor.execute(query, (seq_id,))
                    seq = cursor.fetchone()[0]

                    seqs.append(seq)

                dt = np.dtype([
                    ("seq_id", h5py.string_dtype("utf-8")),
                    ("seq", h5py.string_dtype("utf-8"))
                ])
                data = np.array(list(zip(seq_ids, seqs)), dtype=dt)

                fout.create_dataset(cluster_id, data=data)

                # dset = fout.create_dataset(cluster_id, data=data)
                # create hard link with index
                # 2025-03-24: actually, do this when merging fractions; alternatively, make sure there are not too many keys at the root level, it makes HDF5 very slow
                # fout[str(idx)] = dset

                cursor.close()

    print(f"Done generating fraction {frac} of ESM2 pretraining dataset.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("frac", type=str)

    args = parser.parse_args()

    main(args.frac)