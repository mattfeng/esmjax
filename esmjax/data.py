#!/usr/bin/env python

from typing import Tuple
import random

import h5py

import numpy as np

from torch.utils.data import Dataset, DataLoader

from esmjax.tokenizer import protein_tokenizer
from tokenizers import Tokenizer

class ESM2MaskedResidueDataset(Dataset):
    """
    Dataset format:

    ESM2MaskedResidueDataset is a HDF5 file with the following dataset format:

    - GROUP: group_idx (increments starting from 0, no meaning, simply created to speed up HDF5 file indexing)
        - DATASET: idx (0-9999; ≤10000 [last group may have fewer than 10000 entries] UniRef50 clusters )
            - Each dataset maps to a list of (UniRef90_ID: str, protein_sequence: str) tuples.

    To sample uniformly from the dataset, first a random UniRef50 cluster is sampled, and then a random protein sequence from that cluster is selected.

    """
    CLUSTERS_PER_GROUP = 5000

    def __init__(
        self,
        hdf5_file_path,
        *,
        seed,
        tokenizer: Tokenizer,
        seq_len: int
    ):
        self.seed = seed
        self.tokenizer = tokenizer
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.hdf5_file = h5py.File(hdf5_file_path, "r")
        self.num_ur50_clusters = sum(len(group) for group in self.hdf5_file)
        self.seq_len = seq_len

    def __len__(self):
        return self.num_ur50_clusters

    def _random_crop(self, seq: Tuple[str]) -> Tuple[str]:
        if len(seq) <= self.seq_len:
            return seq
        start = self.rng.randrange(0, len(seq) - self.seq_len)
        return seq[start:start + self.seq_len]

    def _mask(self, seq: Tuple[str]) -> Tuple[str]:
        ret = []

        for c in seq:
            if self.rng.random() < 0.15:
                ret.append("<mask>")
            else:
                ret.append(c)

        return tuple(ret)

    def _tokenize(self, seq: Tuple[str]) -> Tuple[int]:
        # pad and convert to int
        return self.tokenizer.encode(
            "".join(seq),
            add_special_tokens=True
        )

    def __getitem__(self, idx) -> bytes:
        group_idx = idx // self.CLUSTERS_PER_GROUP
        within_group_idx = idx % self.CLUSTERS_PER_GROUP

        seq_id, seq = self.rng.choice(
            self.hdf5_file[str(group_idx)][str(within_group_idx)]
        )
        seq = tuple(["<sos>", *[chr(i) for i in seq], "<eos>"])

        seq = self._random_crop(seq)
        masked_seq = self._mask(seq)
        seq = self._tokenize(seq)
        masked_seq = self._tokenize(masked_seq)

        return {
            "masked_ids": masked_seq.ids,
            "ids": seq.ids,
            "special_tokens_mask": seq.special_tokens_mask
        }

    def collate_fn(self, batch):
        return {
            "masked_ids": np.stack([item["masked_ids"] for item in batch]),
            "ids": np.stack([item["ids"] for item in batch]),
            "special_tokens_mask": np.stack([item["special_tokens_mask"] for item in batch]),
        }

    def __del__(self):
        if self.hdf5_file is not None:
            self.hdf5_file.close()


if __name__ == "__main__":
    hdf5_file_path = "/home/gridsan/mattfeng/datasets/esm2_pretrain_nemo2_fulldata_v1.0/full_esm2_pretrain.h5"

    dataset = ESM2MaskedResidueDataset(
        hdf5_file_path,
        tokenizer=protein_tokenizer(1024),
        seed=100
    )

    # item0_masked, item0 = dataset[0]

    # print(item0.tokens)
    # print(item0.ids)
    # print(item0.special_tokens_mask)

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        drop_last=False,
        persistent_workers=True,
        num_workers=4,
        collate_fn=dataset.collate_fn
    )

    batch = next(iter(dataloader))
    print(batch["masked_ids"])
