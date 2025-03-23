#!/usr/bin/env python

from torch.utils.data import Dataset

class ESM2MaskedResidueDataset(Dataset):

    def __init__(self, data_dir):
        self.data_dir = data_dir


DEFAULT_DIR = "/home/gridsan/mattfeng/datasets/esm2_pretrain_nemo2_fulldata_v1.0"