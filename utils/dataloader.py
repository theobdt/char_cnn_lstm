import torch
from torch.utils.data import Dataset
import os
import numpy as np


class DataLoader(Dataset):
    def __init__(self, dataset_name, mode, batch_size, seq_len):
        if mode not in ["train", "val", "test"]:
            raise ValueError("mode '{mode}' not recognized")

        self.batch_size = batch_size
        self.seq_len = seq_len

        path_objects = os.path.join("data", dataset_name, "objects")
        path_chars = os.path.join(path_objects, mode, f"chars.pt")
        path_words = os.path.join(path_objects, mode, f"words.pt")

        self.indexed_chars = torch.load(path_chars)
        self.indexed_words = torch.load(path_words)
        self.split_batches()
        self.indexes = np.arange(self.n_batches)

    def split_batches(self):
        len_tensor = self.indexed_words.shape[0] // self.batch_size
        self.max_word_len = self.indexed_chars.shape[1]

        idx_chars = self.indexed_chars[: self.batch_size * len_tensor, :]
        batched_chars = idx_chars.view(self.batch_size, -1, self.max_word_len)
        self.batches_input_chars = batched_chars.split(self.seq_len, dim=1)

        idx_words = self.indexed_words[: self.batch_size * len_tensor]
        batched_words = idx_words.view(self.batch_size, -1)
        target_words = torch.cat(
            (batched_words[:, 1:], batched_words[:, 0].view(-1, 1)), dim=1
        )
        self.batches_target_words = target_words.split(self.seq_len, dim=1)

        self.n_batches = np.ceil(len_tensor / self.seq_len).astype(int)

    def shuffle(self):
        self.indexes = np.random.permutation(self.n_batches)

    def __getitem__(self, i):
        idx = self.indexes[i]

        x = self.batches_input_chars[idx].long()
        y = self.batches_target_words[idx].contiguous().view(-1)

        return x, y

    def __len__(self):
        return self.n_batches


if __name__ == "__main__":
    dl = DataLoader("penn-treebank", "train", batch_size=16, seq_len=5)
    print(dl.__len__())
    x, y = dl.__get_item__(10)
    print(x.shape, y.shape)
