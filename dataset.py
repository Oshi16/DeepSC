# !usr/bin/env python
# -*- coding:utf-8 _*-

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

class EurDataset(Dataset):
    def __init__(self, split='train', data_dir='/content/drive/MyDrive/DeepSC_data'):
        """
        Initialize the dataset.

        Args:
            split (str): Specifies the data split, e.g., 'train' or 'test'.
            data_dir (str): Directory where the dataset files are stored.
        """
        # Load the dataset from a pickle file.
        # The file path is constructed based on the split ('train' or 'test') and data directory.
        with open(os.path.join(data_dir, '{}_data.pkl'.format(split)), 'rb') as f:
            self.data = pickle.load(f)

    def __getitem__(self, index):
        """
        Get a single item from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            sents: The data item at the specified index.
        """
        sents = self.data[index]
        return sents

    def __len__(self):
        """
        Get the total number of items in the dataset.

        Returns:
            int: Number of items in the dataset.
        """
        return len(self.data)

def collate_data(batch):
    """
    Collate function to create batches of data.

    Args:
        batch (list): List of data items to be collated.

    Returns:
        torch.Tensor: Padded and batched data as a PyTorch tensor.
    """
    batch_size = len(batch)
    
    # Find the maximum length of sentences in the current batch.
    max_len = max(map(lambda x: len(x), batch))
    
    # Initialize an array to hold padded sentences.
    sents = np.zeros((batch_size, max_len), dtype=np.int64)
    
    # Sort sentences by length in descending order.
    sort_by_len = sorted(batch, key=lambda x: len(x), reverse=True)

    # Pad sentences and add them to the array.
    for i, sent in enumerate(sort_by_len):
        length = len(sent)
        sents[i, :length] = sent  # Pad the sentence with zeros up to max_len

    # Convert the NumPy array to a PyTorch tensor.
    return torch.from_numpy(sents)
