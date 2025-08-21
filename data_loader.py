import numpy as np
import torch
from data.lookup import stoi, itos

train_dataset_path = "data/devanagari_train.npz"
test_dataset_path = "data/devanagari_test.npz"

print("Loading Train Data...")
train_data = np.load(train_dataset_path)['data']

print("Loading Test Data...")
test_data = np.load(test_dataset_path)['data']

vocab_size = len(stoi)

def get_batch(data_set, batch_size, block_size, device):
    indices = torch.randint(len(data_set) - block_size, (batch_size,))
    x = torch.stack([
        torch.from_numpy((data_set[i:i+block_size]).astype(np.int64)) for i in indices
    ])
    y = torch.stack([
        torch.from_numpy((data_set[i+1:i+block_size+1]).astype(np.int64)) for i in indices
    ])

    return x.to(device), y.to(device)
