import torch
import pytest
from data import PackedLMDataset
from typing import List

def test_packed_lm_dataset_simple():
    # 3 docs: [1,2], [3,4], [5,6] (simplified)
    # flattens to [1,2,3,4,5,6]
    # seq_len=2 -> chunk size 3
    # chunks: [1,2,3], [4,5,6]
    # inputs: [1,2], [4,5]
    # labels: [2,3], [5,6]
    
    token_sequences = [
        [1, 2],
        [3, 4],
        [5, 6]
    ]
    seq_len = 2
    ds = PackedLMDataset(token_sequences, seq_len)
    
    assert len(ds) == 2
    
    item0 = ds[0]
    assert torch.equal(item0['input_ids'], torch.tensor([1, 2]))
    assert torch.equal(item0['labels'], torch.tensor([2, 3]))
    
    item1 = ds[1]
    assert torch.equal(item1['input_ids'], torch.tensor([4, 5]))
    assert torch.equal(item1['labels'], torch.tensor([5, 6]))

def test_packed_lm_dataset_discard():
    # flattens to [1,2,3,4]
    # seq_len=2 -> chunk size 3
    # chunks: [1,2,3] ... 4 is left over and discarded
    token_sequences = [[1, 2, 3, 4]]
    seq_len = 2
    ds = PackedLMDataset(token_sequences, seq_len)
    
    assert len(ds) == 1
    item0 = ds[0]
    assert torch.equal(item0['input_ids'], torch.tensor([1, 2]))
    assert torch.equal(item0['labels'], torch.tensor([2, 3]))
