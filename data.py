from typing import Iterator, List, Dict
import random
import itertools
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer


class PackedLMDataset(Dataset):
    """
    Concatenate tokenized text and pack into fixed-length sequences for causal LM.
    Produces items with fields: input_ids, labels (next token prediction).
    """
    def __init__(self, token_sequences: List[List[int]], seq_len: int):
        super().__init__()
        # Flatten all token lists with EOS between docs for safety
        eos = None
        flat: List[int] = []
        for toks in token_sequences:
            flat.extend(toks)
        total = len(flat)
        # Create chunks of length seq_len+1 (so labels are shifted)
        self.seq_len = seq_len
        self.inputs: List[torch.Tensor] = []
        self.labels: List[torch.Tensor] = []
        n_chunks = total // (seq_len + 1)
        for i in range(n_chunks):
            start = i * (seq_len + 1)
            end = start + (seq_len + 1)
            chunk = flat[start:end]
            x = torch.tensor(chunk[:-1], dtype=torch.long)
            y = torch.tensor(chunk[1:], dtype=torch.long)
            self.inputs.append(x)
            self.labels.append(y)

        # For resumable dataloader support
        self._current_epoch = 0

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx],
            'labels': self.labels[idx]
        }

    def state_dict(self):
        """Support for resumable dataloader."""
        return {'current_epoch': self._current_epoch}

    def load_state_dict(self, state_dict):
        """Support for resumable dataloader."""
        self._current_epoch = state_dict.get('current_epoch', 0)


def load_text_dataset(name: str, split: str, text_field: str, streaming: bool=False, max_shards=None):
    """Load a HF dataset and yield raw text strings."""
    ds = load_dataset(name, split=split, streaming=streaming)
    if max_shards and hasattr(ds, 'shard') and not streaming:
        ds = ds.shuffle(seed=1234).shard(num_shards=max_shards, index=0)
    for ex in ds:
        text = ex[text_field]
        if isinstance(text, str) and len(text) > 0:
            yield text


def build_token_sequences(tokenizer, texts: Iterator[str], max_docs: int = 20000):
    """Tokenize texts into lists of token ids, ensuring EOS between documents."""
    tokens: List[List[int]] = []
    eos_id = getattr(tokenizer, 'eos_token_id', None)
    for i, t in enumerate(texts):
        enc = tokenizer(t, add_special_tokens=True, truncation=False)
        ids = enc['input_ids']
        # Ensure documents are terminated with EOS to prevent cross-doc leakage
        if eos_id is not None and (len(ids) == 0 or ids[-1] != eos_id):
            ids = ids + [eos_id]
        tokens.append(ids)
        if (i + 1) >= max_docs:
            break
    return tokens

def collate(batch: List[Dict[str, torch.Tensor]]):
    x = torch.stack([b['input_ids'] for b in batch], dim=0)
    y = torch.stack([b['labels'] for b in batch], dim=0)
    return {'input_ids': x, 'labels': y}

def make_dataloaders(cfg, tokenizer):
    # Load training data
    train_texts = load_text_dataset(
        name=cfg['data']['dataset'],
        split=cfg['data']['split'],
        text_field=cfg['data']['text_field'],
        streaming=cfg['data']['streaming'],
        max_shards=cfg['data']['max_shards']
    )
    train_token_seqs = build_token_sequences(tokenizer, train_texts, max_docs=cfg['data']['train_docs'])
    train_ds = PackedLMDataset(train_token_seqs, seq_len=cfg['training']['seq_len'])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['training']['micro_batch_size'],
        shuffle=True,
        num_workers=cfg['hardware']['num_workers'],
        pin_memory=False,
        persistent_workers=False,
        collate_fn=collate
    )

    # Load validation data (separate batch from same dataset)
    val_texts = load_text_dataset(
        name=cfg['data']['dataset'],
        split=cfg['data']['split'],
        text_field=cfg['data']['text_field'],
        streaming=cfg['data']['streaming'],
        max_shards=cfg['data']['max_shards']
    )
    # Skip first train_docs documents, take next val_docs for validation
    val_texts_sliced = itertools.islice(val_texts, cfg['data']['train_docs'], cfg['data']['train_docs'] + cfg['data']['val_docs'])
    val_token_seqs = build_token_sequences(tokenizer, val_texts_sliced, max_docs=cfg['data']['val_docs'])
    val_ds = PackedLMDataset(val_token_seqs, seq_len=cfg['training']['seq_len'])

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg['training']['micro_batch_size'],
        shuffle=False,
        num_workers=cfg['hardware']['num_workers'],
        pin_memory=False,
        persistent_workers=False,
        collate_fn=collate
    )

    return train_loader, val_loader


def get_tokenizer(cfg):
    tok_name = cfg['data']['tokenizer_name']
    tok = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token is not None else tok.unk_token
    return tok
