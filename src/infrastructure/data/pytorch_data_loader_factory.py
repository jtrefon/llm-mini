"""PyTorch DataLoader factory using HuggingFace datasets and tokenizers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List

import torch
from torch.utils.data import DataLoader, Dataset

from transformers import AutoTokenizer
from datasets import load_dataset, IterableDatasetDict, DatasetDict

from src.infrastructure.config.config_loader import DataConfig
from src.domain.entities.training_config import TrainingConfiguration


# Top-level collate function so DataLoader workers can pickle it
def _lm_collate(batch: List[torch.Tensor]) -> dict[str, torch.Tensor]:
    input_ids = torch.stack(batch, dim=0)
    return {
        'input_ids': input_ids,
        'labels': input_ids.clone(),
    }


@dataclass
class _TokenizedSequenceDataset(Dataset):
    sequences: List[torch.Tensor]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.sequences[idx]


class PyTorchDataLoaderFactory:
    """Factory that builds PyTorch DataLoaders from DataConfig/TrainingConfiguration."""

    def __init__(self, data_config: DataConfig, num_workers: int = 0):
        self.data_config = data_config
        self._num_workers = max(0, int(num_workers))
        self._tokenizer = AutoTokenizer.from_pretrained(self.data_config.tokenizer_name, use_fast=True)
        if self._tokenizer.pad_token is None:
            # Ensure pad token exists for collation
            self._tokenizer.pad_token = self._tokenizer.eos_token or self._tokenizer.unk_token

    def _load_hf_splits(self) -> Tuple[Iterable | None, Iterable | None]:
        ds = load_dataset(self.data_config.dataset, split=None, streaming=self.data_config.streaming)
        # Expect common split names; fall back to using config.split uniformly
        train_split_name = 'train' if 'train' in ds else self.data_config.split
        val_split_name = 'validation' if 'validation' in ds else ('test' if 'test' in ds else self.data_config.split)

        train_iter = ds[train_split_name]
        val_iter = ds[val_split_name] if val_split_name in ds and val_split_name != train_split_name else None
        return train_iter, val_iter

    def _stream_to_sequences(self, itr: Iterable, limit_docs: int, seq_len: int) -> List[torch.Tensor]:
        text_field = self.data_config.text_field
        pack = self.data_config.pack_sequences
        toks: List[int] = []
        sequences: List[torch.Tensor] = []
        count = 0
        for ex in itr:
            if limit_docs is not None and count >= limit_docs:
                break
            text = ex[text_field]
            enc = self._tokenizer(text, add_special_tokens=True, return_attention_mask=False)
            if pack:
                toks.extend(enc['input_ids'])
                while len(toks) >= seq_len:
                    seq = toks[:seq_len]
                    toks = toks[seq_len:]
                    sequences.append(torch.tensor(seq, dtype=torch.long))
            else:
                seq = enc['input_ids'][:seq_len]
                if len(seq) < seq_len:
                    # pad
                    seq = seq + [self._tokenizer.pad_token_id] * (seq_len - len(seq))
                sequences.append(torch.tensor(seq, dtype=torch.long))
            count += 1
        return sequences

    def _build_dataloader(self, sequences: List[torch.Tensor], micro_batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
        ds = _TokenizedSequenceDataset(sequences)
        return DataLoader(
            ds,
            batch_size=micro_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=num_workers > 0,
            collate_fn=_lm_collate,
        )

    def create_loaders(self, training: TrainingConfiguration) -> Tuple[DataLoader, DataLoader | None]:
        train_iter, val_iter = self._load_hf_splits()
        train_sequences = self._stream_to_sequences(train_iter, self.data_config.train_docs, training.seq_len)

        if val_iter is None:
            # Reuse a slice of train for validation if no split available
            val_sequences = train_sequences[: max(1, min(len(train_sequences) // 10, 100))]
        else:
            val_sequences = self._stream_to_sequences(val_iter, self.data_config.val_docs, training.seq_len)

        train_loader = self._build_dataloader(train_sequences, training.micro_batch_size, num_workers=self._num_workers, shuffle=True)
        val_loader = self._build_dataloader(val_sequences, training.micro_batch_size, num_workers=self._num_workers, shuffle=False)
        return train_loader, val_loader
