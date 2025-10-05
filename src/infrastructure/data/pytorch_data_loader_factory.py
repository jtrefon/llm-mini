"""PyTorch DataLoader factory using HuggingFace datasets and tokenizers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info

from transformers import AutoTokenizer
from datasets import load_dataset

from src.infrastructure.config.config_loader import DataConfig
from src.domain.entities.training_config import TrainingConfiguration


# Top-level collate function so DataLoader workers can pickle it
# Expects a batch of dicts with 'input_ids' and 'labels' tensors
def _lm_collate(batch: List[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    input_ids = torch.stack([b['input_ids'] for b in batch], dim=0)
    labels = torch.stack([b['labels'] for b in batch], dim=0)
    return {'input_ids': input_ids, 'labels': labels}


@dataclass
class _TokenizedSequenceDataset(Dataset):
    """Materialized dataset of pre-tokenized (x,y) pairs for non-streaming use."""
    items: List[dict[str, torch.Tensor]]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.items[idx]


class _StreamingPackedLMDataset(IterableDataset):
    """Iterable dataset that tokenizes and packs sequences on the fly.

    Yields fixed-length (seq_len) input_ids and next-token labels without
    holding the entire corpus in memory. Supports optional document packing
    across boundaries.
    """

    def __init__(self, examples_iter: Iterable, tokenizer: AutoTokenizer, text_field: str,
                 seq_len: int, max_docs: int | None, pack_sequences: bool):
        super().__init__()
        self.examples_iter = examples_iter
        self.tokenizer = tokenizer
        self.text_field = text_field
        self.seq_len = int(seq_len)
        self.max_docs = int(max_docs) if max_docs is not None else None
        self.pack_sequences = bool(pack_sequences)

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1

        eos_id = getattr(self.tokenizer, 'eos_token_id', None)
        pad_id = getattr(self.tokenizer, 'pad_token_id', None)
        if pad_id is None:
            pad_id = eos_id if eos_id is not None else 0

        buffer: List[int] = []
        docs_seen = 0

        for idx, ex in enumerate(self.examples_iter):
            # Simple worker sharding by example index
            if idx % num_workers != worker_id:
                continue
            if self.max_docs is not None and docs_seen >= self.max_docs:
                break

            text = ex[self.text_field]
            if not isinstance(text, str) or len(text) == 0:
                continue
            enc = self.tokenizer(text, add_special_tokens=True, return_attention_mask=False)
            ids = enc['input_ids']
            # Ensure EOS termination for safety when packing
            if eos_id is not None and (len(ids) == 0 or ids[-1] != eos_id):
                ids = ids + [eos_id]
            docs_seen += 1

            if self.pack_sequences:
                buffer.extend(ids)
                target = self.seq_len + 1
                while len(buffer) >= target:
                    chunk = buffer[:target]
                    buffer = buffer[target:]
                    x = torch.tensor(chunk[:-1], dtype=torch.long)
                    y = torch.tensor(chunk[1:], dtype=torch.long)
                    yield {'input_ids': x, 'labels': y}
            else:
                # Per-document, pad/truncate to seq_len+1, then shift
                target = self.seq_len + 1
                if len(ids) < target:
                    ids = ids + [pad_id] * (target - len(ids))
                else:
                    ids = ids[:target]
                x = torch.tensor(ids[:-1], dtype=torch.long)
                y = torch.tensor(ids[1:], dtype=torch.long)
                yield {'input_ids': x, 'labels': y}


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
        val_iter = ds[val_split_name] if (val_split_name in ds and val_split_name != train_split_name) else None
        return train_iter, val_iter

    def _materialize_pairs(self, itr: Iterable, limit_docs: int, seq_len: int) -> List[dict[str, torch.Tensor]]:
        """Materialize a limited number of (input_ids, labels) pairs for non-streaming use.

        WARNING: This holds all returned pairs in memory. Only use when dataset size is modest.
        """
        text_field = self.data_config.text_field
        pack = self.data_config.pack_sequences
        eos_id = getattr(self._tokenizer, 'eos_token_id', None)
        pad_id = getattr(self._tokenizer, 'pad_token_id', None)
        if pad_id is None:
            pad_id = eos_id if eos_id is not None else 0

        toks: List[int] = []
        items: List[dict[str, torch.Tensor]] = []
        count = 0
        for ex in itr:
            if limit_docs is not None and count >= limit_docs:
                break
            text = ex[text_field]
            enc = self._tokenizer(text, add_special_tokens=True, return_attention_mask=False)
            ids = enc['input_ids']
            if eos_id is not None and (len(ids) == 0 or ids[-1] != eos_id):
                ids = ids + [eos_id]
            if pack:
                toks.extend(ids)
                target = seq_len + 1
                while len(toks) >= target:
                    chunk = toks[:target]
                    toks = toks[target:]
                    x = torch.tensor(chunk[:-1], dtype=torch.long)
                    y = torch.tensor(chunk[1:], dtype=torch.long)
                    items.append({'input_ids': x, 'labels': y})
            else:
                target = seq_len + 1
                if len(ids) < target:
                    ids = ids + [pad_id] * (target - len(ids))
                else:
                    ids = ids[:target]
                x = torch.tensor(ids[:-1], dtype=torch.long)
                y = torch.tensor(ids[1:], dtype=torch.long)
                items.append({'input_ids': x, 'labels': y})
            count += 1
        return items

    def _build_dataloader(self, items: List[dict[str, torch.Tensor]], micro_batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
        ds = _TokenizedSequenceDataset(items)
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

        # True streaming path: no full-materialization. Use IterableDataset.
        if self.data_config.streaming:
            # Ensure tokenizer has a pad token and reasonable max length
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token or self._tokenizer.unk_token
            # Training stream
            train_ds = _StreamingPackedLMDataset(
                examples_iter=train_iter,
                tokenizer=self._tokenizer,
                text_field=self.data_config.text_field,
                seq_len=training.seq_len,
                max_docs=self.data_config.train_docs,
                pack_sequences=self.data_config.pack_sequences,
            )
            train_loader = DataLoader(
                train_ds,
                batch_size=training.micro_batch_size,
                shuffle=False,  # IterableDataset cannot be shuffled
                num_workers=self._num_workers,
                pin_memory=False,
                persistent_workers=False,
                collate_fn=_lm_collate,
            )

            # Validation stream (reuse validation split when available; otherwise re-open train stream)
            if val_iter is None:
                # Open a fresh iterator for validation to avoid exhausting the train iterator
                val_train_iter, _ = self._load_hf_splits()
                val_iter = val_train_iter
            val_ds = _StreamingPackedLMDataset(
                examples_iter=val_iter,
                tokenizer=self._tokenizer,
                text_field=self.data_config.text_field,
                seq_len=training.seq_len,
                max_docs=self.data_config.val_docs,
                pack_sequences=self.data_config.pack_sequences,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=training.micro_batch_size,
                shuffle=False,
                num_workers=self._num_workers,
                pin_memory=False,
                persistent_workers=False,
                collate_fn=_lm_collate,
            )
            return train_loader, val_loader

        # Non-streaming path (materializes a limited set). Use with caution on memory.
        train_items = self._materialize_pairs(train_iter, self.data_config.train_docs, training.seq_len)
        if val_iter is None:
            # Reuse a slice of train for validation if no split available
            n = max(1, min(len(train_items) // 10, 100))
            val_items = train_items[:n]
        else:
            val_items = self._materialize_pairs(val_iter, self.data_config.val_docs, training.seq_len)

        train_loader = self._build_dataloader(train_items, training.micro_batch_size, num_workers=self._num_workers, shuffle=True)
        val_loader = self._build_dataloader(val_items, training.micro_batch_size, num_workers=self._num_workers, shuffle=False)
        return train_loader, val_loader
