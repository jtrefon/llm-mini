from typing import Iterator, List, Dict, Optional, Iterable, Callable, Union, Any, Tuple
import itertools
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from transformers import PreTrainedTokenizer


class PackedLMDataset(Dataset):
    """Dataset that concatenates tokenized text and packs into fixed-length sequences.

    Produces items with fields: 'input_ids', 'labels' (next token prediction).
    """

    def __init__(self, token_sequences: List[List[int]], seq_len: int):
        """Initializes the dataset.

        Args:
            token_sequences: List of lists of token IDs (one list per document).
            seq_len: Target sequence length for training.
        """
        super().__init__()
        # Flatten all token lists with EOS between docs for safety
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

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.inputs[idx],
            "labels": self.labels[idx],
        }

    def state_dict(self) -> Dict[str, Any]:
        """Support for resumable dataloader."""
        return {"current_epoch": self._current_epoch}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Support for resumable dataloader."""
        self._current_epoch = state_dict.get("current_epoch", 0)


class StreamingPackedLMDataset(IterableDataset):
    """Iterable dataset that tokenizes and packs sequences on the fly.

    Avoids holding the entire corpus in memory. Accumulates a small token buffer
    and yields fixed-length (seq_len) sequences with next-token labels.

    Parameters:
        texts_factory: Callable returning an iterable of raw text strings per epoch.
        tokenizer: HuggingFace tokenizer.
        seq_len: Target sequence length.
        max_docs: Optional limit of documents to read each epoch.
        add_eos: Whether to append EOS to each document.
    """

    def __init__(
        self,
        texts_factory: Callable[[], Iterable[str]],
        tokenizer: Any,
        seq_len: int,
        max_doc_len: Optional[int] = None,
        max_docs: Optional[int] = None,
        add_eos: bool = True,
    ):
        super().__init__()
        if not callable(texts_factory):
            raise ValueError(
                "StreamingPackedLMDataset requires a callable that yields a fresh iterator per epoch"
            )
        self.texts_factory = texts_factory
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.max_doc_len = int(max_doc_len) if max_doc_len is not None else int(seq_len)
        self.max_docs = int(max_docs) if max_docs is not None else None
        self.add_eos = bool(add_eos)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker = get_worker_info()
        if worker is not None:
            worker_id = worker.id
            num_workers = worker.num_workers
        else:
            worker_id = 0
            num_workers = 1

        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        buffer: List[int] = []
        docs_seen = 0

        texts_iter = self.texts_factory()
        if texts_iter is None:
            return

        # Shard by document index among workers
        for idx, text in enumerate(texts_iter):
            if idx % num_workers != worker_id:
                continue
            if self.max_docs is not None and docs_seen >= self.max_docs:
                break
            if not isinstance(text, str) or len(text) == 0:
                continue

            enc = self.tokenizer(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_doc_len,
            )
            ids = enc["input_ids"]
            if (
                self.add_eos
                and eos_id is not None
                and (len(ids) == 0 or ids[-1] != eos_id)
            ):
                ids = ids + [eos_id]
            buffer.extend(ids)
            docs_seen += 1

            # Yield as many full sequences as possible
            # We pack into (seq_len+1) to create shifted labels
            target = self.seq_len + 1
            while len(buffer) >= target:
                chunk = buffer[:target]
                buffer = buffer[target:]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": x, "labels": y}


def load_text_dataset(
    name: str,
    config_name: Optional[str] = None,
    split: str = "train",
    text_field: str = "text",
    streaming: bool = False,
    max_shards: Optional[int] = None,
    verbose: bool = True,
) -> Iterator[str]:
    """Load a HF dataset and yield raw text strings."""
    from datasets import load_dataset

    ds = load_dataset(name, config_name, split=split, streaming=streaming)
    if verbose:
        print(
            f"Loading dataset: {name}/{config_name}, split: {split}, streaming: {streaming}"
        )
    # IMPORTANT: limit shuffle buffer for streaming to avoid large in-memory buffers of long texts
    if streaming:
        try:
            ds = ds.shuffle(buffer_size=512, seed=42)
        except TypeError:
            # Fallback if signature differs; attempt positional
            ds = ds.shuffle(512, seed=42)
    else:
        ds = ds.shuffle(seed=42)
    if max_shards and hasattr(ds, "shard") and not streaming:
        ds = ds.shard(num_shards=max_shards, index=0)
    if verbose:
        print(f"Dataset loaded and shuffled. Starting tokenization...")
    count = 0
    for ex in ds:
        text = ex[text_field]
        if isinstance(text, str) and len(text) > 0:
            count += 1
            if verbose and count % 5000 == 0:
                print(f"Processed {count} documents...", end="\r")
            yield text
    if verbose:
        print(f"Completed processing {count} documents.")


def build_token_sequences(
    tokenizer: Any,
    texts: Iterator[str],
    max_docs: int = 20000,
    seq_len: int = 1024,
    max_doc_len: Optional[int] = None,
) -> List[List[int]]:
    """Tokenize texts into lists of token ids, ensuring EOS between documents."""
    tokens: List[List[int]] = []
    eos_id = getattr(tokenizer, "eos_token_id", None)
    total_tokens = 0
    max_doc_len = int(max_doc_len) if max_doc_len is not None else int(seq_len)
    for i, t in enumerate(texts):
        if i % 1000 == 0:
            print(
                f"Tokenizing document {i+1}/{max_docs if max_docs else 'inf'}", end="\r"
            )
        enc = tokenizer(
            t, add_special_tokens=True, truncation=True, max_length=max_doc_len
        )
        ids = enc["input_ids"]
        total_tokens += len(ids)
        # Ensure documents are terminated with EOS to prevent cross-doc leakage
        if eos_id is not None and (len(ids) == 0 or ids[-1] != eos_id):
            ids = ids + [eos_id]
        tokens.append(ids)
        if max_docs is not None and (i + 1) >= max_docs:
            break
    print(f"Total tokens tokenized: {total_tokens}")
    return tokens


def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collates a batch of input/label tensors."""
    x = torch.stack([b["input_ids"] for b in batch], dim=0)
    y = torch.stack([b["labels"] for b in batch], dim=0)
    return {"input_ids": x, "labels": y}


def make_dataloaders(
    cfg: Dict[str, Any], tokenizer: Any
) -> Tuple[DataLoader, DataLoader]:
    """Prepares training and validation DataLoaders based on configuration."""
    seq_len = cfg["training"]["seq_len"]
    print("Starting data preparation with seq_len=", seq_len)
    max_doc_len = int(cfg["data"].get("max_doc_len") or min(int(seq_len) * 4, 8192))

    streaming = bool(cfg["data"]["streaming"])
    accelerator = cfg["hardware"].get("accelerator", "auto")
    num_workers_cfg = int(cfg["hardware"].get("num_workers", 0) or 0)
    train_split = cfg["data"]["split"]
    val_split = cfg["data"].get("val_split", train_split)
    max_shards = cfg["data"]["max_shards"]

    # Heuristic: on MPS, keep num_workers=0 and pin_memory=False to reduce host memory footprint
    if accelerator == "mps":
        num_workers = 0
        pin_memory = False
        persistent = False
        prefetch = None
    else:
        num_workers = num_workers_cfg
        pin_memory = True
        persistent = bool(num_workers > 0)
        prefetch = 1 if persistent else None

    if streaming:
        print("Using streaming IterableDataset to avoid loading full corpus into memory")

        def make_stream_factory(split_name: str, skip_docs: int = 0):
            def factory():
                iterator = load_text_dataset(
                    name=cfg["data"]["dataset"],
                    config_name=cfg["data"].get("config", None),
                    split=split_name,
                    text_field=cfg["data"]["text_field"],
                    streaming=True,
                    max_shards=max_shards,
                    verbose=False,
                )
                if skip_docs > 0:
                    iterator = itertools.islice(iterator, skip_docs, None)
                return iterator

            return factory

        train_docs = int(cfg["data"].get("train_docs") or 900000)
        train_ds = StreamingPackedLMDataset(
            make_stream_factory(train_split),
            tokenizer,
            seq_len,
            max_doc_len=max_doc_len,
            max_docs=train_docs,
        )
        dl_kwargs = dict(
            batch_size=cfg["training"]["micro_batch_size"],
            shuffle=False,  # IterableDataset cannot be shuffled by DataLoader
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate,
        )
        if persistent:
            dl_kwargs["persistent_workers"] = True
        if prefetch is not None:
            dl_kwargs["prefetch_factor"] = prefetch
        train_loader = DataLoader(train_ds, **dl_kwargs)

        # Validation stream: separate loader; for simplicity use a separate dataset read
        print("Preparing validation streaming dataset...")
        skip_for_val = train_docs if val_split == train_split else 0
        val_docs = int(cfg["data"].get("val_docs") or 100000)
        val_ds = StreamingPackedLMDataset(
            make_stream_factory(val_split, skip_docs=skip_for_val),
            tokenizer,
            seq_len,
            max_doc_len=max_doc_len,
            max_docs=val_docs,
        )
        dl_kwargs = dict(
            batch_size=cfg["training"]["micro_batch_size"],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate,
        )
        if persistent:
            dl_kwargs["persistent_workers"] = True
        if prefetch is not None:
            dl_kwargs["prefetch_factor"] = prefetch
        val_loader = DataLoader(val_ds, **dl_kwargs)
        return train_loader, val_loader

    # Non-streaming path (materializes data) - keep for compatibility, but note memory use
    print("Starting training data loading...")
    train_texts = load_text_dataset(
        name=cfg["data"]["dataset"],
        config_name=cfg["data"].get("config", None),
        split=train_split,
        text_field=cfg["data"]["text_field"],
        streaming=False,
        max_shards=max_shards,
        verbose=True,
    )
    train_docs = cfg["data"].get("train_docs") or 900000
    print(f"Tokenizing {train_docs} training documents (non-streaming)...")
    train_token_seqs = build_token_sequences(
        tokenizer,
        train_texts,
        max_docs=train_docs,
        seq_len=seq_len,
        max_doc_len=max_doc_len,
    )
    train_ds = PackedLMDataset(train_token_seqs, seq_len=seq_len)
    print(f"Created training dataset with {len(train_ds)} sequences")

    dl_kwargs = dict(
        batch_size=cfg["training"]["micro_batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate,
    )
    if persistent:
        dl_kwargs["persistent_workers"] = True
    if prefetch is not None:
        dl_kwargs["prefetch_factor"] = prefetch
    train_loader = DataLoader(train_ds, **dl_kwargs)

    # Validation data
    print("Starting validation data loading...")
    val_texts = load_text_dataset(
        name=cfg["data"]["dataset"],
        config_name=cfg["data"].get("config", None),
        split=val_split,
        text_field=cfg["data"]["text_field"],
        streaming=False,
        max_shards=max_shards,
        verbose=True,
    )
    val_docs = cfg["data"].get("val_docs") or 100000
    if val_split == train_split:
        val_texts = itertools.islice(val_texts, train_docs, None)
    print(f"Tokenizing {val_docs} validation documents (non-streaming)...")
    val_token_seqs = build_token_sequences(
        tokenizer,
        val_texts,
        max_docs=val_docs,
        seq_len=seq_len,
        max_doc_len=max_doc_len,
    )
    val_ds = PackedLMDataset(val_token_seqs, seq_len=seq_len)
    print(f"Created validation dataset with {len(val_ds)} sequences")

    dl_kwargs = dict(
        batch_size=cfg["training"]["micro_batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate,
    )
    if persistent:
        dl_kwargs["persistent_workers"] = True
    if prefetch is not None:
        dl_kwargs["prefetch_factor"] = prefetch
    val_loader = DataLoader(val_ds, **dl_kwargs)

    return train_loader, val_loader


def get_tokenizer(cfg: Dict[str, Any]):
    """Loads and configures the tokenizer."""
    from transformers import AutoTokenizer

    tok_name = cfg["data"]["tokenizer_name"]
    tok = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token is not None else tok.unk_token
    try:
        tok.model_max_length = int(cfg["training"]["seq_len"])
    except Exception:
        tok.model_max_length = 1024
    return tok
