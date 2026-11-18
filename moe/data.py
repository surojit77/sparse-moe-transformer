"""WikiText-2 (raw) tokenised with the GPT-2 BPE tokenizer, cached as uint16 .npy."""
import os

import numpy as np
import torch

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_cache")


def _load_wikitext2():
    from datasets import load_dataset  # pip install datasets
    # Newer huggingface_hub versions require "namespace/name"; the old "wikitext" alias fails.
    err = None
    for repo in ("Salesforce/wikitext", "wikitext"):
        try:
            ds = load_dataset(repo, "wikitext-2-raw-v1")
            break
        except Exception as e:  # noqa: BLE001
            err = e
    else:
        raise RuntimeError(f"Could not download WikiText-2: {err}")
    return {s: "".join(ds[s]["text"]) for s in ("train", "validation", "test")}


def get_splits(dataset="wikitext2"):
    """Returns dict split -> 1-D LongTensor of token ids, and vocab size."""
    if dataset == "synthetic":           # for smoke tests without internet
        g = torch.Generator().manual_seed(0)
        # Markov-ish synthetic data so loss actually drops
        V = 512
        trans = torch.randint(0, V, (V, 4), generator=g)
        def make(n):
            x = torch.empty(n, dtype=torch.long); x[0] = 0
            r = torch.randint(0, 4, (n,), generator=g)
            for i in range(1, n):
                x[i] = trans[x[i - 1], r[i]]
            return x
        return {"train": make(200_000), "validation": make(20_000), "test": make(20_000)}, V

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    os.makedirs(CACHE, exist_ok=True)
    out = {}
    paths = {s: os.path.join(CACHE, f"wikitext2_{s}.npy") for s in ("train", "validation", "test")}
    if not all(os.path.exists(p) for p in paths.values()):
        texts = _load_wikitext2()
        for s, t in texts.items():
            ids = np.array(enc.encode_ordinary(t), dtype=np.uint16)
            np.save(paths[s], ids)
    for s, p in paths.items():
        out[s] = torch.from_numpy(np.load(p).astype(np.int64))
    return out, enc.n_vocab


def random_batch(data, batch_size, seq_len, device, gen=None):
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,), generator=gen)
    x = torch.stack([data[i:i + seq_len] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + seq_len] for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def eval_batches(data, batch_size, seq_len, device):
    """Non-overlapping windows covering the whole split (deterministic)."""
    n = (len(data) - 1) // seq_len
    x = data[: n * seq_len].view(n, seq_len)
    y = data[1: n * seq_len + 1].view(n, seq_len)
    for i in range(0, n, batch_size):
        yield x[i:i + batch_size].to(device), y[i:i + batch_size].to(device)


def amp_dtype_for(device):
    """bf16 only on GPUs with native support (Ampere+, compute capability >= 8).
    T4/V100 report bf16 as 'supported' but emulate it slowly -> use fp16 there."""
    if not str(device).startswith("cuda"):
        return None
    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16
