"""Deep-learning models (paper Section 3.4), re-implemented in PyTorch.

CNN-1D : Embedding(50) -> Conv1d(128, k=3) + ReLU -> MaxPool(3) -> global max -> Dense 256 -> Dropout 0.2
LSTM   : Embedding(50) -> LSTM(100) -> Dense 256 -> Dropout 0.2
BiGRU  : Embedding(50) -> bidirectional GRU(32) -> Dense 128 -> Dropout 0.2
All use Adam, 10 epochs and hold out 10% of the training data for validation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import yaml

from .data import LABELS, load_data, split
from .metrics import evaluate, save, summary_line

PAD, UNK = 0, 1


def build_vocab(texts: List[str], max_size: int) -> Dict[str, int]:
    counts = Counter(t for s in texts for t in s.split())
    vocab = {"<pad>": PAD, "<unk>": UNK}
    for w, _ in counts.most_common(max_size - 2):
        vocab[w] = len(vocab)
    return vocab


def encode(texts: List[str], vocab: Dict[str, int], max_len: int) -> torch.Tensor:
    X = np.zeros((len(texts), max_len), dtype="int64")
    for i, s in enumerate(texts):
        ids = [vocab.get(t, UNK) for t in s.split()[:max_len]]
        X[i, :len(ids)] = ids
    return torch.from_numpy(X)


class CNN1D(nn.Module):
    def __init__(self, vocab_size, c):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, c["embedding_dim"], padding_idx=PAD)
        self.conv = nn.Conv1d(c["embedding_dim"], c["filters"], c["kernel_size"], padding=c["kernel_size"] // 2)
        self.pool = nn.MaxPool1d(c["pool_size"], ceil_mode=True)
        self.head = nn.Sequential(nn.Linear(c["filters"], c["dense"]), nn.ReLU(), nn.Dropout(c["dropout"]),
                                  nn.Linear(c["dense"], 2))

    def forward(self, x):
        h = self.pool(torch.relu(self.conv(self.emb(x).transpose(1, 2))))
        return self.head(h.max(-1).values)


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, c, cell: str, bidirectional: bool):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, c["embedding_dim"], padding_idx=PAD)
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(c["embedding_dim"], c["units"], batch_first=True, bidirectional=bidirectional)
        out = c["units"] * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.Linear(out, c["dense"]), nn.ReLU(), nn.Dropout(c["dropout"]),
                                  nn.Linear(c["dense"], 2))

    def forward(self, x):
        lengths = (x != PAD).sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(self.emb(x), lengths, batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)
        h = h[0] if isinstance(h, tuple) else h  # LSTM returns (h, c)
        last = torch.cat([h[-2], h[-1]], -1) if self.rnn.bidirectional else h[-1]
        return self.head(last)


def build_model(name: str, vocab_size: int, c: dict) -> nn.Module:
    if name == "cnn":
        return CNN1D(vocab_size, c)
    if name == "lstm":
        return RNNClassifier(vocab_size, c, "lstm", bidirectional=False)
    if name == "bigru":
        return RNNClassifier(vocab_size, c, "gru", bidirectional=True)
    raise ValueError(name)


@torch.no_grad()
def predict(model, X, device, bs=256):
    model.eval()
    return torch.cat([model(X[i:i + bs].to(device)).argmax(-1).cpu() for i in range(0, len(X), bs)]).numpy()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", choices=["cnn", "lstm", "bigru"], default="bigru")
    ap.add_argument("--config", default="configs/dl.yaml")
    ap.add_argument("--text-col", default="tweet")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/dl")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    c = {**cfg["common"], **cfg[args.model]}
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train, test = split(load_data(args.data, args.text_col, args.label_col), cfg["test_size"], args.seed)
    train, val = split(train, c["validation_split"], args.seed)
    vocab = build_vocab(train["clean"].tolist(), c["vocab_size"])
    enc = lambda df: encode(df["clean"].tolist(), vocab, c["max_len"])  # noqa: E731
    Xtr, ytr = enc(train), torch.as_tensor(train["y"].values)
    Xva, Xte = enc(val), enc(test)

    model = build_model(args.model, len(vocab), c).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(c["learning_rate"]))
    loss_fn = nn.CrossEntropyLoss()
    bs = c["batch_size"]
    for epoch in range(1, c["epochs"] + 1):
        model.train()
        perm, total = torch.randperm(len(Xtr)), 0.0
        for i in range(0, len(Xtr), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx].to(device)), ytr[idx].to(device))
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        val_acc = evaluate(val["y"], predict(model, Xva, device))["accuracy"]
        print(f"epoch {epoch:>2}  loss {total / len(Xtr):.4f}  val acc {val_acc:.4f}")

    pred = predict(model, Xte, device)
    res = evaluate(test["y"], pred)
    print(f"\n{args.model} test: {summary_line(res)}")
    out = Path(args.out)
    save(res, out / f"{args.model}_metrics.json")
    test.assign(prediction=[LABELS[i] for i in pred]).to_csv(out / f"{args.model}_predictions.csv", index=False)


if __name__ == "__main__":
    main()
