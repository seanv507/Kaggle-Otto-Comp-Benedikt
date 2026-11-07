"""Benny's GRU next-item model (01d_FE_GRU/train_gru.py) in plain PyTorch.

    item embedding (width w) | type embedding (8) | rank embedding (8)
    -> Linear(w + 16, w) -> GRU(w, w, layers) -> the hidden state of the last input
    -> scores against the item embeddings

Training is a sampled softmax over the batch's targets plus neg_factor * batch
random items, at a temperature, with label smoothing. Candidates that are items of
the row's input other than its target get logit 0 (Benny's mask, as in the
transformer's remove_false_neg). With use_pretrained, the item embeddings start from
the fold's Word2Vec vectors (width 50) and stay frozen.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

PAD = 0
FIRST_ITEM = 2


@dataclass(frozen=True)
class Config:
    """One line of Benny's start.sh."""

    target: str
    use_pretrained: bool
    batch_size: int
    lr: float
    neg_factor: int
    label_smoothing: float
    temperature: float
    width: int
    layers: int
    dropout: float

    @property
    def name(self) -> str:
        """Benny's column: gru_<T|F>_<cl|ca|or>."""
        return f"gru_{'T' if self.use_pretrained else 'F'}_{self.target[:2]}"


CONFIGS = [
    Config("clicks", False, 2048, 0.001, 2, 0.0, 1.0, 128, 3, 0.001),
    Config("clicks", True, 2048, 0.001, 4, 0.0, 0.5310532915929775, 50, 2, 0.001),
    Config("carts", False, 2048, 0.001, 4, 0.0, 1.0, 256, 1, 0.001),
    Config(
        "carts",
        True,
        2048,
        0.001309877857990581,
        4,
        0.0,
        1.0,
        50,
        1,
        0.2374017172678567,
    ),
    Config(
        "orders",
        False,
        1024,
        0.000569116341404674,
        0,
        0.16475386028657832,
        1.0,
        256,
        2,
        0.001,
    ),
    Config("orders", True, 2048, 0.0018792499869370312, 4, 0.0, 1.0, 50, 2, 0.001),
]


def false_negative_mask(
    inputs: torch.Tensor, labels: torch.Tensor, candidates: torch.Tensor
) -> torch.Tensor:
    """(n, n_candidates) bool: candidates that are in the row's input but aren't its label.

    `candidates` must be sorted and unique.
    """
    pos = torch.searchsorted(candidates, inputs).clamp(max=len(candidates) - 1)
    hit = (candidates[pos] == inputs) & (inputs != PAD)
    rows = (
        torch.arange(len(inputs), device=inputs.device).unsqueeze(1).expand_as(inputs)
    )
    mask = torch.zeros(
        len(inputs), len(candidates), dtype=torch.bool, device=inputs.device
    )
    mask[rows[hit], pos[hit]] = True
    return mask & (candidates.unsqueeze(0) != labels.unsqueeze(1))


class GRU(nn.Module):
    def __init__(
        self, n_items: int, config: Config, pretrained: np.ndarray | None = None
    ):
        super().__init__()
        self.config = config
        w = config.width
        self.item_emb = nn.Embedding(n_items, w)  # N(0, 1), as Benny's (no padding_idx)
        with torch.no_grad():
            if pretrained is not None:
                self.item_emb.weight.copy_(torch.from_numpy(pretrained))
            self.item_emb.weight[PAD].zero_()
        self.item_emb.weight.requires_grad = pretrained is None
        self.type_emb = nn.Embedding(10, 8)
        self.rank_emb = nn.Embedding(50, 8)
        for e in (self.type_emb, self.rank_emb):
            with torch.no_grad():
                e.weight[PAD].zero_()
        self.inp = nn.Linear(w + 16, w)
        self.rnn = nn.GRU(
            w, w, num_layers=config.layers, dropout=config.dropout, batch_first=True
        )

    def encode(
        self,
        item: torch.Tensor,
        type_: torch.Tensor,
        rank: torch.Tensor,
        length: torch.Tensor,
    ) -> torch.Tensor:
        """The hidden state at each row's last input (padded on the right)."""
        x = torch.cat(
            [self.item_emb(item), self.type_emb(type_), self.rank_emb(rank)], dim=2
        )
        out, _ = self.rnn(self.inp(x))
        return out[torch.arange(len(item), device=item.device), length - 1]

    def loss(
        self,
        item,
        type_,
        rank,
        length,
        target,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        c = self.config
        h = self.encode(item, type_, rank, length)
        if c.neg_factor > 0:
            negatives = torch.randint(
                FIRST_ITEM,
                self.item_emb.num_embeddings,
                (c.neg_factor * len(target),),
                device=h.device,
                generator=generator,
            )
            candidates = torch.unique(torch.cat([target, negatives]))
        else:
            candidates = torch.unique(target)
        logits = h @ self.item_emb.weight[candidates].T
        logits = logits.masked_fill(false_negative_mask(item, target, candidates), 0.0)
        y = torch.searchsorted(candidates, target)
        return F.cross_entropy(
            logits / c.temperature, y, label_smoothing=c.label_smoothing
        )

    @torch.no_grad()
    def predict(
        self, item, type_, rank, length, k: int = 50
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Top-k item ids (padding and the unused id 1 excluded) and their logits."""
        logits = self.encode(item, type_, rank, length) @ self.item_emb.weight.T
        logits[:, :FIRST_ITEM] = -torch.inf
        return logits.topk(k, dim=1)
