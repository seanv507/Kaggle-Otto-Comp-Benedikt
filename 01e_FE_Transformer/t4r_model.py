"""Plain-PyTorch version of Benny's Transformers4Rec model (02_T4R_Train-v2.ipynb).

The model that Benny's final parameters build in T4R (using_type=False,
multi_task_emb=0, item_correction=False, loss_types='Simple'):

    item embedding (padding_idx 0)
    -> masked positions replaced by a learned mask embedding
    -> Linear(d_model, d_model), no activation
    -> XLNet (bidirectional, relative attention)
    -> scores against the item embeddings (weight tying, no bias)

Training is masked-item prediction (T4R's MaskedLanguageModeling). The loss is a
cross-entropy over the masked items of the batch plus `neg_factor` random items per
masked item (Benny's custom_t4r.py, deleted in commit 1a76a2e; see
`git show 83e570e:01e_FE_Transformer/custom_t4r.py`). To predict the next item, a
mask is put after the last item, and the hidden state there is scored against all
items.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from t4r_data import FIRST_ITEM, PAD
from torch import nn
from transformers import XLNetConfig, XLNetModel


@dataclass
class Params:
    """Benny's final `params` from 02_T4R_Train-v2.ipynb, minus the switched-off options."""

    d_model: int = 256
    n_head: int = 32
    n_layer: int = 3
    dropout: float = 0.05096800263401626
    mlm_probability: float = 0.35044384745899415
    neg_factor: int = 4
    temperature: float = 1.5734215681668653
    remove_false_neg: bool = True
    seq_len: int = 20
    batch_size: int = 1024
    lr: float = 0.0005
    cosine_cycles: float = 1.5  # T4R's learning_rate_num_cosine_cycles_by_epoch


@torch.no_grad()
def mlm_targets(items: torch.Tensor, probability: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """Items to predict in each sequence; PAD elsewhere (T4R's MaskedLanguageModeling).

    Each item is masked with `probability`; each sequence gets at least one masked item,
    and a sequence with every item masked gets one unmasked again. So a sequence of
    length 1 has no target.
    """
    rows = torch.arange(items.shape[0], device=items.device)
    real = items != PAD
    masked = (torch.rand(items.shape, generator=generator, device=items.device) < probability) & real
    # at least one per sequence
    one = torch.multinomial(real.float(), 1, generator=generator).squeeze(1)
    masked[rows, one] = True
    # but not all of them
    all_masked = masked.sum(1) == real.sum(1)
    unmask = torch.multinomial(masked.float(), 1, generator=generator).squeeze(1)
    masked[rows[all_masked], unmask[all_masked]] = False
    return torch.where(masked, items, PAD)


def false_negative_mask(sequences: torch.Tensor, labels: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    """(n, n_candidates) bool: candidates that are in the row's sequence but aren't its label.

    `candidates` must be sorted and unique (torch.unique's output). Benny's
    remove_false_neg: items the session interacted with shouldn't count as negatives.
    """
    pos = torch.searchsorted(candidates, sequences).clamp(max=len(candidates) - 1)
    hit = (candidates[pos] == sequences) & (sequences != PAD)
    rows = torch.arange(len(sequences), device=sequences.device).unsqueeze(1).expand_as(sequences)
    mask = torch.zeros(len(sequences), len(candidates), dtype=torch.bool, device=sequences.device)
    mask[rows[hit], pos[hit]] = True
    return mask & (candidates.unsqueeze(0) != labels.unsqueeze(1))


class SessionTransformer(nn.Module):
    def __init__(self, n_items: int, params: Params):
        super().__init__()
        self.n_items = n_items
        self.params = params
        d = params.d_model
        self.item_embedding = nn.Embedding(n_items, d, padding_idx=PAD)
        nn.init.normal_(self.item_embedding.weight, std=0.05)  # T4R's default table initializer
        with torch.no_grad():
            self.item_embedding.weight[PAD].zero_()
        self.mask_embedding = nn.Parameter(torch.randn(d) * 0.001)  # as T4R's masked_item_embedding
        self.dense = nn.Linear(d, d)
        # T4R's XLNetConfig.build
        config = XLNetConfig(
            d_model=d,
            d_inner=4 * d,
            n_layer=params.n_layer,
            n_head=params.n_head,
            ff_activation="gelu",
            untie_r=True,
            bi_data=False,
            attn_type="bi",
            layer_norm_eps=0.03,
            dropout=params.dropout,
            initializer_range=0.02,
            vocab_size=1,
            pad_token_id=0,
            bos_token_id=None,
            eos_token_id=None,
            mem_len=None,
            use_mems_train=False,
            use_mems_eval=False,
        )
        self.transformer = XLNetModel(config)

    def encode(self, items: torch.Tensor, masked: torch.Tensor) -> torch.Tensor:
        """Hidden states (batch, length, d_model); `masked` positions get the mask embedding.

        As in T4R, no attention mask is passed, so padding positions are attended to.
        """
        x = self.item_embedding(items)
        x = torch.where(masked.unsqueeze(-1), self.mask_embedding.to(x.dtype), x)
        x = self.dense(x)
        return self.transformer(inputs_embeds=x).last_hidden_state

    def loss(self, items: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        """Masked-item loss for a batch of padded sequences (batch, seq_len)."""
        p = self.params
        labels = mlm_targets(items, p.mlm_probability, generator)
        target_positions = labels != PAD
        hidden = self.encode(items, target_positions)[target_positions]
        y = labels[target_positions]
        # candidates: the batch's targets plus random items; Benny drew from 1..1825323,
        # a hard-coded vocabulary size from an earlier run
        negatives = torch.randint(
            FIRST_ITEM, self.n_items, (p.neg_factor * len(y),), device=items.device, generator=generator
        )
        candidates = torch.unique(torch.cat([y, negatives]))
        logits = hidden @ self.item_embedding.weight[candidates].T
        if p.remove_false_neg:
            rows = torch.arange(len(items), device=items.device).unsqueeze(1).expand_as(items)
            false_neg = false_negative_mask(items[rows[target_positions]], y, candidates)
            # Benny multiplied these logits by 0 rather than dropping them
            logits = logits.masked_fill(false_neg, 0.0)
        target = torch.searchsorted(candidates, y)
        return F.cross_entropy(logits / p.temperature, target)

    @torch.no_grad()
    def predict(self, items: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Top-k next items (batch, k) and their scores, for padded sequences (batch, length).

        One column is added so that a full sequence still has room for the mask. As in
        T4R, every padding position gets the mask embedding, and the prediction is read
        at the first one. Padding and unknown items are never predicted.
        """
        items = F.pad(items, (0, 1), value=PAD)
        n = (items != PAD).sum(1)
        hidden = self.encode(items, items == PAD)
        hidden = hidden[torch.arange(len(items), device=items.device), n]
        logits = hidden @ self.item_embedding.weight.T
        logits[:, :FIRST_ITEM] = -torch.inf
        return logits.topk(k, dim=1)
