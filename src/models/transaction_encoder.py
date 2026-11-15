import torch
import torch.nn as nn

from src.data.sequences import MASK_ID, N_CATEGORIES, N_TIME_BUCKETS, PAD_ID

# The masked-token head only predicts real categories, so PAD and MASK drop out.
N_TARGETS = N_CATEGORIES - 2


class TransactionEncoder(nn.Module):
    def __init__(self, embedding_dim=32, attention_heads=4, layers=2, max_length=128):
        super().__init__()
        self.category = nn.Embedding(N_CATEGORIES, embedding_dim, padding_idx=PAD_ID)
        self.amount = nn.Linear(1, embedding_dim)
        self.relative_amount = nn.Linear(1, embedding_dim)
        self.time = nn.Embedding(N_TIME_BUCKETS, embedding_dim)
        self.direction = nn.Embedding(2, embedding_dim)
        self.position = nn.Embedding(max_length, embedding_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=attention_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(embedding_dim)
        self.category_head = nn.Linear(embedding_dim, N_TARGETS)

    def forward(self, category, amount, relative_amount, time, direction, padding):
        batch, length = category.shape
        pos = torch.arange(length, device=category.device).unsqueeze(0).expand(batch, -1)
        x = (
            self.category(category)
            + self.amount(amount.unsqueeze(-1))
            + self.relative_amount(relative_amount.unsqueeze(-1))
            + self.time(time)
            + self.direction(direction)
            + self.position(pos)
        )
        x = self.encoder(x, src_key_padding_mask=padding)
        x = self.norm(x)

        # Mean-pool over real transactions only, padding must not dilute it.
        keep = (~padding).unsqueeze(-1)
        pooled = (x * keep).sum(1) / keep.sum(1).clamp(min=1)
        return {"embedding": pooled, "category_logits": self.category_head(x)}


def masked_category_loss(model, batch, mask_probability=0.15, device="cpu"):
    category = batch["category"].to(device)
    amount = batch["amount"].to(device)
    relative_amount = batch["relative_amount"].to(device)
    time = batch["time"].to(device)
    direction = batch["direction"].to(device)
    padding = batch["padding"].to(device)

    targets = category.clone()
    eligible = (~padding) & (category < MASK_ID)
    mask = (torch.rand(category.shape, device=device) < mask_probability) & eligible

    if not mask.any():
        # Cross-entropy on an empty selection returns NaN, so force one token.
        candidates = torch.nonzero(eligible)
        if len(candidates) == 0:
            return None
        row, col = candidates[torch.randint(len(candidates), (1,), device=device)[0]]
        mask[row, col] = True

    masked = category.clone()
    masked[mask] = MASK_ID

    out = model(masked, amount, relative_amount, time, direction, padding)
    return nn.functional.cross_entropy(out["category_logits"][mask], targets[mask])
