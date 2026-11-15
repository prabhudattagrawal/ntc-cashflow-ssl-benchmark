from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.models.transaction_encoder import TransactionEncoder, masked_category_loss


class SequenceDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors

    def __len__(self):
        return self.tensors["category"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tensors.items()}


def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def pretrain(tensors, output_path, embedding_dim=32, heads=4, layers=2,
             epochs=8, batch_size=64, learning_rate=1e-3, seed=42,
             validation_tensors=None, early_stopping_patience=4, max_length=128):
    torch.manual_seed(seed)
    device = select_device()

    model = TransactionEncoder(
        embedding_dim, heads, layers, max_length=max_length
    ).to(device)

    loader = DataLoader(SequenceDataset(tensors), batch_size=batch_size, shuffle=True)
    val_loader = None
    if validation_tensors is not None:
        val_loader = DataLoader(
            SequenceDataset(validation_tensors), batch_size=batch_size, shuffle=False
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    history = []
    best_loss = float("inf")
    best_state = None
    stale_epochs = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = masked_category_loss(model, batch, device=device)
            if loss is None:
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        entry = {
            "epoch": epoch + 1,
            "loss": float(np.mean(train_losses)) if train_losses else float("nan"),
        }

        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    loss = masked_category_loss(model, batch, device=device)
                    if loss is not None:
                        val_losses.append(float(loss.cpu()))
            val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
            entry["validation_loss"] = val_loss

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1

        history.append(entry)

        suffix = ""
        if "validation_loss" in entry:
            suffix = f", validation_loss={entry['validation_loss']:.4f}"
        print(f"pretrain epoch {epoch + 1:02d}/{epochs}: "
              f"loss={entry['loss']:.4f}{suffix}")

        if (val_loader is not None and early_stopping_patience > 0
                and stale_epochs >= early_stopping_patience):
            print(f"early stop at epoch {epoch + 1}, best val loss {best_loss:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model.cpu()
    torch.save({
        "state_dict": model.state_dict(),
        "embedding_dim": embedding_dim,
        "heads": heads,
        "layers": layers,
        "max_length": max_length,
    }, output_path)

    return model, history


@torch.no_grad()
def extract_embeddings(model, tensors, batch_size=128):
    device = select_device()
    model = model.to(device).eval()

    loader = DataLoader(SequenceDataset(tensors), batch_size=batch_size, shuffle=False)
    chunks = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(
            batch["category"],
            batch["amount"],
            batch["relative_amount"],
            batch["time"],
            batch["direction"],
            batch["padding"],
        )
        chunks.append(out["embedding"].cpu().numpy())
    return np.vstack(chunks)
