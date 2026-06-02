"""
Two-Tower Neural Retrieval Model (PyTorch).

Architecture:
  User tower: Embedding(user_id) -> Linear -> ReLU -> Linear -> L2-normalize
  Item tower: Embedding(item_id) -> Linear -> ReLU -> Linear -> L2-normalize

Training:
  BPR (Bayesian Personalized Ranking) loss with in-batch negatives.
  For each (user, pos_item) pair, negative items are drawn randomly
  from items not in the user's training history.

Retrieval:
  Score = dot(user_emb, item_emb) (= cosine similarity after L2 normalization).
  Top-K items retrieved via matrix multiply against all item embeddings.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.config import BATCH_SIZE, EMBEDDING_DIM, EPOCHS, LEARNING_RATE, N_NEGATIVES, SEED


class TwoTowerModel(nn.Module):
    def __init__(self, n_users: int, n_items: int, embedding_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embedding_dim, sparse=False)
        self.item_emb = nn.Embedding(n_items, embedding_dim, sparse=False)

        self.user_tower = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim // 2),
        )
        self.item_tower = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim // 2),
        )
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def encode_users(self, user_ids: torch.Tensor) -> torch.Tensor:
        x = self.user_emb(user_ids)
        x = self.user_tower(x)
        return F.normalize(x, dim=-1)

    def encode_items(self, item_ids: torch.Tensor) -> torch.Tensor:
        x = self.item_emb(item_ids)
        x = self.item_tower(x)
        return F.normalize(x, dim=-1)

    def forward(self, user_ids: torch.Tensor, pos_ids: torch.Tensor,
                neg_ids: torch.Tensor) -> torch.Tensor:
        u   = self.encode_users(user_ids)          # (B, D)
        pos = self.encode_items(pos_ids)           # (B, D)
        neg = self.encode_items(neg_ids)           # (B, D)

        pos_score = (u * pos).sum(dim=-1)          # (B,)
        neg_score = (u * neg).sum(dim=-1)          # (B,)
        loss = -F.logsigmoid(pos_score - neg_score).mean()
        return loss

    @torch.no_grad()
    def get_all_item_embeddings(self, device: str = "cpu") -> np.ndarray:
        """Compute embeddings for all items — used to build retrieval index."""
        n = self.item_emb.num_embeddings
        ids = torch.arange(n, device=device)
        emb = self.encode_items(ids)
        return emb.cpu().numpy()

    @torch.no_grad()
    def get_user_embedding(self, user_idx: int, device: str = "cpu") -> np.ndarray:
        uid = torch.tensor([user_idx], device=device)
        return self.encode_users(uid).cpu().numpy()[0]


class InteractionDataset(Dataset):
    def __init__(self, user_idxs: np.ndarray, item_idxs: np.ndarray,
                 n_items: int, train_history: dict[int, set[int]],
                 n_negatives: int = N_NEGATIVES):
        self.users     = user_idxs
        self.items     = item_idxs
        self.n_items   = n_items
        self.history   = train_history
        self.n_neg     = n_negatives
        self.rng       = np.random.default_rng(SEED)

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int):
        user = self.users[idx]
        pos  = self.items[idx]
        # uniform random negative (retry if in history)
        seen = self.history.get(user, set())
        neg  = int(self.rng.integers(0, self.n_items))
        for _ in range(10):
            if neg not in seen:
                break
            neg = int(self.rng.integers(0, self.n_items))
        return user, pos, neg


def train_two_tower(
    train_df,
    n_users: int,
    n_items: int,
    train_history: dict[int, set[int]],
    device: str | None = None,
) -> TwoTowerModel:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training Two-Tower on {device} ...")

    torch.manual_seed(SEED)
    model = TwoTowerModel(n_users, n_items).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    dataset = InteractionDataset(
        train_df["user_idx"].values,
        train_df["item_idx"].values,
        n_items, train_history,
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, pin_memory=(device == "cuda"))

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()
        for users, pos, neg in loader:
            users = users.to(device)
            pos   = pos.to(device)
            neg   = neg.to(device)
            loss  = model(users, pos, neg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg = total_loss / len(loader)
        print(f"  Epoch {epoch:2d}/{EPOCHS} | loss={avg:.4f} | {time.time()-t0:.1f}s")

    return model
