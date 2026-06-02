"""Unit tests for the Two-Tower model architecture."""
import pytest
import torch

from src.two_tower import TwoTowerModel


@pytest.fixture
def small_model():
    return TwoTowerModel(n_users=10, n_items=20, embedding_dim=16)


class TestTwoTowerModel:
    def test_output_shape(self, small_model):
        users = torch.tensor([0, 1, 2])
        pos   = torch.tensor([3, 5, 7])
        neg   = torch.tensor([4, 6, 8])
        loss  = small_model(users, pos, neg)
        assert loss.shape == torch.Size([])  # scalar

    def test_loss_positive(self, small_model):
        users = torch.tensor([0, 1, 2])
        pos   = torch.tensor([3, 5, 7])
        neg   = torch.tensor([4, 6, 8])
        loss  = small_model(users, pos, neg)
        assert float(loss) > 0

    def test_encode_users_normalized(self, small_model):
        users = torch.tensor([0, 1, 2])
        emb   = small_model.encode_users(users)
        norms = torch.norm(emb, dim=-1)
        assert torch.allclose(norms, torch.ones(3), atol=1e-5)

    def test_encode_items_normalized(self, small_model):
        items = torch.tensor([0, 5, 10])
        emb   = small_model.encode_items(items)
        norms = torch.norm(emb, dim=-1)
        assert torch.allclose(norms, torch.ones(3), atol=1e-5)

    def test_get_all_item_embeddings_shape(self, small_model):
        embs = small_model.get_all_item_embeddings()
        assert embs.shape == (20, 8)  # embedding_dim // 2 = 8

    def test_get_user_embedding_shape(self, small_model):
        emb = small_model.get_user_embedding(0)
        assert emb.shape == (8,)  # embedding_dim // 2

    def test_dot_product_scores(self, small_model):
        """User-item dot product should be in [-1, 1] after L2 normalization."""
        u_emb = torch.tensor(small_model.get_user_embedding(0))
        i_embs = torch.tensor(small_model.get_all_item_embeddings())
        scores = i_embs @ u_emb
        assert scores.min() >= -1.01
        assert scores.max() <=  1.01
