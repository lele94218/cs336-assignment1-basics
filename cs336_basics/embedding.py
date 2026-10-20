import torch
import torch.nn.functional as F
from einops import einsum


class Embedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        emb = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        self.weight = torch.nn.Parameter(emb)
        std = 1
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        onehot = F.one_hot(token_ids, num_classes=self.weight.shape[0])
        onehot = onehot.to(self.weight.dtype)
        return einsum(onehot, self.weight, "... num_embeddings, num_embeddings embedding_dim -> ... embedding_dim")
        """
        return self.weight[token_ids]
