import torch
from einops import einsum, rearrange


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_v = torch.max(x, dim=dim, keepdim=True).values
    exp_v = torch.exp(x - max_v)
    return exp_v / torch.sum(exp_v, dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    dk = Q.shape[-1]
    scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / (dk**0.5)
    scores = scores.masked_fill(~mask, float("-inf"))
    out = einsum(softmax(scores, -1), V, "... query key, ... key d_v -> ... query d_v")
    return out
