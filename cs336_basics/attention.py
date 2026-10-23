import torch
from einops import einsum, rearrange
from cs336_basics import linear, rope


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_v = torch.max(x, dim=dim, keepdim=True).values
    exp_v = torch.exp(x - max_v)
    return exp_v / torch.sum(exp_v, dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    dk = Q.shape[-1]
    scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / (dk**0.5)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    out = einsum(softmax(scores, -1), V, "... query key, ... key d_v -> ... query d_v")
    return out


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rp: rope.RotaryPositionalEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.rp = rp

        self.q_proj = linear.Linear(d_model, d_model, device, dtype)
        self.k_proj = linear.Linear(d_model, d_model, device, dtype)
        self.v_proj = linear.Linear(d_model, d_model, device, dtype)
        self.o_proj = linear.Linear(d_model, d_model, device, dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        seq_len = x.shape[-2]

        # divide
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = rearrange(q, "... query (num_heads d_head) -> ... num_heads query d_head", num_heads=self.num_heads)
        k = rearrange(k, "... key (num_heads d_head) -> ... num_heads key d_head", num_heads=self.num_heads)
        v = rearrange(v, "... value (num_heads d_head) -> ... num_heads value d_head", num_heads=self.num_heads)
        # rope
        if self.rp is not None and token_positions is not None:
            q = self.rp(q, token_positions)
            k = self.rp(k, token_positions)
        # mask
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))

        # attention
        out = scaled_dot_product_attention(q, k, v, mask)

        # combine
        out = rearrange(out, "... num_heads query d_head -> ... query (num_heads d_head)")
        return self.o_proj(out)
