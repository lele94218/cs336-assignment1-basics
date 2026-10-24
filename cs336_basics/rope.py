import torch
from einops import einsum, rearrange


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        assert d_k % 2 == 0
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        pairs = torch.arange(d_k // 2, device=device, dtype=torch.float32)
        inv_theta = theta ** (-2 * pairs / d_k)
        angles = einsum(positions, inv_theta, "i, k -> i k")
        self.register_buffer("cos_cache", angles.cos(), persistent=False)
        self.register_buffer("sin_cache", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        if token_positions is None:
            token_positions = torch.arange(x.shape[-2], device=x.device, dtype=int)
        token_cos = self.cos_cache[token_positions]
        token_sin = self.sin_cache[token_positions]
        a = x[..., 0::2]
        b = x[..., 1::2]
        a_h = a * token_cos - b * token_sin
        b_h = a * token_sin + b * token_cos

        return rearrange(torch.stack((a_h, b_h), dim=-1), "... p1 p2 -> ... (p1 p2)")
