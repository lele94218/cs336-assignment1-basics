import torch

from cs336_basics import rmsnorm, rope, ffn, attention


class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        super().__init__()
        print(d_model, num_heads, d_ff, max_seq_len, theta)
        self.ln1 = rmsnorm.RMSNorm(d_model)
        self.ln2 = rmsnorm.RMSNorm(d_model)
        rp = rope.RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len)
        self.attn = attention.MultiHeadSelfAttention(d_model, num_heads, rp=rp)
        self.ffn = ffn.SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.attn(self.ln1(x))
        y = h + self.ffn(self.ln2(h))
        return y
