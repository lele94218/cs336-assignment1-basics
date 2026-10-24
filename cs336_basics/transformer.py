import torch

from cs336_basics import linear, embedding, rmsnorm, rope, ffn, attention


class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        super().__init__()
        self.ln1 = rmsnorm.RMSNorm(d_model)
        self.ln2 = rmsnorm.RMSNorm(d_model)
        rp = rope.RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len)
        self.attn = attention.MultiHeadSelfAttention(d_model, num_heads, rp=rp)
        self.ffn = ffn.SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.attn(self.ln1(x))
        y = h + self.ffn(self.ln2(h))
        return y


class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()
        self.token_embeddings = embedding.Embedding(vocab_size, d_model)
        self.layers = torch.nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta) for _ in range(num_layers)]
        )
        self.ln_final = rmsnorm.RMSNorm(d_model)
        self.lm_head = linear.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)
