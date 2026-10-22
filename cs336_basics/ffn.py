import torch
from einops import einsum
from cs336_basics import linear


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.w1 = linear.Linear(d_ff, d_model, device, dtype)
        self.w2 = linear.Linear(d_model, d_ff, device, dtype)
        self.w3 = linear.Linear(d_ff, d_model, device, dtype)

    @staticmethod
    def silu(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.silu(einsum(self.w1.weight, x, "d_ff d_model, ... d_model -> ... d_ff"))
        v = einsum(self.w3.weight, x, "d_ff d_model, ... d_model -> ... d_ff")
        return einsum(self.w2.weight, u * v, "d_model d_ff, ... d_ff -> ... d_model")
