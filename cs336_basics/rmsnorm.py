import torch
from einops import einsum, reduce


class RMSNorm(torch.nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)
        # rms = (x ** 2).sum(dim=-1)
        rms = (reduce(x**2, "... d_model -> ... 1", "sum") / self.d_model + self.eps) ** 0.5
        res = (x / rms) * self.weight
        return res.to(in_type)
