import torch


def softmax(x: torch.Tensor, dim: int):
    max_v = torch.max(x, dim=dim, keepdim=True).values
    exp_v = torch.exp(x - max_v)
    return exp_v / torch.sum(exp_v, dim=dim, keepdim=True)
