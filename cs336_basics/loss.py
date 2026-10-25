import torch


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    max_v = torch.max(logits, dim=-1, keepdim=True).values
    z = logits - max_v
    exp_v = torch.exp(z)
    target_scores = z.gather(dim=-1, index=targets.unsqueeze(-1))
    loss_v = torch.log(torch.sum(exp_v, dim=-1, keepdim=True)) - target_scores
    return loss_v.mean()
