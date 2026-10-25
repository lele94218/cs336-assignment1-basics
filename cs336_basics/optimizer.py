from collections.abc import Callable
import torch
import math


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay: float, betas: tuple[float, float], eps: float):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                state["t"] += 1  # Increment iteration number.
                m = state["m"]
                v = state["v"]
                t = state["t"]  # Get iteration number from the state, or 0.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr * weight_decay * p.data

                m = betas[0] * m + (1 - betas[0]) * grad
                v = betas[1] * v + (1 - betas[1]) * (grad**2)
                lr_t = lr * ((1 - betas[1] ** t) ** 0.5) / (1 - betas[0] ** t)
                p.data -= lr_t * m / (torch.sqrt(v) + eps)  # Update weight tensor in-place.

                state["m"] = m
                state["v"] = v

        return loss
