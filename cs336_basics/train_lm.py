import argparse
import time
import torch

import numpy as np
import numpy.typing as npt

from cs336_basics.config import GPTConfig, config_args
from cs336_basics import (
    transformer,
    loss,
    optimizer,
    data,
    checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--val-data", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-lr", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=100)

    return parser.parse_args()


def evaluate(
    model: torch.nn.Module, dataset: npt.NDArray, batch_size: int, context_length: int, device: str, num_batches=10
):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = data.get_batch(dataset, batch_size, context_length, device)
            logits = model(x)
            loss_value = loss.cross_entropy(logits, y)
            total_loss += loss_value.item()

    model.train()
    return total_loss / num_batches


def synchronize_device(device: str) -> None:
    device_type = torch.device(device).type
    if device_type == "mps":
        torch.mps.synchronize()
    elif device_type == "cuda":
        torch.cuda.synchronize(device)


def main():
    args = parse_args()
    config = GPTConfig(**config_args["tiny-stories"])
    model = transformer.TransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.block_size,
        d_model=config.n_embd,
        num_layers=config.n_layer,
        num_heads=config.n_head,
        d_ff=config.d_ff,
        rope_theta=10000.0,
    )
    model = model.to(args.device)
    optim = optimizer.AdamW(
        params=model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay, betas=(0.9, 0.95), eps=1e-8
    )

    dataset = np.load(args.train_data, mmap_mode="r")
    val_dataset = np.load(args.val_data, mmap_mode="r")

    start_step = 0
    model.train()
    synchronize_device(args.device)
    run_start = time.perf_counter()

    for step in range(start_step, args.max_steps):
        synchronize_device(args.device)
        step_start = time.perf_counter()
        # 3. 计算本步 lr，设置到 optimizer.param_groups
        lr = optimizer.get_lr_cosine_schedule(
            it=step,
            max_learning_rate=args.max_lr,
            min_learning_rate=args.max_lr * 0.1,
            warmup_iters=10,
            cosine_cycle_iters=args.max_steps,
        )
        for group in optim.param_groups:
            group["lr"] = lr

        # 4. get_batch 获取 x、y
        x, y = data.get_batch(dataset, args.batch_size, config.block_size, args.device)

        # 5. zero_grad → forward → loss → backward
        optim.zero_grad()
        logits = model(x)
        loss_value = loss.cross_entropy(logits, y)
        loss_value.backward()

        # 6. 梯度裁剪 → optimizer.step()
        optimizer.gradient_clipping(parameters=model.parameters(), max_l2_norm=1.0)
        optim.step()
        synchronize_device(args.device)
        step_seconds = time.perf_counter() - step_start
        tokens_per_second = x.numel() / step_seconds
        elapsed_seconds = time.perf_counter() - run_start

        # 7. 定期打印 step、loss.item()、lr
        # 8. 定期调用 evaluate
        # 9. 定期保存 checkpoint
        # 建议 iteration 存已完成更新次数，即 step + 1
        completed_steps = step + 1
        print(
            f"step={completed_steps}, train_loss={loss_value.item():.4f}, lr={lr:.6g}, "
            f"step_time={step_seconds:.3f}s, tok/s={tokens_per_second:,.0f}, "
            f"elapsed={elapsed_seconds:.1f}s",
            flush=True,
        )
        if completed_steps % 100 == 0:
            val_loss = evaluate(model, val_dataset, args.batch_size, config.block_size, args.device, num_batches=10)
            print(f"step={completed_steps}, train_loss={loss_value.item():.4f}, val_loss={val_loss:.4f}, lr={lr:.6g}")

        if completed_steps % 1000 == 0:
            checkpoint.save_chekpoint(
                model=model,
                optimizer=optim,
                iteration=completed_steps,
                out=f"checkpoint_{completed_steps}.pt",
            )

    # 10. 保存最终 checkpoint
    checkpoint.save_chekpoint(
        model=model,
        optimizer=optim,
        iteration=completed_steps,
        out="checkpoint_final.pt",
    )


if __name__ == "__main__":
    main()
