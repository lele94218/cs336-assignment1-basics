import argparse
import torch
import numpy as np


from cs336_basics import (
    bpe,
    tokenizer,
    linear,
    embedding,
    rmsnorm,
    ffn,
    rope,
    attention,
    transformer,
    loss,
    optimizer,
    data,
    checkpoint,
)
from dataclasses import dataclass


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--val-data", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-lr", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)

    return parser.parse_args()


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


def main():
    args = parse_args()
    config = GPTConfig()
    model = transformer.TransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.block_size,
        d_model=config.n_embd,
        num_layers=config.n_layer,
        num_heads=config.n_head,
        d_ff=config.n_embd * 8 // 3,
        rope_theta=10000.0,
    )
    optimizer = optimizer.AdamW(
        params=model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay, betas=(0.9, 0.95), eps=1e-8
    )

    dataset = np.memmap(args.train_data, dtype=np.uint16, mode="r")
