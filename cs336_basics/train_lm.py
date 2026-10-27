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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--val-data", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)

    return parser.parse_args()


def main():
    args = parse_args()

    model = transformer.TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)

    dataset = np.memmap(args.train_data, dtype=np.uint16, mode="r")
