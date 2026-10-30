from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 10000  # 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    d_ff: int = 2048


config_args = {
    "tiny-stories": dict(
        block_size=256,
        vocab_size=10000,
        n_layer=4,
        n_head=16,
        n_embd=512,
        d_ff=1344,
    ),
    "gpt2": dict(
        block_size=1024,
        vocab_size=10000,
        n_layer=12,
        n_head=12,
        n_embd=768,
        d_ff=2048,
    ),
    "gpt2-medium": dict(
        block_size=1024,
        vocab_size=10000,
        n_layer=24,
        n_head=16,
        n_embd=1024,
        d_ff=2752,
    ),
    "gpt2-large": dict(
        block_size=1024,
        vocab_size=10000,
        n_layer=36,
        n_head=20,
        n_embd=1280,
        d_ff=3392,
    ),
    "gpt2-xl": dict(
        block_size=1024,
        vocab_size=10000,
        n_layer=48,
        n_head=25,
        n_embd=1600,
        d_ff=4288,
    ),
}
