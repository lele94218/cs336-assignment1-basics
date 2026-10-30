import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--vocab", type=str, default="data/vocab.pkl")
    parser.add_argument("--merges", type=str, default="data/merges.pkl")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)

    return parser.parse_args()


def load_model(checkpoint_path: str, device: str) -> torch.nn.Module:
    pass


def generate(
    model,
    tokenizer,
    prompt: str,
    context_length: int,
    device: str,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> str:
    pass


def main():
    args = parse_args()


if __name__ == "__main__":
    main()
