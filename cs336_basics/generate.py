"""Command-line interface and skeleton for autoregressive text generation."""

import argparse

import torch

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer import TransformerLM
from cs336_basics.config import GPTConfig, config_args
from cs336_basics.attention import softmax
from cs336_basics import checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained Transformer LM.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--vocab", default="data/vocab.pkl")
    parser.add_argument("--merges", default="data/merges.pkl")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str) -> torch.nn.Module:
    """Create the training architecture, load model weights, and enter eval mode.

    The current checkpoint stores weights under "model", but not model config.
    Use the same configuration as the run that produced this checkpoint.
    No optimizer is needed for generation.
    """
    raise NotImplementedError


def generate(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    context_length: int,
    device: str,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> str:
    """Return the prompt plus its generated completion as text.

    Start with a nonempty prompt and basic sampling (temperature=1, top_p=1).
    Then add temperature scaling and top-p filtering.
    Limit model input to context_length tokens while retaining the full output.
    Stop at <|endoftext|> or max_new_tokens newly generated tokens.
    Run in eval mode without recording gradients.
    """
    model = model.to(device)
    model.eval()

    ids = tokenizer.encode(prompt)
    x = torch.tensor(ids, dtype=torch.long, device=device)
    x = x.unsqueeze(0)
    eos_id = tokenizer.token_to_id[b"<|endoftext|>"]
    with torch.no_grad():
        for _ in range(max_new_tokens):
            x_context = x[:, -context_length:]
            logits = model(x_context)
            last = logits[:, -1, :]
            probs = softmax(last / temperature, dim=-1)

            # top p
            sorted_probs, sorted_ids = torch.sort(probs, dim=-1, descending=True)
            cumul_probs = torch.cumsum(sorted_probs, dim=-1)
            prev_probs = cumul_probs - sorted_probs
            remove_mask = prev_probs >= top_p
            sorted_probs = sorted_probs.masked_fill(remove_mask, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            sample_index = torch.multinomial(sorted_probs, num_samples=1)
            ix = sorted_ids.gather(dim=-1, index=sample_index)
            if ix.item() == eos_id:
                break
            x = torch.cat((x, ix), dim=1)

    return tokenizer.decode(x[0].tolist())


def main() -> None:
    args = parse_args()
    # TODO: Load the tokenizer and model.
    config = GPTConfig(**config_args["tiny-stories"])
    tokenizer = Tokenizer.from_files(args.vocab, args.merges, ["<|endoftext|>"])
    model = TransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.block_size,
        d_model=config.n_embd,
        num_layers=config.n_layer,
        num_heads=config.n_head,
        d_ff=config.d_ff,
        rope_theta=10000.0,
    )
    checkpoint.load_checkpoint(src=args.checkpoint, model=model)
    # TODO: Call generate with args and the training configuration's context length.
    text = generate(
        model,
        tokenizer,
        args.prompt,
        config.block_size,
        args.device,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
    )
    # TODO: Print the returned text.
    print(">", text)


if __name__ == "__main__":
    main()
