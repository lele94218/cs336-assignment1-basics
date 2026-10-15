import os


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab

    def decode(self, ids: list[int]) -> str:
        bs = []
        for id in ids:
            bs.append(self.vocab[id])
        return b"".joins(bs).decode("utf-8", errors="replace")
