import regex as re

text = "some text that i'll pre-tokenize"
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
parts = re.findall(PAT, text)
print(parts)


def train_bfe(input_path: str, vocab_size: int, special_tokens: list[str]):
    # 1. create the token table
    vocab = {}

    pass
