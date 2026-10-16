import pickle
import regex as re

from collections.abc import Iterable, Iterator

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab.copy()
        self.special_tokens = list(special_tokens or [])
        self.token_to_id = {token: token_id for token_id, token in self.vocab.items()}
        next_id = max(self.vocab, default=-1) + 1
        for special in self.special_tokens:
            token = special.encode("utf-8")
            if token not in self.token_to_id:
                self.vocab[next_id] = token
                self.token_to_id[token] = next_id
                next_id += 1
        self.merge_rank = {pair: rank for rank, pair in enumerate(merges)}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> "Tokenizer":
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def decode(self, ids: list[int]) -> str:
        bs = []
        for id in ids:
            bs.append(self.vocab[id])
        return b"".join(bs).decode("utf-8", errors="replace")

    def encode(self, text: str) -> list[int]:
        if self.special_tokens:
            split_pat = "(" + "|".join(re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True)) + ")"
            chunks = re.split(split_pat, text)
        else:
            chunks = [text]

        # print("chunks: ", chunks)

        tokens = []
        for chunk in chunks:
            if self.special_tokens and chunk in self.special_tokens:
                tokens.append(self.token_to_id[chunk.encode("utf-8")])
                continue
            for word in re.findall(PAT, chunk):
                # print("w: ", word)
                tokens.extend(self.encode_word(word.encode("utf-8")))
        return tokens

    def encode_word(self, word: bytes) -> list[int]:
        # print(word)
        word_bytes = [bytes([c]) for c in word]
        # print("origin: ", word_bytes)

        for _ in range(len(word) - 1):
            best_merge = (None, len(self.merge_rank))
            for p in zip(word_bytes, word_bytes[1:]):
                if p in self.merge_rank:
                    rank = self.merge_rank[p]
                    if rank < best_merge[1]:
                        best_merge = (p, self.merge_rank[p])
            if not best_merge[0]:
                break

            new_word_bytes = []
            i = 0
            best = best_merge[0]
            while i < len(word_bytes):
                if i + 1 < len(word_bytes) and word_bytes[i] == best[0] and word_bytes[i + 1] == best[1]:
                    new_word_bytes.append(best[0] + best[1])
                    i += 2
                else:
                    new_word_bytes.append(word_bytes[i])
                    i += 1

            if word_bytes == new_word_bytes:
                break

            word_bytes = new_word_bytes

        word_tokens = []
        # print("done: ", word_bytes)
        for word_byte in word_bytes:
            # print("in loop:", word_byte)
            # print(self.token_to_id[word_byte])
            word_tokens.append(self.token_to_id[word_byte])
        # print("done tokens: ", word_tokens)
        return word_tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)
