import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def read_from_file(input_path: str) -> str:
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    # input:             "low low low lower<endoftext>lowerest"
    # special_tokens:    "<endoftext>"
    # vocab_size:        260

    # 1. create the token table
    # e.g. 1 - 255 -> byte
    # 256 -> b'<endoftext>'
    vocab = {}
    merges = []
    for i in range(256):
        vocab[i] = bytes([i])
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")

    # 2. cut the words by special tokens
    # "low low low lower<endoftext>lowerest"
    # chunks: ["low low low lower", "lowerest"]
    raw = read_from_file(input_path)
    if special_tokens:
        split_pat = "|".join(re.escape(t) for t in special_tokens)
        chunks = re.split(split_pat, raw)
    else:
        chunks = [raw]

    # 3. tokenized
    # counter:
    # { "low": 1, " low": 2, "lower": 1, "lowerest": 1}
    counts = {}
    for chunk in chunks:
        for m in re.finditer(PAT, chunk):
            key = m.group().encode("utf-8")
            counts[key] = counts.get(key, 0) + 1

    # 4. to single bytes -> frequency
    # { "l o w": 1, " l o w": 2, "l o w e r": 1, "l o w e r e s t": 1}
    byte_freqs = []
    byte_words = []
    for k, v in counts.items():
        byte_words.append([bytes([c]) for c in k])
        byte_freqs.append(v)

    # 5. merge adjacent bytes
    # (l, o)  = 1 + 2 + 1 + 1 = 5
    # (o, w)  = 5
    # (␣, l)  = 2 + 1 = 3
    # (w, e)  = 2
    # (e, r)、(e, s)、(s, t) = 1

    merged_count = {}
    merged_time = vocab_size - len(vocab)

    for p in zip(byte_words, byte_freqs):
        byte_word = p[0]
        for j in range(len(byte_word) - 1):
            word_pair = (byte_word[j], byte_word[j + 1])
            merged_count[word_pair] = merged_count.get(word_pair, 0) + p[1]

    # find the maximum freq
    # (l, ow) = 5    ← 最多
    # (␣, l)  = 3
    # (ow, e) = 2
    # ...
    for _ in range(merged_time):
        if not merged_count:
            break
        # print("merged count: ", merged_count)
        best = max(merged_count, key=lambda p: (merged_count[p], p))
        merges.append(best)
        # print("best: ", best)
        best_word = best[0] + best[1]
        vocab[len(vocab)] = best_word

        # merge ow
        # print("before merged words: ", byte_words)
        for i in range(len(byte_words)):
            byte_word = byte_words[i]
            merged_word = []
            j = 0
            while j < len(byte_word):
                # merge if same token pair
                if j + 1 < len(byte_word) and byte_word[j] == best[0] and byte_word[j + 1] == best[1]:
                    merged_word.append(best[0] + best[1])
                    j += 1
                else:
                    merged_word.append(byte_word[j])
                j += 1
            byte_words[i] = merged_word
        # print("after merged words: ", byte_words)

        # recalculate merged_count
        merged_count.clear()
        for p in zip(byte_words, byte_freqs):
            byte_word = p[0]
            for j in range(len(byte_word) - 1):
                word_pair = (byte_word[j], byte_word[j + 1])
                merged_count[word_pair] = merged_count.get(word_pair, 0) + p[1]

    return vocab, merges
