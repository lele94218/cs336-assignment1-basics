import time
import numpy as np
from cs336_basics import tokenizer

tokenizer = tokenizer.Tokenizer.from_files("data/vocab.pkl", "data/merges.pkl", ["<|endoftext|>"])

with open("data/TinyStoriesV2-GPT4-valid.txt", encoding="utf-8") as f:
    text = f.read()

docs = text.split("<|endoftext|>")[10:]
sample = "".join(d + "<|endoftext|>" for d in docs)
n_bytes = len(sample.encode("utf-8"))


t0 = time.perf_counter()
ids = tokenizer.encode(sample)
elapsed = time.perf_counter() - t0

arr = np.array(ids, dtype=np.uint16)
np.save("data/tinystories_valid.npy", arr)

print(f"bytes:     {n_bytes:,}")
print(f"tokens:    {len(ids):,}")
print(f"ratio:     {n_bytes / len(ids):.2f} bytes/token")
print(f"time:      {elapsed:.4f}s")
print(f"throughput:{n_bytes / elapsed:,.0f} bytes/s")

assert tokenizer.decode(ids) == sample
