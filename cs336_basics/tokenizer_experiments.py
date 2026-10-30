import argparse
import time
import numpy as np
from cs336_basics import tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--split", choices=("train", "val"), default="val")
args = parser.parse_args()
split = "valid" if args.split == "val" else "train"
input_path = f"data/TinyStoriesV2-GPT4-{split}.txt"
output_path = f"data/tinystories_{split}.npy"

tokenizer = tokenizer.Tokenizer.from_files("data/vocab.pkl", "data/merges.pkl", ["<|endoftext|>"])

with open(input_path, encoding="utf-8") as f:
    text = f.read()

sample = text
n_bytes = len(sample.encode("utf-8"))


t0 = time.perf_counter()
ids = tokenizer.encode(sample)
elapsed = time.perf_counter() - t0

arr = np.array(ids, dtype=np.uint16)
np.save(output_path, arr)
print(f"saved:     {output_path}")

print(f"bytes:     {n_bytes:,}")
print(f"tokens:    {len(ids):,}")
print(f"ratio:     {n_bytes / len(ids):.2f} bytes/token")
print(f"time:      {elapsed:.4f}s")
print(f"throughput:{n_bytes / elapsed:,.0f} bytes/s")

assert tokenizer.decode(ids) == sample
