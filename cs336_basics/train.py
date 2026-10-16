import os
import pickle
import time

from multiprocessing import Pool
from typing import BinaryIO
from cs336_basics import bpe


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def counter_worker(start: int, end: int, special_tokens: list[str]) -> dict:
    # Run pre-tokenization on your chunk and store the counts for each pre-token
    with open("data/TinyStoriesV2-GPT4-train.txt", "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    return bpe.cal_counter(chunk, special_tokens)


if __name__ == "__main__":
    ## Usage
    t0 = time.perf_counter()
    with open("data/TinyStoriesV2-GPT4-train.txt", "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
    args = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        args.append((start, end, ["<|endoftext|>"]))

    print(f"[{time.perf_counter() - t0:6.4f}s] prepare multiprocessing", flush=True)
    with Pool(processes=4) as pool:
        results = pool.starmap(counter_worker, args)

    # merge counter
    total = {}
    for i, counter in enumerate(results):
        print(f"[{time.perf_counter() - t0:6.4f}s] chunk {i}/{len(args)}", flush=True)
        for k, v in counter.items():
            total[k] = total.get(k, 0) + v

    print(f"[{time.perf_counter() - t0:6.4f}s] finish multiprocessing", flush=True)
    vocab, merges = bpe.train_bpe_with_counter(total, 10000, ["<|endoftext|>"])
    # print(list(vocab.items())[:20])
    # print(merges[:20])
    with open("data/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("data/merges.pkl", "wb") as f:
        pickle.dump(merges, f)
    print(f"[{time.perf_counter() - t0:6.4f}s] finish training", flush=True)
