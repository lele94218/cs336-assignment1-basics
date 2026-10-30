# CS336 Spring 2026 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Study progress (2026-10-30)

- Implemented byte-level BPE training and tokenization, Transformer LM components, cross-entropy, AdamW, learning-rate scheduling, gradient clipping, data loading, and checkpointing.
- End-to-end training, validation, and text generation work; generation supports temperature and top-p sampling. Training logs include loss, learning rate, step time, and tokens/second.
- TinyStories: completed a 2,500-step run; final reported train loss **1.7948**, validation loss **1.8055**. Generated text has recognizable story structure but still has coherence and repetition issues. This run is shorter than the handout's full training budget.
- OpenWebText: **training in progress** on the RTX 3060. Configuration: 4 layers, model dimension 512, 16 heads, context length 256, batch size 64, and 20,000 steps (327,680,000 processed tokens). Uses the existing 10K BPE tokenizer. Final validation loss and generation examples are pending.
- Dataset preparation helpers live in the [parent workspace](../tools/): streaming FineWeb and official OWT subsets into `.npy` arrays. Data, logs, and checkpoints are kept outside Git. Not all written experiments or ablations have been completed.

OWT run directory on the GPU host: `~/remote-runs/assignment1-basics/outputs/owt-20261030-133145/`. After completion, record the final validation loss and a generated-text sample here.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

