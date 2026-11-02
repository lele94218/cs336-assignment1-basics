# CS336 Spring 2026 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Study progress (2026-11-02)

- Implemented byte-level BPE training and tokenization, Transformer LM components, cross-entropy, AdamW, learning-rate scheduling, gradient clipping, data loading, and checkpointing.
- End-to-end training, validation, and text generation work; generation supports temperature and top-p sampling. Training logs include loss, learning rate, step time, and tokens/second.
- TinyStories: completed a 2,500-step run; final reported train loss **1.7948**, validation loss **1.8055**. Generated text has recognizable story structure but still has coherence and repetition issues. This run is shorter than the handout's full training budget.
- OpenWebText: **completed** on the RTX 3060. Configuration: 4 layers, model dimension 512, 16 heads, context length 256, batch size 64, and 20,000 steps (327,680,000 processed tokens). Uses the existing 10K BPE tokenizer. Final train-batch loss **3.0625**, validation loss **3.0927** (averaged over 10 sampled batches). Logged training time: **11,460 seconds**, approximately 3 hours 11 minutes; steady throughput: approximately 29,500 tokens/second.
- Dataset preparation helpers live in the [parent workspace](https://github.com/lele94218/cs336-assignments/tree/main/tools): streaming FineWeb and official OWT subsets into `.npy` arrays. Data, logs, and checkpoints are kept outside Git. Not all written experiments or ablations have been completed.

OWT run directory on the GPU host: `~/remote-runs/assignment1-basics/outputs/owt-20261030-133145/`. Final checkpoint: `checkpoint_final.pt`.

### OWT generation result

Prompt: `Artificial intelligence is`. Sampling: temperature **0.8**, top-p **0.9**, maximum **200 new tokens**. The following is the observed output from the final checkpoint:

> Artificial intelligence is becoming increasingly unaware of the value of the Middle East as a national security and international crisis. The U.S. is a major source of energy and energy, and has shown that all countries in the region are part of the solution.

The model learned recognizable news-style phrasing, but this sample drifts off topic, repeats words, and lacks coherent meaning. Lower validation loss does not establish factual accuracy or reliable generation. These losses are not directly comparable to the course leaderboard: this run uses a TinyStories-trained 10K tokenizer, context length 256, and a sampled validation subset.

**Study closeout:** the implementation, training, checkpoint loading, and generation workflow is complete for this self-study pass. Written experiments, ablations, and leaderboard submission were not all completed. Moving on to Assignment 2: Systems.

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

