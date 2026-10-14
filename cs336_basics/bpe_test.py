from cs336_basics.bpe import train_bpe

EOT = "<|endoftext|>"


def run(tmp_path, text, vocab_size, special_tokens=()):
    p = tmp_path / "data.txt"
    p.write_text(text, encoding="utf-8")
    return train_bpe(str(p), vocab_size, list(special_tokens))


def test_worked_example(tmp_path):
    # 之前手算的例子
    vocab, merges = run(tmp_path, f"low low low lower{EOT}lowest", 260, [EOT])
    assert merges == [(b"o", b"w"), (b"l", b"ow"), (b" ", b"low")]
    assert vocab[257] == b"ow"
    assert vocab[259] == b" low"


def test_initial_vocab(tmp_path):
    # 名额刚好只够 256 字节 + 1 个特殊 token，不应发生合并
    vocab, merges = run(tmp_path, "hello hello", 257, [EOT])
    assert merges == []
    assert len(vocab) == 257
    assert all(vocab[i] == bytes([i]) for i in range(256))
    assert vocab[256] == EOT.encode("utf-8")


def test_tie_break(tmp_path):
    # (a,b) (␣,c) (c,d) 都是 1 次，字典序最大的是 (c,d)
    _, merges = run(tmp_path, "ab cd", 257)
    assert merges[0] == (b"c", b"d")


def test_repeated_pair(tmp_path):
    # aaaa: (a,a)=3 → [aa,aa] → (aa,aa)=1 → [aaaa]
    _, merges = run(tmp_path, "aaaa", 258)
    assert merges == [(b"a", b"a"), (b"aa", b"aa")]


def test_overlap(tmp_path):
    # aaa: 从左往右合并 → [aa, a]，不是 [a, aa]
    _, merges = run(tmp_path, "aaa", 258)
    assert merges == [(b"a", b"a"), (b"aa", b"a")]


def test_no_merge_across_special(tmp_path):
    # 每段只有一个字符，没有任何相邻对；vocab_size 很大也要正常停下
    vocab, merges = run(tmp_path, f"x{EOT}x{EOT}x", 1000, [EOT])
    assert merges == []
    assert len(vocab) == 257


def test_invariants(tmp_path):
    text = "the cat sat on the mat. the dog sat on the log. " * 20
    vocab, merges = run(tmp_path, text, 300, [EOT])
    assert len(vocab) <= 300
    assert sorted(vocab) == list(range(len(vocab)))  # id 连续
    assert len(set(vocab.values())) == len(vocab)  # 没有重复 token
    # 第 k 个合并产生的 token 就是两边拼接
    for k, (a, b) in enumerate(merges):
        assert vocab[257 + k] == a + b
