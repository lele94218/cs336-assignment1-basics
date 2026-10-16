import os
import pickle
import tempfile
from cs336_basics.tokenizer import Tokenizer

EOT = "<|endoftext|>"

# 和之前手算的 train_bpe 例子一致
VOCAB = {i: bytes([i]) for i in range(256)}
VOCAB[256] = EOT.encode()
VOCAB[257] = b"ow"
VOCAB[258] = b"low"
VOCAB[259] = b" low"
MERGES = [(b"o", b"w"), (b"l", b"ow"), (b" ", b"low")]


def make(special=(EOT,)):
    return Tokenizer(VOCAB, MERGES, list(special))


def test_basic():
    # "low" + " low"
    assert make().encode("low low") == [258, 259]


def test_merge_order():
    # lower: l o w e r → l ow e r → low e r
    assert make().encode("lower") == [258, ord("e"), ord("r")]


def test_empty():
    assert make().encode("") == []


def test_special_token():
    assert make().encode(f"low{EOT} low") == [258, 256, 259]


def test_no_special_tokens():
    # 不注册特殊 token 时，它就是普通文本，不应编码成 256
    tok = make(special=())
    ids = tok.encode(EOT)
    assert 256 not in ids
    assert tok.decode(ids) == EOT


def test_overlapping_special_tokens():
    double = EOT + EOT
    vocab = dict(VOCAB)
    vocab[260] = double.encode()
    tok = Tokenizer(vocab, MERGES, [EOT, double])
    # 长的优先匹配
    assert tok.encode(double) == [260]
    assert tok.encode(double + EOT) == [260, 256]


def test_roundtrip():
    tok = make()
    for s in ["low low lower", "Hello, world!", "你好 世界", "emoji 😀 ok", f"a{EOT}b", "  spaces  \n\n"]:
        assert tok.decode(tok.encode(s)) == s, s


def test_decode_invalid_utf8():
    # 半个中文字符，不应抛异常，应替换成 U+FFFD
    assert "\ufffd" in make().decode([0xE4])


def test_encode_iterable_matches_encode():
    tok = make()
    lines = ["low low\n", "lower\n", f"{EOT} low\n", "\n", "你好\n"]
    assert list(tok.encode_iterable(lines)) == tok.encode("".join(lines))


def test_from_files():
    with tempfile.TemporaryDirectory() as d:
        vp, mp = os.path.join(d, "vocab.pkl"), os.path.join(d, "merges.pkl")
        with open(vp, "wb") as f:
            pickle.dump(VOCAB, f)
        with open(mp, "wb") as f:
            pickle.dump(MERGES, f)
        tok = Tokenizer.from_files(vp, mp, [EOT])
        assert tok.encode(f"low{EOT} low") == [258, 256, 259]
