import pytest

from tinyformer.tokenizer import EOT, Tokenizer

TEXT = (
    "Once upon a time, there was a little girl named Lily. She loved to play outside.\n"
    "One day, she saw a big red ball! It's 2024 and the ball costs $3.50... "
    "Ünïcödé, emoji \U0001F600, CJK 漢字, tabs\tand   spaces.\n\n"
) * 20


@pytest.fixture(scope="module")
def tok():
    return Tokenizer.train(TEXT, vocab_size=350)


def test_vocab_size(tok):
    assert tok.vocab_size == 350
    assert tok.eot_id == 349


@pytest.mark.parametrize("s", [
    "", "a", "hello world", "  leading and trailing  ", "\n\n\t", "Ünïcödé \U0001F600 漢字",
    "never seen: zxqv 9876543210 ~~~", "bytes\x00\x01\x7f", "Lily's ball isn't red.",
])
def test_round_trip(tok, s):
    assert tok.decode(tok.encode(s)) == s


def test_round_trip_training_text(tok):
    assert tok.decode(tok.encode(TEXT)) == TEXT


def test_compresses(tok):
    assert len(tok.encode(TEXT)) < 0.6 * len(TEXT.encode("utf-8"))


def test_special_token(tok):
    ids = tok.encode(f"hi{EOT}there")
    assert tok.eot_id in ids
    assert tok.decode(ids) == f"hi{EOT}there"
    assert tok.eot_id not in tok.encode(f"hi{EOT}there", allow_special=False)


def test_merges_do_not_cross_chunks(tok):
    # every token decodes to bytes that the pre-tokenizer would keep together
    for b in tok.ranks:
        s = b.decode("utf-8", errors="ignore")
        if s:
            assert len(tok._pat.findall(s)) <= 1 or not s.strip()


def test_save_load(tok, tmp_path):
    p = tmp_path / "tok.json"
    tok.save(p)
    t2 = Tokenizer.load(p)
    assert t2.encode(TEXT) == tok.encode(TEXT)


def test_deterministic():
    a = Tokenizer.train(TEXT, 320)
    b = Tokenizer.train(TEXT, 320)
    assert a.ranks == b.ranks


def test_matches_tiktoken_gpt2():
    """Same encoder, GPT-2's published ranks -> identical ids to tiktoken."""
    tiktoken = pytest.importorskip("tiktoken")
    try:
        ref = tiktoken.get_encoding("gpt2")
    except Exception as e:  # offline CI
        pytest.skip(f"tiktoken data unavailable: {e}")
    mine = Tokenizer.from_tiktoken("gpt2")
    for s in TEXT.splitlines() + [TEXT, "The quick brown fox, 1234567 times!!", "  x\n\n y"]:
        assert mine.encode_ordinary(s) == ref.encode_ordinary(s)
