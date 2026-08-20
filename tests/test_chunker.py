from ergo.rag.chunker import chunk_document, corpus_idf


def test_word_aligned_2000():
    text = " ".join(f"word{i}" for i in range(2000))
    chunks = chunk_document("d", text, max_chars=2000, overlap_chars=0)
    assert all(len(c.text) <= 2000 for c in chunks)
    # never breaks a word: every chunk's edge tokens are intact words
    vocab = set(text.split())
    for c in chunks:
        toks = c.text.split()
        assert toks[0] in vocab and toks[-1] in vocab
    # coverage: all words present across chunks
    seen = set(w for c in chunks for w in c.text.split())
    assert seen == vocab


def test_overlap_word_aligned():
    text = " ".join(f"w{i}" for i in range(3000))
    chunks = chunk_document("d", text, max_chars=2000, overlap_chars=200)
    assert len(chunks) >= 2
    for a, b in zip(chunks, chunks[1:]):
        assert b.char_start < a.char_end          # overlapping
        assert b.text.split()[0] in set(text.split())


def test_ids_and_metadata():
    chunks = chunk_document("mydoc", "hello world " * 500, max_chars=2000)
    assert chunks[0].id == "mydoc::0"
    assert chunks[1].chunk_idx == 1


def test_idf_specificity():
    idf = corpus_idf(["apple banana", "apple cherry", "apple durian"])
    assert idf["apple"] < idf["banana"]  # ubiquitous word -> low idf
