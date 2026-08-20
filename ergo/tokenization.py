"""Tokenizer protocol so the core loop is backbone-agnostic.

HF tokenizers (LLaDA/Dream) are wrapped by ``HFTokenizerWrapper``;
tests and the mock backbone use ``SimpleTokenizer`` (whitespace, CPU-only).
"""
from __future__ import annotations

from typing import Protocol, Sequence


class TokenizerLike(Protocol):
    mask_id: int
    pad_id: int

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str: ...
    def tokens(self, ids: Sequence[int]) -> list[str]: ...
    def is_word_start(self, ids: Sequence[int], position: int) -> bool: ...
    @property
    def vocab_size(self) -> int: ...


class SimpleTokenizer:
    """Whitespace tokenizer with a growable vocab. One token == one word."""

    def __init__(self) -> None:
        self._tok2id: dict[str, int] = {"<pad>": 0, "<mask>": 1}
        self._id2tok: list[str] = ["<pad>", "<mask>"]
        self.pad_id = 0
        self.mask_id = 1

    def _add(self, tok: str) -> int:
        if tok not in self._tok2id:
            self._tok2id[tok] = len(self._id2tok)
            self._id2tok.append(tok)
        return self._tok2id[tok]

    def encode(self, text: str) -> list[int]:
        return [self._add(t) for t in text.split()]

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        toks = []
        for i in ids:
            t = self._id2tok[i] if 0 <= i < len(self._id2tok) else "<unk>"
            if skip_special and t in ("<pad>", "<mask>"):
                continue
            toks.append(t)
        return " ".join(toks)

    def tokens(self, ids: Sequence[int]) -> list[str]:
        return [self._id2tok[i] if 0 <= i < len(self._id2tok) else "<unk>" for i in ids]

    def is_word_start(self, ids: Sequence[int], position: int) -> bool:
        return True  # every token is a full word

    @property
    def vocab_size(self) -> int:
        return len(self._id2tok)


class HFTokenizerWrapper:
    """Adapts a HuggingFace tokenizer to TokenizerLike."""

    def __init__(self, hf_tokenizer, mask_id: int | None = None):
        self.hf = hf_tokenizer
        mid = mask_id if mask_id is not None else getattr(hf_tokenizer, "mask_token_id", None)
        if mid is None:
            raise ValueError("mask token id required (pass mask_id explicitly for LLaDA: 126336)")
        self.mask_id = int(mid)
        self.pad_id = int(hf_tokenizer.pad_token_id or 0)

    def encode(self, text: str) -> list[int]:
        return self.hf.encode(text, add_special_tokens=False)

    def decode(self, ids, skip_special: bool = True) -> str:
        ids = [i for i in ids if i != self.mask_id] if skip_special else list(ids)
        return self.hf.decode(ids, skip_special_tokens=skip_special)

    def tokens(self, ids) -> list[str]:
        return self.hf.convert_ids_to_tokens(list(ids))

    def is_word_start(self, ids, position: int) -> bool:
        tok = self.hf.convert_ids_to_tokens([ids[position]])[0]
        if tok.startswith("##"):          # WordPiece continuation
            return False
        if tok.startswith(("Ġ", "▁")):   # BPE / SentencePiece word starts
            return True
        return position == 0  # BPE continuation pieces lack the space marker

    @property
    def vocab_size(self) -> int:
        return len(self.hf)
