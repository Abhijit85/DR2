"""Lightweight answer cleanup shared by all backbones."""
from __future__ import annotations

import re

_FIELD_START_RE = re.compile(r"\[(?:thought|action(?:_input)?|critique|observation|answer)[^\n\]]*", re.IGNORECASE)
_PREFIX_RE = re.compile(r"^(?:final\s+answer|answer)\s*[:\-]\s*", re.IGNORECASE)


def clean_answer(text: str) -> str:
    text = (text or "").replace("<pad>", " ").strip()
    text = _PREFIX_RE.sub("", text)
    m = _FIELD_START_RE.search(text)
    if m:
        text = text[:m.start()]
    text = re.sub(r"\s+", " ", text).strip(" \n:\t")
    return text
