from __future__ import annotations

from dataclasses import dataclass
from typing import List

from tokenizers import Tokenizer


@dataclass
class PythonTokenizer:
    tokenizer: Tokenizer

    @classmethod
    def from_file(cls, path: str) -> "PythonTokenizer":
        return cls(Tokenizer.from_file(path))

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)
