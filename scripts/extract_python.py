#!/usr/bin/env python3
"""Extract Python-only code from raw dumps into JSONL.

Supported inputs:
- .py, .pyi files
- Jupyter notebooks (.ipynb) with code cells
"""
from __future__ import annotations

import argparse
import ast
import json
import pathlib
import re
import sys
from typing import Iterable


PY_EXTENSIONS = {".py", ".pyi"}
NOTEBOOK_EXTENSIONS = {".ipynb"}
MIN_LINES = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Python code into JSONL.")
    parser.add_argument("--input", required=True, help="Input directory")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    return parser.parse_args()


def iter_paths(root: pathlib.Path) -> Iterable[pathlib.Path]:
    for path in root.rglob("*"):
        if path.is_file():
            suffix = path.suffix.lower()
            if suffix in PY_EXTENSIONS or suffix in NOTEBOOK_EXTENSIONS:
                yield path


def clean_code(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_valid_python(text: str) -> bool:
    try:
        ast.parse(text)
    except SyntaxError:
        return False
    return True


def extract_notebook(path: pathlib.Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    cells = data.get("cells", [])
    code_chunks = []
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        code_chunks.append(source)
    return "\n\n".join(code_chunks)


def load_code(path: pathlib.Path) -> str:
    if path.suffix.lower() in NOTEBOOK_EXTENSIONS:
        return extract_notebook(path)
    return path.read_text(encoding="utf-8", errors="ignore")


def main() -> int:
    args = parse_args()
    root = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for path in iter_paths(root):
            code = clean_code(load_code(path))
            if not code:
                continue
            if code.count("\n") + 1 < MIN_LINES:
                continue
            if not is_valid_python(code):
                continue
            record = {
                "id": str(path),
                "text": f"<file_sep>\n{code}",
            }
            handle.write(json.dumps(record) + "\n")
            count += 1

    print(f"Wrote {count} records to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
