#!/usr/bin/env python3
"""Download raw data sources into a staging directory.

This script is intentionally conservative: it expects a YAML manifest of URLs
and does not embed any dataset credentials. That keeps the pipeline auditable
and allows you to plug in private mirrors as needed.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import urllib.request
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download dataset URLs.")
    parser.add_argument("--manifest", required=True, help="YAML file with a list of URLs")
    parser.add_argument("--output-dir", required=True, help="Directory to store downloads")
    return parser.parse_args()


def download(url: str, output_dir: pathlib.Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1]
    dest = output_dir / filename
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)  # noqa: S310 - intended for controlled manifests.


def main() -> int:
    args = parse_args()
    manifest_path = pathlib.Path(args.manifest)
    output_dir = pathlib.Path(args.output_dir)

    data = yaml.safe_load(manifest_path.read_text())
    urls = data.get("urls", [])
    if not urls:
        print("Manifest contains no URLs.", file=sys.stderr)
        return 1

    for url in urls:
        download(url, output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
