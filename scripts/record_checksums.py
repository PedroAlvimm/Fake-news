#!/usr/bin/env python3
"""Gerar checksums SHA256 para arquivos em `modelos/` e `resultados/`.

Uso:
    python3 scripts/record_checksums.py --output resultados/artifacts_checksums.json
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(root: Path):
    data = {}
    if not root.exists():
        return data
    for p in sorted(root.rglob("**/*")):
        if p.is_file():
            rel = str(p.relative_to(Path.cwd()))
            try:
                data[rel] = {
                    "sha256": sha256_of_file(p),
                    "size_bytes": p.stat().st_size,
                    "mtime": int(p.stat().st_mtime),
                }
            except Exception as e:
                data[rel] = {"error": str(e)}
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Arquivo JSON de saída (ex: resultados/artifacts_checksums.json)")
    args = parser.parse_args()

    modelos = collect(Path("modelos"))
    resultados = collect(Path("resultados"))

    out = {"modelos": modelos, "resultados": resultados}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Checksums escritos em: {out_path}")


if __name__ == "__main__":
    main()
