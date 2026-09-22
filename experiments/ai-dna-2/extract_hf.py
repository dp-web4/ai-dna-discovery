#!/usr/bin/env python3
"""Extract hidden states from one Hugging Face causal LM, one model per process."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ai_dna2.hf_extract import extract_hf_model, parse_depths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--anchors", default="data/anchors/pilot.jsonl")
    parser.add_argument("--output-dir", default="results/pilot")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument(
        "--dtype", default="float32", choices=["float32", "float16", "bfloat16"]
    )
    parser.add_argument("--depths", default="0.25,0.50,0.75,1.00")
    parser.add_argument("--pooling", default="last", choices=["last", "mean"])
    parser.add_argument("--max-tokens", type=int, default=192)
    args = parser.parse_args()

    activation, manifest = extract_hf_model(
        model_id=args.model,
        revision=args.revision,
        anchors_path=args.anchors,
        output_dir=args.output_dir,
        device=args.device,
        dtype=args.dtype,
        depths=parse_depths(args.depths),
        pooling=args.pooling,
        max_tokens=args.max_tokens,
    )
    print(f"activations: {activation}")
    print(f"manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
