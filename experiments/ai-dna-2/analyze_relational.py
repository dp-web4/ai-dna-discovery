#!/usr/bin/env python3
"""Compare two hidden-state manifests using coordinate-free relational metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ai_dna2.controls import permutation_test
from ai_dna2.relational import linear_cka, pairwise_cosine_rdm, rdm_spearman
from ai_dna2.representations import RepresentationBatch


def load_manifest(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    arrays_path = path.parent / manifest["activation_file"]
    with np.load(arrays_path) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    return manifest, arrays


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--permutations", type=int, default=499)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    left_meta, left_arrays = load_manifest(args.left)
    right_meta, right_arrays = load_manifest(args.right)
    if left_meta["anchor_sha256"] != right_meta["anchor_sha256"]:
        raise SystemExit("anchor hashes differ; comparison refused")
    if left_meta["sample_ids"] != right_meta["sample_ids"]:
        raise SystemExit("sample IDs differ; comparison refused")

    ids = left_meta["sample_ids"]
    common = sorted(set(left_arrays) & set(right_arrays))
    results = []
    for key in common:
        depth = f"{int(key[-3:]) / 100:.2f}"
        li = left_meta["selected_layers"][depth]
        ri = right_meta["selected_layers"][depth]
        a = RepresentationBatch(
            left_arrays[key],
            ids,
            f"{left_meta['model_id']}@{left_meta.get('model_commit_hash')}:{li}:{left_meta['pooling']}",
            left_meta["model_id"],
            str(li),
        )
        b = RepresentationBatch(
            right_arrays[key],
            ids,
            f"{right_meta['model_id']}@{right_meta.get('model_commit_hash')}:{ri}:{right_meta['pooling']}",
            right_meta["model_id"],
            str(ri),
        )
        ra = pairwise_cosine_rdm(a)
        rb = pairwise_cosine_rdm(b)
        observed = rdm_spearman(ra, rb)
        perm = permutation_test(
            observed,
            lambda p: rdm_spearman(ra, rb[np.ix_(p, p)]),
            len(ids),
            n_permutations=args.permutations,
            seed=args.seed,
        )
        results.append({
            "depth_key": key,
            "left_layer": li,
            "right_layer": ri,
            "rdm_spearman": observed,
            "linear_cka": linear_cka(a, b),
            "permutation_p": perm["p_value"],
            "null_mean": perm["null_mean"],
            "null_std": perm["null_std"],
        })

    report = {
        "schema": "ai-dna-2.relational-report.v1",
        "left_model": left_meta["model_id"],
        "right_model": right_meta["model_id"],
        "anchor_sha256": left_meta["anchor_sha256"],
        "n_samples": len(ids),
        "results": results,
    }
    output_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text, encoding="utf-8")
    print(output_text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
