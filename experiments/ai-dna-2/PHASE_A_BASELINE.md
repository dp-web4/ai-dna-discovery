# Phase A Baseline — Synthetic Rotation Sanity Check

**Date:** 2026-09-21  
**Runner:** GPT local validation environment  
**Code:** AI-DNA 2.0 branch `research/ai-dna-2-universality`

## Result

```text
7 passed
coordinate_guard_raised: true
n_samples: 256
dimension: 24
naive_coordinate_cosine_mean: -0.06351083338178212
naive_coordinate_cosine_abs_mean: 0.17914616273434827
rdm_spearman: 1.0
linear_cka: 1.0
heldout_retrieval.top1: 1.0
heldout_retrieval.top5: 1.0
heldout_retrieval.mrr: 1.0
permutation_test.p_value: 0.005
permutation_test.null_mean: 0.00029179851802360715
permutation_test.null_std: 0.005845166598667792
```

## Interpretation

The same synthetic representation was multiplied by a random orthogonal matrix. This destroys coordinate identity while preserving all inner products and relational geometry.

The deliberately naive coordinate-wise cosine therefore changes substantially, despite the two spaces containing exactly the same geometry. In contrast:

- RDM Spearman remains exactly 1;
- linear CKA remains exactly 1;
- an explicit Procrustes map learned on training samples retrieves every held-out counterpart correctly;
- shuffled sample identities collapse to a null distribution centered near zero;
- the protected API refuses direct cross-coordinate cosine rather than silently truncating or padding dimensions.

This is the minimum sanity condition for every later AI-DNA 2.0 experiment.

## Reproduction

```bash
cd experiments/ai-dna-2
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pytest -q
.venv/bin/python run_phase_a.py
```

The exact naive-cosine values depend on the fixed RNG implementation and floating-point library but should remain close. The invariance metrics and held-out retrieval should be numerically exact or effectively exact.

## Status after baseline

Phase A has since been independently reproduced on Pub, and the real-model C0 pilot has run on Qwen2.5-0.5B-Instruct, Pythia-410M, and TinyLlama-1.1B. Phase A is therefore closed as an instrumentation gate.

The next claim-bearing gate is C1 Semantic Closure Horizon; see `STATUS.md`, `SEMANTIC_CLOSURE_HORIZONS.md`, and `PRH_RELEVANCE.md`.
