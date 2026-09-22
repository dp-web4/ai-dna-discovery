# PRD: AI-DNA 2.0 Experimental Harness

**Status:** proposed  
**Parent:** \`README.md\` in this directory  
**Principle:** compare invariants, not arbitrary coordinates.

## 1. Objective

Build a reproducible harness that distinguishes:

1. raw-coordinate similarity;
2. coordinate-free relational similarity;
3. learned cross-space alignment;
4. functional transfer after alignment;
5. same-architecture weight-subspace structure.

The harness must make it difficult to accidentally reproduce the original 2025 methodological error.

It must also distinguish **statistical correspondence** from **semantic correspondence**. RDM, CKA, retrieval, and permutation statistics are instruments; none is semantic evidence unless the measured structure responds correctly to controlled meaning-preserving and meaning-changing interventions. See `SEMANTIC_CLOSURE_HORIZONS.md`.

## 2. Non-goals

- proving a universal latent language from a single metric;
- optimizing benchmark performance;
- training large new base models;
- treating model self-report as evidence of internal similarity;
- using consciousness/sentience claims as experimental variables.

## 3. Proposed layout

\`\`\`text
experiments/ai-dna-2/
  README.md
  PRD.md
  config/
    models.yaml
    anchors.yaml
    experiment.yaml
  data/
    anchors/
      concepts.jsonl
      splits.json
  src/
    providers.py
    representations.py
    relational.py
    align.py
    retrieval.py
    functional.py
    weight_subspace.py
    controls.py
    statistics.py
    provenance.py
  tests/
    test_rotation_invariance.py
    test_permutation_null.py
    test_no_cross_arch_truncation.py
    test_split_leakage.py
    test_alignment_holdout.py
  results/
    .gitkeep
  run.py
\`\`\`

Generated model activations and large tensors should not be committed. Commit compact summaries, hashes, configs, and plots/tables sufficient to audit the run.

## 4. Hard methodological guardrails

### 4.1 No coordinate-wise cross-architecture cosine

The API should refuse operations equivalent to:

\`\`\`python
a = a[:min(len(a), len(b))]
b = b[:min(len(a), len(b))]
cosine(a, b)
\`\`\`

unless the experiment explicitly marks the pair as sharing a coordinate system.

### 4.2 Train/test separation for mappings

All learned alignment functions receive explicit train indices and may not inspect held-out anchor vectors during fitting.

Splits are made at the semantic-concept level before paraphrase expansion.

At least one full semantic domain is held out from every primary alignment fit.

### 4.3 Mapping complexity is reported

For every mapper, record:

- parameter count;
- effective rank;
- anchors used for fitting;
- ratio of fit parameters to independent training anchors;
- regularization;
- whether the mapping is invertible/orthogonal/linear/nonlinear.

A mapping that has enough capacity to memorize the dictionary is a translation system, not evidence of a shared latent geometry.

### 4.4 Primary metrics are fixed before the first full run

Primary:

- RDM Spearman correlation;
- linear CKA;
- held-out top-k retrieval / MRR;
- neighborhood overlap;
- functional effect-sign agreement;
- functional effect-size correlation.

Secondary analyses may be exploratory but must be labeled as such.

## 5. Anchor dataset

Each JSONL record should minimally contain:

\`\`\`json
{
  "concept_id": "causality.cause",
  "domain": "causality",
  "label": "cause",
  "definition": "an event or condition that produces an effect",
  "paraphrases": [
    "the thing that makes another event happen",
    "a condition responsible for a later result"
  ],
  "relations": [
    {"target": "causality.effect", "type": "paired"},
    {"target": "logic.implication", "type": "related"}
  ]
}
\`\`\`

Target initial scale: 500 concepts, 5-10 domains, 3+ surface forms per concept.

The first run can use a smaller pilot set (100-200 concepts) only to validate the harness. Claims require the larger set.

## 6. Representation protocol

For causal LMs:

- run a fixed prompt template plus paraphrase variants;
- extract hidden states at normalized layer-depth positions;
- pool over concept span and/or final prompt token using declared methods;
- store native dimension;
- L2 normalization may be applied only after the raw vector is preserved;
- record tokenizer output and span mapping.

Recommended depth fractions:

- 0.25
- 0.50
- 0.75
- 1.00

Do not choose the "best" layer on the test set. A development split may select a layer rule; final tests then use that rule unchanged.

## 7. Phase A acceptance: synthetic rotation sanity test

Construct matrix X of synthetic or real same-model representations.

Create Y = XQ where Q is random orthogonal.

Required behavior:

- coordinate-wise cosine between corresponding rows changes substantially;
- pairwise Euclidean/cosine-distance ordering is preserved up to numerical tolerance;
- RDM correlation ≈ 1;
- linear CKA ≈ 1;
- Procrustes recovers correspondence with near-perfect held-out retrieval.

If the harness fails this phase, no model experiments proceed.

Also construct a shuffled-row version Y_perm.

Required behavior:

- invariance metrics tied to sample identity collapse appropriately;
- permutation test identifies the correct pairing as highly non-random.

## 8. Phase B acceptance: same-architecture weight-subspace control

Input: N LoRA/fine-tune deltas from one exact base architecture.

For each comparable module:

1. reconstruct or load delta W;
2. compute group mean μ;
3. center residuals ΔW_i - μ;
4. decompose residual bank;
5. measure explained variance;
6. compare to matched random low-rank matrices;
7. hold out some adapters from basis estimation.

Report three reconstruction/function conditions separately:

- **mean only**;
- **mean + k-dimensional basis**;
- **full adapter**.

If mean-only explains the functional gain, do not describe the residual basis as carrying task-specific universal directions.

## 9. Phase C acceptance: relational cross-architecture test

Phase C is now split into:

- **C0 — geometry instrumentation validation:** establish that non-random relational structure is measurable across architectures. The initial pub pilot belongs here and is explicitly pre-semantic.
- **C1 — semantic-closure invariance/sensitivity:** establish that the measured structure tracks coherent meaning rather than prompt/token nuisance variables.

C1 is a gate for claim-grade relational universality and for Phase D alignment. **Semantic closure horizon (SCH)** is the C1 term; Web4 MRH remains a separate declared/signed graph-neighborhood construct with traversal depth.

For every architecture pair and layer-depth pair:

- compute RDM Spearman;
- compute linear CKA from sample Gram matrices;
- compute k-nearest-neighbor overlap;
- compute semantic triplet ordering accuracy.

Run permutation nulls with at least 1,000 identity shuffles for final claims.

A C0 candidate signal is worth further study if it:

- beats null with corrected significance;
- appears in at least three architecture pairs.

It does **not** advance to semantic or alignment claims on those facts alone.

For C1, each semantic target must include controlled probe families:

- paraphrase/surface variation with meaning preserved;
- outside-closure distractors that should not change meaning;
- inside-closure evidence that should change or disambiguate meaning;
- lexical near-neighbor / meaning-flip pairs that should separate;
- where useful, progressive-context probes that estimate a semantic closure horizon.

A candidate advances beyond C1 only if it:

- is invariant under meaning-preserving surface changes;
- is insensitive to outside-closure distractors;
- changes appropriately under meaning-relevant evidence;
- separates lexically similar but semantically different cases;
- preserves semantic relations or transformations across architectures;
- survives held-out concepts and at least one held-out domain.

Token count, character count, frequency, syntax, and related variables are nuisance metadata / confound controls, never target signals.

## 10. Phase D acceptance: learned alignment

**Entry gate:** Phase D should not begin as a claim-bearing experiment until C1 has identified structure that tracks semantic manipulations rather than merely non-random geometry.

Implement in order:

1. orthogonal Procrustes;
2. ridge linear map;
3. CCA-style shared space;
4. low-rank linear map.

Only add nonlinear maps after the linear results are established.

Primary test is **concept retrieval**:

Given mapped source representation, retrieve the target-model representation for the same concept among all held-out candidates.

Report:

- top-1;
- top-5;
- MRR;
- domain-stratified scores;
- confidence intervals.

Nulls:

- shuffled fit pairs;
- random map of matched rank;
- random representation;
- same-capacity map fit to nonsense labels.

Advance to functional tests only if held-out retrieval is reproducibly above null and not limited to lexical overlap.

## 11. Phase E acceptance: functional transfer

Start with tasks that expose a continuous or binary scalar output where intervention effects are measurable.

For each source direction d_A:

1. derive d_A using training examples only;
2. map d_A through the alignment into target model B;
3. intervene at the corresponding normalized layer depth;
4. compare target behavior with:
   - no intervention;
   - random direction matched in norm;
   - shuffled mapped direction;
   - native direction learned directly in B, if available.

Primary outputs:

- sign agreement between source and mapped-target effects;
- standardized effect size;
- per-example effect correlation;
- fraction of tasks where mapped direction beats random controls.

A single dramatic example is not sufficient.

## 12. Statistics

Use bootstrap confidence intervals over concepts/examples.

Use permutation tests for alignment/relational metrics where analytic assumptions are weak.

Correct for multiple architecture/layer comparisons in confirmatory results.

Report effect sizes, not only p-values.

All random seeds are recorded.

## 13. Provenance and reproducibility

Each run emits a manifest with:

- git commit SHA;
- timestamp;
- host/runtime;
- model identifier and hash;
- quantization;
- tokenizer identifier/hash;
- config hash;
- anchor dataset hash;
- random seed;
- code path/version for every metric;
- output artifact hashes.

The manifest is part of the result, not optional bookkeeping.

## 14. Result vocabulary

Use these labels in reports:

- **coordinate similarity** — only if coordinates are shared/aligned;
- **relational similarity** — coordinate-free geometry;
- **alignment generalization** — learned map performance on held-out concepts;
- **functional transfer** — intervention/task effect survives mapping;
- **within-architecture spectral structure** — weight-space result.

Avoid "universal representation" unless at least relational + held-out alignment criteria are satisfied across multiple architecture families.

Avoid "universal functional subspace" unless functional-transfer criteria are satisfied.

## 15. First implementation milestone

The first PR after this design should contain only enough code to pass Phase A:

- representation container with explicit coordinate-system metadata;
- RDM;
- linear CKA;
- Procrustes;
- permutation null;
- synthetic rotation test;
- hard failure for accidental dimension-truncation cosine.

That small milestone validates the scientific plumbing before GPU/model complexity is introduced.

## 16. SAGE handoff condition

The preferred architecture-neutral primitive, if C1 succeeds, is not a generic latent vector. It is a semantic entity/relation plus its relevance horizon: what evidence established it, what lies inside/outside the semantic closure horizon, which transformations preserve identity, and which change meaning.

No SAGE architecture change is justified by this experiment until Phase D or E succeeds.

If Phase D succeeds but Phase E fails, a discovered mapping may still be useful for:

- cross-model retrieval;
- dictionary lookup;
- memory indexing;
- semantic handshakes.

If Phase E succeeds, then evaluate a genuine architecture-adapter layer for SAGE beings/nodes, with provenance identifying source model, mapping version, confidence, and failure domain.

That integration should live in SAGE; this repository remains the experimental evidence base.
