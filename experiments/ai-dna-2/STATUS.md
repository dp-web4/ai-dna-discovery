# AI-DNA 2.0 — Current Status

**Date:** 2026-09-22  
**Branch:** `research/ai-dna-2-universality`  
**PR:** #1  
**Research state:** instrumentation validated; semantic experiment design active

## Executive status

AI-DNA 2.0 has moved from correcting the 2025 coordinate-comparison error to a more precise question:

> **Do different architectures conserve local semantic relations and state transitions under controlled changes in evidence?**

The project no longer treats global representational similarity as the target phenomenon. The current target is **semantic conservation**, operationalized through predeclared relations and **Semantic Closure Horizons (SCH)**.

## Completed

### Phase A — synthetic coordinate sanity

Complete and independently reproduced. Arbitrary coordinate rotation destroys raw coordinate cosine while preserving relational geometry; the harness also refuses direct coordinate-wise comparison across independently declared spaces.

See `PHASE_A_BASELINE.md`.

### C0 — first real hidden-state panel

Complete as an instrumentation pilot on:

- Qwen2.5-0.5B-Instruct
- Pythia-410M-deduped
- TinyLlama-1.1B-Chat

Across four normalized layer depths, all three architecture pairs showed non-random relational correspondence in the initial RDM/CKA analysis. Reported ranges from Pub were approximately:

| Pair | RDM Spearman | Linear CKA |
|---|---:|---:|
| Qwen ↔ Pythia | 0.34–0.66 | 0.63–0.76 |
| Qwen ↔ TinyLlama | 0.52–0.63 | 0.61–0.78 |
| Pythia ↔ TinyLlama | 0.44–0.64 | 0.64–0.76 |

With 499 identity permutations, all 12 cells reached the finite-resolution floor `p=0.002`.

**Interpretation:** the pilot contains non-random relational geometry. Nothing stronger is claimed.

## Confounders identified

C0 is explicitly pre-semantic because:

1. every prompt shared a common suffix;
2. extraction used last-token pooling rather than a declared semantic span/closure point;
3. token lengths were strongly correlated across tokenizers;
4. the anchor bank contained only 40 concepts;
5. instruction/base/chat training regimes were mixed;
6. there were not yet sufficient paraphrase/minimal-pair families;
7. raw global metric baselines were not calibrated for width/depth and layer-selection effects.

PRH independently reports weak alignment for simple last-token LM representations and stronger results from token averaging, reinforcing the need to revisit extraction for C1.

## External literature update

### Platonic Representation Hypothesis (PRH)

Huh et al. (2024) is now a first-class conceptual prior because it compares induced relations/kernels rather than raw coordinates and reports relational convergence across models/modalities.

### 2026 Aristotelian re-analysis

Gröger, Wen, and Brbić show that raw global representation-similarity metrics can be inflated by representation width and layer-selection depth. After permutation-based calibration, global convergence weakens substantially while **local neighborhood agreement remains robust**.

This is now built into the C1 methodology.

## Current working hypothesis

> **The best current candidate is a shared local topology of semantic relations / transitions, not a globally matching latent geometry.**

Examples of claim-bearing questions:

- Does paraphrase preserve a predeclared SAME-CONCEPT relation?
- Does disambiguating evidence move `bank` into FINANCIAL or RIVER neighborhoods?
- Does changing `A before B` to `A after B` invert a predeclared TEMPORAL-ORDER relation?
- Do irrelevant distractors leave the semantic relation unchanged after closure?
- Do these response patterns recur across architectures?

## Next gate: C1 Semantic Closure Horizon

Before any claim-bearing learned alignment:

1. build versioned semantic probe families;
2. predeclare semantic relations before runs;
3. add multiple surface forms per concept;
4. add progressive ambiguity/disambiguation probes;
5. remove common fixed-tail dependence;
6. compare declared pooling/extraction strategies;
7. implement local-neighborhood metrics (mKNN / CKNNA-style where appropriate);
8. implement permutation-based width-aware calibration;
9. calibrate the final layer aggregation/selection statistic, not only individual cells;
10. use restricted permutations for grouped paraphrase/minimal-pair families;
11. report nuisance metadata separately.

### C1 success means

A candidate signal:

- remains stable when meaning remains stable;
- changes when meaning changes;
- resists irrelevant evidence;
- preserves named semantic relations across architectures;
- generalizes to held-out concepts/domains;
- survives calibrated nulls.

Only then does Phase D become claim-bearing.

## Gated later work

### Phase B

Same-architecture UWSH-style weight-subspace positive/control track. Useful but not required before C1.

### Phase D

Learned cross-space alignment on training anchors, evaluated on held-out concepts/domains.

### Phase E

Functional transfer of latent directions/interventions.

### SAGE integration

No SAGE architecture change is justified by C0. Application should be revisited only after D/E evidence exists.

## Open methodological questions

- How should SCH closure be detected without defining it circularly using the same similarity metric under test?
- Which local-neighborhood sizes remain stable across sample scale and concept density?
- What restricted permutation scheme is valid once paraphrases/minimal pairs create grouped dependence?
- How much neighborhood agreement comes from common linguistic/statistical training rather than a deeper invariant?
- Does functional transfer track local semantic conservation better than global geometry?

## No current claim

As of this status:

- no universal latent alphabet has been shown;
- no universal semantic representation has been shown;
- no functional cross-architecture transfer has been shown.

What has been shown is that the corrected harness works and that a small real-model pilot contains enough non-random relational structure to justify the stricter semantic experiment.
