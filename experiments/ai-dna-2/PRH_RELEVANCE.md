# PRH / Aristotelian Representation Hypothesis — Relevance to AI-DNA 2.0

**Date:** 2026-09-22  
**Purpose:** map the closest external prior art onto AI-DNA 2.0 without overstating equivalence.

## Papers

### The Platonic Representation Hypothesis

Minyoung Huh, Brian Cheung, Tongzhou Wang, Phillip Isola. ICML 2024.

- Project: https://phillipi.github.io/prh/
- Paper: https://arxiv.org/abs/2405.07987

Core proposal: models trained with different objectives, data, and modalities may converge toward a shared statistical model of reality in their representation spaces.

For AI-DNA, the important move is methodological: characterize representations through induced **kernels / relations among samples**, not raw coordinate equality.

### Revisiting the Platonic Representation Hypothesis: An Aristotelian View

Fabian Gröger, Shuo Wen, Maria Brbić, 2026.

- Paper: https://arxiv.org/abs/2602.14486

This work revisits PRH using permutation-based null calibration and finds that:

- representation width can inflate raw similarity scores;
- selecting/aggregating over more layers can inflate the final score under the null;
- after calibration, much apparent **global** convergence weakens;
- **local neighborhood identity** remains significantly aligned;
- local pairwise distances need not agree even where neighborhood membership does.

The implication for AI-DNA is strong: what converges may be local relations — effectively, **who is near whom** — rather than globally matching geometry.

## What PRH independently validates

### 1. Coordinate equality is the wrong requirement

PRH compares sample relations rather than requiring feature axes to align. This independently supports the AI-DNA 2.0 guard against raw coordinate cosine across unrelated architectures.

### 2. Local relations may be more robust than global geometry

The 2026 re-analysis argues against making CKA or a global RDM correlation the phenomenon. Local semantic neighborhoods are the stronger primary candidate.

### 3. Representation extraction is part of the experiment

PRH reports that simply taking the last language-model token did not show strong alignment in its vision-language analysis. Average pooling across valid tokens produced stronger alignment; a prompted last-token variant showed similar trends but lower alignment.

This is directly relevant to Pub C0, which used a common prompt suffix and last-token pooling. C0 remains useful as instrumentation, but that extraction strategy must not be carried into C1 as an assumed semantic representation.

### 4. Information sufficiency matters

PRH's caption-density experiment reports stronger vision-language alignment for richer descriptions.

This is adjacent to the AI-DNA **Semantic Closure Horizon (SCH)** idea. PRH asks approximately whether more informative observation improves relational alignment. SCH asks the sharper question:

> which specific evidence is sufficient to establish a semantic state, which additional evidence is irrelevant, and do independent architectures undergo the same semantic transition?

That distinction matters because more information can also mean more tokens, more shared lexical/statistical structure, changed pooling behavior, or more nuisance correlation.

## What PRH does not establish for AI-DNA

### Statistical alignment is not semantic alignment

A kernel/neighborhood can align for reasons unrelated to the intended semantic variable. AI-DNA C1 therefore requires controlled interventions: paraphrase, syntax change, ambiguity resolution, meaning flips, irrelevant distractors, and held-out domains.

### An absolute similarity score has no intrinsic semantic meaning

PRH itself notes the difficulty of interpreting an alignment value far below the metric maximum. The 2026 re-analysis shows why: raw similarity magnitudes can have nonzero, dimension-dependent null baselines.

AI-DNA must therefore specify the exact statistic, its null, the semantic relation predeclared before measurement, the intervention expected to preserve/change that relation, and the calibrated effect size.

### Global geometry may be the wrong target

AI-DNA should accept a weaker but semantically meaningful invariant if that is what survives:

> local relational topology under controlled semantic transformations.

### PRH does not provide functional transfer

Even strong relational convergence does not establish that a latent direction/intervention can transfer between models. AI-DNA Phase E remains a stronger, separate test.

## Metric implications

### Primary C1 instruments

1. mutual k-nearest-neighbor overlap (mKNN);
2. CKNNA-style local neighborhood comparison where implementation is auditable;
3. predeclared semantic triplet/relation accuracy;
4. held-out neighborhood conservation under intervention.

### Secondary diagnostics

- RDM Spearman / RSA;
- linear CKA;
- global Procrustes/shape metrics.

These remain useful, especially for Phase A and diagnostics, but should not drive semantic claims.

## Null calibration requirements

### Width / sample-size baseline

Raw metric expectations can depend on representation width relative to sample count. C1 reports must calibrate each reported metric against appropriate nulls.

### Layer-selection inflation

If many layer-pair scores are computed and the maximum is reported, the final statistic has a different null from each individual cell.

> **Calibrate the aggregate statistic that will actually be reported.**

Do not select the best layer pair and then quote an unadjusted single-cell p-value.

### Grouped probe dependence

C1 will contain paraphrases and minimal-pair families. Those samples are not fully exchangeable. A naive global permutation can break the data-generating structure and yield an invalid null.

Define permutation blocks at the semantic-family level before claim-bearing runs.

## Relationship to Semantic Closure Horizon

PRH's descriptive-caption result suggests a precursor:

> more relevant information → stronger cross-modal alignment

SCH tests a stronger causal/semantic shape:

> insufficient evidence → ambiguous/unstable relation → relevant evidence arrives → structured semantic transition → closure → irrelevant evidence arrives → relation remains stable

The invariant is not that a vector stops moving. It is that **the named semantic relation/state transition responds similarly to relevant evidence across architectures**.

## Prior-art map

| Work | Main object | Coordinate requirement | Strongest relevant claim | AI-DNA use |
|---|---|---|---|---|
| Original AI-DNA 2025 | native embeddings / symbols / behavior | often incorrectly assumed | exploratory cross-model commonality | historical question + negative results |
| UWSH | weight/update subspaces | shared architecture | low-dimensional same-architecture parameter structure | positive/control track |
| PRH 2024 | kernels / relational geometry | none | increasing relational convergence across models/modalities | primary conceptual prior |
| Aristotelian 2026 | calibrated local neighborhoods | none | local neighbor relations survive where global convergence weakens | primary metric correction |
| World-model PRH 2026 | action-conditioned state-transition representations | none initially; stitching adapters later | predictive consistency can yield transition-compatible latent structure | transition/functional prior |
| AI-DNA 2.0 | intervention-defined semantic relations + later function | none initially; learned map later | semantic conservation / SCH / functional transfer | target research program |

## Concrete changes adopted

Because of PRH + the 2026 re-analysis:

- C0 stays an instrumentation result only.
- Last-token extraction is not assumed valid for C1.
- Local-neighborhood metrics become primary.
- Raw CKA magnitude is downgraded to diagnostic.
- Width-aware null calibration is mandatory.
- Layer aggregation/selection is calibrated as part of the final statistic.
- C1 predeclares semantic relations before execution.
- Restricted permutations are required for grouped paraphrase/minimal-pair probes.
- Caption/information-density ideas become controlled relevant-vs-irrelevant SCH probes.

## Bottom line

PRH is not evidence that AI-DNA 2.0 is already solved. It is strong independent support for the **kind of invariant** now under test.

The 2026 Aristotelian re-analysis is equally valuable because it narrows that invariant and warns against exactly the kind of uninterpreted metric magnitude AI-DNA 2.0 is trying to avoid.

> **Current target: determine whether different architectures conserve predeclared local semantic relations and transitions as relevant evidence crosses a Semantic Closure Horizon; only later ask whether those conserved relations support held-out alignment and functional transfer.**
