# AI-DNA 2.0: Coordinate, Relational, and Functional Universality

**Status:** active experiment — Phase A and C0 complete; C1 Semantic Closure Horizon next  
**Updated:** 2026-09-22  
**Origin:** retrospective on the 2025 AI-DNA experiments; now informed by the Universal Weight Subspace Hypothesis, the Platonic Representation Hypothesis (PRH), and the 2026 calibrated "Aristotelian" re-analysis of PRH

## Why reopen this question?

The original AI-DNA work asked whether independently trained language models share architecture-independent structure: common patterns, semantic relationships, or a latent "language" that survives differences in model family and training.

Some of that work was useful. Some of its earliest evidence was not.

Several archived experiments compared native embedding vectors from different model architectures by truncating them to the same dimensionality and then taking coordinate-wise cosine similarity. For example, \`archive/shared_pattern_creation_experiment.py\` and \`archive/handshake_protocol.py\` use the minimum vector length and compare the first N coordinates directly.

That is not a valid test of cross-architecture representational similarity. Hidden/embedding coordinates in two independently trained models are not canonically aligned. Equivalent representations may be related by rotation, reflection, permutation, scaling, or a more general mapping. Coordinate 137 in one model has no reason to correspond to coordinate 137 in another.

The archived results remain historically useful, but raw cross-model cosine scores from unaligned native spaces should **not** be treated as evidence for or against a universal representation.

AI-DNA 2.0 restates the original question in a falsifiable way:

> Do different neural architectures share measurable invariants in representation or function after removing arbitrary coordinate choice?

The experiment deliberately separates three claims that were previously mixed together:

1. **Coordinate universality** — are raw coordinates directly comparable?
2. **Relational/geometric universality** — do models preserve similar relationships among concepts even when their coordinates differ?
3. **Functional universality** — can a learned mapping transfer a latent intervention or task-relevant direction between architectures?

A fourth track provides a positive/control comparison with recent work:

4. **Within-architecture weight-subspace structure** — do independently fine-tuned instances of one architecture occupy a shared low-dimensional parameter subspace?

## Closest external prior: PRH and its calibrated re-analysis

Huh et al. (2024), *The Platonic Representation Hypothesis*, is the closest published precedent to the corrected AI-DNA 2.0 question. PRH compares representations through their induced kernels / sample-to-sample relations rather than raw coordinate equality, reports cross-model and cross-modal convergence, and shows that representation extraction and information density materially affect measured alignment.

A 2026 re-analysis by Gröger, Wen, and Brbić shows that raw global similarity scores can be inflated by representation width and by selecting over more layers. After permutation-based calibration, much global convergence weakens while **local neighborhood identity remains robust**.

AI-DNA 2.0 therefore treats local semantic neighborhoods as primary, global CKA/RDM magnitude as diagnostic, and null calibration as mandatory. C1 also goes beyond PRH by defining semantic significance through predeclared meaning-preserving and meaning-changing interventions rather than similarity alone.

See `PRH_RELEVANCE.md`.

## Relation to the Universal Weight Subspace Hypothesis

Kaushik et al. analyze more than 1,100 trained networks, including large collections of Mistral LoRAs, Vision Transformers, and LLaMA models. Their central result is spectral concentration in weight space: within shared architectures, trained solutions use a surprisingly small set of principal directions.

That is adjacent to, but not identical with, the original AI-DNA hypothesis.

- **UWSH:** same architecture, different tasks/training, common parameter directions.
- **AI-DNA 2.0:** different architectures, different native coordinates, possible common relational or functional invariants.

The distinction matters. A shared basis is straightforward to define when tensors correspond layer-for-layer inside one architecture. It is not straightforward across Phi, Gemma, Llama, Mistral, Qwen, etc.

Two later stress tests also motivate stricter controls. The \`cahlen/universal-subspace-stress-test\` replication reports that much of the transferable signal in the 500-LoRA corpus can be explained by the group-mean update rather than many independent task-specific universal directions. Project WIZORB reports poor geometric reconstruction of held-out LoRA updates while retaining substantial functional uplift after projection, suggesting that **geometric similarity and functional similarity can diverge**.

AI-DNA 2.0 therefore measures them separately.

## Core hypotheses

### H0 — Coordinate identity is not universal

For unrelated architectures, raw hidden/embedding coordinate cosine should not be interpreted as a semantic comparison.

This is primarily a methodological sanity check, not a discovery target.

### H1 — Relational geometry contains cross-architecture invariants

Given the same set of semantic anchors, models may produce different vectors but preserve similar relations among anchors.

Examples:

- truth is closer to evidence than to temperature;
- cause is related to effect in a way that differs from synonymy;
- "before/after" forms a stable opposition;
- arithmetic, logical, temporal, social, and physical concepts form comparable neighborhood structures.

Evidence for H1 requires held-out generalization and comparison against permutation/null controls.

### H2 — A mapping learned on some anchors generalizes to unseen anchors

If two latent spaces contain a shared structure under different coordinates, an alignment learned on a training set should improve cross-model retrieval or correspondence on concepts and domains withheld from alignment.

Candidate mappings:

- orthogonal Procrustes;
- ridge/linear mapping;
- CCA/SVCCA/PWCCA-style alignment;
- low-rank mapping;
- nonlinear mapping only as a later comparison, because too much mapping capacity can memorize the dictionary and make the claim vacuous.

### H3 — Functional effects can transfer after alignment

A stronger claim than representational similarity is that a direction/intervention identified in model A can be mapped into model B and produce a consistent behavioral effect.

Examples:

- increase/decrease a scalar judgment;
- bias toward one side of a controlled concept contrast;
- transfer a small task-specific steering direction;
- preserve the sign and relative magnitude of a logit/probability effect.

H3 is the most important target. Similar geometry without functional transfer may be descriptive only.

### H4 — Same-architecture spectral structure is a positive control, not proof of H1-H3

For multiple LoRAs or fine-tunes of one base architecture, reproduce a lightweight UWSH-style analysis:

- center weight deltas;
- decompose by SVD/HOSVD or per-module PCA;
- measure explained variance;
- compare to matched random low-rank updates;
- explicitly compare against the **mean-update baseline**;
- test held-out functional behavior, not reconstruction alone.

This establishes that the harness can detect genuine shared structure where aligned coordinates actually exist.

## Experimental design

### Model panel

Use at least four distinct architecture families if locally practical, ideally with two sizes or checkpoints in at least one family.

Candidate families:

- Qwen
- Llama
- Gemma
- Mistral
- Phi

The experiment should not depend on any specific set. Configuration records exact model hashes, quantization, tokenizer, layer choice, pooling rule, and runtime.

### Anchor bank

Build a versioned bank of 500-2,000 paired semantic probes spanning multiple domains:

- logic and quantification;
- arithmetic and magnitude;
- temporal relations;
- spatial relations;
- causality;
- physical properties;
- social relations;
- epistemic concepts;
- lexical synonym/antonym controls;
- nonsense/random-string controls.

Each anchor should have multiple surface forms so the unit of comparison is not a single token spelling.

Split by **concept and domain**, not merely by paraphrase:

- alignment/train anchors;
- in-domain held-out anchors;
- held-out domains;
- adversarial controls.

A mapping that only learns the particular vocabulary used during fitting does not count as evidence of universality.

### Representation extraction

For each model and prompt:

- capture selected hidden states from several normalized depth positions, e.g. 25%, 50%, 75%, final;
- record token-level and pooled variants;
- normalize consistently;
- preserve native dimensionality;
- never truncate unrelated spaces merely to make vector dimensions match.

Tokenization differences are part of the system and must be recorded. Concept representations should use explicit pooling rules rather than assuming matching token boundaries.

## Metric stack

### Level 0: raw-coordinate diagnostics

Record native vector norms/dimensions and, only where coordinates are genuinely shared, raw cosine.

For unrelated architectures, raw cosine is reported only as a demonstration of why it is not an invariant metric.

### Level 1: coordinate-free relational metrics

**Priority order after the PRH re-analysis:** local neighborhood identity/overlap first; semantic triplet/relation tests second; global RDM/CKA as supporting diagnostics. Raw global magnitudes must be calibrated against appropriate width/layer-selection nulls.

Use sample-by-sample geometry instead of coordinate identity:

- mutual/local k-nearest-neighbor overlap (primary);
- CKNNA-style local neighborhood comparison where implementation is auditable;
- predeclared semantic triplet/relation conservation;
- representational dissimilarity matrices (RDMs);
- rank correlation between pairwise-distance matrices;
- linear CKA on Gram matrices;
- nearest-neighbor overlap;
- triplet-order consistency;
- graph/topology preservation.

An orthogonal rotation of one representation should leave these metrics essentially unchanged. That is a **positive invariance control**, not a null.

Nulls include:

- shuffled anchor identities;
- randomly paired models/probes;
- covariance-matched random representations;
- semantic-label permutation;
- domain-mismatched anchors.

### Level 2: learned alignment

Fit alignment on train anchors only.

Primary tests on held-out data:

- top-1 / top-k cross-model concept retrieval;
- mean reciprocal rank;
- neighborhood preservation;
- held-out RDM/CKA after mapping;
- performance on an entirely held-out semantic domain.

Report performance against:

- chance;
- shuffled labels;
- overparameterized memorization baseline;
- same-architecture checkpoint pair;
- unrelated/random representation.

### Level 3: functional transfer

Identify a direction or low-rank intervention in source model A from a controlled task. Map it into target model B using only the alignment learned above.

Measure:

- signed change in target behavior;
- effect size;
- rank correlation of per-example effects;
- success across multiple source/target directions;
- stability across prompt paraphrases.

A representation claim is strongest if geometry and behavior agree. If geometry aligns but effects do not transfer, report that explicitly.

## Controls that must be present

1. **Identity control:** same model, same checkpoint.
2. **Same-family control:** same architecture, different fine-tune/checkpoint.
3. **Orthogonal-rotation control:** geometry preserved, coordinates destroyed.
4. **Permutation null:** anchor identities shuffled.
5. **Random-feature null:** matched shape/covariance where practical.
6. **Mean-update baseline:** mandatory for UWSH/LoRA tests.
7. **Held-out-domain split:** prevents dictionary memorization.
8. **Surface-form robustness:** multiple prompts per concept.
9. **Layer sweep:** prevents cherry-picking one favorable layer.
10. **Pre-registered primary metrics:** avoid selecting whichever statistic looks best afterward.

## Success criteria

AI-DNA 2.0 should use graded conclusions rather than a single "DNA score."

### Evidence for relational universality

Require all of:

- relational metrics significantly above permutation/random nulls;
- replication across at least three independent architecture pairs;
- persistence on held-out concepts;
- at least one held-out semantic domain;
- robustness across paraphrases and more than one layer region.

### Evidence for transferable alignment

Require:

- held-out concept retrieval materially above chance and nulls;
- mapping capacity small enough that memorization is implausible;
- result replicated across more than one mapping method or architecture pair.

### Evidence for functional universality

Require:

- mapped interventions produce the same signed behavioral effect in the target model;
- effect exceeds random-direction and shuffled-mapping controls;
- result replicates over multiple concept/task directions and model pairs.

### Falsification outcomes are valid outcomes

The arc succeeds scientifically even if it concludes:

- only same-architecture structure is measurable;
- relational similarity exists but no useful mapping generalizes;
- mappings generalize geometrically but functional interventions do not;
- apparent universality collapses under held-out-domain controls.

No result should be promoted from "interesting" to "universal" without those controls.

## Phases

### Phase A — Retrospective sanity reproduction

Reproduce the old truncation-plus-cosine method on synthetic data where one space is an arbitrary orthogonal rotation of the other.

Expected result:

- semantics/geometry are identical;
- raw coordinate cosine changes dramatically;
- RDM/CKA remain invariant.

This provides a concrete demonstration of the methodological correction.

### Phase B — Same-architecture positive control

Reproduce a bounded UWSH-style test on one accessible family/LoRA bank.

Questions:

- how sharp is the spectrum versus random low-rank noise?
- how much is explained by the mean update alone?
- does the basis help held-out tasks functionally?
- does geometric reconstruction predict functional performance?

### Phase C — Cross-architecture relational geometry

Run the anchor bank across the model panel and compare RDM, CKA, neighborhood, and triplet structure without fitting a coordinate mapping.

### Phase D — Held-out latent alignment

Fit Procrustes/linear/CCA-style mappings on training anchors. Test unseen concepts and unseen domains.

### Phase E — Functional transfer

Transfer controlled latent directions between architecture pairs. This is the strongest test and should be attempted only after Phase D demonstrates genuine held-out generalization.

### Phase F — SAGE relevance

Only after results exist, evaluate whether any discovered invariants are useful for SAGE:

- dictionary entities;
- node-to-node semantic handshakes;
- architecture adapters;
- latent-state translation;
- model-independent memory indexing.

SAGE integration is an application question, not part of the evidence for universality.

## What this experiment does **not** claim

AI-DNA 2.0 does not use "consciousness" as an explanatory variable and does not treat representational similarity as evidence of consciousness.

It does not assume that common training corpora imply a universal internal language.

It does not assume that a successful nonlinear translator proves shared native representations; a sufficiently expressive translator can simply learn a new code.

It does not erase the 2025 work. The archive records the actual path by which the question evolved.

## References

- Kaushik, P. et al. (2025), *The Universal Weight Subspace Hypothesis*, arXiv:2512.05117: https://arxiv.org/abs/2512.05117
- Official UWSH code: https://github.com/toshi2k2/unisub
- Independent replication/stress test: https://github.com/cahlen/universal-subspace-stress-test
- Project WIZORB: https://github.com/stevenAthompson/WIZORB
- Historical AI-DNA experiments: \`../../archive/\`

## Next artifact

See \`PRD.md\` for the implementation contract, file layout, datasets, metrics, and staged acceptance criteria.
