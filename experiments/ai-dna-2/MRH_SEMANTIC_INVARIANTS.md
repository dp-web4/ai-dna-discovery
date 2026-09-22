# AI-DNA 2.0 — MRH-Semantic Invariants

**Status:** research note / experiment refinement  
**Date:** 2026-09-22  
**Context:** follows the first pub cross-architecture pilot on Qwen2.5-0.5B, Pythia-410M, and TinyLlama-1.1B.

## Why this refinement is needed

The first AI-DNA 2.0 pilot produced non-random cross-model relational geometry under RDM Spearman and linear CKA. That is useful evidence that the harness is detecting structure, but it is **not yet evidence that the shared structure is semantic**.

The distinction is essential.

A statistically significant correspondence can arise from shared nuisance structure:

- prompt template;
- sequence length;
- token position;
- tokenizer behavior;
- word frequency;
- grammatical category;
- common training distribution;
- sentence morphology;
- other regularities unrelated to the meaning we care about.

Therefore:

> **A metric is not meaningful merely because it is significant. Its semantic interpretation must be specified by the experimental manipulation that produced it.**

RDM, CKA, retrieval scores, p-values, and related statistics are measuring instruments. They are not the phenomenon.

The target phenomenon is architecture-independent structure in how coherent meanings are represented, distinguished, stabilized, and transformed.

## The semantic unit: an MRH-bounded concept state

The original pilot uses a fixed prompt:

```text
Concept: X
Definition: Y
Represent the meaning of this concept.
```

and reads the hidden state of the final token.

That state compresses the preceding sequence, but there is no reason to assume it cleanly equals "the representation of concept X."

AI-DNA 2.0 should instead define its unit using the **Markov Relevancy Horizon (MRH)** principle:

> **A semantic representation is the latent state associated with the smallest context horizon sufficient to instantiate a coherent concept, relation, or meaning for the current task.**

The horizon is semantic, not a fixed token count.

Example:

- `bank`
- `deposit money at the bank`
- `sat on the river bank`

The surface token may be identical while the coherent concept differs. The relevant representation emerges only after enough context enters the horizon to resolve the meaning.

Thus the object of study is not "the vector for the token `bank`."

It is:

> **the latent state after the model has crossed a semantic closure boundary for FINANCIAL-BANK or RIVER-BANK.**

Different models may cross that boundary at different token positions, using different tokenizers and entirely different native coordinates.

That is acceptable. The experiment asks what survives those implementation differences.

## Three primary semantic quantities

AI-DNA 2.0 should replace any temptation toward a single "DNA score" with three explicitly interpretable quantities.

### 1. Semantic invariance

**Question:** Does the latent representation remain stable when surface form changes but meaning does not?

Examples:

- synonym substitution;
- paraphrase;
- altered word order;
- equivalent definition;
- irrelevant surrounding context outside the semantic MRH.

Operationally, for two meaning-preserving presentations `c_1` and `c_2` of semantic entity `s`:

```text
D(z_s(c_1), z_s(c_2))
```

should be small relative to distances to semantically distinct entities.

Cross-architecture evidence exists if models preserve the same **relational neighborhood** under those changes even though their raw coordinates differ.

### 2. Semantic sensitivity

**Question:** Does the latent representation change when meaning-relevant evidence enters the MRH?

Examples:

- disambiguating `bank` as financial versus river;
- changing `before` to `after`;
- changing causal direction;
- adding a negation that changes the proposition;
- changing the referent of an ambiguous pronoun.

A useful representation should not be invariant to everything.

The key test is selective sensitivity:

> irrelevant evidence outside the MRH should have little effect; meaning-changing evidence inside the MRH should have a large, structured effect.

### 3. Relational conservation

**Question:** Do semantic relationships or transformations survive across architectures?

Examples:

- cause → effect;
- before → after;
- inside → outside;
- true → false;
- evidence → belief update;
- object → category.

The common invariant may not be a universal concept vector at all.

It may instead be a **universal relation or transformation among semantic states**.

This possibility is especially important because it is coordinate-independent by construction and is closer to what cross-agent communication actually requires.

## MRH-semantic probe families

Each semantic target should be tested through controlled probe families rather than a single sentence.

| Probe family | Manipulation | Desired behavior |
|---|---|---|
| Paraphrase | surface form changes, meaning preserved | latent/relational state stable |
| Outside-MRH noise | irrelevant context changes | state stable |
| Inside-MRH evidence | meaning-relevant context added | state changes appropriately |
| Meaning flip | lexically similar prompt, semantic target changes | state separates |
| Ambiguity resolution | context progressively disambiguates target | state converges after semantic closure |
| Relation transform | one semantic relation changes to another | transformation is structurally reproducible |

These manipulations give numerical metrics an interpretation.

Without them, a high RDM or CKA value means only "the geometries covary."

With them, we can ask whether the covariance tracks meaning.

## Progressive context / semantic closure experiment

MRH itself may be an architecture-independent observable.

For an ambiguous or underspecified target:

1. present minimal context;
2. add context incrementally;
3. extract latent state after each increment;
4. measure changes in semantic neighborhood or classifier/probe behavior;
5. identify the point at which further irrelevant context no longer materially changes the semantic state.

Call this the **semantic closure horizon**.

Example:

```text
bank
the bank
the bank approved
the bank approved the loan
the bank approved the loan after reviewing the applicant
```

versus:

```text
bank
the bank
the bank was covered
the bank was covered with reeds
the river bank was covered with reeds
```

The hypotheses are not that models use the same tokens or layer coordinates.

They are:

- the same evidence causes the same semantic branch;
- irrelevant additions after closure have diminishing effect;
- the resulting concept enters comparable relational neighborhoods;
- the closure horizon may have comparable structure across architectures.

A cross-architecture similarity in **when meaning becomes determined** may itself be more informative than coordinate similarity.

## Interpreting the current metrics

### RDM Spearman

RDM Spearman measures whether pairwise distances among samples are ranked similarly in two spaces.

It is useful because it ignores coordinate identity.

It does **not** tell us why the geometry is similar.

Semantic interpretation requires probe-controlled comparisons.

### Linear CKA

Linear CKA measures similarity between the sample-level Gram structures induced by two representation matrices.

It is invariant to orthogonal feature rotations and isotropic scaling.

It is useful for asking whether two models organize a set of samples similarly.

It does **not** establish that the shared organization encodes the intended semantic variable.

### Permutation p-value

The current identity permutation asks whether the observed correspondence exceeds what occurs when sample identities are scrambled.

It tests non-random correspondence.

It does **not** distinguish semantic identity from shared nuisance variables that remain tied to the sample.

Therefore a small p-value is evidence against the permutation null, not evidence for semantic universality.

### Token length and other metadata

Token count, character count, frequency, syntax, and related quantities should be retained only as **nuisance metadata / confound controls**.

They are not targets.

If controlling one of them destroys an apparent semantic result, that is a useful falsification.

The experiment should never optimize for correspondence in these variables.

## Required next controls

Before interpreting the pub pilot as semantic evidence:

### A. Remove the common suffix dependency

The current last-token extraction reads the state after the identical instruction:

`Represent the meaning of this concept.`

Add extraction conditions that do not share a fixed semantic instruction tail.

Candidates:

- representation at the end of the actual concept span;
- representation immediately after the disambiguating semantic chunk;
- multiple prompt templates;
- mean/attention-weighted pooling over a declared semantic span;
- closure-point extraction in progressive contexts.

### B. Surface-form replication

For each concept, create multiple paraphrases whose token counts and syntax vary substantially.

Require the model to preserve semantic neighborhood despite those differences.

### C. Meaning-flip minimal pairs

Build prompts that share most lexical material but differ semantically.

Examples:

- `A happened before B` / `A happened after B`;
- `X caused Y` / `Y caused X`;
- financial `bank` / river `bank`;
- affirmative / negated proposition.

If the representation follows lexical overlap instead of the meaning flip, the candidate invariant fails.

### D. Outside-MRH distractors

Add semantically irrelevant material before and after the target context.

A robust semantic representation should resist these distractors after closure.

### E. Confound regression / partial analysis

Use token counts and other nuisance variables only to ask:

> how much of the cross-model relation remains after the nuisance structure is removed?

Partial correlations or residualized RDMs may be useful, but these remain secondary instruments. The semantic manipulations above are stronger evidence.

## Stronger success criterion

A candidate cross-architecture invariant should not advance merely because:

```text
RDM correlation > null
CKA > null
p < threshold
```

It should advance only if the same structure shows the correct **semantic response profile**:

1. **invariant** under meaning-preserving surface changes;
2. **insensitive** to outside-MRH distractors;
3. **sensitive** to meaning-changing evidence inside the MRH;
4. **separates** lexical near-neighbors with different meanings;
5. **preserves** semantic relations or transformations across architectures;
6. generalizes to held-out concepts and domains.

Only then do RDM, CKA, retrieval, and alignment scores become evidence about a semantically defined phenomenon.

## Possible deepest outcome

The original AI-DNA intuition may have been framed too much like a search for common symbols or common vectors.

A more plausible universal object is:

> **a shared topology of semantic state transitions under relevant evidence.**

Each architecture may implement that topology in an unrelated coordinate system.

If so, "AI DNA" would not be an identical latent alphabet.

It would be a conserved set of rules for how coherent meanings form, differentiate, relate, and transform.

MRH provides a way to test that directly.

## Immediate experiment revision

Treat the first pub pilot as:

> **Phase C0: geometry instrumentation validation — pre-semantic.**

Do not discard it. It established that:

- coordinate-safe extraction works;
- cross-family geometries contain reproducible non-random structure;
- the pipeline can identify nuisance variables;
- the next experiment needs semantic interventions.

Add a new phase before claim-grade relational universality:

> **Phase C1: MRH-semantic invariance/sensitivity experiment.**

C1 should precede learned cross-architecture alignment. There is little value in learning a mapping between spaces until we know which structure in those spaces tracks meaning rather than formatting.

## Relationship to SAGE

This refinement also fits SAGE more naturally than a generic latent translator.

If successful, the useful primitive for SAGE would be an architecture-neutral **semantic entity plus relevance horizon**, carrying:

- what concept/relation is currently coherent;
- what evidence established it;
- what context lies inside versus outside its MRH;
- what transformations preserve identity;
- what transformations change meaning;
- confidence/provenance.

That could eventually inform:

- dictionary entities;
- memory retrieval;
- model-to-model semantic handshakes;
- context packing;
- cross-model transfer;
- semantic witness comparison.

But AI-DNA 2.0 should establish the phenomenon independently before SAGE consumes it.
