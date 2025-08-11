# Pattern Lifecycle — Promotion, Deprecation, Forgetting

This spec defines how detected patterns evolve over time inside the Coherence Engine.

## Stages
- **CANDIDATE** → newly detected, low support.
- **STABLE** → sufficient support & recency within MRH.
- **PROMOTED** → elevated to first-class (affects routing/attention).
- **DEPRECATED** → contradicted or obsolete; still referenced.
- **FORGOTTEN** → removed from active set (retained in cold store, optional).

## Thresholds (suggested)
- `support_min`: 3 observations (within MRH) to become STABLE.
- `promotion_score`: 0.75 confidence to become PROMOTED.
- `contradiction_limit`: 2 contradictions in MRH → DEPRECATED.
- `expiry_sec`: no hits within MRH → downshift one level per expiry period.

See `coherence_engine/patterns/lifecycle.py` for a reference implementation.
