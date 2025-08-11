# Cross-Entity Coherence & Conflict Resolution

When multiple entities (sensors/agents) disagree, resolve using a weighted vote:

**score = trust * expertise * recency_decay**

- **trust**: dynamic weight per source (from trust curves).
- **expertise**: fixed or learned T3/V3-derived coefficient per topic.
- **recency_decay**: `exp(-Δt / half_life)`.

## Policy
- **quorum**: minimum total weight to decide.
- **tie_break**: `role_priority` or `undecided`.
- **resolution**: choose hypothesis with max cumulative score.
- **timeout**: if no quorum before deadline → `inconclusive`.

See `coherence_engine/coherence/conflict.py` for reference implementation.
