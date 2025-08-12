# Vectorized Trust (T3/V3)

- Represent trust as a **vector** keyed by traits/topics/roles.
- Reduce to a scalar gate with policy (weighted_sum, geometric_mean, min_gate, logistic).
- Store per-feature or per-source vectors; expose `trust_scalar(topic|role)` via reducer.

See: `coherence_engine/trust/trust_vector.py`.
