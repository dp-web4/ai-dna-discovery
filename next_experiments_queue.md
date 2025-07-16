# Next Experiments Queue

## Ready to Run

### 1. Layer Dynamics Investigation (45 mins)
```bash
python layer_dynamics_experiment.py
```
**Theory**: Models may exhibit runtime neural plasticity  
**Focus**: deepseek-coder as primary candidate  
**What to look for**: Changes in embedding variability or response consistency

### 2. Multi-Model Handshakes (2-3 hours)
```bash
python multi_model_handshake_experiment.py  # Need to create this
```
**Goal**: Test 3+ models simultaneously  
**Hypothesis**: gemma could mediate between incompatible pairs  
**Key metric**: Can 3 models reach consensus when 2 cannot?

### 3. Extended Convergence Push (4-6 hours)
```bash
python extended_convergence_experiment.py  # Need to create this
```
**Target**: Push 0.402 → 0.7+ convergence  
**Method**: 1000+ iteration handshakes on successful pairs  
**Question**: Does convergence plateau or continue improving?

### 4. Information-Divergence Validation (2-3 hours)
```bash
python information_divergence_test.py  # Need to create this
```
**Goal**: Test if similarity ∝ 1/Information(pattern)  
**Method**: Create patterns with controlled information content  
**Range**: 1-bit to N-bit patterns

### 5. Temporal Dynamics Study (24 hours)
```bash
python temporal_dynamics_study.py  # Need to create this
```
**GPT's suggestion**: Test time-of-day effects  
**Method**: Run same tests every hour for 24 hours  
**Also test**: Memory decay over extended periods

## Prioritization

Based on potential impact and current discoveries:

1. **Layer Dynamics** - Most revolutionary if true
2. **Multi-Model Handshakes** - Natural next step from pair success
3. **Information-Divergence** - Validate our key theoretical discovery
4. **Extended Convergence** - Push the limits of what we've found
5. **Temporal Dynamics** - Longer term study

## Hardware Considerations

With RTX 4090 and i9-13900HX:
- Can run multiple experiments in parallel
- Layer dynamics won't stress GPU much
- Save GPU for embedding-heavy experiments

## Data to Collect

For each experiment:
- Raw JSON data
- Statistical summaries  
- Visualizations (PNG)
- Markdown reports
- Any anomalies or unexpected behaviors

## Success Criteria

We're looking for:
- Statistically significant deviations from baseline
- Reproducible effects
- Model-specific differences
- Patterns that challenge our current understanding

---

*Ready to explore the unknown edges of AI behavior!*