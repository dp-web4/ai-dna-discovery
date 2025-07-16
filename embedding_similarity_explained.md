# Embedding Similarity Calculation: Technical Details

## Overview

This document provides a detailed explanation of how we calculate embedding similarity scores in the AI DNA Discovery project. Understanding these calculations is crucial for interpreting our results and reproducing our experiments.

## What Are Embeddings?

Embeddings are dense vector representations of text that capture semantic meaning. Each model converts input text into a high-dimensional vector (typically 1024-4096 dimensions) that represents the text in the model's "understanding space."

### Example Dimensions
- **phi3:mini**: 3072 dimensions
- **gemma:2b**: 2048 dimensions
- **tinyllama**: 2048 dimensions
- **deepseek-coder:1.3b**: 2048 dimensions

## Cosine Similarity Formula

We use cosine similarity to measure the similarity between two embedding vectors. This metric is preferred because it:
- Normalizes for vector magnitude
- Ranges from -1 to 1 (we typically see 0 to 1 for embeddings)
- Captures directional similarity regardless of vector length

### Mathematical Formula

```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)

Where:
- A · B = dot product of vectors A and B
- ||A|| = Euclidean norm (magnitude) of vector A
- ||B|| = Euclidean norm (magnitude) of vector B
```

### Python Implementation

```python
import numpy as np

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    # Handle dimension mismatch by truncating to minimum
    min_dim = min(len(vec1), len(vec2))
    vec1 = vec1[:min_dim]
    vec2 = vec2[:min_dim]
    
    # Calculate dot product
    dot_product = np.dot(vec1, vec2)
    
    # Calculate norms
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    # Avoid division by zero
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # Calculate cosine similarity
    similarity = dot_product / (norm1 * norm2)
    
    return similarity
```

## Cross-Model Similarity Calculation

When comparing embeddings across different models, we follow this process:

### 1. Generate Embeddings
```python
# For each model, generate embedding for the same input
embeddings = {}
for model in models:
    embeddings[model] = get_embedding(model, input_text)
```

### 2. Calculate Pairwise Similarities
```python
# Compare all pairs of models
similarities = []
for i, model1 in enumerate(models):
    for j, model2 in enumerate(models):
        if i < j:  # Avoid duplicate pairs
            sim = cosine_similarity(
                embeddings[model1], 
                embeddings[model2]
            )
            similarities.append(sim)
```

### 3. Aggregate Results
```python
# Calculate mean and standard deviation
mean_similarity = np.mean(similarities)
std_similarity = np.std(similarities)
```

## Handling Dimension Mismatches

Different models produce embeddings of different dimensions. We handle this by:

1. **Truncation to minimum dimension**: We take the first N dimensions where N = min(dim1, dim2)
2. **Rationale**: The most important semantic information is typically captured in the earlier dimensions
3. **Alternative approaches** (not used):
   - Zero-padding: Would artificially inflate similarity
   - Projection: Would require training a projection matrix

## Interpretation Guide

### Similarity Score Ranges

- **1.0**: Identical vectors (perfect alignment)
- **0.9-1.0**: Extremely high similarity (rare across different models)
- **0.7-0.9**: High similarity (significant semantic alignment)
- **0.5-0.7**: Moderate similarity
- **0.3-0.5**: Low similarity
- **0.0-0.3**: Very low similarity (essentially unrelated)
- **< 0.0**: Negative correlation (vectors point in opposite directions)

### What the Scores Mean

1. **DNA Score (1.0)**: When we report a "perfect 1.0 DNA score," this means the cosine similarity between embeddings is extremely close to 1.0 (typically > 0.99)

2. **Cross-Model Average**: When comparing a pattern across 6 models, we calculate all 15 pairwise similarities and report the mean

3. **Within-Model Consistency**: Some patterns show high similarity when compared within the same model across different prompts

## Statistical Considerations

### Baseline Establishment

We establish baselines using:
- **Random strings**: "xqz7", "bflm9", "####"
- **Common words**: "the", "and", "hello"
- **Null hypothesis**: Random strings should show low, inconsistent similarity

### Significance Testing

For a pattern to be considered significant:
1. Mean similarity should exceed baseline by at least 2 standard deviations
2. Consistency across multiple model pairs
3. Reproducibility across multiple runs

## Example Calculation

Let's walk through a concrete example:

```python
# Pattern: "∃" (existence quantifier)
# Models: phi3 and gemma

# Step 1: Get embeddings
phi3_embedding = get_embedding("phi3:mini", "∃")
# Returns: [0.234, -0.567, 0.123, ...] (3072 dimensions)

gemma_embedding = get_embedding("gemma:2b", "∃")  
# Returns: [0.245, -0.578, 0.134, ...] (2048 dimensions)

# Step 2: Truncate to minimum dimension (2048)
phi3_embedding = phi3_embedding[:2048]

# Step 3: Calculate cosine similarity
similarity = cosine_similarity(phi3_embedding, gemma_embedding)
# Returns: 0.967

# Step 4: Interpretation
# 0.967 indicates very high similarity between how phi3 and gemma
# represent the existence quantifier symbol
```

## Validation Methods

### 1. Consistency Checks
- Run the same pattern multiple times
- Verify similar scores (within ±0.05)

### 2. Control Patterns
- Test known similar patterns (synonyms)
- Test known different patterns (antonyms)
- Verify expected relationships

### 3. Model Behavior
- Test across different prompt contexts
- Ensure embedding stability

## Common Pitfalls

1. **Not normalizing vectors**: Raw dot product without normalization
2. **Ignoring dimension mismatch**: Comparing vectors of different lengths
3. **Over-interpreting small differences**: 0.95 vs 0.96 may not be meaningful
4. **Context effects**: Embeddings can vary based on surrounding context

## Reproducibility

To reproduce our similarity calculations:

1. Use the exact model versions specified
2. Apply the same preprocessing (no extra spaces, same encoding)
3. Use the truncation method for dimension mismatches
4. Calculate all pairwise similarities for cross-model scores

## Code References

Key implementation files:
- Similarity calculation: `multi_model_dna_test.py:148-165`
- Embedding extraction: `continuous_ai_dna_experiment.py:89-102`
- Statistical analysis: `embedding_space_mapper.py:211-234`

## Further Reading

- [Cosine Similarity in NLP](https://en.wikipedia.org/wiki/Cosine_similarity)
- [Understanding Word Embeddings](https://arxiv.org/abs/1301.3781)
- [Sentence Embeddings Survey](https://arxiv.org/abs/1908.10084)

---

*For questions or clarifications about these calculations, please refer to the main project documentation or open an issue on GitHub.*