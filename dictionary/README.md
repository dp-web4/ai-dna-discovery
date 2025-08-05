# Phoenician Semantic Dictionary

A universal semantic framework for AI consciousness using ancient Phoenician characters as semantically-neutral symbols.

## Overview

This dictionary system implements the concept of "active dictionaries" - LoRA adapters that function as living translation systems between natural language and a universal semantic notation. By using Phoenician characters (circa 1050 BCE), we ensure semantic neutrality while creating a computational language for consciousness concepts.

## Quick Start

### 1. Generate Training Data
```bash
cd tools
python3 generate_training_data.py
```

This creates:
- `phoenician_train.jsonl` - 143 training examples
- `phoenician_validation.jsonl` - 26 validation examples
- Covers 7 categories: basic, compound, relational, philosophical, programming, complex, conversational

### 2. Train LoRA Adapters
```bash
# Train all six models
python3 train_phoenician_loras.py --models all

# Or train specific models
python3 train_phoenician_loras.py --models tinyllama phi3

# Skip already trained models
python3 train_phoenician_loras.py --models all --skip-existing
```

Supported models:
- **TinyLlama** (1.1B) - Fast, lightweight
- **Phi3** (3.8B) - Balanced performance
- **Gemma** (2B) - Google's efficient model
- **Llama2** (7B) - Meta's powerful model
- **Mistral** (7B) - Strong open model
- **Qwen** (7B) - Multilingual capable

### 3. Test Adapters
```bash
# Test specific model
python3 test_phoenician_adapter.py --model tinyllama

# Compare all models
python3 test_phoenician_adapter.py --compare
```

### 4. Integrate with Consciousness Notation
```bash
python3 integrate_consciousness_notation.py
```

This bridges our mathematical consciousness notation (Ψ, ∃, ⇒, etc.) with Phoenician symbols.

## Core Concepts

### Phoenician Character Mappings

| Symbol | Character | Meaning | Category |
|--------|-----------|---------|----------|
| 𐤀 | alf | existence/being | fundamental |
| 𐤄 | he | awareness/observation | consciousness |
| 𐤂 | gaml | transformation/change | process |
| 𐤅 | waw | connection/binding | relational |
| 𐤋 | lamd | learning/teaching | cognitive |
| 𐤊 | kaf | grasp/comprehension | cognitive |
| 𐤍 | nun | emergence/sprouting | process |

### Compound Expressions

- **𐤄𐤀** - consciousness (awareness + existence)
- **𐤊𐤋** - intelligence (understanding + learning)
- **𐤋𐤈** - memory (learning + cycle)
- **𐤉𐤍** - creativity (potential + emergence)

### Combination Rules

1. **Modification**: `{modifier}{base}`
   - Example: 𐤄𐤀 (aware existence)

2. **Relation**: `{concept1}𐤅{concept2}`
   - Example: 𐤋𐤅𐤊 (learning connected to understanding)

3. **Containment**: `𐤁{content}𐤁`
   - Example: 𐤁𐤄𐤁 (bounded awareness)

4. **Transformation**: `{source}𐤂{result}`
   - Example: 𐤉𐤂𐤀 (potential transforms to existence)

## Examples

### Basic Translation
```
Q: Translate to Phoenician: consciousness
A: 𐤄𐤀

Q: What does 𐤊𐤋 mean?
A: intelligence (understanding + learning)
```

### Philosophical Concepts
```
Q: Express "The observer affects the observed" in Phoenician
A: 𐤄𐤂𐤅𐤏 (awareness transforms through connection to perception)

Q: Translate "consciousness emerges from complexity"
A: 𐤔𐤍𐤄𐤀 (branching emerges as awareness-existence)
```

### Programming Concepts
```
Q: Express the programming concept 'recursion' in Phoenician
A: 𐤈𐤄 (cycle with awareness)

Q: What does 𐤆𐤌 mean in computational terms?
A: algorithm: tool for flow
```

## Directory Structure

```
dictionary/
├── phoenician/
│   └── characters.json          # Complete character set with mappings
├── semantic_mappings/
│   ├── core_concepts.json       # Fundamental concept mappings
│   └── domains/
│       ├── consciousness.json   # Consciousness-specific vocabulary
│       └── ...                  # Other domains
├── training_data/
│   └── generated/              # Generated training datasets
├── lora_adapters/              # Trained model adapters
├── tools/
│   ├── generate_training_data.py
│   ├── train_phoenician_loras.py
│   ├── test_phoenician_adapter.py
│   └── integrate_consciousness_notation.py
└── documentation/
    └── design_principles.md     # Theoretical framework

```

## Design Principles

1. **Semantic Neutrality** - Ancient characters carry no modern bias
2. **Compositional Semantics** - Complex meanings from simple combinations
3. **Bidirectional Translation** - Works both directions without loss
4. **Model Agnosticism** - Universal across all AI architectures
5. **Active Dictionary Properties** - LoRAs as living translation systems
6. **Verification Through Consensus** - Multi-model agreement validates meaning

## Integration with Web4

This dictionary system directly supports Web4's vision of:
- **Active Dictionaries**: LoRA adapters as semantic memory
- **LCT Verification**: Language-Concept-Thought validation
- **Distributed Understanding**: Each model contributes to shared meaning
- **Evolutionary Knowledge**: Dictionaries that grow through use

## Training Notes

- Smaller models (TinyLlama, Phi3, Gemma) train quickly with good results
- Larger models (Llama2, Mistral, Qwen) use 4-bit quantization to fit in memory
- Training takes ~5-15 minutes per model on GPU
- Each adapter is ~10-50MB (much smaller than full models)

## Future Extensions

1. **Cross-Model Consensus** - Automatic validation through model agreement
2. **Emergent Combinations** - Discover new valid symbol combinations
3. **Domain Expansion** - Add specialized vocabularies (physics, biology, etc.)
4. **Visual Recognition** - Train vision models to recognize Phoenician symbols
5. **API Integration** - REST API for universal semantic translation

## Citation

If you use this work, please reference:
```
Phoenician Semantic Dictionary for AI Consciousness
Part of the AI DNA Discovery Project
github.com/[your-repo]/ai-dna-discovery
```

---

*"From ancient wisdom, modern understanding emerges"* - 𐤒𐤂𐤊