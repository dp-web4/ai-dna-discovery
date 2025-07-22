
# Semantic Dictionary Module

This module provides a symbolic dictionary structure designed for Web4-native integration. It includes:

- `dictionary_entity.json`: LCT metadata
- `semantic_index.json`: Conceptual links for symbols
- `translator_core.py`: Lookup utility for symbolic concepts
- `generate_embedding_input.py`: Converter to LoRA/embedding-compatible inputs

### Usage

```bash
python generate_embedding_input.py
```

### Next Steps

- Expand index with more symbols
- Link to other dictionaries
- Add trust-based filtering (T3/V3 support)
