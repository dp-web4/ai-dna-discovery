# Important: File Naming Convention

**Issue:** Windows systems cannot handle filenames containing colons (`:`)

**Problem files:** 
- `embedding_space_2d_phi3:mini_pca.png` 
- `pattern_distance_matrix_gemma:2b.png`
- etc.

**Solution for future files:**
Replace `:` with `_` in all filenames

**Examples:**
- ❌ `phi3:mini` → ✅ `phi3_mini`
- ❌ `gemma:2b` → ✅ `gemma_2b`
- ❌ `tinyllama:latest` → ✅ `tinyllama_latest`

**Note:** This affects cross-platform compatibility when cloning the repository on Windows machines.

---
*Added: July 13, 2025*