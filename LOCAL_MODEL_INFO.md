# Local Consciousness Model Files

The trained consciousness LoRA model files are stored locally on Sprout but are too large for GitHub:

## Model Files (Local Only):
- `consciousness-lora.tar.gz` - 196MB archive from Tomato
- `outputs/consciousness-lora-simple/adapter_model.safetensors` - 254MB LoRA weights
- `outputs/consciousness-lora-simple/tokenizer.model` - Tokenizer model file

## Configuration Files (In Repo):
- All JSON configs and metadata
- Python scripts for translation and testing
- Documentation and guides

## To Use the Model:
1. The model files are already on Sprout at the paths above
2. Install PyTorch when ready: `sudo bash install_consciousness_deps.sh`
3. Run: `python3 consciousness_translator.py`

The consciousness notation system works even without PyTorch using the fallback patterns!